"""DSpark / DFlash Draft Model 时间建模

建模 DSpark 三阶段 draft 时间:
1. Parallel backbone (DFlash): 1 次 forward pass, block_size tokens 并行处理
2. Sequential head (Markov/RNN): block_size 步自回归 sampling
3. Confidence head: 线性投影 (可忽略)

依赖层次: Layer 4
  输入: config (DisaggregatedConfig, ModelConfig, DeviceSKUConfig),
        execution (AnalyticalPredictor)
  输出: DraftModelPredictor (被 SpeculativeDecodingEngine 消费)
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from distlmsim.config import DeviceSKUConfig, DisaggregatedConfig, ModelConfig
from distlmsim.execution.execution_time_predictor import AnalyticalPredictor


@dataclass
class DraftTimeBreakdown:
    """Draft 阶段时间分解。"""

    parallel_time_ms: float = 0.0  # DFlash backbone 前向时间
    sequential_time_ms: float = 0.0  # Markov/RNN head 自回归时间
    confidence_time_ms: float = 0.0  # Confidence head 时间
    target_extraction_time_ms: float = 0.0  # 目标模型特征抽取时间
    total_time_ms: float = 0.0


class DraftModelPredictor:
    """Draft model 前向传播时间预测器。

    使用 Roofline 模型为小型 draft model (如 5 层, dim=512) 计算
    各阶段的执行时间。

    DSpark 架构:
    - 5 层 Transformer (与 DFlash 相同的 backbone)
    - Markov head: nn.Embedding(vocab, rank) + nn.Linear(rank, vocab)
    - Confidence head: nn.Linear(hidden, 1) + sigmoid
    - 从目标模型 tap 入 num_target_layer_ids 个中间层

    时间计算:
    - Parallel backbone: 1 次 forward, block_size * batch_size tokens
    - Sequential head: block_size 步, 每步 = embedding lookup + linear
    - Confidence head: 1 次 Linear(hidden, 1), 可忽略
    - Target extraction: cat + projection from target layers
    """

    def __init__(
        self,
        target_model: ModelConfig,
        device_config: DeviceSKUConfig,
        spec_config: DisaggregatedConfig,
    ):
        self._target_model = target_model
        self._device = device_config
        self._spec = spec_config

        # 创建 draft model 的 ModelConfig
        self._draft_model = ModelConfig(
            model_name=f"DSpark-Draft-{spec_config.draft_model_name}",
            num_layers=spec_config.draft_num_layers,
            num_q_heads=max(1, target_model.num_q_heads // (target_model.embedding_dim // spec_config.draft_embedding_dim)),
            num_kv_heads=max(1, target_model.num_kv_heads // (target_model.embedding_dim // spec_config.draft_embedding_dim)),
            embedding_dim=spec_config.draft_embedding_dim,
            num_experts=0,
            top_k_experts=0,
            vocab_size=target_model.vocab_size,
        )
        self._draft_predictor = AnalyticalPredictor(self._draft_model, device_config)

    @property
    def block_size(self) -> int:
        return self._spec.block_size

    @property
    def draft_model(self) -> ModelConfig:
        return self._draft_model

    def get_draft_time(
        self,
        num_tokens: int,
        batch_size: int,
        kv_cache_size: int,
    ) -> DraftTimeBreakdown:
        """计算一轮 draft 的完整时间 (ms)。

        Args:
            num_tokens: 每个请求 draft 的 token 数 (= block_size)
            batch_size: 当前 batch 中的请求数
            kv_cache_size: 平均 KV cache 大小

        Returns:
            DraftTimeBreakdown 包含各阶段时间
        """
        total_tokens = num_tokens * batch_size

        # 1. Parallel backbone: draft model forward (1 pass, all tokens parallel)
        et = self._draft_predictor.get_execution_time(
            num_tokens=total_tokens,
            batch_size=batch_size,
            kv_cache_size=kv_cache_size,
            is_prefill=False,
        )
        parallel_time = et.total_time * self._draft_model.num_layers

        # 2. Sequential head: Markov/RNN, block_size steps
        sequential_time = self._compute_sequential_head_time(num_tokens, batch_size)

        # 3. Confidence head: negligible
        confidence_time = self._compute_confidence_head_time(num_tokens, batch_size)

        # 4. Target model feature extraction
        extraction_time = self._compute_target_extraction_time(num_tokens, batch_size)

        total = parallel_time + sequential_time + confidence_time + extraction_time

        return DraftTimeBreakdown(
            parallel_time_ms=parallel_time,
            sequential_time_ms=sequential_time,
            confidence_time_ms=confidence_time,
            target_extraction_time_ms=extraction_time,
            total_time_ms=total,
        )

    def _compute_sequential_head_time(
        self, num_tokens: int, batch_size: int
    ) -> float:
        """计算 Markov/RNN head 的自回归时间 (ms)。

        Markov head (vanilla):
          W1: Embedding(vocab_size, rank) → lookup O(1)
          W2: Linear(rank, vocab_size) → matrix multiply
          FLOPs per token ≈ 2 * rank * vocab_size

        Markov head (gated):
          gate_proj: Linear(hidden + rank, rank) → gating
          + vanilla Markov
          FLOPs += 2 * (hidden + rank) * rank

        Markov head (rnn):
          joint_proj: Linear(2*rank + hidden, 3*rank) → GRU gates
          + vanilla Markov
          FLOPs += 2 * (2*rank + hidden) * 3*rank
        """
        vocab = self._target_model.vocab_size
        rank = self._spec.markov_rank
        hidden = self._spec.draft_embedding_dim
        head_type = self._spec.markov_head_type

        # Base: vanilla Markov (embedding + projection)
        flops_per_token = 2 * rank * vocab

        if head_type == "gated":
            flops_per_token += 2 * (hidden + rank) * rank + 2 * rank * vocab
        elif head_type == "rnn":
            flops_per_token += 2 * (2 * rank + hidden) * 3 * rank + 2 * rank * vocab

        total_flops = flops_per_token * num_tokens * batch_size

        device = self._device
        peak_flops = device.fp16_tflops * 1e12
        eta_c = 0.85
        compute_time_s = total_flops / (eta_c * peak_flops)

        # Memory-bound: embedding lookup (rank * 2 bytes per token)
        bytes_per_token = rank * 2
        memory_bw = device.memory_bandwidth_gbps * 1e9
        eta_m = 0.90
        memory_time_s = (bytes_per_token * num_tokens * batch_size) / (eta_m * memory_bw)

        return max(compute_time_s, memory_time_s) * 1e3  # s → ms

    def _compute_confidence_head_time(
        self, num_tokens: int, batch_size: int
    ) -> float:
        """计算 Confidence head 时间 (ms)。

        Confidence head: Linear(hidden, 1) + Sigmoid
        FLOPs per token = 2 * hidden * 1 = 2 * hidden
        非常轻量，通常 < 0.001 ms
        """
        hidden = self._spec.draft_embedding_dim
        total_flops = 2 * hidden * num_tokens * batch_size

        peak_flops = self._device.fp16_tflops * 1e12
        eta_c = 0.85
        return (total_flops / (eta_c * peak_flops)) * 1e3

    def _compute_target_extraction_time(
        self, num_tokens: int, batch_size: int
    ) -> float:
        """计算目标模型特征抽取时间 (ms)。

        从目标模型的 num_target_layer_ids 个中间层提取隐藏状态:
        1. cat([h_1, h_9, h_17, h_25, h_33]) → [B*T, num_layers*dim]
        2. Linear(num_layers * dim, draft_dim) → [B*T, draft_dim]
        """
        total_tokens = num_tokens * batch_size
        target_dim = self._target_model.embedding_dim
        num_layers = self._spec.num_target_layer_ids
        draft_dim = self._spec.draft_embedding_dim

        concat_dim = num_layers * target_dim
        flops = 2 * concat_dim * draft_dim * total_tokens

        peak_flops = self._device.fp16_tflops * 1e12
        eta_c = 0.85
        return (flops / (eta_c * peak_flops)) * 1e3


def load_sps_curve(path: str) -> Dict[int, float]:
    """加载 SPS(B) 曲线。

    Args:
        path: CSV 文件路径 (batch_size,steps_per_second)

    Returns:
        Dict mapping batch_size → steps_per_second
    """
    sps: Dict[int, float] = {}
    if not os.path.exists(path):
        return sps
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bs = int(row["batch_size"])
            rate = float(row["steps_per_second"])
            sps[bs] = rate
    return sps
