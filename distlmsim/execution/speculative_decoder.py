"""Speculative Decoding 模块

将投机解码的计算时间建模逻辑从 main.py 中抽取为独立模块。
负责:
- Draft model 单步 decode 时间计算
- Target model 验证 (K tokens) 时间计算
- 接受率采样
- 完整投机周期时间计算

依赖层次: Layer 4
  输入: config (ModelConfig, DisaggregatedConfig), entities (Request),
        execution (AnalyticalPredictor), context (SimContext)
  输出: SpeculativeDecoder (被 simulator 消费)
"""

from __future__ import annotations

import logging
import random
from typing import List

from distlmsim.config import DisaggregatedConfig, ModelConfig
from distlmsim.context import SimContext
from distlmsim.entities import Request
from distlmsim.execution.execution_time_predictor import AnalyticalPredictor

logger = logging.getLogger(__name__)


class SpeculativeDecoder:
    """Speculative Decoding 时间建模。

    管理 draft model 和 target model 的执行时间计算，
    以及基于接受率的 token 接受数采样。
    """

    def __init__(
        self,
        ctx: SimContext,
        spec_config: DisaggregatedConfig,
        rng: random.Random,
    ):
        self._ctx = ctx
        self._cfg = spec_config
        self._rng = rng

        # 预创建 draft model predictor
        target = ctx.model_config
        self._draft_model = ModelConfig(
            model_name=f"Draft-{target.model_name}",
            num_layers=spec_config.draft_num_layers,
            num_q_heads=max(1, target.num_q_heads // (target.embedding_dim // spec_config.draft_embedding_dim)),
            num_kv_heads=max(1, target.num_kv_heads // (target.embedding_dim // spec_config.draft_embedding_dim)),
            embedding_dim=spec_config.draft_embedding_dim,
            num_experts=0,
            top_k_experts=0,
            vocab_size=target.vocab_size,
        )
        self._draft_predictor = AnalyticalPredictor(self._draft_model, ctx.device_config)

    @property
    def K(self) -> int:
        return self._cfg.speculation_length

    @property
    def alpha(self) -> float:
        return self._cfg.acceptance_rate

    def compute_draft_step_time(self, batch_requests: List[Request]) -> float:
        """计算 draft model 单步 decode 时间 (ms)。"""
        batch_size = len(batch_requests)
        num_tokens = batch_size

        avg_kv = sum(r.prefill_tokens + r.num_generated_tokens
                     for r in batch_requests) // max(1, batch_size)

        exec_time = self._draft_predictor.get_execution_time(
            num_tokens=num_tokens, batch_size=batch_size,
            kv_cache_size=avg_kv, is_prefill=False,
        )

        per_layer = exec_time.total_time

        # Draft model TP 通信
        tp = self._ctx.tp_size
        if tp > 1:
            data_size = num_tokens * self._cfg.draft_embedding_dim * 2
            tp_comm = self._ctx.nvlink_model.get_allreduce_time(tp, data_size) * 2
        else:
            tp_comm = 0.0

        total_per_layer = _apply_overlap(per_layer, tp_comm, self._ctx)
        return total_per_layer * self._cfg.draft_num_layers

    def compute_verify_time(self, batch_requests: List[Request], K: int) -> float:
        """计算 target model 验证 K 个 candidate tokens 的时间 (ms)。"""
        batch_size = len(batch_requests)
        num_tokens = batch_size * K
        model = self._ctx.model_config

        avg_kv = sum(r.prefill_tokens + r.num_generated_tokens
                     for r in batch_requests) // max(1, batch_size)

        exec_time = self._ctx.time_predictor.get_execution_time(
            num_tokens=num_tokens, batch_size=batch_size,
            kv_cache_size=avg_kv, is_prefill=True,
        )

        per_layer = exec_time.total_time

        # Target model TP 通信
        tp = self._ctx.tp_size
        if tp > 1:
            data_size = num_tokens * model.embedding_dim * 2
            tp_comm = self._ctx.nvlink_model.get_allreduce_time(tp, data_size) * 2
        else:
            tp_comm = 0.0

        total_per_layer = _apply_overlap(per_layer, tp_comm, self._ctx)
        return total_per_layer * model.num_layers

    def sample_acceptance(self, K: int, alpha: float) -> int:
        """采样一轮投机解码的接受 token 数。

        每个 candidate token 独立以概率 alpha 被接受，
        第一个被拒绝后后续全部丢弃。
        """
        accepted = 0
        for _ in range(K):
            if self._rng.random() < alpha:
                accepted += 1
            else:
                break
        return accepted

    def compute_cycle_time(
        self, batch_requests: List[Request]
    ) -> tuple:
        """计算一轮完整投机解码周期的时间和接受 token 数。

        Returns:
            (cycle_time_ms, accepted_tokens): 周期总时间和接受的 token 数
        """
        K = self.K
        alpha = self.alpha

        # Draft: K 步
        draft_time = 0.0
        for _ in range(K):
            draft_time += self.compute_draft_step_time(batch_requests)

        # 采样接受数
        accepted = self.sample_acceptance(K, alpha)
        effective_K = max(1, accepted)

        # Verify: 1 步
        verify_time = self.compute_verify_time(batch_requests, effective_K)

        return draft_time + verify_time, accepted


class DSparkSpeculativeDecoder:
    """DSpark / DFlash 投机解码时间建模 (DeepSeek 方案)。

    相比标准投机解码的关键差异：
    1. **块级半自回归起草**: 每轮生成 block_size 个 token，
       块内通过 Markov head 自回归，块间并行
    2. **Markov head**: 低秩嵌入 (vocab→rank→vocab) 建模
       token 间依赖，比堆叠 Transformer 层更参数高效
    3. **目标模型特征抽取**: 草稿模型 tap 入目标模型的
       num_target_layer_ids 个中间层，共享部分前向计算
    4. **置信度调度验证**: AcceptRatePredictor 线性头预测
       接受率，低于 confidence_threshold 时早停

    参考: DeepSpec (github.com/deepseek-ai/DeepSpec)

    时间模型:
    - T_draft = num_draft_layers * T_draft_layer(block_size tokens)
              + block_size * T_markov_head
              + num_target_layer_ids * T_target_layer_extraction
    - T_verify = T_target_forward(block_size tokens)
    - T_cycle = T_draft + T_verify
    - accepted = block_size * α (置信度调度可提前截断)
    """

    def __init__(
        self,
        ctx: SimContext,
        spec_config: DisaggregatedConfig,
        rng: random.Random,
    ):
        self._ctx = ctx
        self._cfg = spec_config
        self._rng = rng

        # Draft model config (同标准版，但层数更少因为有 Markov head 辅助)
        target = ctx.model_config
        self._draft_model = ModelConfig(
            model_name=f"DSpark-Draft-{target.model_name}",
            num_layers=spec_config.draft_num_layers,
            num_q_heads=max(1, target.num_q_heads // (target.embedding_dim // spec_config.draft_embedding_dim)),
            num_kv_heads=max(1, target.num_kv_heads // (target.embedding_dim // spec_config.draft_embedding_dim)),
            embedding_dim=spec_config.draft_embedding_dim,
            num_experts=0,
            top_k_experts=0,
            vocab_size=target.vocab_size,
        )
        self._draft_predictor = AnalyticalPredictor(self._draft_model, ctx.device_config)

    @property
    def block_size(self) -> int:
        return self._cfg.block_size

    @property
    def alpha(self) -> float:
        return self._cfg.acceptance_rate

    @property
    def markov_rank(self) -> int:
        return self._cfg.markov_rank

    def compute_markov_head_time(self, num_tokens: int) -> float:
        """计算 Markov head 处理 num_tokens 个 token 的时间 (ms)。

        Markov head 操作:
        - W1: nn.Embedding(vocab_size, markov_rank) → 查表 O(1)
        - W2: nn.Linear(markov_rank, vocab_size) → 矩阵乘
        - 每 token 的 FLOPs ≈ 2 * markov_rank * vocab_size

        对于 gated 类型额外有:
        - gate_proj: Linear(hidden_size + markov_rank, markov_rank)
        - sigmoid 激活

        对于 rnn 类型额外有:
        - joint_proj: Linear(2*rank + hidden, 3*rank)
        - GRU gate/candidate 计算
        """
        vocab_size = self._ctx.model_config.vocab_size
        rank = self.markov_rank
        hidden = self._cfg.draft_embedding_dim

        # Vanilla Markov: embedding lookup + rank→vocab projection
        flops_per_token = 2 * rank * vocab_size

        # Gated: additional gate projection
        if self._cfg.markov_head_type == "gated":
            flops_per_token += 2 * (hidden + rank) * rank + 2 * rank * vocab_size

        # RNN: additional GRU-like state update
        elif self._cfg.markov_head_type == "rnn":
            flops_per_token += 2 * (2 * rank + hidden) * 3 * rank + 2 * rank * vocab_size

        # 使用 AnalyticalPredictor 的 roofline 模型估算
        # Markov head 是 compute-bound (大 vocab 矩阵乘)
        device = self._ctx.device_config
        peak_flops = device.fp16_tflops * 1e12  # TFLOPS → FLOPS
        eta_c = 0.85

        total_flops = flops_per_token * num_tokens
        compute_time_s = total_flops / (eta_c * peak_flops)

        # 加上 memory-bound 的 embedding lookup
        # W1: vocab_size * rank * 2 bytes (fp16), 每 token 查一次
        bytes_per_token = rank * 2  # embedding lookup
        memory_bw = device.memory_bandwidth_gbps * 1e9  # GB/s → B/s
        eta_m = 0.90
        memory_time_s = (bytes_per_token * num_tokens) / (eta_m * memory_bw)

        return max(compute_time_s, memory_time_s) * 1e3  # s → ms

    def compute_target_layer_extraction_time(
        self, batch_requests: List[Request], num_layers: int
    ) -> float:
        """计算从目标模型抽取中间层特征的时间 (ms)。

        DSpark 的草稿模型需要从目标模型的 num_target_layer_ids 个
        中间层提取隐藏状态，这会产生额外的前向计算开销。
        由于草稿模型的 forward pass 已经覆盖了这些层的计算，
        这里的开销主要是特征拼接 (cat) 和投影 (linear)。

        Args:
            batch_requests: 当前 batch 中的请求列表
            num_layers: 需要抽取的目标模型层数
        """
        batch_size = len(batch_requests)
        num_tokens = batch_size * self.block_size
        target_dim = self._ctx.model_config.embedding_dim

        # 特征拼接: cat([h_1, h_9, h_17, h_25, h_33]) → shape [B*T, num_layers*dim]
        # 投影: Linear(num_layers * dim, draft_dim)
        concat_dim = num_layers * target_dim
        flops = 2 * concat_dim * self._cfg.draft_embedding_dim * num_tokens

        device = self._ctx.device_config
        peak_flops = device.fp16_tflops * 1e12
        eta_c = 0.85
        return (flops / (eta_c * peak_flops)) * 1e3

    def compute_draft_block_time(self, batch_requests: List[Request]) -> float:
        """计算 DSpark 一个 draft block 的时间 (ms)。

        包括:
        1. Draft model 前向 (num_draft_layers 层, 处理 block_size tokens)
        2. Markov head 块内自回归 (block_size 步)
        3. 目标模型特征抽取
        """
        batch_size = len(batch_requests)
        num_tokens = batch_size * self.block_size

        avg_kv = sum(r.prefill_tokens + r.num_generated_tokens
                     for r in batch_requests) // max(1, batch_size)

        # 1. Draft model forward: 每层处理 block_size tokens
        exec_time = self._draft_predictor.get_execution_time(
            num_tokens=num_tokens, batch_size=batch_size,
            kv_cache_size=avg_kv, is_prefill=False,
        )
        per_layer = exec_time.total_time

        # Draft model TP 通信
        tp = self._ctx.tp_size
        if tp > 1:
            data_size = num_tokens * self._cfg.draft_embedding_dim * 2
            tp_comm = self._ctx.nvlink_model.get_allreduce_time(tp, data_size) * 2
        else:
            tp_comm = 0.0

        draft_layer_time = _apply_overlap(per_layer, tp_comm, self._ctx)
        total_draft = draft_layer_time * self._cfg.draft_num_layers

        # 2. Markov head: block_size 步自回归
        markov_time = self.compute_markov_head_time(self.block_size)

        # 3. Target layer feature extraction
        extraction_time = self.compute_target_layer_extraction_time(
            batch_requests, self._cfg.num_target_layer_ids
        )

        return total_draft + markov_time + extraction_time

    def compute_verify_time(self, batch_requests: List[Request]) -> float:
        """计算 target model 验证 block_size 个 tokens 的时间 (ms)。"""
        batch_size = len(batch_requests)
        num_tokens = batch_size * self.block_size
        model = self._ctx.model_config

        avg_kv = sum(r.prefill_tokens + r.num_generated_tokens
                     for r in batch_requests) // max(1, batch_size)

        exec_time = self._ctx.time_predictor.get_execution_time(
            num_tokens=num_tokens, batch_size=batch_size,
            kv_cache_size=avg_kv, is_prefill=True,
        )
        per_layer = exec_time.total_time

        tp = self._ctx.tp_size
        if tp > 1:
            data_size = num_tokens * model.embedding_dim * 2
            tp_comm = self._ctx.nvlink_model.get_allreduce_time(tp, data_size) * 2
        else:
            tp_comm = 0.0

        total_per_layer = _apply_overlap(per_layer, tp_comm, self._ctx)
        return total_per_layer * model.num_layers

    def sample_acceptance(self) -> int:
        """采样一轮 DSpark 投机解码的接受 token 数。

        标准模式: block_size 个 token 逐个独立验证，
        第一个被拒绝后后续全部丢弃。

        置信度调度模式: 当启用 enable_confidence_scheduling 时，
        根据 AcceptRatePredictor 预测的接受率决定是否提前截断。
        低置信度块直接跳过验证（节省 target model 计算）。
        """
        accepted = 0
        for i in range(self.block_size):
            # 置信度调度: 每个位置的有效接受率随位置递减
            # DSpark 论文中，块内位置越靠后接受率越低
            position_decay = self.alpha * (1.0 - 0.05 * i)
            effective_alpha = max(0.1, min(1.0, position_decay))

            if self._rng.random() < effective_alpha:
                accepted += 1
            else:
                break

            # 置信度早停: 如果当前接受率低于阈值，停止起草
            if (self._cfg.enable_confidence_scheduling
                    and self._cfg.confidence_threshold > 0
                    and effective_alpha < self._cfg.confidence_threshold):
                break

        return accepted

    def compute_cycle_time(
        self, batch_requests: List[Request]
    ) -> tuple:
        """计算一轮完整 DSpark 投机解码周期的时间和接受 token 数。

        Returns:
            (cycle_time_ms, accepted_tokens): 周期总时间和接受的 token 数
        """
        # Draft: 1 个 block
        draft_time = self.compute_draft_block_time(batch_requests)

        # 采样接受数
        accepted = self.sample_acceptance()
        effective_K = max(1, accepted)

        # Verify: target model forward (block_size tokens)
        verify_time = self.compute_verify_time(batch_requests)

        # DFlash 并行起草: 如果有多个 block 并行，verify 可以流水线化
        if self._cfg.speculative_mode == "dflash":
            # DFlash 使用并行 block 起草，draft 和 verify 可部分重叠
            overlap_factor = 0.7  # DFlash 的并行加速比
            cycle_time = draft_time + verify_time * overlap_factor
        else:
            cycle_time = draft_time + verify_time

        return cycle_time, accepted


def create_speculative_decoder(
    ctx: SimContext,
    spec_config: DisaggregatedConfig,
    rng: random.Random,
) -> SpeculativeDecoder | DSparkSpeculativeDecoder:
    """根据 speculative_mode 创建对应的投机解码器。

    Args:
        ctx: 模拟运行时上下文
        spec_config: 存算分离配置（含 speculative_mode）
        rng: 随机数生成器

    Returns:
        SpeculativeDecoder (mode="standard")
        或 DSparkSpeculativeDecoder (mode="dspark" | "dflash")
    """
    mode = spec_config.speculative_mode
    if mode in ("dspark", "dflash"):
        return DSparkSpeculativeDecoder(ctx, spec_config, rng)
    return SpeculativeDecoder(ctx, spec_config, rng)


def _apply_overlap(compute_ms: float, comm_ms: float, ctx: SimContext) -> float:
    """通信-计算重叠辅助函数。"""
    if comm_ms <= 0:
        return compute_ms
    proc = ctx.overlap_processor
    pair = proc.make_compute_comm_pair(compute_ms, comm_ms)
    pair = proc.apply_ratio_slowdown(pair)
    return max(pair.adjusted_a_ms, pair.adjusted_b_ms)


# ─── 统一 Speculative Decoding Engine ────────────────────────────────────────


from dataclasses import dataclass
from distlmsim.execution.draft_model import DraftModelPredictor, load_sps_curve
from distlmsim.scheduling.prefix_scheduler import (
    HardwareAwarePrefixScheduler,
    generate_confidence_scores,
)
import os


@dataclass
class CycleResult:
    """一轮 decode cycle 的结果。

    K=0 (标准 decode): accepted_tokens=1, scheduled_length=1
    K>0 (投机解码): accepted_tokens=采样接受数, scheduled_length=前缀调度长度
    """

    cycle_time_ms: float = 0.0
    accepted_tokens: int = 1
    scheduled_length: int = 1
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    is_speculative: bool = False


class SpeculativeDecodingEngine:
    """统一的投机解码引擎，覆盖标准 decode 和所有投机解码模式。

    一个 cycle 的完整流程:
    1. [Draft] DraftModelPredictor.get_draft_time() → draft block
    2. [Sample] 根据 acceptance_rate 和 position_decay 采样接受数
    3. [Confidence] 生成 confidence scores (如果启用)
    4. [Schedule] HardwareAwarePrefixScheduler.schedule() → 裁剪 prefix
    5. [Verify] target model forward(scheduled_length) → 验证

    K=0 / disabled 时: 退化为标准 decode (1 token/step, 无 draft)
    """

    def __init__(
        self,
        ctx: SimContext,
        config: DisaggregatedConfig,
        rng: random.Random,
    ):
        self._ctx = ctx
        self._config = config
        self._rng = rng
        self._enabled = config.enable_speculative_decoding

        if self._enabled:
            self._draft_predictor = DraftModelPredictor(
                ctx.model_config, ctx.device_config, config
            )
            self._block_size = config.block_size

            # 加载 SPS 曲线
            sps_path = config.sps_profile_path
            if not sps_path:
                # 自动查找
                candidates = [
                    "data/profiling/system/a800/sps_curve.csv",
                    os.path.join(os.path.dirname(__file__), "../../data/profiling/system/a800/sps_curve.csv"),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        sps_path = c
                        break
            sps_curve = load_sps_curve(sps_path) if sps_path else {}

            self._prefix_scheduler = HardwareAwarePrefixScheduler(
                sps_curve=sps_curve,
                confidence_threshold=config.confidence_threshold,
            )
        else:
            self._draft_predictor = None
            self._block_size = 0
            self._prefix_scheduler = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def block_size(self) -> int:
        return self._block_size

    def compute_cycle(
        self,
        batch_requests: List[Request],
        current_time: float,
        compute_standard_step_fn=None,
    ) -> CycleResult:
        """计算一轮 decode cycle。

        Args:
            batch_requests: 当前 batch 中所有活跃请求
            current_time: 当前模拟时间 (ms)
            compute_standard_step_fn: 标准 decode 单步时间计算函数
                (用于 K=0 退化时调用 main.py 的 _compute_decode_step_time)

        Returns:
            CycleResult
        """
        if not self._enabled:
            # 标准 decode: 1 token/step
            if compute_standard_step_fn:
                step_time = compute_standard_step_fn(batch_requests)
            else:
                step_time = self._compute_standard_step_time(batch_requests)
            return CycleResult(
                cycle_time_ms=step_time,
                accepted_tokens=1,
                scheduled_length=1,
                draft_time_ms=0.0,
                verify_time_ms=step_time,
                is_speculative=False,
            )

        # 投机解码 cycle
        return self._compute_speculative_cycle(batch_requests)

    def _compute_standard_step_time(
        self, batch_requests: List[Request]
    ) -> float:
        """标准 decode 单步时间 (fallback)。"""
        model = self._ctx.model_config
        batch_size = len(batch_requests)
        avg_kv = sum(
            r.prefill_tokens + r.num_generated_tokens for r in batch_requests
        ) // max(1, batch_size)

        et = self._ctx.time_predictor.get_execution_time(
            num_tokens=1, batch_size=batch_size,
            kv_cache_size=avg_kv, is_prefill=False,
        )
        per_layer = et.total_time
        tp = self._ctx.tp_size
        if tp > 1:
            data_size = batch_size * model.embedding_dim * 2
            tp_comm = self._ctx.nvlink_model.get_allreduce_time(tp, data_size) * 2
        else:
            tp_comm = 0.0
        return _apply_overlap(per_layer, tp_comm, self._ctx) * model.num_layers

    def _compute_speculative_cycle(
        self, batch_requests: List[Request]
    ) -> CycleResult:
        """投机解码 cycle: draft → sample → schedule → verify。"""
        batch_size = len(batch_requests)
        avg_kv = sum(
            r.prefill_tokens + r.num_generated_tokens for r in batch_requests
        ) // max(1, batch_size)

        # 1. Draft phase
        draft_breakdown = self._draft_predictor.get_draft_time(
            num_tokens=self._block_size,
            batch_size=batch_size,
            kv_cache_size=avg_kv,
        )
        draft_time = draft_breakdown.total_time_ms

        # 2. Sample acceptance (position-dependent)
        alpha = self._config.acceptance_rate
        accepted = 0
        for k in range(self._block_size):
            pos_alpha = alpha * (1.0 - 0.05 * k)
            if self._rng.random() < pos_alpha:
                accepted += 1
            else:
                break

        # 3. Generate confidence scores (for prefix scheduler)
        if self._config.enable_confidence_scheduling:
            conf_scores_per_req = []
            for _ in batch_requests:
                conf_scores_per_req.append(
                    generate_confidence_scores(self._block_size, alpha)
                )
            scheduled_lengths = self._prefix_scheduler.schedule(
                batch_requests, conf_scores_per_req, self._block_size
            )
            # 取平均 scheduled length (简化: 同一 batch 使用相同长度)
            scheduled_length = max(1, sum(scheduled_lengths) // batch_size)
        else:
            scheduled_length = self._block_size

        # Cap accepted by scheduled length
        effective_accepted = min(accepted, scheduled_length)

        # 4. Bonus token (标准投机解码保证: 验证后额外接受 1 个 target-generated token)
        if self._config.bonus_token:
            effective_accepted = min(effective_accepted + 1, scheduled_length + 1)

        effective_accepted = max(1, effective_accepted)

        # 5. Verify phase: target model forward with scheduled_length tokens
        num_verify_tokens = scheduled_length * batch_size
        model = self._ctx.model_config
        et = self._ctx.time_predictor.get_execution_time(
            num_tokens=num_verify_tokens, batch_size=batch_size,
            kv_cache_size=avg_kv, is_prefill=True,
        )
        verify_per_layer = et.total_time
        tp = self._ctx.tp_size
        if tp > 1:
            data_size = num_verify_tokens * model.embedding_dim * 2
            tp_comm = self._ctx.nvlink_model.get_allreduce_time(tp, data_size) * 2
        else:
            tp_comm = 0.0
        verify_time = _apply_overlap(verify_per_layer, tp_comm, self._ctx) * model.num_layers

        # DFlash: draft-verify pipelining
        if self._config.speculative_mode == "dflash":
            cycle_time = draft_time + verify_time * 0.7
        else:
            cycle_time = draft_time + verify_time

        return CycleResult(
            cycle_time_ms=cycle_time,
            accepted_tokens=effective_accepted,
            scheduled_length=scheduled_length,
            draft_time_ms=draft_time,
            verify_time_ms=verify_time,
            is_speculative=True,
        )
