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


def _apply_overlap(compute_ms: float, comm_ms: float, ctx: SimContext) -> float:
    """通信-计算重叠辅助函数。"""
    if comm_ms <= 0:
        return compute_ms
    proc = ctx.overlap_processor
    pair = proc.make_compute_comm_pair(compute_ms, comm_ms)
    pair = proc.apply_ratio_slowdown(pair)
    return max(pair.adjusted_a_ms, pair.adjusted_b_ms)
