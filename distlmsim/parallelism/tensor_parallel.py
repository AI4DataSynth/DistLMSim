"""张量并行 (Tensor Parallelism) 模型

TP 将模型参数按层拆分到多个 GPU，通常在同一节点内通过 NVLink 通信。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from distlmsim.config import ModelConfig


@dataclass
class TPExecutionParams:
    """TP 执行参数。"""
    q_heads_per_worker: int       # 每 GPU 的 Q head 数
    kv_heads_per_worker: int      # 每 GPU 的 KV head 数
    mlp_hidden_per_worker: int    # 每 GPU 的 MLP hidden dim
    embedding_per_worker: int     # 每 GPU 的 embedding dim
    num_allreduce_per_layer: int  # 每层 all-reduce 次数


class TensorParallelModel:
    """张量并行模型。

    计算 TP 下的:
    1. 每 GPU 的计算量 (FLOPs)
    2. 通信数据量 (all-reduce bytes)
    3. 通信时间 (通过 CommunicationCostCalculator)
    """

    def __init__(self, model_config: ModelConfig, tp_size: int):
        self._model = model_config
        self._tp_size = tp_size

    def get_execution_params(self) -> TPExecutionParams:
        """计算 TP 拆分后的执行参数。"""
        return TPExecutionParams(
            q_heads_per_worker=self._model.num_q_heads // self._tp_size,
            kv_heads_per_worker=max(1, self._model.num_kv_heads // self._tp_size),
            mlp_hidden_per_worker=self._get_mlp_hidden_dim() // self._tp_size,
            embedding_per_worker=self._model.embedding_dim,
            num_allreduce_per_layer=2,  # attention + MLP 各一次
        )

    def get_allreduce_data_size(self, num_tokens: int) -> int:
        """计算单次 All-Reduce 的数据量 (bytes)。

        All-Reduce 的数据 = 激活张量 = [num_tokens, hidden_dim]
        TP 下 hidden_dim 被拆分，all-reduce 的是完整 hidden_dim。

        Args:
            num_tokens: batch 中的 token 数

        Returns:
            数据量 (bytes), float16 = 2 bytes/element
        """
        if self._tp_size <= 1:
            return 0

        # all-reduce 数据: [num_tokens, embedding_dim] * 2 bytes (float16)
        return num_tokens * self._model.embedding_dim * 2

    def get_compute_flops_per_layer(self, num_tokens: int) -> int:
        """计算每层每 GPU 的计算量 (FLOPs)。

        Attention: 4 * num_tokens * embedding_dim * (q_heads_per_worker * head_dim)
        MLP (gated): 3 * num_tokens * embedding_dim * mlp_hidden_per_worker
        """
        params = self.get_execution_params()
        head_dim = self._model.embedding_dim // self._model.num_q_heads

        # Attention FLOPs
        attn_flops = 4 * num_tokens * self._model.embedding_dim * (
            params.q_heads_per_worker * head_dim
        )

        # MLP FLOPs (gated: up + gate + down = 3 matmuls)
        mlp_flops = 3 * num_tokens * self._model.embedding_dim * params.mlp_hidden_per_worker

        return attn_flops + mlp_flops

    def _get_mlp_hidden_dim(self) -> int:
        """获取 MLP hidden dim。"""
        if self._model.mlp_hidden_dim > 0:
            return self._model.mlp_hidden_dim
        # 默认: 8/3 * embedding_dim (LLaMA 风格 gated MLP)
        return int(self._model.embedding_dim * 8 / 3)
