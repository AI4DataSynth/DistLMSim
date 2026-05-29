"""专家并行 (Expert Parallelism) 模型

EP 将 MoE 模型的专家分布到多个 GPU/节点，
通过 All-to-All 通信进行 token 分发和收集。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from distlmsim.config import ModelConfig


@dataclass
class ExpertPlacement:
    """专家放置方案。"""
    expert_id: int
    gpu_id: int
    node_id: int
    is_redundant: bool = False  # 是否为冗余副本


@dataclass
class ExpertRoutingResult:
    """专家路由结果。"""
    token_expert_assignment: np.ndarray   # shape: [num_tokens, top_k]
    expert_loads: np.ndarray             # shape: [num_experts] 每个专家的 token 数
    gpu_loads: np.ndarray                # shape: [num_gpus] 每个 GPU 的 token 数
    max_gpu_load: int                    # 最大 GPU 负载
    load_imbalance: float                # 负载不均衡度 (std/mean)


class ExpertParallelModel:
    """专家并行模型。

    负责:
    1. 专家放置策略 (均匀分布 + 冗余 + 节点亲和性)
    2. Token 路由 (Top-K + 负载均衡)
    3. All-to-All 通信数据量计算
    4. EPLB 负载均衡调度
    """

    def __init__(
        self,
        model_config: ModelConfig,
        ep_size: int,
        num_gpus_per_node: int = 8,
        redundant_experts: int = 0,
    ):
        self._model = model_config
        self._ep_size = ep_size
        self._num_gpus_per_node = num_gpus_per_node
        self._redundant_experts = redundant_experts
        self._placement: List[ExpertPlacement] = []

    def create_expert_placement(
        self,
        gpu_node_mapping: Dict[int, int],
    ) -> List[ExpertPlacement]:
        """创建专家放置方案。

        策略:
        1. 均匀分布专家到所有 GPU
        2. 冗余专家优先放在不同节点 (容错 + 负载均衡)
        3. 同节点内专家连续放置 (减少跨节点通信)

        Args:
            gpu_node_mapping: gpu_id -> node_id

        Returns:
            ExpertPlacement 列表
        """
        num_experts = self._model.num_experts
        placements = []

        # 基础放置: 均匀分配
        for expert_id in range(num_experts):
            gpu_id = expert_id % self._ep_size
            node_id = gpu_node_mapping.get(gpu_id, 0)
            placements.append(ExpertPlacement(
                expert_id=expert_id,
                gpu_id=gpu_id,
                node_id=node_id,
            ))

        # 冗余专家
        for i in range(self._redundant_experts):
            expert_id = i % num_experts
            gpu_id = (expert_id + self._ep_size // 2) % self._ep_size  # 放在不同位置
            node_id = gpu_node_mapping.get(gpu_id, 0)
            placements.append(ExpertPlacement(
                expert_id=expert_id,
                gpu_id=gpu_id,
                node_id=node_id,
                is_redundant=True,
            ))

        self._placement = placements
        return placements

    def route_tokens(
        self,
        expert_distribution: np.ndarray,
        top_k: int,
    ) -> ExpertRoutingResult:
        """执行 token 路由。

        根据专家分布矩阵，为每个 token 选择 top_k 个专家。

        Args:
            expert_distribution: shape [num_tokens, num_experts]，路由权重
            top_k: 每 token 选择的专家数

        Returns:
            ExpertRoutingResult
        """
        num_tokens = expert_distribution.shape[0]

        # Top-K 选择
        top_k_indices = np.argsort(expert_distribution, axis=1)[:, -top_k:]

        # 计算专家负载
        expert_loads = np.zeros(self._model.num_experts, dtype=np.int64)
        for token_idx in range(num_tokens):
            for k in range(top_k):
                expert_id = top_k_indices[token_idx, k]
                expert_loads[expert_id] += 1

        # 计算 GPU 负载 (基于专家放置)
        gpu_loads = np.zeros(self._ep_size, dtype=np.int64)
        for placement in self._placement:
            if placement.gpu_id < self._ep_size:
                gpu_loads[placement.gpu_id] += expert_loads[placement.expert_id]

        max_gpu_load = int(np.max(gpu_loads))
        mean_load = np.mean(gpu_loads[gpu_loads > 0])
        std_load = np.std(gpu_loads[gpu_loads > 0])
        imbalance = float(std_load / mean_load) if mean_load > 0 else 0.0

        return ExpertRoutingResult(
            token_expert_assignment=top_k_indices,
            expert_loads=expert_loads,
            gpu_loads=gpu_loads,
            max_gpu_load=max_gpu_load,
            load_imbalance=imbalance,
        )

    def get_alltoall_data_size(
        self,
        num_tokens: int,
        top_k: int,
    ) -> int:
        """计算 All-to-All 通信的数据量 (bytes)。

        Dispatch: 每 token 发送 top_k 份到目标专家
        Combine: 收集结果

        Args:
            num_tokens: batch 中的 token 数
            top_k: Top-K 路由

        Returns:
            每 GPU 的 All-to-All 数据量 (bytes)
        """
        if self._ep_size <= 1:
            return 0

        # 每 token: hidden_dim * 2 bytes (float16) * top_k
        per_token_bytes = self._model.embedding_dim * 2 * top_k
        # 均匀分布到 ep_size 个 GPU
        return int(num_tokens * per_token_bytes / self._ep_size)
