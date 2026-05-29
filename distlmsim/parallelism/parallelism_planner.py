"""并行策略规划器

给定模型配置和集群拓扑，推荐最优的 3D/4D/5D 并行方案。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from distlmsim.config import ModelConfig, ClusterConfig
from distlmsim.topology.communication_cost import CommunicationCostCalculator


@dataclass
class ParallelismPlan:
    """并行策略方案。"""
    tp_size: int                  # 张量并行度
    pp_size: int                  # 流水线并行度
    dp_size: int                  # 数据并行度
    ep_size: int                  # 专家并行度
    gpus_per_replica: int         # 每副本 GPU 数
    num_replicas: int             # 副本数
    # 节点映射
    stage_node_mapping: dict[int, List[int]]  # pp_stage -> node_ids
    # 估计指标
    estimated_throughput: float = 0.0   # tokens/s
    estimated_latency_ms: float = 0.0   # 估计延迟
    comm_overhead_ratio: float = 0.0    # 通信开销比例


class ParallelismPlanner:
    """并行策略规划器。

    输入: 模型配置 + 集群拓扑
    输出: 推荐的 (TP, PP, DP, EP) 组合 + 节点映射

    约束条件:
    1. TP 必须同节点 (NVLink): TP <= gpus_per_node
    2. PP 可跨节点 (RDMA): PP * TP <= total_gpus
    3. DP = total_gpus / (TP * PP)
    4. EP 需要跨节点 All-to-All: EP <= DP * TP
    5. 模型层数必须能被 PP 整除

    优化目标:
    - 最小化通信开销比例
    - 最大化 GPU 利用率
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cluster_config: ClusterConfig,
    ):
        self._model = model_config
        self._cluster = cluster_config
        self._total_gpus = cluster_config.num_nodes * cluster_config.node_sku.num_gpus
        self._gpus_per_node = cluster_config.node_sku.num_gpus

    def recommend_plan(
        self,
        target_batch_size: int = 256,
        target_seq_len: int = 2048,
    ) -> ParallelismPlan:
        """推荐并行策略。

        启发式规则:
        1. TP 优先填满节点内 GPU (最大化 NVLink 利用)
        2. PP 用于跨节点 (当 TP 不够时)
        3. DP 复制模型以提高吞吐
        4. EP 仅在 MoE 模型且 EP > 1 时启用

        Args:
            target_batch_size: 目标 batch size
            target_seq_len: 目标序列长度

        Returns:
            ParallelismPlan
        """
        # 估算模型内存需求
        model_memory_gb = self._estimate_model_memory()
        gpu_memory_gb = self._cluster.node_sku.device_sku.memory_gb

        # 确定 TP
        tp_size = self._determine_tp_size(model_memory_gb, gpu_memory_gb)

        # 确定 PP
        pp_size = self._determine_pp_size(model_memory_gb, gpu_memory_gb, tp_size)

        # 确定 DP
        dp_size = self._total_gpus // (tp_size * pp_size)

        # 确定 EP
        ep_size = 1
        if self._model.num_experts > 0 and self._cluster.replica.enable_expert_parallel:
            ep_size = min(
                self._cluster.replica.expert_parallel_size,
                dp_size * tp_size,
            )

        # 创建 stage 到节点的映射
        stage_mapping = self._create_stage_mapping(tp_size, pp_size)

        return ParallelismPlan(
            tp_size=tp_size,
            pp_size=pp_size,
            dp_size=dp_size,
            ep_size=ep_size,
            gpus_per_replica=tp_size * pp_size,
            num_replicas=dp_size,
            stage_node_mapping=stage_mapping,
        )

    def _estimate_model_memory(self) -> float:
        """估算模型参数内存 (GB)。"""
        # 粗略估算: embedding + layers * (attn + mlp)
        emb_params = self._model.vocab_size * self._model.embedding_dim
        head_dim = self._model.embedding_dim // self._model.num_q_heads
        attn_params = (
            self._model.embedding_dim * self._model.num_q_heads * head_dim * 4
            + self._model.embedding_dim * self._model.num_q_heads * head_dim
        )
        mlp_hidden = self._model.mlp_hidden_dim or int(self._model.embedding_dim * 8 / 3)
        mlp_params = 3 * self._model.embedding_dim * mlp_hidden

        total_params = emb_params + self._model.num_layers * (attn_params + mlp_params)
        # float16: 2 bytes per param
        return total_params * 2 / (1024 ** 3)

    def _determine_tp_size(self, model_memory_gb: float, gpu_memory_gb: float) -> int:
        """确定张量并行度。

        原则: 模型必须能放进 TP 个 GPU 的显存中。
        """
        for tp in [8, 4, 2, 1]:
            if tp > self._gpus_per_node:
                continue
            if model_memory_gb / tp < gpu_memory_gb * 0.8:  # 80% 显存利用率
                return tp
        return 1

    def _determine_pp_size(
        self,
        model_memory_gb: float,
        gpu_memory_gb: float,
        tp_size: int,
    ) -> int:
        """确定流水线并行度。

        PP 用于模型太大单节点放不下时。
        """
        if self._model.num_layers <= 1:
            return 1

        per_gpu_memory = model_memory_gb / tp_size
        if per_gpu_memory < gpu_memory_gb * 0.8:
            return 1

        # 需要 PP
        max_pp = self._total_gpus // tp_size
        for pp in range(2, max_pp + 1):
            if self._model.num_layers % pp == 0:
                if per_gpu_memory / pp < gpu_memory_gb * 0.8:
                    return pp

        return 1

    def _create_stage_mapping(
        self, tp_size: int, pp_size: int
    ) -> dict[int, List[int]]:
        """创建 PP stage 到节点的映射。"""
        mapping = {}
        gpus_per_stage = tp_size
        for stage_id in range(pp_size):
            start_gpu = stage_id * gpus_per_stage
            node_ids = []
            for g in range(gpus_per_stage):
                node_id = (start_gpu + g) // self._gpus_per_node
                if node_id not in node_ids:
                    node_ids.append(node_id)
            mapping[stage_id] = node_ids
        return mapping

    def enumerate_plans(self) -> List[ParallelismPlan]:
        """枚举所有可行的并行策略方案。

        用于搜索最优方案。

        Returns:
            所有可行方案列表
        """
        plans = []

        for tp in [1, 2, 4, 8]:
            if tp > self._gpus_per_node:
                continue
            for pp in range(1, self._model.num_layers + 1):
                if self._model.num_layers % pp != 0:
                    continue
                if tp * pp > self._total_gpus:
                    continue

                dp = self._total_gpus // (tp * pp)
                if dp < 1:
                    continue

                stage_mapping = self._create_stage_mapping(tp, pp)
                plans.append(ParallelismPlan(
                    tp_size=tp,
                    pp_size=pp,
                    dp_size=dp,
                    ep_size=1,
                    gpus_per_replica=tp * pp,
                    num_replicas=dp,
                    stage_node_mapping=stage_mapping,
                ))

        return plans
