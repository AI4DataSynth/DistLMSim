"""资源管理器

负责 GPU 资源分配和副本放置策略。
"""

from __future__ import annotations

from typing import List, Optional

from distlmsim.cluster.node import PhysicalNode
from distlmsim.topology.network_topology import NetworkTopology


class ResourceManager:
    """GPU 资源管理器。

    管理集群中所有 GPU 资源的分配和释放，支持：
    - 副本到节点/GPU 的映射
    - 连续 GPU 分配（同节点 TP 亲和性）
    - 存算分离模式下的节点角色分配
    """

    def __init__(self, nodes: List[PhysicalNode], topology: NetworkTopology):
        self._nodes = nodes
        self._topology = topology
        self._gpu_allocation: dict[int, int] = {}  # global_gpu_id -> replica_id

    def allocate_replica(
        self,
        replica_id: int,
        num_gpus: int,
        tp_size: int,
        preferred_node_ids: Optional[List[int]] = None,
    ) -> List[int]:
        """为副本分配 GPU 资源。

        策略：
        1. TP 组内的 GPU 必须在同一节点（NVLink 亲和性）
        2. 不同 TP 组尽量在同一节点（减少跨节点通信）
        3. 如果单节点放不下，跨节点分配（PP 可跨节点）

        Args:
            replica_id: 副本 ID
            num_gpus: 需要的总 GPU 数 (= TP * PP)
            tp_size: 张量并行度
            preferred_node_ids: 优先使用的节点

        Returns:
            分配的全局 GPU ID 列表
        """
        allocated: List[int] = []
        gpus_per_node = self._nodes[0].num_gpus if self._nodes else 8

        # 计算需要的节点数和每节点 GPU 数
        num_pp_groups = num_gpus // tp_size

        # 优先同节点分配
        candidate_nodes = (
            [self._nodes[nid] for nid in (preferred_node_ids or range(len(self._nodes)))]
            if preferred_node_ids
            else self._nodes
        )

        for pp_group_idx in range(num_pp_groups):
            # 每个 PP group 需要 tp_size 个连续 GPU (同节点)
            placed = False
            for node in candidate_nodes:
                available = [
                    gpu.global_id for gpu in node.get_available_gpus()
                    if gpu.global_id not in self._gpu_allocation
                ]
                if len(available) >= tp_size:
                    # 分配 tp_size 个 GPU
                    for gpu_id in available[:tp_size]:
                        self._gpu_allocation[gpu_id] = replica_id
                        allocated.append(gpu_id)
                    placed = True
                    break

            if not placed:
                raise RuntimeError(
                    f"无法为副本 {replica_id} 分配足够的 GPU 资源。"
                    f"需要 {num_gpus} GPU (TP={tp_size})，"
                    f"已分配 {len(allocated)}。"
                )

        return allocated

    def release_replica(self, replica_id: int) -> None:
        """释放副本占用的 GPU 资源。"""
        to_remove = [
            gid for gid, rid in self._gpu_allocation.items()
            if rid == replica_id
        ]
        for gid in to_remove:
            del self._gpu_allocation[gid]

    def get_gpu_allocation(self) -> dict[int, int]:
        """获取当前 GPU 分配映射。"""
        return dict(self._gpu_allocation)

    def get_available_gpu_count(self) -> int:
        """获取可用 GPU 数量。"""
        total = sum(n.num_gpus for n in self._nodes)
        return total - len(self._gpu_allocation)

    def assign_node_roles(
        self,
        prefill_node_ids: List[int],
        decode_node_ids: List[int],
    ) -> None:
        """存算分离模式：分配节点角色。"""
        from distlmsim.types import NodeRole

        for node in self._nodes:
            if node.id in prefill_node_ids:
                node.role = NodeRole.PREFILL
            elif node.id in decode_node_ids:
                node.role = NodeRole.DECODE
