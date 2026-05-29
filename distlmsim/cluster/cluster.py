"""集群管理

管理多个物理节点和网络拓扑，提供统一的集群视图。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from distlmsim.config import ClusterConfig
from distlmsim.cluster.node import PhysicalNode
from distlmsim.cluster.resource_manager import ResourceManager
from distlmsim.entities import Replica
from distlmsim.topology.network_topology import NetworkTopology
from distlmsim.types import NodeRole


class Cluster:
    """分布式集群。

    包含多个物理节点（通过 RDMA 互联）和网络拓扑。
    负责创建副本（Replica）并将副本映射到物理节点/GPU。
    """

    def __init__(
        self,
        nodes: List[PhysicalNode],
        topology: NetworkTopology,
        resource_manager: ResourceManager,
    ):
        self._nodes = nodes
        self._topology = topology
        self._resource_manager = resource_manager
        self._replicas: Dict[int, Replica] = {}

    @classmethod
    def from_config(cls, config: ClusterConfig) -> "Cluster":
        """根据配置构建集群。"""
        # 创建节点
        nodes = []
        gpu_offset = 0
        for node_id in range(config.num_nodes):
            node = PhysicalNode(
                node_id=node_id,
                sku=config.node_sku,
                global_gpu_id_offset=gpu_offset,
            )
            nodes.append(node)
            gpu_offset += config.node_sku.num_gpus

        # 创建网络拓扑
        topology = NetworkTopology.from_config(config.network, config.num_nodes)

        # 创建资源管理器
        resource_manager = ResourceManager(nodes, topology)

        cluster = cls(nodes, topology, resource_manager)

        # 创建副本
        cluster._create_replicas(config)

        return cluster

    def _create_replicas(self, config: ClusterConfig) -> None:
        """创建模型副本并分配到节点。

        每个副本需要的 GPU 数 = TP * PP
        """
        tp = config.replica.tensor_parallel_size
        pp = config.replica.num_pipeline_stages
        gpus_per_replica = tp * pp

        for replica_id in range(config.num_replicas):
            # 资源管理器分配 GPU
            gpu_assignment = self._resource_manager.allocate_replica(
                replica_id, gpus_per_replica, tp
            )

            replica = Replica(
                id=replica_id,
                model_name=config.replica.model.model_name,
                tensor_parallel_size=tp,
                num_pipeline_stages=pp,
                expert_parallel_size=config.replica.expert_parallel_size,
                node_ids=list(set(
                    self._gpu_to_node(gid) for gid in gpu_assignment
                )),
                gpu_ids=gpu_assignment,
            )
            self._replicas[replica_id] = replica

    def _gpu_to_node(self, global_gpu_id: int) -> int:
        """全局 GPU ID -> 节点 ID。"""
        gpus_per_node = self._nodes[0].num_gpus if self._nodes else 8
        return global_gpu_id // gpus_per_node

    @property
    def nodes(self) -> List[PhysicalNode]:
        return self._nodes

    @property
    def topology(self) -> NetworkTopology:
        return self._topology

    @property
    def replicas(self) -> Dict[int, Replica]:
        return self._replicas

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def total_gpus(self) -> int:
        return sum(n.num_gpus for n in self._nodes)

    def get_node(self, node_id: int) -> PhysicalNode:
        return self._nodes[node_id]

    def get_replica(self, replica_id: int) -> Replica:
        return self._replicas[replica_id]

    def get_replicas_on_node(self, node_id: int) -> List[Replica]:
        """获取某节点上运行的所有副本。"""
        return [
            r for r in self._replicas.values()
            if node_id in r.node_ids
        ]

    def are_gpus_on_same_node(self, gpu_id_a: int, gpu_id_b: int) -> bool:
        """判断两个 GPU 是否在同一节点。"""
        return self._gpu_to_node(gpu_id_a) == self._gpu_to_node(gpu_id_b)
