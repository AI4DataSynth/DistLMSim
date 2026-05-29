"""网络拓扑图定义

描述集群中节点间的网络连接关系，支持拓扑感知的路径和带宽查询。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from distlmsim.config import NetworkTopologyConfig
from distlmsim.types import RDMAProtocolType


@dataclass
class NetworkLink:
    """一条网络链路。"""
    src_id: int
    dst_id: int
    bandwidth_gbps: float        # 链路带宽 (Gbps)
    latency_us: float            # 链路延迟 (微秒)
    link_type: str               # "nvlink", "rdma", "switch_uplink"


@dataclass
class SwitchNode:
    """交换机节点 (用于建模 fat-tree / leaf-spine 拓扑)。"""
    id: int
    layer: int                   # 0=leaf, 1=spine, ...
    uplinks: List[int] = field(default_factory=list)    # 上联端口
    downlinks: List[int] = field(default_factory=list)  # 下联端口


class NetworkTopology:
    """集群网络拓扑。

    维护节点间所有链路信息，提供：
    - 任意两节点间的通信路径和有效带宽
    - 链路拥塞检测
    - 拓扑类型支持: fat-tree, leaf-spine, torus
    """

    def __init__(self, config: NetworkTopologyConfig):
        self._config = config
        self._links: Dict[Tuple[int, int], NetworkLink] = {}
        self._switches: List[SwitchNode] = []
        self._node_to_switch: Dict[int, int] = {}  # 节点 -> 接入交换机

    @classmethod
    def from_config(cls, config: NetworkTopologyConfig, num_nodes: int) -> "NetworkTopology":
        """根据配置构建网络拓扑。"""
        topo = cls(config)
        topo._build_topology(num_nodes)
        return topo

    def _build_topology(self, num_nodes: int) -> None:
        """构建网络拓扑图。

        简化 fat-tree 拓扑: 每个节点通过一条上行链路连接到交换机，
        节点间通信经过 RDMA 链路。
        """
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                self.add_link(NetworkLink(
                    src_id=i,
                    dst_id=j,
                    bandwidth_gbps=self._config.rdma.bandwidth_gbps,
                    latency_us=self._config.rdma.latency_us,
                    link_type="rdma",
                ))

    def add_link(self, link: NetworkLink) -> None:
        """添加一条链路。"""
        self._links[(link.src_id, link.dst_id)] = link
        self._links[(link.dst_id, link.src_id)] = NetworkLink(
            src_id=link.dst_id,
            dst_id=link.src_id,
            bandwidth_gbps=link.bandwidth_gbps,
            latency_us=link.latency_us,
            link_type=link.link_type,
        )

    def get_path(self, src_node: int, dst_node: int) -> List[NetworkLink]:
        """获取两节点间的通信路径（链路列表）。

        Returns:
            从 src 到 dst 的链路列表。同节点返回空列表。
        """
        if src_node == dst_node:
            return []
        link = self._links.get((src_node, dst_node))
        if link:
            return [link]
        return []

    def get_effective_bandwidth(self, src_node: int, dst_node: int) -> float:
        """获取两节点间的有效带宽 (Gbps)。

        考虑路径上所有链路的最小带宽和收敛比。
        """
        if src_node == dst_node:
            # 同节点走 NVLink
            return self._config.nvlink.nvswitch_bandwidth_gbps
        # TODO: 计算跨节点路径带宽
        return self._config.rdma.bandwidth_gbps

    def get_latency(self, src_node: int, dst_node: int) -> float:
        """获取两节点间的总延迟 (微秒)。"""
        if src_node == dst_node:
            return self._config.nvlink.latency_us
        return self._config.rdma.latency_us

    def is_same_node(self, gpu_id_a: int, gpu_id_b: int, gpus_per_node: int) -> bool:
        """判断两个 GPU 是否在同一节点内。"""
        return gpu_id_a // gpus_per_node == gpu_id_b // gpus_per_node

    def get_all_links(self) -> List[NetworkLink]:
        """获取所有链路。"""
        return list(self._links.values())
