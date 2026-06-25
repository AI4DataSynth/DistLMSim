"""RDMA 互联模型

建模节点间的 RDMA 通信开销，支持 RoCEv2 和 InfiniBand 协议。
"""

from __future__ import annotations

from typing import Optional

from distlmsim.config import RDMAConfig
from distlmsim.types import RDMAProtocolType


class RDMAModel:
    """RDMA 网络通信模型。

    建模节点间通过 RDMA NIC 的数据传输开销。
    支持:
    - RoCEv2 (RDMA over Converged Ethernet v2)
    - InfiniBand
    - TCP/IP (降级对比)

    两种建模模式:
    1. 解析模型: 带宽/延迟公式 + 拥塞建模
    2. Profiling 模型: 从预采集数据查表
    """

    def __init__(self, config: RDMAConfig, congestion_alpha: float = 0.05):
        self._config = config
        self._congestion_alpha = congestion_alpha

    def get_transfer_time(
        self,
        data_size_bytes: int,
        src_node_id: int = 0,
        dst_node_id: int = 1,
        use_profiling: bool = False,
        concurrent_transfers: int = 1,
    ) -> float:
        """计算节点间 RDMA 数据传输时间 (ms)。

        Args:
            data_size_bytes: 数据量 (bytes)
            src_node_id: 源节点 ID
            dst_node_id: 目标节点 ID
            use_profiling: 是否使用 profiling 数据
            concurrent_transfers: 同时进行的传输流数量 (用于拥塞建模)

        Returns:
            传输时间 (ms)
        """
        if use_profiling:
            return self._get_transfer_from_profiling(data_size_bytes)
        return self._get_transfer_analytical(data_size_bytes, concurrent_transfers)

    def _get_transfer_analytical(self, data_size_bytes: int, concurrent_transfers: int = 1) -> float:
        """解析模型计算 RDMA 传输时间。

        模型: T = latency + data_size / effective_bandwidth
        考虑:
        - 协议开销 (RoCEv2 header, IB header)
        - 拥塞控制 (DCQCN 降速因子，基于并发流数量)
        - MTU 分片
        """
        # 基础延迟
        latency_ms = self._config.latency_us / 1e3

        # 有效带宽 (考虑协议开销)
        protocol_overhead = self._get_protocol_overhead_ratio()
        effective_bw_gbps = self._config.bandwidth_gbps * (1 - protocol_overhead)

        # 拥塞控制降速 (基于并发传输数量)
        congestion_factor = self._get_congestion_factor(concurrent_transfers)
        effective_bw_gbps *= congestion_factor

        effective_bw_Bps = effective_bw_gbps * 1e9 / 8
        transfer_time_ms = data_size_bytes / effective_bw_Bps * 1e3

        return latency_ms + transfer_time_ms

    def _get_protocol_overhead_ratio(self) -> float:
        """协议开销比例。"""
        if self._config.protocol == RDMAProtocolType.ROCE_V2:
            # RoCEv2: Ethernet(14) + IP(20) + UDP(8) + BTH(12) + RETH(16) = 70 bytes
            # MTU=1500, 有效载荷 = 1500 - 70 = 1430, overhead ≈ 4.7%
            return 0.047
        elif self._config.protocol == RDMAProtocolType.INFINIBAND:
            # IB: LRH(8) + GRH(40) + BTH(12) + RETH(16) = 76 bytes
            # MTU=4096, overhead ≈ 1.8%
            return 0.018
        else:
            # TCP/IP: 开销较大
            return 0.10

    def _get_congestion_factor(self, concurrent_transfers: int = 1) -> float:
        """拥塞控制降速因子 (0-1, 1=无拥塞)。

        基于 alpha-fair 拥塞模型:
        factor = 1 / (1 + alpha * max(0, concurrent - 1))
        
        当多流共享同一 RDMA 链路时，DCQCN 拥塞控制会降低每流的有效带宽。
        alpha 参数控制拥塞敏感度 (默认 0.05，即每增加一个并发流带宽降低约 5%)。

        Args:
            concurrent_transfers: 同时进行的传输流数量

        Returns:
            带宽衰减因子 (0-1)
        """
        if concurrent_transfers <= 1:
            return 1.0
        return 1.0 / (1.0 + self._congestion_alpha * (concurrent_transfers - 1))

    def _get_transfer_from_profiling(self, data_size_bytes: int) -> float:
        """从 profiling 数据查询传输时间。

        TODO: 读取 data/profiling/network/ 下对应协议的 CSV
        """
        raise NotImplementedError("Profiling-based RDMA model not yet implemented")

    def get_allreduce_time(
        self,
        num_nodes: int,
        data_size_bytes: int,
        use_profiling: bool = False,
    ) -> float:
        """计算跨节点 All-Reduce 通信时间 (ms)。

        用于跨节点的张量并行或梯度同步。
        使用 Ring All-Reduce 算法。

        Args:
            num_nodes: 参与通信的节点数
            data_size_bytes: 每节点的数据量

        Returns:
            通信时间 (ms)
        """
        if num_nodes <= 1:
            return 0.0

        # Ring All-Reduce: 2*(N-1)/N 轮
        ring_factor = 2.0 * (num_nodes - 1) / num_nodes
        total_bytes = data_size_bytes * ring_factor

        return self.get_transfer_time(total_bytes, use_profiling=use_profiling)

    def get_alltoall_time(
        self,
        num_nodes: int,
        data_size_per_node_bytes: int,
        use_profiling: bool = False,
    ) -> float:
        """计算跨节点 All-to-All 通信时间 (ms)。

        用于跨节点的专家并行 token 分发。

        Args:
            num_nodes: 参与通信的节点数
            data_size_per_node_bytes: 每节点发送的总数据量

        Returns:
            通信时间 (ms)
        """
        if num_nodes <= 1:
            return 0.0

        # All-to-All: 每节点向其他 N-1 个节点发送 1/(N-1) 的数据
        # 但总出口带宽受限于 NIC 带宽
        total_send_bytes = data_size_per_node_bytes * (num_nodes - 1) / num_nodes
        return self.get_transfer_time(int(total_send_bytes), use_profiling=use_profiling)

    def get_send_recv_time(
        self,
        data_size_bytes: int,
        use_profiling: bool = False,
    ) -> float:
        """计算跨节点点对点 Send/Recv 时间 (ms)。

        用于跨节点的流水线并行 stage 间通信。
        """
        return self.get_transfer_time(data_size_bytes, use_profiling=use_profiling)
