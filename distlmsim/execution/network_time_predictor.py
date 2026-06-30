"""网络时间预测器

统一封装 NVLink 和 RDMA 的通信时间预测，
支持解析模型和 profiling 数据混合模式。
"""

from __future__ import annotations

from typing import Optional

from distlmsim.config import NetworkTopologyConfig
from distlmsim.topology.nvlink_model import NVLinkModel
from distlmsim.topology.rdma_model import RDMAModel
from distlmsim.types import NetworkModelMode


class NetworkTimePredictor:
    """网络时间预测器。

    根据通信类型（节点内/节点间）自动选择 NVLink 或 RDMA 模型。
    支持混合模式：优先使用 profiling 数据，回退到解析模型。
    """

    def __init__(self, config: NetworkTopologyConfig, num_gpus_per_node: int = 8, profiling_dir: Optional[str] = None):
        self._config = config
        self._nvlink = NVLinkModel(config.nvlink, num_gpus_per_node, profiling_dir=profiling_dir)
        self._rdma = RDMAModel(config.rdma, profiling_dir=profiling_dir)
        self._mode = config.model_mode
        self._num_gpus_per_node = num_gpus_per_node

    def get_tp_allreduce_time(
        self, tp_size: int, data_size_bytes: int
    ) -> float:
        """TP All-Reduce 时间 (ms)。同节点 NVLink。"""
        use_profiling = self._mode in (NetworkModelMode.PROFILING, NetworkModelMode.HYBRID)
        try:
            return self._nvlink.get_allreduce_time(tp_size, data_size_bytes, use_profiling)
        except NotImplementedError:
            return self._nvlink.get_allreduce_time(tp_size, data_size_bytes, False)

    def get_pp_send_recv_time(
        self,
        data_size_bytes: int,
        is_cross_node: bool = False,
    ) -> float:
        """PP Send/Recv 时间 (ms)。"""
        if is_cross_node:
            return self._get_rdma_time(data_size_bytes)
        else:
            return self._nvlink.get_send_recv_time(data_size_bytes)

    def get_ep_alltoall_time(
        self,
        ep_size: int,
        data_size_per_gpu_bytes: int,
        is_cross_node: bool = False,
    ) -> float:
        """EP All-to-All 时间 (ms)。"""
        if is_cross_node:
            gpus_per_node = self._num_gpus_per_node
            num_nodes = (ep_size + gpus_per_node - 1) // gpus_per_node
            data_per_node = data_size_per_gpu_bytes * gpus_per_node
            return self._rdma.get_alltoall_time(
                num_nodes, data_per_node
            ) * 2  # dispatch + combine
        else:
            return self._nvlink.get_alltoall_time(ep_size, data_size_per_gpu_bytes) * 2

    def get_kv_cache_transfer_time(
        self,
        kv_cache_size_bytes: int,
        compression_ratio: float = 1.0,
    ) -> float:
        """KV Cache RDMA 传输时间 (ms)。"""
        effective_size = int(kv_cache_size_bytes / compression_ratio)
        return self._get_rdma_time(effective_size)

    def _get_rdma_time(self, data_size_bytes: int) -> float:
        """RDMA 传输时间，自动选择 profiling 或解析模型。"""
        use_profiling = self._mode in (NetworkModelMode.PROFILING, NetworkModelMode.HYBRID)
        try:
            return self._rdma.get_transfer_time(data_size_bytes, use_profiling=use_profiling)
        except NotImplementedError:
            return self._rdma.get_transfer_time(data_size_bytes, use_profiling=False)
