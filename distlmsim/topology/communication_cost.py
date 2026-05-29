"""通信开销计算器

根据并行策略和网络拓扑，计算各通信阶段的开销。
统一封装 NVLink (节点内) 和 RDMA (节点间) 的通信时间计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from distlmsim.config import NetworkTopologyConfig
from distlmsim.topology.nvlink_model import NVLinkModel
from distlmsim.topology.rdma_model import RDMAModel
from distlmsim.topology.network_topology import NetworkTopology
from distlmsim.types import NetworkModelMode


@dataclass
class CommunicationBreakdown:
    """一次通信的时间分解。"""
    compute_time_ms: float = 0.0       # 计算重叠时间 (可隐藏)
    transfer_time_ms: float = 0.0      # 数据传输时间
    latency_ms: float = 0.0            # 基础延迟
    nccl_overhead_ms: float = 0.0      # NCCL CPU 启动开销
    congestion_penalty_ms: float = 0.0 # 拥塞惩罚

    @property
    def total_time_ms(self) -> float:
        return (
            self.transfer_time_ms
            + self.latency_ms
            + self.nccl_overhead_ms
            + self.congestion_penalty_ms
        )


class CommunicationCostCalculator:
    """通信开销计算器。

    根据并行策略配置和网络拓扑，自动选择 NVLink 或 RDMA 模型，
    计算 TP all-reduce、PP send/recv、EP all-to-all 的通信时间。
    """

    def __init__(
        self,
        network_config: NetworkTopologyConfig,
        topology: NetworkTopology,
        num_gpus_per_node: int = 8,
    ):
        self._config = network_config
        self._topology = topology
        self._num_gpus_per_node = num_gpus_per_node
        self._nvlink = NVLinkModel(network_config.nvlink, num_gpus_per_node)
        self._rdma = RDMAModel(network_config.rdma)
        self._use_profiling = network_config.model_mode == NetworkModelMode.PROFILING

    def tensor_parallel_allreduce(
        self,
        tp_size: int,
        data_size_bytes: int,
        node_id: int = 0,
    ) -> float:
        """计算张量并行 All-Reduce 通信时间 (ms)。

        TP 通常在同一节点内，走 NVLink。
        如果 TP 跨节点 (不推荐)，走 RDMA。

        Args:
            tp_size: 张量并行度
            data_size_bytes: 每次 all-reduce 的数据量
            node_id: 所在节点 (用于判断是否同节点)
        """
        if tp_size <= 1:
            return 0.0

        # TP 默认同节点，走 NVLink
        time_ms = self._nvlink.get_allreduce_time(
            tp_size, data_size_bytes, use_profiling=self._use_profiling
        )

        # NCCL CPU 开销
        nccl_overhead = self._config.nccl_cpu_launch_overhead_ms
        skew_overhead = (
            self._config.nccl_cpu_skew_overhead_per_device_ms
            * (tp_size ** 1.25)
        )

        return time_ms + nccl_overhead + skew_overhead

    def pipeline_parallel_send_recv(
        self,
        data_size_bytes: int,
        src_stage_node_id: int,
        dst_stage_node_id: int,
    ) -> float:
        """计算流水线并行 Send/Recv 通信时间 (ms)。

        PP stage 间通信：同节点走 NVLink，跨节点走 RDMA。

        Args:
            data_size_bytes: stage 间传输的激活数据量
            src_stage_node_id: 源 stage 所在节点
            dst_stage_node_id: 目标 stage 所在节点
        """
        if src_stage_node_id == dst_stage_node_id:
            return self._nvlink.get_send_recv_time(data_size_bytes)
        else:
            return self._rdma.get_send_recv_time(
                data_size_bytes, use_profiling=self._use_profiling
            )

    def expert_parallel_alltoall(
        self,
        ep_size: int,
        data_size_per_gpu_bytes: int,
        node_ids: list[int],
    ) -> float:
        """计算专家并行 All-to-All 通信时间 (ms)。

        EP 的 token 分发和收集。可能跨节点。
        分两阶段: dispatch (发送到目标专家) + combine (收集结果)。

        Args:
            ep_size: 专家并行度
            data_size_per_gpu_bytes: 每 GPU 发送的数据量
            node_ids: 参与的节点 ID 列表

        Returns:
            总通信时间 (ms)，包含 dispatch + combine
        """
        if ep_size <= 1:
            return 0.0

        # 判断是否所有 GPU 在同一节点
        unique_nodes = set(node_ids)
        if len(unique_nodes) == 1:
            # 同节点，走 NVLink
            return self._nvlink.get_alltoall_time(ep_size, data_size_per_gpu_bytes) * 2
        else:
            # 跨节点，走 RDMA (dispatch + combine = 2x)
            num_nodes = len(unique_nodes)
            return self._rdma.get_alltoall_time(
                num_nodes, data_size_per_gpu_bytes * ep_size,
                use_profiling=self._use_profiling
            ) * 2

    def kv_cache_transfer(
        self,
        kv_cache_size_bytes: int,
        src_node_id: int,
        dst_node_id: int,
        compression_ratio: float = 1.0,
    ) -> float:
        """计算 KV Cache RDMA 传输时间 (ms)。

        用于存算分离模式下，Prefill 节点向 Decode 节点传输 KV Cache。

        Args:
            kv_cache_size_bytes: KV Cache 原始大小
            src_node_id: Prefill 节点 ID
            dst_node_id: Decode 节点 ID
            compression_ratio: 压缩比 (如 2.0 表示压缩一半)
        """
        effective_size = int(kv_cache_size_bytes / compression_ratio)
        return self._rdma.get_transfer_time(
            effective_size, src_node_id, dst_node_id,
            use_profiling=self._use_profiling,
        )

    def get_communication_breakdown(
        self,
        tp_size: int,
        pp_size: int,
        ep_size: int,
        tp_data_bytes: int,
        pp_data_bytes: int,
        ep_data_bytes: int,
        src_node_id: int = 0,
        dst_node_id: int = 0,
    ) -> dict:
        """获取完整的通信时间分解。

        Returns:
            dict 包含 tp_time, pp_time, ep_time (均为 ms)
        """
        return {
            "tp_time_ms": self.tensor_parallel_allreduce(tp_size, tp_data_bytes),
            "pp_time_ms": self.pipeline_parallel_send_recv(
                pp_data_bytes, src_node_id, dst_node_id
            ),
            "ep_time_ms": self.expert_parallel_alltoall(
                ep_size, ep_data_bytes, [src_node_id]
            ),
        }
