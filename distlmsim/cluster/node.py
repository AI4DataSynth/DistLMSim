"""物理节点定义

描述单个 DGX 节点的硬件配置和运行状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from distlmsim.config import NodeSKUConfig
from distlmsim.types import NodeRole


@dataclass
class GPUDevice:
    """单个 GPU 设备。"""
    global_id: int                     # 全局 GPU ID
    local_id: int                      # 节点内 GPU ID
    node_id: int                       # 所属节点 ID
    memory_used_gb: float = 0.0        # 已用显存 (GB)
    memory_total_gb: float = 80.0      # 总显存 (GB)
    is_available: bool = True          # 是否可用

    @property
    def memory_free_gb(self) -> float:
        return self.memory_total_gb - self.memory_used_gb


class PhysicalNode:
    """物理节点。

    一个 DGX 节点包含多个 GPU，通过 NVLink/NVSwitch 互联。
    节点间通过 RDMA 网络连接。
    """

    def __init__(
        self,
        node_id: int,
        sku: NodeSKUConfig,
        global_gpu_id_offset: int = 0,
        role: NodeRole = NodeRole.MIXED,
    ):
        self._id = node_id
        self._sku = sku
        self._role = role
        self._gpus: List[GPUDevice] = []

        # 创建 GPU 设备
        for local_id in range(sku.num_gpus):
            gpu = GPUDevice(
                global_id=global_gpu_id_offset + local_id,
                local_id=local_id,
                node_id=node_id,
                memory_total_gb=sku.device_sku.memory_gb,
            )
            self._gpus.append(gpu)

        self._replica_ids: List[int] = []  # 此节点上运行的副本 ID
        self._rdma_nic_bandwidth_gbps = 200.0  # 默认，后续从配置读取

    @property
    def id(self) -> int:
        return self._id

    @property
    def role(self) -> NodeRole:
        return self._role

    @role.setter
    def role(self, value: NodeRole) -> None:
        self._role = value

    @property
    def num_gpus(self) -> int:
        return len(self._gpus)

    @property
    def gpus(self) -> List[GPUDevice]:
        return self._gpus

    @property
    def global_gpu_ids(self) -> List[int]:
        return [gpu.global_id for gpu in self._gpus]

    @property
    def replica_ids(self) -> List[int]:
        return self._replica_ids

    def add_replica(self, replica_id: int) -> None:
        if replica_id not in self._replica_ids:
            self._replica_ids.append(replica_id)

    def remove_replica(self, replica_id: int) -> None:
        if replica_id in self._replica_ids:
            self._replica_ids.remove(replica_id)

    def get_available_gpus(self) -> List[GPUDevice]:
        """获取可用的 GPU 列表。"""
        return [gpu for gpu in self._gpus if gpu.is_available]

    def get_total_memory_free_gb(self) -> float:
        """获取节点总可用显存。"""
        return sum(gpu.memory_free_gb for gpu in self._gpus)

    def allocate_gpu_memory(self, local_gpu_id: int, size_gb: float) -> bool:
        """在指定 GPU 上分配显存。"""
        gpu = self._gpus[local_gpu_id]
        if gpu.memory_free_gb < size_gb:
            return False
        gpu.memory_used_gb += size_gb
        return True

    def free_gpu_memory(self, local_gpu_id: int, size_gb: float) -> None:
        """释放指定 GPU 的显存。"""
        self._gpus[local_gpu_id].memory_used_gb = max(
            0.0, self._gpus[local_gpu_id].memory_used_gb - size_gb
        )
