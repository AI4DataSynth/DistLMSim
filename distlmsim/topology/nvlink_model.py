"""A800 NVLink/NVSwitch 互联模型

建模节点内 GPU 间的 NVLink 通信开销。
A800 DGX: 8 GPU 通过 NVSwitch 全互联，每 GPU 12 条 NVLink3 链路。
"""

from __future__ import annotations

import csv
import math
import os
from typing import Optional

from distlmsim.config import NVLinkConfig
from distlmsim.types import InterconnectType


class NVLinkModel:
    """NVLink/NVSwitch 通信模型。

    支持两种建模模式:
    1. 解析模型: 基于带宽公式计算通信时间
    2. Profiling 模型: 从预采集的 CSV 数据查表 (后续实现)
    """

    def __init__(self, config: NVLinkConfig, num_gpus_per_node: int = 8, profiling_dir: Optional[str] = None):
        self._config = config
        self._num_gpus = num_gpus_per_node
        self._profiling_dir = profiling_dir
        self._allreduce_profiling_cache: Optional[dict] = None

    def get_allreduce_time(
        self,
        num_gpus: int,
        data_size_bytes: int,
        use_profiling: bool = False,
    ) -> float:
        """计算 All-Reduce 通信时间 (ms)。

        用于张量并行 (TP) 的梯度/激活同步。
        Ring All-Reduce: 2 * (N-1)/N * data_size / bandwidth

        Args:
            num_gpus: 参与通信的 GPU 数量 (TP size)
            data_size_bytes: 单次通信的数据量 (bytes)
            use_profiling: 是否使用 profiling 数据 (需要预采集)

        Returns:
            通信时间 (ms)
        """
        if num_gpus <= 1:
            return 0.0

        if use_profiling:
            return self._get_allreduce_from_profiling(num_gpus, data_size_bytes)

        return self._get_allreduce_analytical(num_gpus, data_size_bytes)

    def _get_allreduce_analytical(self, num_gpus: int, data_size_bytes: int) -> float:
        """解析模型计算 All-Reduce 时间。

        NVSwitch 全互联: 所有 GPU 同时收发，带宽受限于单 GPU NVLink 带宽。
        有效带宽 = min(NVSwitch 端口带宽, NVLink 链路总带宽)
        """
        # NVSwitch 模式下，有效带宽 = NVSwitch 端口带宽
        effective_bw_gbps = self._config.nvswitch_bandwidth_gbps
        effective_bw_Bps = effective_bw_gbps * 1e9 / 8  # 转换为 bytes/s

        # Ring All-Reduce: 2 * (N-1)/N 轮通信
        ring_factor = 2.0 * (num_gpus - 1) / num_gpus
        transfer_time_ms = (data_size_bytes * ring_factor) / effective_bw_Bps * 1e3

        # 加上基础延迟
        latency_ms = self._config.latency_us / 1e3

        return transfer_time_ms + latency_ms

    def _load_allreduce_profiling(self) -> dict:
        """加载 All-Reduce profiling CSV 数据到内存缓存。

        Returns:
            嵌套字典 {num_workers: {size: median_time_ms}}
        """
        if self._allreduce_profiling_cache is not None:
            return self._allreduce_profiling_cache
        if not self._profiling_dir:
            raise FileNotFoundError("No profiling_dir configured")
        csv_path = os.path.join(self._profiling_dir, "network", "a800_dgx", "all_reduce.csv")
        cache = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                nw = int(row['num_workers'])
                size = int(row['size'])
                time_ms = float(row['time_stats.all_reduce.median'])
                if nw not in cache:
                    cache[nw] = {}
                cache[nw][size] = time_ms
        self._allreduce_profiling_cache = cache
        return cache

    def _get_allreduce_from_profiling(self, num_gpus: int, data_size_bytes: int) -> float:
        """从 profiling CSV 数据查询 All-Reduce 时间。

        使用 log-linear 插值处理不在 profiling 数据中的数据大小。

        Args:
            num_gpus: 参与通信的 GPU 数量 (TP size)
            data_size_bytes: 数据量 (bytes)

        Returns:
            通信时间 (ms)
        """
        cache = self._load_allreduce_profiling()
        if num_gpus not in cache:
            # Fallback to analytical if no profiling for this TP size
            return self._get_allreduce_analytical(num_gpus, data_size_bytes)
        size_map = cache[num_gpus]
        if data_size_bytes in size_map:
            return size_map[data_size_bytes]
        sizes = sorted(size_map.keys())
        if data_size_bytes < sizes[0]:
            return size_map[sizes[0]] * (data_size_bytes / sizes[0])
        if data_size_bytes > sizes[-1]:
            return size_map[sizes[-1]] * (data_size_bytes / sizes[-1])
        for i in range(len(sizes) - 1):
            if sizes[i] <= data_size_bytes <= sizes[i + 1]:
                lo, hi = sizes[i], sizes[i + 1]
                t_lo, t_hi = size_map[lo], size_map[hi]
                frac = math.log(data_size_bytes / lo) / math.log(hi / lo)
                return t_lo + frac * (t_hi - t_lo)
        return size_map[sizes[-1]]

    def get_send_recv_time(
        self,
        data_size_bytes: int,
        num_gpus: int = 2,
    ) -> float:
        """计算点对点 Send/Recv 通信时间 (ms)。

        用于流水线并行 (PP) 的 stage 间数据传输。
        同节点内走 NVLink。

        Args:
            data_size_bytes: 数据量 (bytes)
            num_gpus: 发送方和接收方涉及的 GPU 数

        Returns:
            通信时间 (ms)
        """
        # NVSwitch 点对点带宽
        effective_bw_Bps = self._config.nvswitch_bandwidth_gbps * 1e9 / 8
        transfer_time_ms = data_size_bytes / effective_bw_Bps * 1e3
        latency_ms = self._config.latency_us / 1e3
        return transfer_time_ms + latency_ms

    def get_alltoall_time(
        self,
        num_gpus: int,
        data_size_per_gpu_bytes: int,
    ) -> float:
        """计算 All-to-All 通信时间 (ms)。

        用于专家并行 (EP) 的 token 分发和收集。
        同节点内走 NVLink。

        Args:
            num_gpus: 参与通信的 GPU 数
            data_size_per_gpu_bytes: 每 GPU 发送的数据量

        Returns:
            通信时间 (ms)
        """
        if num_gpus <= 1:
            return 0.0

        # All-to-All: 每 GPU 向其他 N-1 个 GPU 发送数据
        # NVSwitch 下可以全并行，瓶颈在单 GPU 出口带宽
        total_send_bytes = data_size_per_gpu_bytes * (num_gpus - 1)
        effective_bw_Bps = self._config.nvswitch_bandwidth_gbps * 1e9 / 8
        transfer_time_ms = total_send_bytes / effective_bw_Bps * 1e3
        latency_ms = self._config.latency_us / 1e3
        return transfer_time_ms + latency_ms
