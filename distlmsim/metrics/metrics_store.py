"""指标收集与统计

收集和统计模拟过程中的各项性能指标。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from distlmsim.config import MetricsConfig


@dataclass
class RequestMetrics:
    """单个请求的指标。"""
    request_id: int
    arrival_time: float = 0.0
    scheduled_time: float = 0.0
    prefill_start_time: float = 0.0
    prefill_end_time: float = 0.0
    decode_start_time: float = 0.0
    decode_end_time: float = 0.0
    kv_cache_transfer_start: float = 0.0
    kv_cache_transfer_end: float = 0.0
    prefill_node_id: int = -1
    decode_node_id: int = -1
    prefill_tokens: int = 0
    decode_tokens: int = 0

    @property
    def ttft(self) -> float:
        """Time to First Token (ms): prefill 完成 - 请求到达"""
        if self.prefill_end_time <= 0:
            return 0.0
        return self.prefill_end_time - self.arrival_time

    @property
    def tbt(self) -> float:
        """Time Between Tokens (ms): decode 阶段平均每 token 间隔"""
        if self.decode_end_time <= self.decode_start_time or self.decode_tokens <= 1:
            return 0.0
        return (self.decode_end_time - self.decode_start_time) / (self.decode_tokens - 1)

    @property
    def e2e_latency(self) -> float:
        """端到端延迟 (ms): decode 完成 - 请求到达"""
        if self.decode_end_time <= 0:
            return 0.0
        return self.decode_end_time - self.arrival_time

    @property
    def scheduling_delay(self) -> float:
        return self.scheduled_time - self.arrival_time if self.scheduled_time > 0 else 0.0

    @property
    def kv_cache_transfer_time(self) -> float:
        return self.kv_cache_transfer_end - self.kv_cache_transfer_start

    @property
    def prefill_time(self) -> float:
        return self.prefill_end_time - self.prefill_start_time

    @property
    def decode_time(self) -> float:
        return self.decode_end_time - self.decode_start_time


class MetricsStore:
    """指标收集存储。"""

    def __init__(self, config: MetricsConfig):
        self._config = config
        self._request_metrics: Dict[int, RequestMetrics] = {}
        self._prefill_node_counts: Dict[int, int] = defaultdict(int)
        self._decode_node_counts: Dict[int, int] = defaultdict(int)

    def record_request_arrival(self, request_id: int, time: float) -> None:
        self._request_metrics[request_id] = RequestMetrics(
            request_id=request_id,
            arrival_time=time,
        )

    def record_request_scheduled(self, request_id: int, time: float) -> None:
        if request_id in self._request_metrics:
            self._request_metrics[request_id].scheduled_time = time

    def record_prefill_start(self, request_id: int, time: float, node_id: int) -> None:
        if request_id in self._request_metrics:
            m = self._request_metrics[request_id]
            m.prefill_start_time = time
            m.prefill_node_id = node_id
            self._prefill_node_counts[node_id] += 1

    def record_prefill_end(self, request_id: int, time: float) -> None:
        if request_id in self._request_metrics:
            self._request_metrics[request_id].prefill_end_time = time

    def record_kv_cache_transfer_start(self, request_id: int, time: float) -> None:
        if request_id in self._request_metrics:
            self._request_metrics[request_id].kv_cache_transfer_start = time

    def record_kv_cache_transfer_end(self, request_id: int, time: float) -> None:
        if request_id in self._request_metrics:
            self._request_metrics[request_id].kv_cache_transfer_end = time

    def record_decode_start(self, request_id: int, time: float, node_id: int) -> None:
        if request_id in self._request_metrics:
            m = self._request_metrics[request_id]
            m.decode_start_time = time
            m.decode_node_id = node_id
            self._decode_node_counts[node_id] += 1

    def record_decode_end(self, request_id: int, time: float) -> None:
        if request_id in self._request_metrics:
            self._request_metrics[request_id].decode_end_time = time

    def set_request_tokens(self, request_id: int, prefill_tokens: int, decode_tokens: int) -> None:
        if request_id in self._request_metrics:
            m = self._request_metrics[request_id]
            m.prefill_tokens = prefill_tokens
            m.decode_tokens = decode_tokens

    def finalize(self) -> None:
        pass

    def print_summary(self) -> None:
        completed = [
            m for m in self._request_metrics.values()
            if m.decode_end_time > 0
        ]

        if not completed:
            print("没有完成的请求。")
            return

        ttfts = [m.ttft for m in completed]
        e2e_latencies = [m.e2e_latency for m in completed]
        tbts = [m.tbt for m in completed if m.tbt > 0]
        kv_times = [m.kv_cache_transfer_time for m in completed if m.kv_cache_transfer_time > 0]
        prefill_times = [m.prefill_time for m in completed if m.prefill_time > 0]
        decode_times = [m.decode_time for m in completed if m.decode_time > 0]

        # 吞吐量
        total_decode_tokens = sum(m.decode_tokens for m in completed if m.decode_tokens > 0)
        total_prefill_tokens = sum(m.prefill_tokens for m in completed if m.prefill_tokens > 0)
        wall_time = max(m.decode_end_time for m in completed)
        first_arrival = min(m.arrival_time for m in completed)

        print(f"\n{'='*64}")
        print(f"  DistLMSim 模拟结果汇总")
        print(f"{'='*64}")
        print(f"  完成请求数:    {len(completed)}")
        print(f"  总请求数:      {len(self._request_metrics)}")
        print(f"  模拟时长:      {wall_time:.1f} ms ({wall_time/1000:.2f} s)")

        if total_prefill_tokens > 0:
            print(f"\n  --- Prefill ---")
            print(f"  总 prefill tokens: {total_prefill_tokens}")
            if prefill_times:
                print(f"  Prefill 延迟 (ms): mean={np.mean(prefill_times):.2f}, "
                      f"P50={np.percentile(prefill_times, 50):.2f}")

        if kv_times:
            print(f"\n  --- KV Cache 传输 (RDMA) ---")
            print(f"  传输延迟 (ms):  mean={np.mean(kv_times):.2f}, "
                  f"P50={np.percentile(kv_times, 50):.2f}, "
                  f"P99={np.percentile(kv_times, 99):.2f}")

        print(f"\n  --- TTFT (Time to First Token, ms) ---")
        for p in self._config.percentiles:
            print(f"    P{p}: {np.percentile(ttfts, p):.2f}")

        if tbts:
            print(f"\n  --- TBT (Time Between Tokens, ms) ---")
            for p in self._config.percentiles:
                print(f"    P{p}: {np.percentile(tbts, p):.2f}")

        if decode_times:
            print(f"\n  --- Decode 阶段延迟 (ms) ---")
            print(f"    mean={np.mean(decode_times):.2f}, "
                  f"P50={np.percentile(decode_times, 50):.2f}")

        print(f"\n  --- E2E Latency (端到端延迟, ms) ---")
        for p in self._config.percentiles:
            print(f"    P{p}: {np.percentile(e2e_latencies, p):.2f}")

        if wall_time > first_arrival:
            effective_time_s = (wall_time - first_arrival) / 1000.0
            if total_decode_tokens > 0:
                throughput = total_decode_tokens / effective_time_s
                print(f"\n  --- 吞吐量 ---")
                print(f"    Decode tokens/s: {throughput:.1f}")
            if total_prefill_tokens > 0:
                prefill_throughput = total_prefill_tokens / effective_time_s
                print(f"    Prefill tokens/s: {prefill_throughput:.1f}")

        print(f"\n  --- 节点负载 ---")
        print(f"    Prefill 节点: {dict(self._prefill_node_counts)}")
        print(f"    Decode  节点: {dict(self._decode_node_counts)}")
        print(f"{'='*64}\n")

    def get_completed_count(self) -> int:
        return sum(
            1 for m in self._request_metrics.values()
            if m.decode_end_time > 0
        )
