"""请求生成器

生成模拟推理请求，支持合成流量和 trace 回放两种模式。
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from distlmsim.config import RequestGeneratorConfig
from distlmsim.entities import Request
from distlmsim.events import BaseEvent, RequestArrivalEvent


_request_id_counter = itertools.count()


class BaseRequestGenerator(ABC):
    """请求生成器基类。"""

    @classmethod
    def from_config(cls, config: RequestGeneratorConfig) -> "BaseRequestGenerator":
        """根据配置创建请求生成器。"""
        if config.generator_type == "trace_replay":
            return TraceReplayRequestGenerator(config)
        return SyntheticRequestGenerator(config)

    @abstractmethod
    def generate_initial_events(self) -> List[BaseEvent]:
        """生成初始请求到达事件。"""
        ...

    @abstractmethod
    def generate_next_request(self, current_time: float) -> Optional[BaseEvent]:
        """生成下一个请求到达事件。"""
        ...


class SyntheticRequestGenerator(BaseRequestGenerator):
    """合成请求生成器。

    使用泊松过程生成请求到达间隔，正态/对数正态分布生成长度。
    """

    def __init__(self, config: RequestGeneratorConfig):
        self._config = config
        self._rng = np.random.default_rng(42)
        self._next_arrival_time = 0.0

    def generate_initial_events(self) -> List[BaseEvent]:
        """生成第一批请求。"""
        events = []
        request_id = next(_request_id_counter)
        events.append(RequestArrivalEvent(0.0, request_id))
        # 预计算下一次到达时间
        interval_ms = self._sample_interval()
        self._next_arrival_time = interval_ms
        return events

    def generate_next_request(self, current_time: float) -> Optional[BaseEvent]:
        interval_ms = self._sample_interval()
        arrival_time = current_time + interval_ms
        request_id = next(_request_id_counter)
        return RequestArrivalEvent(arrival_time, request_id)

    def _sample_interval(self) -> float:
        """采样请求到达间隔 (ms)。泊松过程 -> 指数分布。"""
        qps = self._config.qps
        if qps <= 0:
            return float("inf")
        interval_s = self._rng.exponential(1.0 / qps)
        return interval_s * 1e3  # 转换为 ms

    def create_request(self, arrival_time: float) -> Request:
        """创建一个合成请求。"""
        prefill_tokens = self._sample_length(self._config.prefill_length)
        decode_tokens = self._sample_length(self._config.decode_length)

        return Request(
            id=next(_request_id_counter),
            arrival_time=arrival_time,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
        )

    def _sample_length(self, mean: int) -> int:
        """采样请求长度。"""
        dist = self._config.length_distribution
        if dist == "fixed":
            return mean
        elif dist == "normal":
            std = mean * self._config.length_cv
            return max(1, int(self._rng.normal(mean, std)))
        elif dist == "lognormal":
            # 对数正态: 给定 mean 和 cv，反推 mu 和 sigma
            cv = self._config.length_cv
            sigma = np.sqrt(np.log(1 + cv ** 2))
            mu = np.log(mean) - sigma ** 2 / 2
            return max(1, int(self._rng.lognormal(mu, sigma)))
        return mean


class TraceReplayRequestGenerator(BaseRequestGenerator):
    """Trace 回放请求生成器。

    从 trace 文件中读取请求序列，按时间戳回放。

    TODO: 实现 trace 文件解析。
    """

    def __init__(self, config: RequestGeneratorConfig):
        self._config = config
        self._trace_index = 0

    def generate_initial_events(self) -> List[BaseEvent]:
        # TODO: 从 trace 文件加载
        return []

    def generate_next_request(self, current_time: float) -> Optional[BaseEvent]:
        # TODO: 读取下一条 trace
        return None
