"""请求生成器

生成模拟推理请求，支持合成流量和 trace 回放两种模式。
支持多种到达间隔分布（Poisson/Gamma）和长度分布（Fixed/Normal/Lognormal/Zipf）。
"""

from __future__ import annotations

import csv
import itertools
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from distlmsim.config import RequestGeneratorConfig
from distlmsim.entities import Request
from distlmsim.events import BaseEvent, RequestArrivalEvent


_request_id_counter = itertools.count()


# ─── 到达间隔生成器 ────────────────────────────────────────────────────────────


class BaseIntervalGenerator(ABC):
    """到达间隔生成器基类。"""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> float:
        """采样一个到达间隔（秒）。"""
        ...


class PoissonIntervalGenerator(BaseIntervalGenerator):
    """泊松过程到达间隔（指数分布）。"""

    def __init__(self, qps: float):
        self._qps = qps

    def sample(self, rng: np.random.Generator) -> float:
        if self._qps <= 0:
            return float("inf")
        return float(rng.exponential(1.0 / self._qps))


class GammaIntervalGenerator(BaseIntervalGenerator):
    """Gamma 分布到达间隔。

    Gamma(k, scale) 其中 scale = 1 / (k * qps)，使得均值 = 1/qps。
    shape k 越大，间隔越集中（接近确定性）；k=1 退化为指数分布。
    """

    def __init__(self, qps: float, shape: float = 2.0):
        self._qps = qps
        self._shape = shape  # k

    def sample(self, rng: np.random.Generator) -> float:
        if self._qps <= 0:
            return float("inf")
        scale = 1.0 / (self._shape * self._qps)
        return float(rng.gamma(self._shape, scale))


# ─── 长度生成器 ────────────────────────────────────────────────────────────────


class BaseLengthGenerator(ABC):
    """请求长度生成器基类。"""

    @abstractmethod
    def sample(self, mean: int, rng: np.random.Generator) -> int:
        """采样一个请求长度。"""
        ...


class FixedLengthGenerator(BaseLengthGenerator):
    """固定长度。"""

    def sample(self, mean: int, rng: np.random.Generator) -> int:
        return mean


class NormalLengthGenerator(BaseLengthGenerator):
    """正态分布长度。"""

    def __init__(self, cv: float = 0.3):
        self._cv = cv

    def sample(self, mean: int, rng: np.random.Generator) -> int:
        std = mean * self._cv
        return max(1, int(rng.normal(mean, std)))


class LognormalLengthGenerator(BaseLengthGenerator):
    """对数正态分布长度。"""

    def __init__(self, cv: float = 0.3):
        self._cv = cv

    def sample(self, mean: int, rng: np.random.Generator) -> int:
        cv = self._cv
        sigma = np.sqrt(np.log(1 + cv ** 2))
        mu = np.log(mean) - sigma ** 2 / 2
        return max(1, int(rng.lognormal(mu, sigma)))


class ZipfLengthGenerator(BaseLengthGenerator):
    """Zipf 分布长度。

    Zipf 分布产生幂律分布的长尾值。
    采样后缩放使得均值接近目标 mean。
    """

    def __init__(self, alpha: float = 1.5):
        self._alpha = alpha

    def sample(self, mean: int, rng: np.random.Generator) -> int:
        # numpy zipf 采样 (a > 1)，值从 1 开始
        raw = rng.zipf(self._alpha)
        # Zipf(a) 的期望 = zeta(a-1) / zeta(a) (a > 2 时有限)
        # 简单方法：缩放使得期望接近 mean
        # 对于 a=1.5, E[X] 约 3.6，缩放因子 = mean / E[X_approx]
        # 使用近似缩放
        if self._alpha > 2:
            from scipy.special import zeta as _zeta
            expected = _zeta(self._alpha - 1) / _zeta(self._alpha)
        else:
            # 近似：对于 a=1.5，经验值约 3.6
            expected = max(1.0, 1.0 / (1.0 - 1.0 / self._alpha) if self._alpha > 1 else 1.0)
        scaled = max(1, int(raw * mean / expected))
        return scaled


class BaseRequestGenerator(ABC):
    """请求生成器基类。"""

    @classmethod
    def from_config(cls, config: RequestGeneratorConfig) -> "BaseRequestGenerator":
        """根据配置创建请求生成器。"""
        if config.generator_type == "trace_replay":
            return TraceReplayRequestGenerator(config, trace_file=config.trace_file or "")
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

    支持多种到达间隔分布（poisson/gamma）和长度分布（fixed/normal/lognormal/zipf）。
    """

    def __init__(self, config: RequestGeneratorConfig):
        self._config = config
        self._rng = np.random.default_rng(42)
        self._next_arrival_time = 0.0

        # 初始化到达间隔生成器
        self._interval_gen: BaseIntervalGenerator = self._create_interval_generator(config)
        # 初始化长度生成器
        self._length_gen: BaseLengthGenerator = self._create_length_generator(config)

    @staticmethod
    def _create_interval_generator(config: RequestGeneratorConfig) -> BaseIntervalGenerator:
        """根据配置创建到达间隔生成器。"""
        if config.interval_distribution == "gamma":
            return GammaIntervalGenerator(config.qps, shape=config.gamma_shape)
        return PoissonIntervalGenerator(config.qps)

    @staticmethod
    def _create_length_generator(config: RequestGeneratorConfig) -> BaseLengthGenerator:
        """根据配置创建长度生成器。"""
        gen_type = config.length_generator_type
        if gen_type == "fixed":
            return FixedLengthGenerator()
        elif gen_type == "zipf":
            return ZipfLengthGenerator(alpha=config.zipf_alpha)
        elif gen_type == "lognormal":
            return LognormalLengthGenerator(cv=config.length_cv)
        else:  # "normal" (default)
            return NormalLengthGenerator(cv=config.length_cv)

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
        """采样请求到达间隔 (ms)。"""
        interval_s = self._interval_gen.sample(self._rng)
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
            domain=self._sample_domain(),
        )

    def _sample_domain(self) -> str:
        """按论文训练数据比例采样 workload domain。

        比例来源: Open-PerfectBlend (DSpark 论文 Section 4.1)
        - math: 39.4%
        - code: 38.9%
        - chat: 17.6%
        - instruction: 4.1% (归入 chat)
        """
        r = self._rng.random()
        if r < 0.394:
            return "math"
        elif r < 0.783:
            return "code"
        else:
            return "chat"

    def _sample_length(self, mean: int) -> int:
        """采样请求长度。委托给长度生成器。"""
        return self._length_gen.sample(mean, self._rng)

    def generate_expert_distributions(
        self,
        num_layers: int = 48,
        num_experts: int = 128,
        alpha: float = 1.5,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """生成 MoE 专家路由分布。

        使用 Zipf-like 分布模拟专家热门程度：少数专家被频繁选中，
        大部分专家较少被选中。

        Args:
            num_layers: 模型层数
            num_experts: 每层专家数
            alpha: Zipf 分布参数（越大越不均匀）

        Returns:
            (prefill_expert_distribution, decode_expert_distributions)
            - prefill_expert_distribution: shape [num_layers, num_experts]
            - decode_expert_distributions: list of [num_layers, num_experts]，
              每个 decode step 一个分布
        """
        # 生成基础 Zipf 权重: w_i = 1 / i^alpha，然后归一化
        ranks = np.arange(1, num_experts + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, alpha)
        weights /= weights.sum()

        # prefill 分布：每层打乱专家顺序（模拟不同层的 router 偏好）
        prefill_dist = np.zeros((num_layers, num_experts), dtype=np.float64)
        for layer in range(num_layers):
            perm = self._rng.permutation(num_experts)
            prefill_dist[layer] = weights[perm]

        # decode 分布：每个 decode step 略有变化（模拟动态路由）
        # 生成一个基础分布，然后在各 step 间添加微小扰动
        num_decode_steps = self._config.decode_length
        decode_dists: List[np.ndarray] = []
        for _step in range(num_decode_steps):
            step_dist = np.zeros((num_layers, num_experts), dtype=np.float64)
            for layer in range(num_layers):
                # 在基础权重上添加微小噪声
                noise = self._rng.dirichlet(np.ones(num_experts) * 10.0)
                mixed = 0.8 * weights + 0.2 * noise
                mixed /= mixed.sum()
                perm = self._rng.permutation(num_experts)
                step_dist[layer] = mixed[perm]
            decode_dists.append(step_dist)

        return prefill_dist, decode_dists


class TraceReplayRequestGenerator(BaseRequestGenerator):
    """Trace 回放请求生成器。

    从 CSV trace 文件中读取请求序列，按时间戳回放。
    Trace 文件格式: arrival_time_ms,prefill_tokens,decode_tokens
    """

    def __init__(self, config: RequestGeneratorConfig, trace_file: str = ""):
        self._config = config
        self._trace_index = 0
        self._trace_entries: List[dict] = []
        if trace_file and os.path.exists(trace_file):
            self._load_trace(trace_file)

    def _load_trace(self, trace_file: str) -> None:
        """Load trace entries from CSV file."""
        with open(trace_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._trace_entries.append({
                    'arrival_time_ms': float(row.get('arrival_time_ms', 0)),
                    'prefill_tokens': int(row.get('prefill_tokens', 128)),
                    'decode_tokens': int(row.get('decode_tokens', 128)),
                })

    def generate_initial_events(self) -> List[BaseEvent]:
        """Generate RequestArrivalEvents from trace entries at their arrival times."""
        events = []
        for i, entry in enumerate(self._trace_entries):
            # Only create events for requests arriving at time 0
            if entry['arrival_time_ms'] <= 0:
                request_id = next(_request_id_counter)
                evt = RequestArrivalEvent(time=0.0, request_id=request_id)
                events.append(evt)
        self._trace_index = len(events)
        return events

    def generate_next_request(self, current_time: float) -> Optional[BaseEvent]:
        """Get the next trace entry whose arrival time is >= current_time."""
        while self._trace_index < len(self._trace_entries):
            entry = self._trace_entries[self._trace_index]
            self._trace_index += 1
            if entry['arrival_time_ms'] >= current_time:
                request_id = next(_request_id_counter)
                return RequestArrivalEvent(
                    time=entry['arrival_time_ms'],
                    request_id=request_id,
                )
        return None
