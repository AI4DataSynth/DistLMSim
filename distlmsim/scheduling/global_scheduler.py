"""全局调度器

将到达的请求分配到集群中的某个副本。
支持轮询、随机、最少未完成请求、拓扑感知等策略。

依赖层次: Layer 6
  输入: config (SchedulingConfig), types (GlobalSchedulerType), interfaces (ClusterView)
  输出: BaseGlobalScheduler 及其子类 (被 simulator 消费)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from distlmsim.config import SchedulingConfig
from distlmsim.interfaces import ClusterView
from distlmsim.types import GlobalSchedulerType


class BaseGlobalScheduler(ABC):
    """全局调度器基类。

    管理所有副本级调度器，将请求路由到合适的副本。
    """

    def __init__(self, cluster: ClusterView):
        self._cluster = cluster
        self._replica_schedulers: Dict[int, object] = {}

    @classmethod
    def from_config(
        cls, config: SchedulingConfig, cluster: ClusterView
    ) -> BaseGlobalScheduler:
        """根据配置创建全局调度器。"""
        scheduler_type = config.global_scheduler_type
        if scheduler_type == GlobalSchedulerType.ROUND_ROBIN:
            return RoundRobinGlobalScheduler(cluster)
        elif scheduler_type == GlobalSchedulerType.RANDOM:
            return RandomGlobalScheduler(cluster)
        elif scheduler_type == GlobalSchedulerType.LEAST_OUTSTANDING:
            return LeastOutstandingGlobalScheduler(cluster)
        elif scheduler_type == GlobalSchedulerType.TOPOLOGY_AWARE:
            return TopologyAwareGlobalScheduler(cluster)
        else:
            return RoundRobinGlobalScheduler(cluster)

    @abstractmethod
    def select_replica(self, request_id: int) -> int:
        """为请求选择一个副本。

        Args:
            request_id: 请求 ID

        Returns:
            选中的副本 ID
        """
        ...

    @property
    def replica_schedulers(self) -> Dict[int, object]:
        return self._replica_schedulers


class RoundRobinGlobalScheduler(BaseGlobalScheduler):
    """轮询全局调度器。"""

    def __init__(self, cluster: ClusterView):
        super().__init__(cluster)
        self._next_replica_id = 0

    def select_replica(self, request_id: int) -> int:
        num_replicas = len(self._cluster.replicas)
        replica_id = self._next_replica_id % num_replicas
        self._next_replica_id += 1
        return replica_id


class RandomGlobalScheduler(BaseGlobalScheduler):
    """随机全局调度器。"""

    def __init__(self, cluster: ClusterView):
        super().__init__(cluster)
        import random
        self._random = random

    def select_replica(self, request_id: int) -> int:
        replica_ids = list(self._cluster.replicas.keys())
        return self._random.choice(replica_ids)


class LeastOutstandingGlobalScheduler(BaseGlobalScheduler):
    """最少未完成请求全局调度器。

    将请求分配到当前排队请求最少的副本。
    """

    def __init__(self, cluster: ClusterView):
        super().__init__(cluster)
        self._outstanding_counts: Dict[int, int] = {
            rid: 0 for rid in cluster.replicas
        }

    def select_replica(self, request_id: int) -> int:
        min_count = float("inf")
        best_replica = 0
        for replica_id, count in self._outstanding_counts.items():
            if count < min_count:
                min_count = count
                best_replica = replica_id

        self._outstanding_counts[best_replica] += 1
        return best_replica

    def release_request(self, replica_id: int) -> None:
        """请求完成时调用。"""
        if replica_id in self._outstanding_counts:
            self._outstanding_counts[replica_id] = max(
                0, self._outstanding_counts[replica_id] - 1
            )


class TopologyAwareGlobalScheduler(BaseGlobalScheduler):
    """拓扑感知全局调度器。

    考虑请求来源（如客户端位置）和副本所在节点的网络距离，
    优先选择网络距离最近的副本。

    TODO: 需要引入客户端位置信息。
    """

    def __init__(self, cluster: ClusterView):
        super().__init__(cluster)
        self._fallback = RoundRobinGlobalScheduler(cluster)

    def select_replica(self, request_id: int) -> int:
        replicas = list(self._cluster.replicas.keys())
        if not replicas:
            return self._fallback.select_replica(request_id)

        # Use request_id as a hash to select a replica, providing
        # some affinity (same request patterns go to same replica).
        # This simulates topology-aware routing without actual client position info.
        idx = request_id % len(replicas)
        return replicas[idx]
