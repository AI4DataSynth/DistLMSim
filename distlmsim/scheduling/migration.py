"""请求迁移管理

支持在运行中将请求从一个副本迁移到另一个副本，实现负载均衡。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from distlmsim.entities import Request

if TYPE_CHECKING:
    from distlmsim.cluster.cluster import Cluster
    from distlmsim.events import BaseEvent


@dataclass
class MigrationPlan:
    """迁移计划。"""
    request_id: int
    src_replica_id: int
    dst_replica_id: int
    estimated_cost_ms: float           # 迁移开销估计 (KV Cache 传输时间)
    estimated_benefit_ms: float        # 预期收益 (等待时间减少)


class RequestMigrationManager:
    """请求迁移管理器。

    周期性地评估集群负载，决定是否迁移请求以平衡负载。

    迁移条件:
    1. 源副本队列深度显著高于目标副本
    2. 迁移成本 (KV Cache 传输) 小于预期收益
    3. 目标副本有足够资源接收

    迁移开销:
    - KV Cache 数据通过 RDMA 传输
    - 跨节点迁移开销更大
    """

    def __init__(
        self,
        cluster: "Cluster",
        migration_interval_ms: float = 1000.0,
        min_benefit_ratio: float = 2.0,
    ):
        self._cluster = cluster
        self._migration_interval_ms = migration_interval_ms
        self._min_benefit_ratio = min_benefit_ratio
        self._migration_history: List[MigrationPlan] = []

    def evaluate_migrations(
        self,
        current_time: float,
        replica_queue_depths: Dict[int, int],
    ) -> List[MigrationPlan]:
        """评估是否需要迁移请求。

        Args:
            current_time: 当前模拟时间
            replica_queue_depths: 各副本的队列深度

        Returns:
            建议的迁移计划列表
        """
        plans: List[MigrationPlan] = []

        # 找出过载和空闲的副本
        sorted_replicas = sorted(
            replica_queue_depths.items(), key=lambda x: x[1], reverse=True
        )

        overloaded = [r for r, d in sorted_replicas if d > self._get_threshold()]
        underloaded = [r for r, d in sorted_replicas if d < self._get_threshold() // 2]

        for src_replica in overloaded:
            for dst_replica in underloaded:
                # 估算迁移成本和收益
                cost = self._estimate_migration_cost(src_replica, dst_replica)
                benefit = self._estimate_migration_benefit(
                    src_replica, dst_replica, replica_queue_depths
                )

                if benefit > cost * self._min_benefit_ratio:
                    # TODO: 选择具体的请求进行迁移
                    pass

        return plans

    def _get_threshold(self) -> int:
        """获取队列深度阈值。"""
        # TODO: 基于系统负载动态调整
        return 32

    def _estimate_migration_cost(
        self, src_replica_id: int, dst_replica_id: int
    ) -> float:
        """估算迁移成本 (ms)。

        主要是 KV Cache 传输时间。
        """
        # TODO: 计算 KV Cache 大小 + RDMA 传输时间
        return 0.0

    def _estimate_migration_benefit(
        self,
        src_replica_id: int,
        dst_replica_id: int,
        queue_depths: Dict[int, int],
    ) -> float:
        """估算迁移收益 (ms)。"""
        # TODO: 基于排队论估计等待时间减少
        return 0.0

    def execute_migration(self, plan: MigrationPlan) -> List["BaseEvent"]:
        """执行迁移计划。

        Returns:
            迁移相关的事件列表
        """
        self._migration_history.append(plan)
        # TODO: 生成迁移事件 (KV Cache 传输等)
        return []
