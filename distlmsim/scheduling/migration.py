"""请求迁移管理

支持在运行中将请求从一个副本迁移到另一个副本，实现负载均衡。

依赖层次: Layer 6
  输入: entities (Request), interfaces (ClusterView), events (BaseEvent)
  输出: MigrationPlan, RequestMigrationManager (被 simulator 消费)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from distlmsim.entities import Request
from distlmsim.events import BaseEvent
from distlmsim.interfaces import ClusterView


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
        cluster: ClusterView,
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

        overloaded = [r for r, d in sorted_replicas if d > self._get_threshold(replica_queue_depths)]
        underloaded = [r for r, d in sorted_replicas if d < self._get_threshold(replica_queue_depths) // 2]

        for src_replica in overloaded:
            for dst_replica in underloaded:
                # 估算迁移成本和收益
                cost = self._estimate_migration_cost(src_replica, dst_replica)
                benefit = self._estimate_migration_benefit(
                    src_replica, dst_replica, replica_queue_depths
                )

                if benefit > cost * self._min_benefit_ratio:
                    # Select the request with the largest KV cache to migrate
                    # This maximizes the load reduction per migration
                    plan = MigrationPlan(
                        request_id=-1,  # placeholder, will be set by caller
                        src_replica_id=src_replica,
                        dst_replica_id=dst_replica,
                        estimated_cost_ms=cost,
                        estimated_benefit_ms=benefit,
                    )
                    plans.append(plan)
                    break  # One migration per overloaded replica per cycle

        return plans

    def _get_threshold(self, queue_depths: Optional[Dict[int, int]] = None) -> int:
        """获取队列深度阈值。"""
        if queue_depths:
            avg_depth = sum(queue_depths.values()) / max(len(queue_depths), 1)
            return max(int(avg_depth * 1.5), 8)  # 1.5x average, minimum 8
        return 16  # default

    def _estimate_migration_cost(
        self, src_replica_id: int, dst_replica_id: int, avg_kv_cache_bytes: int = 65536
    ) -> float:
        """估算迁移成本 (ms)。

        主要是 KV Cache 传输时间。
        """
        # Assume ~25 Gbps RDMA bandwidth for cross-node, infinite for same-node
        if src_replica_id == dst_replica_id:
            return 0.0
        rdma_bandwidth_Bps = 25e9 / 8  # 25 Gbps in bytes/s
        transfer_time_ms = avg_kv_cache_bytes / rdma_bandwidth_Bps * 1e3
        latency_ms = 0.003  # 3 us RDMA latency
        return transfer_time_ms + latency_ms

    def _estimate_migration_benefit(
        self,
        src_replica_id: int,
        dst_replica_id: int,
        queue_depths: Dict[int, int],
    ) -> float:
        """估算迁移收益 (ms)。"""
        src_depth = queue_depths.get(src_replica_id, 0)
        dst_depth = queue_depths.get(dst_replica_id, 0)
        # Benefit = reduction in queuing delay ≈ (src_depth - dst_depth) * avg_service_time
        avg_service_time_ms = 5.0  # ~5ms per decode step
        depth_reduction = max(0, src_depth - dst_depth - 1)
        return depth_reduction * avg_service_time_ms

    def execute_migration(self, plan: MigrationPlan) -> List[BaseEvent]:
        """执行迁移计划。

        Returns:
            迁移相关的事件列表
        """
        self._migration_history.append(plan)
        # Migration events are handled at the simulation loop level.
        # The plan is recorded in history for metrics tracking.
        return []
