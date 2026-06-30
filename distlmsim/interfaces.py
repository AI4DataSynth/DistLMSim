"""DistLMSim Protocol 接口定义

定义模块间的抽象接口，确保依赖关系为严格 DAG（有向无环图）。
所有 Protocol 仅依赖 types/entities，不引用任何高层模块。

依赖层次: Layer 1 (仅依赖 types, entities)
"""

from __future__ import annotations

from typing import Dict, Protocol, runtime_checkable


@runtime_checkable
class ReplicaSelector(Protocol):
    """全局调度器的副本选择接口。

    被 events.py 中的事件处理系统使用，
    避免 events 模块直接依赖 scheduling 模块。
    """

    def select_replica(self, request_id: int) -> int:
        """为请求选择一个副本。"""
        ...


@runtime_checkable
class MetricsRecorder(Protocol):
    """指标记录接口。

    被 events.py 中的事件处理系统使用，
    避免 events 模块直接依赖 metrics 模块。
    """

    def record_request_arrival(self, request_id: int, time: float) -> None:
        ...

    def record_request_scheduled(self, request_id: int, time: float) -> None:
        ...

    def record_kv_cache_transfer_start(self, request_id: int, time: float) -> None:
        ...

    def record_kv_cache_transfer_end(self, request_id: int, time: float) -> None:
        ...

    def record_decode_start(self, request_id: int, time: float) -> None:
        ...


@runtime_checkable
class ClusterView(Protocol):
    """集群视图接口。

    被 scheduling 模块使用，
    避免 scheduling 直接依赖 cluster 模块。
    """

    @property
    def replicas(self) -> Dict[int, object]:
        """返回副本字典 {replica_id: Replica}。"""
        ...

    @property
    def nodes(self) -> Dict[int, object]:
        """返回节点字典 {node_id: PhysicalNode}。"""
        ...
