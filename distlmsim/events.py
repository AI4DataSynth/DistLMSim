"""DistLMSim 事件系统

离散事件模拟的核心：每个事件在 handle_event() 中处理自身逻辑并返回后续事件列表。
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from distlmsim.types import EventType

if TYPE_CHECKING:
    from distlmsim.metrics.metrics_store import MetricsStore
    from distlmsim.scheduling.global_scheduler import BaseGlobalScheduler


_event_id_counter = itertools.count()


class BaseEvent(ABC):
    """事件基类。

    所有事件继承此类，实现 handle_event() 方法。
    事件通过 heapq 按 (time, id, event_type_priority) 排序。
    """

    def __init__(self, time: float, event_type: EventType):
        self._time = time
        self._id = next(_event_id_counter)
        self._event_type = event_type

    @property
    def time(self) -> float:
        return self._time

    @property
    def id(self) -> int:
        return self._id

    @property
    def event_type(self) -> EventType:
        return self._event_type

    @abstractmethod
    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List["BaseEvent"]:
        """处理事件，返回后续事件列表。"""
        ...

    def __lt__(self, other: "BaseEvent") -> bool:
        if self._time != other._time:
            return self._time < other._time
        if self._id != other._id:
            return self._id < other._id
        return self._event_type.value < other._event_type.value

    def __le__(self, other: "BaseEvent") -> bool:
        return self == other or self < other


# ─── 标准事件流 ────────────────────────────────────────────────────────────────


class RequestArrivalEvent(BaseEvent):
    """请求到达事件。触发全局调度。"""

    def __init__(self, time: float, request_id: int):
        super().__init__(time, EventType.REQUEST_ARRIVAL)
        self._request_id = request_id

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        metrics_store.record_request_arrival(self._request_id, self._time)
        return [GlobalScheduleEvent(self._time, self._request_id)]


class GlobalScheduleEvent(BaseEvent):
    """全局调度事件。将请求分配到某个副本。"""

    def __init__(self, time: float, request_id: int):
        super().__init__(time, EventType.GLOBAL_SCHEDULE)
        self._request_id = request_id

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        replica_id = scheduler.select_replica(self._request_id)
        return [ReplicaScheduleEvent(self._time, self._request_id, replica_id)]


class ReplicaScheduleEvent(BaseEvent):
    """副本级调度事件。将请求加入某个副本的批处理队列。"""

    def __init__(self, time: float, request_id: int, replica_id: int):
        super().__init__(time, EventType.REPLICA_SCHEDULE)
        self._request_id = request_id
        self._replica_id = replica_id

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        """副本级调度：将请求加入副本的批处理队列。

        当前 main.py 使用自己的仿真循环，此事件系统集成 planned for future work。
        实现时需：调用副本级调度器，形成 Batch，生成 BatchStageArrivalEvent。
        """
        metrics_store.record_request_scheduled(self._request_id, self._time)
        return []  # Future: 返回 BatchStageArrivalEvent


class BatchStageArrivalEvent(BaseEvent):
    """批处理到达某个 pipeline stage。"""

    def __init__(self, time: float, batch_id: int, stage_id: int, replica_id: int):
        super().__init__(time, EventType.BATCH_STAGE_ARRIVAL)
        self._batch_id = batch_id
        self._stage_id = stage_id
        self._replica_id = replica_id

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        # Future: 计算执行时间，生成 BatchStageEndEvent
        # 当前 main.py 使用自己的仿真循环处理 pipeline stage 执行
        return [
            BatchStageEndEvent(
                self._time, self._batch_id, self._stage_id,
                self._replica_id, num_stages=1  # 默认单 stage
            )
        ]


class BatchStageEndEvent(BaseEvent):
    """Pipeline stage 执行完成。"""

    def __init__(self, time: float, batch_id: int, stage_id: int, replica_id: int,
                 num_stages: int):
        super().__init__(time, EventType.BATCH_STAGE_END)
        self._batch_id = batch_id
        self._stage_id = stage_id
        self._replica_id = replica_id
        self._num_stages = num_stages

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        if self._stage_id < self._num_stages - 1:
            # 还有下一个 stage → PP 通信 → 下一个 stage
            # TODO: 计算 PP 通信时间 (可能跨节点 RDMA)
            next_stage_time = self._time  # + pp_comm_time
            return [
                BatchStageArrivalEvent(
                    next_stage_time, self._batch_id,
                    self._stage_id + 1, self._replica_id
                )
            ]
        else:
            # 最后一个 stage → BatchEndEvent
            return [BatchEndEvent(self._time, self._batch_id, self._replica_id)]


class BatchEndEvent(BaseEvent):
    """一个批处理的所有 pipeline stage 执行完成。"""

    def __init__(self, time: float, batch_id: int, replica_id: int):
        super().__init__(time, EventType.BATCH_END)
        self._batch_id = batch_id
        self._replica_id = replica_id

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        # Future: 更新请求状态，判断是否继续 decode 或完成
        # 当前 main.py 使用自己的仿真循环处理 batch 完成逻辑
        return []


# ─── 存算分离事件 ──────────────────────────────────────────────────────────────


class PrefillCompleteEvent(BaseEvent):
    """Prefill 阶段完成事件 (存算分离模式)。

    Prefill 节点完成计算后，需要将 KV Cache 传输到 Decode 节点。
    """

    def __init__(self, time: float, request_id: int, prefill_node_id: int,
                 kv_cache_size_bytes: int):
        super().__init__(time, EventType.PREFILL_COMPLETE)
        self._request_id = request_id
        self._prefill_node_id = prefill_node_id
        self._kv_cache_size_bytes = kv_cache_size_bytes

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        # 触发 KV Cache 传输
        return [
            KVCacheTransferStartEvent(
                self._time, self._request_id,
                self._prefill_node_id, self._kv_cache_size_bytes
            )
        ]


class KVCacheTransferStartEvent(BaseEvent):
    """KV Cache RDMA 传输开始事件。"""

    def __init__(self, time: float, request_id: int, src_node_id: int,
                 kv_cache_size_bytes: int):
        super().__init__(time, EventType.KV_CACHE_TRANSFER_START)
        self._request_id = request_id
        self._src_node_id = src_node_id
        self._kv_cache_size_bytes = kv_cache_size_bytes

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        """通过 RDMA 模型计算 KV Cache 传输时间。

        使用 scheduler 中的 rdma_model 计算传输延迟，
        考虑协议开销和拥塞因子。
        """
        rdma_model = getattr(scheduler, '_rdma_model', None)
        if rdma_model is not None:
            transfer_time_ms = rdma_model.get_transfer_time(
                self._kv_cache_size_bytes
            )
        else:
            # 回退: 简单带宽延迟模型 (200 Gbps RDMA)
            bandwidth_bps = 200e9 / 8  # 200 Gbps → B/s
            latency_ms = 2e-3  # 2 μs
            transfer_time_ms = self._kv_cache_size_bytes / bandwidth_bps * 1e3 + latency_ms

        metrics_store.record_kv_cache_transfer_start(self._request_id, self._time)
        end_time = self._time + transfer_time_ms
        return [
            KVCacheTransferEndEvent(
                end_time, self._request_id, self._kv_cache_size_bytes
            )
        ]


class KVCacheTransferEndEvent(BaseEvent):
    """KV Cache RDMA 传输完成事件。"""

    def __init__(self, time: float, request_id: int, kv_cache_size_bytes: int):
        super().__init__(time, EventType.KV_CACHE_TRANSFER_END)
        self._request_id = request_id
        self._kv_cache_size_bytes = kv_cache_size_bytes

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        metrics_store.record_kv_cache_transfer_end(self._request_id, self._time)
        # 传输完成，开始在 Decode 节点上调度 decode
        return [DecodeStartEvent(self._time, self._request_id)]


class DecodeStartEvent(BaseEvent):
    """Decode 阶段开始事件 (存算分离模式)。"""

    def __init__(self, time: float, request_id: int):
        super().__init__(time, EventType.DECODE_START)
        self._request_id = request_id

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        # Future: 将请求分配到 Decode 节点的副本
        # 当前 main.py 使用自己的仿真循环处理 decode 调度
        metrics_store.record_decode_start(self._request_id, self._time)
        return []


# ─── 专家并行事件 ──────────────────────────────────────────────────────────────


class ExpertAssignmentEvent(BaseEvent):
    """专家分配事件。根据路由规则确定每个 token 的目标专家。"""

    def __init__(self, time: float, batch_id: int, replica_id: int):
        super().__init__(time, EventType.EXPERT_ASSIGNMENT)
        self._batch_id = batch_id
        self._replica_id = replica_id

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        # Future: 执行 Top-K 路由 + 负载均衡，生成 ExpertCommStartEvent
        # 当前 main.py 通过 _compute_moe_imbalance_factor() 处理 MoE 路由
        return [ExpertCommStartEvent(self._time, self._batch_id, self._replica_id)]


class ExpertCommStartEvent(BaseEvent):
    """专家通信开始事件。跨节点 all-to-all 传输 token 到目标专家所在节点。"""

    def __init__(self, time: float, batch_id: int, replica_id: int):
        super().__init__(time, EventType.EXPERT_COMM_START)
        self._batch_id = batch_id
        self._replica_id = replica_id

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        # Future: 通过 RDMA 模型计算 all-to-all 通信时间
        # 当前 main.py 通过 expert_parallel_comm_time 处理 EP 通信
        rdma_model = getattr(scheduler, '_rdma_model', None)
        comm_time_ms = 0.0
        if rdma_model is not None:
            # 估算 all-to-all 数据量: batch_size * top_k * hidden_dim * 2 bytes
            estimated_bytes = 1024 * 8 * 2048 * 2  # 典型 MoE 配置
            comm_time_ms = rdma_model.get_alltoall_time(2, estimated_bytes)
        return [ExpertCommEndEvent(
            self._time + comm_time_ms, self._batch_id, self._replica_id
        )]


class ExpertCommEndEvent(BaseEvent):
    """专家通信完成事件。"""

    def __init__(self, time: float, batch_id: int, replica_id: int):
        super().__init__(time, EventType.EXPERT_COMM_END)
        self._batch_id = batch_id
        self._replica_id = replica_id

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        # Future: 通信完成，继续后续 pipeline stage
        # 当前 main.py 使用自己的仿真循环处理后续流程
        return []
