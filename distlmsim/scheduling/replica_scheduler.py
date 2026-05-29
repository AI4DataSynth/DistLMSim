"""副本级调度器

管理单个副本内的请求排队、批处理和执行调度。
复用 TRADIOS 的调度器接口设计。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from distlmsim.entities import Batch, Request

if TYPE_CHECKING:
    from distlmsim.events import BaseEvent


class BaseReplicaScheduler(ABC):
    """副本级调度器基类。

    管理单个模型副本内的请求队列和批处理形成。
    """

    def __init__(
        self,
        replica_id: int,
        max_batch_size: int = 256,
        max_num_tokens: int = 16384,
    ):
        self._replica_id = replica_id
        self._max_batch_size = max_batch_size
        self._max_num_tokens = max_num_tokens
        self._waiting_queue: List[Request] = []
        self._running_batches: List[Batch] = []

    @abstractmethod
    def on_request_arrival(self, request: Request) -> List["BaseEvent"]:
        """新请求到达时的处理逻辑。"""
        ...

    @abstractmethod
    def on_batch_end(self, batch_id: int) -> List["BaseEvent"]:
        """批处理完成时的处理逻辑。"""
        ...

    @abstractmethod
    def form_batch(self, current_time: float) -> Optional[Batch]:
        """形成新的批处理。

        Args:
            current_time: 当前模拟时间

        Returns:
            新形成的 Batch，如果无法形成则返回 None
        """
        ...

    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def num_waiting_requests(self) -> int:
        return len(self._waiting_queue)

    @property
    def num_running_batches(self) -> int:
        return len(self._running_batches)


class SarathiReplicaScheduler(BaseReplicaScheduler):
    """Sarathi 调度器。

    支持块级 (chunked) prefill 分块和迭代级调度。
    Prefill 请求被分成多个 chunk，与 decode 请求混合执行。
    """

    def __init__(
        self,
        replica_id: int,
        max_batch_size: int = 256,
        max_num_tokens: int = 16384,
        chunk_size: int = 4096,
    ):
        super().__init__(replica_id, max_batch_size, max_num_tokens)
        self._chunk_size = chunk_size

    def on_request_arrival(self, request: Request) -> List["BaseEvent"]:
        self._waiting_queue.append(request)
        return []

    def on_batch_end(self, batch_id: int) -> List["BaseEvent"]:
        # TODO: 更新 batch 中的请求状态，判断是否完成
        return []

    def form_batch(self, current_time: float) -> Optional[Batch]:
        # TODO: 实现 chunked prefill + decode 混合批处理
        raise NotImplementedError("SarathiReplicaScheduler.form_batch")


class VllmReplicaScheduler(BaseReplicaScheduler):
    """vLLM 调度器。

    基于 PagedAttention 的内存管理，支持 preemption。
    """

    def __init__(
        self,
        replica_id: int,
        max_batch_size: int = 256,
        max_num_tokens: int = 16384,
        block_size: int = 16,
    ):
        super().__init__(replica_id, max_batch_size, max_num_tokens)
        self._block_size = block_size

    def on_request_arrival(self, request: Request) -> List["BaseEvent"]:
        self._waiting_queue.append(request)
        return []

    def on_batch_end(self, batch_id: int) -> List["BaseEvent"]:
        return []

    def form_batch(self, current_time: float) -> Optional[Batch]:
        raise NotImplementedError("VllmReplicaScheduler.form_batch")


class OrcaReplicaScheduler(BaseReplicaScheduler):
    """Orca 调度器。迭代级调度，每步形成 batch。"""

    def on_request_arrival(self, request: Request) -> List["BaseEvent"]:
        self._waiting_queue.append(request)
        return []

    def on_batch_end(self, batch_id: int) -> List["BaseEvent"]:
        return []

    def form_batch(self, current_time: float) -> Optional[Batch]:
        raise NotImplementedError("OrcaReplicaScheduler.form_batch")
