"""副本级调度器

管理单个副本内的请求排队、批处理和执行调度。
复用 TRADIOS 的调度器接口设计。

依赖层次: Layer 6
  输入: entities (Batch, Request, RequestStatus), events (BaseEvent)
  输出: BaseReplicaScheduler 及其子类 (被 simulator 消费)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from distlmsim.entities import Batch, Request, RequestStatus
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
    def on_request_arrival(self, request: Request) -> List[BaseEvent]:
        """新请求到达时的处理逻辑。"""
        ...

    @abstractmethod
    def on_batch_end(self, batch_id: int) -> List[BaseEvent]:
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
        self._next_batch_id = 0

    def on_request_arrival(self, request: Request) -> List[BaseEvent]:
        self._waiting_queue.append(request)
        return []

    def on_batch_end(self, batch_id: int) -> List[BaseEvent]:
        # Remove completed batches
        completed = []
        still_running = []
        for batch in self._running_batches:
            if batch.id == batch_id:
                # Update request states
                for req in batch.requests:
                    req.num_generated_tokens += 1
                    if req.num_generated_tokens >= req.decode_tokens:
                        req.status = RequestStatus.COMPLETED
                completed.append(batch)
            else:
                still_running.append(batch)
        self._running_batches = still_running
        return []

    def form_batch(self, current_time: float) -> Optional[Batch]:
        if not self._waiting_queue and not self._running_batches:
            return None

        batch_requests: List[Request] = []
        batch_tokens: List[int] = []
        total_tokens = 0

        # 1. Add decode requests from running batches (iteration-level scheduling)
        for batch in self._running_batches:
            for req in batch.requests:
                if req.status != RequestStatus.COMPLETED:
                    batch_requests.append(req)
                    batch_tokens.append(1)  # 1 decode token per request
                    total_tokens += 1

        # 2. Fill remaining token budget with prefill chunks from waiting queue
        remaining_budget = self._max_num_tokens - total_tokens
        prefill_requests = []
        while self._waiting_queue and remaining_budget > 0:
            req = self._waiting_queue[0]
            tokens_needed = req.prefill_tokens - (
                req.num_generated_tokens if req.prefill_end_time is None else 0
            )
            chunk = min(self._chunk_size, tokens_needed, remaining_budget)
            if chunk <= 0:
                self._waiting_queue.pop(0)
                continue
            prefill_requests.append((req, chunk))
            remaining_budget -= chunk
            if chunk >= tokens_needed:
                self._waiting_queue.pop(0)
            else:
                # Partial chunk - keep in queue for next iteration
                req.num_generated_tokens += chunk
                break

        for req, chunk in prefill_requests:
            batch_requests.append(req)
            batch_tokens.append(chunk)

        if not batch_requests:
            return None

        batch_id = self._next_batch_id
        self._next_batch_id += 1

        batch = Batch(
            id=batch_id,
            replica_id=self._replica_id,
            requests=batch_requests,
            num_tokens=batch_tokens,
            creation_time=current_time,
            is_prefill_batch=len(prefill_requests) > 0,
        )
        self._running_batches.append(batch)
        return batch


class VllmReplicaScheduler(BaseReplicaScheduler):
    """vLLM 调度器。

    基于 PagedAttention 的块级内存管理，支持 preemption。
    每个请求的 KV cache 按固定大小的 block 分配；当可用 block
    不足时，低优先级的 decode 请求被 preempt（swap out），
    释放 block 给新到达的 prefill 请求。

    form_batch 逻辑:
    1. 将所有 running decode 请求加入 batch（每个 1 token）
    2. 从 waiting queue 中取 prefill 请求，检查 block 可用性
    3. 若 block 不足，preempt 最早到达的 running 请求
    """

    def __init__(
        self,
        replica_id: int,
        max_batch_size: int = 256,
        max_num_tokens: int = 16384,
        block_size: int = 16,
        num_blocks: int = 10000,
    ):
        super().__init__(replica_id, max_batch_size, max_num_tokens)
        self._block_size = block_size
        self._num_blocks = num_blocks
        self._free_blocks = num_blocks
        self._request_blocks: Dict[int, int] = {}  # req_id -> allocated blocks
        self._running_requests: List[Request] = []
        self._next_batch_id = 0

    def _blocks_needed(self, req: Request) -> int:
        """计算请求需要的 KV cache block 数。"""
        seq_len = req.prefill_tokens + req.decode_tokens
        return (seq_len + self._block_size - 1) // self._block_size

    def _preempt(self) -> Optional[Request]:
        """Preempt 最早到达的 running 请求，释放其 block。"""
        if not self._running_requests:
            return None
        # Preempt the request with earliest arrival time
        victim = min(self._running_requests, key=lambda r: r.arrival_time)
        freed = self._request_blocks.pop(victim.id, 0)
        self._free_blocks += freed
        self._running_requests.remove(victim)
        victim.status = RequestStatus.WAITING
        self._waiting_queue.insert(0, victim)
        return victim

    def on_request_arrival(self, request: Request) -> List[BaseEvent]:
        self._waiting_queue.append(request)
        return []

    def on_batch_end(self, batch_id: int) -> List[BaseEvent]:
        completed = []
        still_running = []
        for req in self._running_requests:
            if req.num_generated_tokens >= req.decode_tokens:
                req.status = RequestStatus.COMPLETED
                freed = self._request_blocks.pop(req.id, 0)
                self._free_blocks += freed
                completed.append(req)
            else:
                still_running.append(req)
        self._running_requests = still_running
        return []

    def form_batch(self, current_time: float) -> Optional[Batch]:
        if not self._waiting_queue and not self._running_requests:
            return None

        batch_requests: List[Request] = []
        batch_tokens: List[int] = []

        # 1. Add all running decode requests (1 token each)
        for req in self._running_requests:
            if req.status != RequestStatus.COMPLETED:
                batch_requests.append(req)
                batch_tokens.append(1)

        # 2. Admit prefill requests from waiting queue
        while self._waiting_queue and len(batch_requests) < self._max_batch_size:
            req = self._waiting_queue[0]
            needed = self._blocks_needed(req)

            # Try preemption if not enough blocks
            while self._free_blocks < needed and self._running_requests:
                self._preempt()

            if self._free_blocks < needed:
                break  # Cannot admit even after preemption

            # Allocate blocks and admit
            self._free_blocks -= needed
            self._request_blocks[req.id] = needed
            self._waiting_queue.pop(0)
            self._running_requests.append(req)
            batch_requests.append(req)
            batch_tokens.append(req.prefill_tokens)

        if not batch_requests:
            return None

        batch_id = self._next_batch_id
        self._next_batch_id += 1
        batch = Batch(
            id=batch_id,
            replica_id=self._replica_id,
            requests=batch_requests,
            num_tokens=batch_tokens,
            creation_time=current_time,
            is_prefill_batch=any(
                r.status == RequestStatus.WAITING for r in batch_requests
            ),
        )
        return batch


class OrcaReplicaScheduler(BaseReplicaScheduler):
    """Orca 调度器。

    迭代级调度 (iteration-level scheduling)：每个 iteration
    将所有 running 请求和新到达的请求组成一个 batch。
    不支持 chunked prefill — prefill 请求整体处理。

    form_batch 逻辑:
    1. 将所有 running 请求加入 batch（每个 1 decode token）
    2. 从 waiting queue 取新请求，执行完整 prefill
    3. 不超过 max_batch_size 限制
    """

    def __init__(
        self,
        replica_id: int,
        max_batch_size: int = 256,
        max_num_tokens: int = 16384,
    ):
        super().__init__(replica_id, max_batch_size, max_num_tokens)
        self._running_requests: List[Request] = []
        self._next_batch_id = 0

    def on_request_arrival(self, request: Request) -> List[BaseEvent]:
        self._waiting_queue.append(request)
        return []

    def on_batch_end(self, batch_id: int) -> List[BaseEvent]:
        still_running = []
        for req in self._running_requests:
            req.num_generated_tokens += 1
            if req.num_generated_tokens >= req.decode_tokens:
                req.status = RequestStatus.COMPLETED
            else:
                still_running.append(req)
        self._running_requests = still_running
        return []

    def form_batch(self, current_time: float) -> Optional[Batch]:
        if not self._waiting_queue and not self._running_requests:
            return None

        batch_requests: List[Request] = []
        batch_tokens: List[int] = []

        # 1. All running requests get 1 decode token
        for req in self._running_requests:
            if req.status != RequestStatus.COMPLETED:
                batch_requests.append(req)
                batch_tokens.append(1)

        # 2. Admit new requests from waiting queue (full prefill, no chunking)
        remaining_slots = self._max_batch_size - len(batch_requests)
        while self._waiting_queue and remaining_slots > 0:
            req = self._waiting_queue.pop(0)
            self._running_requests.append(req)
            batch_requests.append(req)
            batch_tokens.append(req.prefill_tokens)
            remaining_slots -= 1

        if not batch_requests:
            return None

        batch_id = self._next_batch_id
        self._next_batch_id += 1
        batch = Batch(
            id=batch_id,
            replica_id=self._replica_id,
            requests=batch_requests,
            num_tokens=batch_tokens,
            creation_time=current_time,
            is_prefill_batch=any(
                r.status == RequestStatus.WAITING for r in batch_requests
            ),
        )
        return batch
