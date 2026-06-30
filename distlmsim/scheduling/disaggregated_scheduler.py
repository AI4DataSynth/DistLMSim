"""存算分离 (Disaggregated Prefill/Decode) 调度器

支持 Prefill 和 Decode 阶段部署在不同节点上，
通过 RDMA 传输 KV Cache。

依赖层次: Layer 6
  输入: config (DisaggregatedConfig, ModelConfig), entities (Request),
        types (NodeRole), interfaces (ClusterView), events (BaseEvent)
  输出: DisaggregatedScheduler (被 simulator 消费)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from distlmsim.config import DisaggregatedConfig, ModelConfig
from distlmsim.entities import Request
from distlmsim.events import BaseEvent
from distlmsim.interfaces import ClusterView
from distlmsim.types import NodeRole


class DisaggregatedScheduler:
    """存算分离调度器。

    管理 Prefill 节点和 Decode 节点的协同调度：
    1. 新请求先分配到 Prefill 节点执行 prefill
    2. Prefill 完成后，KV Cache 通过 RDMA 传输到 Decode 节点
    3. Decode 节点执行后续的 decode 阶段

    调度策略:
    - Prefill 节点间：轮询或负载均衡
    - Decode 节点间：按剩余显存或队列长度分配
    - KV Cache 传输：直接/流水线/存储转发
    """

    def __init__(
        self,
        config: DisaggregatedConfig,
        cluster: ClusterView,
        model_config: Optional[ModelConfig] = None,
    ):
        self._config = config
        self._cluster = cluster
        self._model_config = model_config
        self._prefill_queue: List[Request] = []
        self._decode_queue: List[Request] = []
        self._prefill_node_ids: List[int] = []
        self._decode_node_ids: List[int] = []
        self._next_prefill_node_idx = 0
        # 跟踪每个 Decode 节点的负载（已分配的请求数）
        self._decode_load: Dict[int, int] = {}

    def initialize(self) -> None:
        """初始化节点角色分配。"""
        from distlmsim.types import NodeRole

        all_node_ids = [n.id for n in self._cluster.nodes]

        if self._config.enabled:
            # 前 N 个节点为 Prefill，后 M 个为 Decode
            self._prefill_node_ids = all_node_ids[: self._config.num_prefill_nodes]
            self._decode_node_ids = all_node_ids[
                self._config.num_prefill_nodes :
                self._config.num_prefill_nodes + self._config.num_decode_nodes
            ]
            # 设置节点角色
            self._cluster._resource_manager.assign_node_roles(
                self._prefill_node_ids, self._decode_node_ids
            )
        else:
            # 混合模式，所有节点同时处理 prefill 和 decode
            self._prefill_node_ids = all_node_ids
            self._decode_node_ids = all_node_ids

        # 初始化 Decode 节点负载计数
        self._decode_load = {node_id: 0 for node_id in self._decode_node_ids}

    def schedule_prefill(self, request: Request) -> int:
        """为请求选择 Prefill 节点。

        使用轮询策略。

        Returns:
            选中的 Prefill 节点 ID
        """
        if not self._prefill_node_ids:
            raise RuntimeError("没有可用的 Prefill 节点")

        node_id = self._prefill_node_ids[
            self._next_prefill_node_idx % len(self._prefill_node_ids)
        ]
        self._next_prefill_node_idx += 1
        return node_id

    def schedule_decode(self, request: Request) -> int:
        """为请求选择 Decode 节点。

        策略：选择当前负载（已分配请求数）最低的 Decode 节点。
        相同负载时优先选择列表中靠前的节点。

        Returns:
            选中的 Decode 节点 ID
        """
        if not self._decode_node_ids:
            raise RuntimeError("没有可用的 Decode 节点")

        best_node = self._decode_node_ids[0]
        min_load = self._decode_load.get(best_node, 0)
        for node_id in self._decode_node_ids[1:]:
            load = self._decode_load.get(node_id, 0)
            if load < min_load:
                min_load = load
                best_node = node_id

        # 更新负载计数
        self._decode_load[best_node] = self._decode_load.get(best_node, 0) + 1
        return best_node

    def on_prefill_complete(self, request: Request) -> List[BaseEvent]:
        """Prefill 完成后的处理。

        生成 KV Cache 传输事件和 Decode 调度事件。
        """
        from distlmsim.events import (
            KVCacheTransferStartEvent,
            DecodeStartEvent,
        )

        events: List[BaseEvent] = []

        # 计算 KV Cache 大小
        kv_cache_size = self._estimate_kv_cache_size(request)
        request.kv_cache_size_bytes = kv_cache_size

        # 选择 Decode 节点
        decode_node_id = self.schedule_decode(request)
        request.decode_node_id = decode_node_id

        # 生成 KV Cache 传输事件
        transfer_event = KVCacheTransferStartEvent(
            time=request.prefill_end_time,
            request_id=request.id,
            src_node_id=request.prefill_node_id,
            kv_cache_size_bytes=kv_cache_size,
        )
        events.append(transfer_event)

        return events

    def _estimate_kv_cache_size(self, request: Request) -> int:
        """估算请求的 KV Cache 大小 (bytes)。

        KV Cache 大小 = 2 * num_layers * num_kv_heads * head_dim * seq_len * sizeof(float16)

        其中:
        - 2 对应 K 和 V 两个张量
        - head_dim = embedding_dim / num_q_heads
        - sizeof(float16) = 2 bytes
        - seq_len = request.prefill_tokens

        若未提供 model_config，则回退到 128 bytes/token 的粗略估算。
        """
        seq_len = request.prefill_tokens

        if self._model_config is not None:
            cfg = self._model_config
            head_dim = cfg.embedding_dim // cfg.num_q_heads
            # 2(K+V) * layers * kv_heads * head_dim * seq_len * 2(float16)
            return 2 * cfg.num_layers * cfg.num_kv_heads * head_dim * seq_len * 2

        # 回退: 粗略估算 ~128 bytes per token
        return seq_len * 128

    @property
    def prefill_node_ids(self) -> List[int]:
        return self._prefill_node_ids

    @property
    def decode_node_ids(self) -> List[int]:
        return self._decode_node_ids
