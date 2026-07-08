"""DistLMSim 核心实体

定义模拟器中的主要数据结构：Request, Batch, BatchStage, ExecutionTime, Node, Replica 等。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class RequestStatus(Enum):
    """请求生命周期状态"""
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    DRAFTING = "drafting"           # 投机解码: Draft 阶段
    KV_CACHE_TRANSFERRING = "kv_cache_transferring"  # 存算分离：KV Cache 传输中
    COMPLETED = "completed"
    PREEMPTED = "preempted"


@dataclass
class ExecutionTime:
    """单层执行时间的完整分解。

    记录模型前向传播中各子阶段的执行时间（单位：ms）。
    """
    # Attention 子阶段
    attn_pre_proj_time: float = 0.0
    attn_rope_time: float = 0.0
    attn_kv_cache_save_time: float = 0.0
    attn_prefill_time: float = 0.0
    attn_decode_time: float = 0.0
    attn_post_proj_time: float = 0.0
    # MLP 子阶段
    mlp_up_proj_time: float = 0.0
    mlp_act_time: float = 0.0
    mlp_down_proj_time: float = 0.0
    # LayerNorm
    input_layernorm_time: float = 0.0
    post_attention_layernorm_time: float = 0.0
    # Add
    add_time: float = 0.0
    # 通信
    tensor_parallel_comm_time: float = 0.0     # TP all-reduce (NVLink)
    pipeline_parallel_comm_time: float = 0.0   # PP send/recv (可跨节点)
    expert_parallel_comm_time: float = 0.0     # EP all-to-all (RDMA)
    # MoE
    expert_mlp_time: float = 0.0
    eplb_overhead_time: float = 0.0
    # CPU
    cpu_overhead_time: float = 0.0

    @property
    def attention_time(self) -> float:
        return (
            self.attn_pre_proj_time
            + self.attn_rope_time
            + self.attn_kv_cache_save_time
            + self.attn_prefill_time
            + self.attn_decode_time
            + self.attn_post_proj_time
        )

    @property
    def mlp_time(self) -> float:
        return self.mlp_up_proj_time + self.mlp_act_time + self.mlp_down_proj_time

    @property
    def comm_time(self) -> float:
        return (
            self.tensor_parallel_comm_time
            + self.pipeline_parallel_comm_time
            + self.expert_parallel_comm_time
        )

    @property
    def layer_time(self) -> float:
        return (
            self.attention_time
            + self.mlp_time
            + self.expert_mlp_time
            + self.input_layernorm_time
            + self.post_attention_layernorm_time
            + self.add_time
        )

    @property
    def total_time(self) -> float:
        return self.layer_time + self.comm_time + self.eplb_overhead_time + self.cpu_overhead_time


@dataclass
class Request:
    """单个推理请求。"""
    id: int
    arrival_time: float                          # 到达时间 (ms)
    prefill_tokens: int                          # Prefill token 数
    decode_tokens: int                           # Decode token 数
    # MoE 专家路由分布 (可选)
    prefill_expert_distribution: Optional[np.ndarray] = None  # shape: [num_layers, num_experts]
    decode_expert_distributions: Optional[List[np.ndarray]] = None
    # 生命周期
    status: RequestStatus = RequestStatus.WAITING
    scheduled_time: Optional[float] = None
    prefill_start_time: Optional[float] = None
    prefill_end_time: Optional[float] = None
    decode_start_time: Optional[float] = None
    decode_end_time: Optional[float] = None
    # 存算分离相关
    prefill_node_id: Optional[int] = None        # Prefill 所在节点
    decode_node_id: Optional[int] = None         # Decode 所在节点
    kv_cache_size_bytes: int = 0                 # KV Cache 大小
    # 分配信息
    replica_id: Optional[int] = None
    num_generated_tokens: int = 0
    # 投机解码 cycle 追踪
    draft_tokens_generated: int = 0              # 当前轮 draft 的 token 数
    accepted_tokens_last_cycle: int = 0           # 上一轮接受的 token 数
    confidence_scores: Optional[List[float]] = None  # 每位置置信度 (Prefix Scheduler)

    @property
    def is_prefill_complete(self) -> bool:
        return self.prefill_end_time is not None

    @property
    def is_complete(self) -> bool:
        return self.status == RequestStatus.COMPLETED

    @property
    def e2e_latency(self) -> float:
        """端到端延迟 (ms)"""
        if self.decode_end_time is None:
            return 0.0
        return self.decode_end_time - self.arrival_time

    @property
    def ttft(self) -> float:
        """Time to First Token (ms)"""
        if self.prefill_end_time is None:
            return 0.0
        return self.prefill_end_time - self.arrival_time


@dataclass
class Batch:
    """一组请求的批处理。"""
    id: int
    replica_id: int
    requests: List[Request] = field(default_factory=list)
    num_tokens: List[int] = field(default_factory=list)  # 每请求当前轮 token 数
    creation_time: float = 0.0
    is_prefill_batch: bool = True  # True=Prefill 批，False=Decode 批

    @property
    def batch_size(self) -> int:
        return len(self.requests)

    @property
    def total_tokens(self) -> int:
        return sum(self.num_tokens)

    def add_request(self, request: Request, num_tokens: int) -> None:
        self.requests.append(request)
        self.num_tokens.append(num_tokens)

    def remove_request(self, request: Request) -> None:
        idx = self.requests.index(request)
        self.requests.pop(idx)
        self.num_tokens.pop(idx)


@dataclass
class BatchStage:
    """批处理在某个 pipeline stage 的执行单元。"""
    id: int
    batch: Batch
    stage_id: int
    execution_time: Optional[ExecutionTime] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration(self) -> float:
        if self.execution_time is None:
            return 0.0
        return self.execution_time.total_time


@dataclass
class Replica:
    """一个模型副本，可能跨多个节点。"""
    id: int
    model_name: str
    # 并行配置
    tensor_parallel_size: int = 1
    num_pipeline_stages: int = 1
    expert_parallel_size: int = 1
    # 节点映射：哪些节点/GPU 属于此副本
    node_ids: List[int] = field(default_factory=list)
    gpu_ids: List[int] = field(default_factory=list)  # 全局 GPU ID 列表
    # 调度器引用（运行时设置）
    scheduler: Optional[object] = None
    # 运行状态
    num_running_batches: int = 0
    max_running_batches: int = 4


@dataclass
class Node:
    """物理节点。"""
    id: int
    node_sku_name: str
    num_gpus: int = 8
    gpu_ids: List[int] = field(default_factory=list)
    # 网络
    rdma_nic_bandwidth_gbps: float = 200.0  # RDMA NIC 带宽
    # 运行状态
    role: str = "mixed"  # mixed / prefill / decode
    current_replica_ids: List[int] = field(default_factory=list)
