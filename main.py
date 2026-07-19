"""DistLMSim 入口点：分布式推理模拟器

提供两种运行方式：
1. DisaggregatedSimulator: 存算分离模式的完整模拟 (推荐入门)
2. DistributedInferenceSimulator: 通用分布式模拟 (事件驱动)
3. ColocatedSimulator: Colocated (非分离) 模式模拟

依赖层次: Layer 7 (顶层组装)
  输入: 所有下层模块
  输出: DisaggregatedSimulator, DistributedInferenceSimulator, ColocatedSimulator
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from distlmsim.config import (
    SimulationConfig,
    ClusterConfig,
    NodeSKUConfig,
    DeviceSKUConfig,
    NetworkTopologyConfig,
    NVLinkConfig,
    RDMAConfig,
    ReplicaConfig,
    ModelConfig,
    DisaggregatedConfig,
    RequestGeneratorConfig,
    MetricsConfig,
)
from distlmsim.context import SimContext
from distlmsim.entities import Request, RequestStatus, ExecutionTime
from distlmsim.execution.execution_time_predictor import (
    AnalyticalPredictor,
    ExecutionTimePredictor,
    create_predictor,
)
from distlmsim.topology.nvlink_model import NVLinkModel
from distlmsim.topology.rdma_model import RDMAModel
from distlmsim.topology.overlap_processor import OverlapProcessor, OverlapConfig
from distlmsim.parallelism.expert_parallel import ExpertParallelModel
from distlmsim.metrics.metrics_store import MetricsStore
from distlmsim.scheduling.advanced_schedulers import AdvancedSchedulers

logger = logging.getLogger(__name__)


# SimContext 已从 distlmsim.context 导入 (消除循环依赖)


# ─── 存算分离模拟器 ────────────────────────────────────────────────────────────


class DisaggregatedSimulator:
    """存算分离 (Disaggregated Prefill/Decode) 模拟器。

    模拟 1 个 Prefill 节点 + 1 个 Decode 节点的推理服务：
    1. 请求到达 → Prefill 节点执行 prefill (批量)
    2. Prefill 完成 → KV Cache 通过 RDMA 传输到 Decode 节点
    3. Decode 节点执行 continuous batching (iteration-level scheduling):
       每步 admit 新请求 → 运行 ONE step → 移除完成请求
    4. 所有 decode token 生成完毕 → 请求完成

    通信建模：
    - 节点内 (TP all-reduce): NVLink/NVSwitch
    - 节点间 (KV Cache 传输): RDMA (RoCEv2/InfiniBand)

    调度策略 (prefill_schedule_policy):
    - fcfs:   First-Come-First-Served (按到达时间, 默认)
    - sjf:    Shortest-Job-First (按 prefill tokens 升序)
    - ljf:    Longest-Job-First (按 prefill tokens 降序)
    - srtf:   Shortest-Remaining-Time-First (按 decode tokens 升序)
    - random: 随机排序
    - mlfq:   Multi-Level Feedback Queue (多级反馈队列)
    - po:     Priority Ordering (短作业 FCFS + 长作业 SJF)
    - opt:    Optimal (Score = remaining_tokens × noise_factor)
    - lightllm: LightLLM (分离 prefill/decode batch)
    """

    SUPPORTED_SCHEDULERS = ("fcfs", "sjf", "ljf", "srtf", "random", "mlfq", "po", "opt", "lightllm")

    def __init__(
        self,
        ctx: SimContext,
        config: SimulationConfig,
        prefill_schedule_policy: str = "fcfs",
        decode_schedule_policy: str = "fcfs",
    ):
        self.ctx = ctx
        self.config = config
        self._request_counter = 0
        self._rng = np.random.default_rng(config.seed)
        self._prefill_policy = prefill_schedule_policy
        self._decode_policy = decode_schedule_policy
        self._advanced_schedulers = AdvancedSchedulers(seed=config.seed)

    def _compute_moe_imbalance_factor(self, num_tokens: int) -> float:
        """计算 MoE 专家负载不均衡因子。

        使用 Zipf 分布模拟 token→expert 路由，返回 max_gpu_load / avg_gpu_load。
        该因子用于调整 expert_mlp_time：有效专家时间 = expert_mlp_time × imbalance_factor。

        论文 §3.5: "最慢专家决定 all-to-all 通信完成时间"

        Args:
            num_tokens: 当前 batch 的总 token 数

        Returns:
            不均衡因子 (≥ 1.0，1.0 表示完全均衡)
        """
        model = self.ctx.model_config
        if model.num_experts <= 0:
            return 1.0

        alpha = self.config.disaggregated.moe_expert_load_zipf_alpha
        num_experts = model.num_experts
        top_k = model.top_k_experts

        # 生成 Zipf 分布的专家路由权重
        rng = self._rng
        ranks = np.arange(1, num_experts + 1, dtype=float)
        weights = 1.0 / (ranks ** alpha)
        weights /= weights.sum()

        # 模拟 token 路由: 每 token 选 top_k 专家
        expert_loads = np.zeros(num_experts, dtype=np.int64)
        for _ in range(num_tokens):
            chosen = rng.choice(num_experts, size=top_k, replace=False, p=weights)
            expert_loads[chosen] += 1

        # 计算 GPU 负载 (假设均匀放置: 每 GPU 放 num_experts/ep_size 个专家)
        ep_size = min(num_experts, self.ctx.tp_size)
        experts_per_gpu = num_experts // ep_size
        gpu_loads = np.zeros(ep_size, dtype=np.int64)
        for gpu_id in range(ep_size):
            start = gpu_id * experts_per_gpu
            end = start + experts_per_gpu
            gpu_loads[gpu_id] = expert_loads[start:end].sum()

        # Per-expert latency (Roofline 模型)
        ep_model = ExpertParallelModel(model, ep_size, device_config=self.ctx.device_config)
        per_expert_latencies = ep_model._compute_per_expert_latencies(expert_loads)

        # GPU 级延迟: 每 GPU 上所有专家延迟之和 (串行执行)
        gpu_latencies = np.zeros(ep_size, dtype=np.float64)
        for gpu_id in range(ep_size):
            start = gpu_id * experts_per_gpu
            end = start + experts_per_gpu
            gpu_latencies[gpu_id] = per_expert_latencies[start:end].sum()

        max_latency = float(gpu_latencies.max())
        avg_latency = float(gpu_latencies[gpu_latencies > 0].mean())

        if avg_latency <= 0:
            return 1.0

        return max_latency / avg_latency

    def _apply_tp_overlap(self, compute_ms: float, comm_ms: float) -> float:
        """应用通信-计算重叠模型，返回单层墙钟时间 (ms)。

        论文 §3.6: 构建 3D timeline，计算和通信在不同硬件单元上并发执行。
        重叠部分双方因资源竞争而变慢 (compute_slowdown=1.15, comm_slowdown=1.20)。
        墙钟时间 = max(adjusted_compute, adjusted_comm)。

        当 comm_ms == 0 (TP=1) 时，退化为纯计算时间。
        """
        if comm_ms <= 0:
            return compute_ms

        proc = self.ctx.overlap_processor
        pair = proc.make_compute_comm_pair(compute_ms, comm_ms)
        pair = proc.apply_ratio_slowdown(pair)
        return max(pair.adjusted_a_ms, pair.adjusted_b_ms)

    def _create_request(self, arrival_time: float) -> Request:
        """创建一个合成请求。"""
        req_config = self.config.request
        prefill_tokens = self._sample_length(
            req_config.prefill_length, req_config.prefill_length_cv
        )
        decode_tokens = self._sample_length(
            req_config.decode_length, req_config.decode_length_cv
        )
        req = Request(
            id=self._request_counter,
            arrival_time=arrival_time,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
        )
        self._request_counter += 1
        return req

    def _sample_length(self, mean: int, phase_cv: float = -1.0) -> int:
        dist = self.config.request.length_distribution
        if dist == "fixed":
            return mean
        cv = phase_cv if phase_cv >= 0 else self.config.request.length_cv
        std = mean * cv
        return max(1, int(self._rng.normal(mean, std)))

    def _generate_requests(self) -> List[Request]:
        """生成所有请求 (泊松到达过程)。"""
        requests = []
        current_time = 0.0
        qps = self.config.request.qps
        time_limit_ms = self.config.time_limit_s * 1e3

        while current_time < time_limit_ms:
            req = self._create_request(current_time)
            requests.append(req)
            self.ctx.requests[req.id] = req
            # 泊松到达间隔
            interval_s = self._rng.exponential(1.0 / qps) if qps > 0 else float("inf")
            current_time += interval_s * 1e3

        return requests

    def _compute_kv_cache_size(self, request: Request) -> int:
        """计算 KV Cache 大小 (bytes)。

        KV Cache = 2(K+V) * num_layers * num_kv_heads * head_dim * seq_len * 2(float16)
        """
        model = self.ctx.model_config
        head_dim = model.embedding_dim // model.num_q_heads
        kv_per_token = (
            2 * model.num_layers * model.num_kv_heads * head_dim * 2
        )
        return request.prefill_tokens * kv_per_token

    def _compute_tp_allreduce_time(self, num_tokens: int) -> float:
        """计算 TP all-reduce 通信时间 (ms)。"""
        tp = self.ctx.tp_size
        if tp <= 1:
            return 0.0
        data_size = num_tokens * self.ctx.model_config.embedding_dim * 2
        # 每层 2 次 all-reduce (attention + MLP)
        return self.ctx.nvlink_model.get_allreduce_time(tp, data_size) * 2

    def _compute_prefill_time(self, batch_requests: List[Request]) -> float:
        """计算一个 prefill batch 的执行时间 (ms)。

        支持 chunked prefill (论文 §3.4): 当 enable_chunked_prefill=True 时，
        将每个请求的 prefill tokens 拆分为多个 chunk 顺序处理，
        每个 chunk 的 attention 需要 attend 到之前积累的 KV cache。
        这降低了峰值内存但增加了 prefill 延迟。

        时间 = sum(每 chunk 的 [每层计算时间 + 每层 TP 通信时间]) * num_layers
        """
        chunk_times = self._compute_prefill_time_per_chunk(batch_requests)
        return sum(chunk_times)

    def _compute_prefill_time_per_chunk(self, batch_requests: List[Request]) -> List[float]:
        """计算每个 prefill chunk 的执行时间列表 (ms)。

        返回每个 chunk 的执行时间，用于 PIPELINED 传输策略的重叠计算。
        """
        total_tokens = sum(r.prefill_tokens for r in batch_requests)
        batch_size = len(batch_requests)
        model = self.ctx.model_config

        # Chunked prefill 配置
        dcfg = self.config.disaggregated
        if dcfg.enable_chunked_prefill and dcfg.prefill_chunk_size > 0:
            chunk_size = dcfg.prefill_chunk_size
        else:
            chunk_size = total_tokens  # 不拆分

        if total_tokens <= chunk_size:
            # 无需拆分，单次处理
            return [self._compute_prefill_chunk(
                batch_requests, total_tokens, batch_size, kv_cache_size=0
            )]

        # 拆分: 按 chunk_size 逐步处理，KV cache 逐步积累
        chunk_times = []
        processed_tokens = 0
        while processed_tokens < total_tokens:
            this_chunk = min(chunk_size, total_tokens - processed_tokens)
            chunk_time = self._compute_prefill_chunk(
                batch_requests, this_chunk, batch_size, kv_cache_size=processed_tokens
            )
            chunk_times.append(chunk_time)
            processed_tokens += this_chunk

        return chunk_times

    def _compute_prefill_chunk(
        self,
        batch_requests: List[Request],
        num_tokens: int,
        batch_size: int,
        kv_cache_size: int,
    ) -> float:
        """计算单个 prefill chunk 的执行时间 (一层的时间，不含 ×num_layers)。

        Args:
            kv_cache_size: 之前 chunk 积累的 KV cache 长度 (tokens)
        """
        model = self.ctx.model_config

        exec_time = self.ctx.time_predictor.get_execution_time(
            num_tokens=num_tokens,
            batch_size=batch_size,
            kv_cache_size=kv_cache_size,
            is_prefill=True,
        )

        # MoE 负载不均衡调整: expert_mlp_time × (max_load / avg_load)
        if exec_time.expert_mlp_time > 0:
            imbalance_factor = self._compute_moe_imbalance_factor(num_tokens)
            adjusted_total = exec_time.total_time + exec_time.expert_mlp_time * (imbalance_factor - 1.0)
        else:
            adjusted_total = exec_time.total_time

        per_layer_time = adjusted_total
        tp_comm = self._compute_tp_allreduce_time(num_tokens)
        total_per_layer = self._apply_tp_overlap(per_layer_time, tp_comm)

        return total_per_layer * model.num_layers

    def _compute_decode_step_time(self, batch_requests: List[Request]) -> float:
        """计算一步 decode 的执行时间 (ms)。

        Decode 每步生成 1 个 token per request。
        MoE 负载不均衡通过 _compute_moe_imbalance_factor() 调整 expert_mlp_time。
        通信-计算重叠通过 _apply_tp_overlap() 建模 (论文 §3.6)。
        """
        batch_size = len(batch_requests)
        num_tokens = batch_size  # 每请求 1 token
        model = self.ctx.model_config

        avg_kv_cache = sum(r.prefill_tokens + r.num_generated_tokens for r in batch_requests) // max(1, batch_size)

        exec_time = self.ctx.time_predictor.get_execution_time(
            num_tokens=num_tokens,
            batch_size=batch_size,
            kv_cache_size=avg_kv_cache,
            is_prefill=False,
        )

        # MoE 负载不均衡调整: expert_mlp_time × (max_load / avg_load)
        if exec_time.expert_mlp_time > 0:
            imbalance_factor = self._compute_moe_imbalance_factor(num_tokens)
            adjusted_total = exec_time.total_time + exec_time.expert_mlp_time * (imbalance_factor - 1.0)
        else:
            adjusted_total = exec_time.total_time

        per_layer_time = adjusted_total
        tp_comm = self._compute_tp_allreduce_time(num_tokens)
        total_per_layer = self._apply_tp_overlap(per_layer_time, tp_comm)

        return total_per_layer * model.num_layers

    # ─── Speculative Decoding (委托给 SpeculativeDecoder 模块) ──────────────

    def _get_spec_decoder(self) -> "SpeculativeDecoder":
        """懒初始化 SpeculativeDecoder。"""
        if not hasattr(self, '_spec_decoder') or self._spec_decoder is None:
            from distlmsim.execution.speculative_decoder import SpeculativeDecoder
            self._spec_decoder = SpeculativeDecoder(
                self.ctx, self.config.disaggregated, self._rng
            )
        return self._spec_decoder

    def _get_spec_engine(self) -> "SpeculativeDecodingEngine":
        """懒初始化统一的 SpeculativeDecodingEngine。"""
        if not hasattr(self, '_spec_engine') or self._spec_engine is None:
            from distlmsim.execution.speculative_decoder import SpeculativeDecodingEngine
            self._spec_engine = SpeculativeDecodingEngine(
                self.ctx, self.config.disaggregated, self._rng
            )
        return self._spec_engine

    def _compute_dynamic_batch_sizes(self) -> tuple:
        """基于 GPU 显存容量动态计算 prefill 和 decode 的最大 batch size。

        Returns:
            (prefill_bs, decode_bs): 受 GPU 显存约束的实际 batch size
        """
        model = self.ctx.model_config
        device = self.ctx.device_config
        dcfg = self.config.disaggregated
        num_gpus = self.ctx.tp_size

        # 模型参数大小 (GB): 近似 active 参数量 × 2 bytes (FP16)
        if model.num_experts > 0:
            active_params = (model.num_layers * model.embedding_dim *
                             (4 * model.embedding_dim) * 2)
        else:
            active_params = model.num_layers * model.embedding_dim * (4 * model.embedding_dim) * 2
        model_size_gb = active_params / 1e9 / num_gpus

        total_gpu_mem_gb = device.memory_gb
        available_mem_gb = total_gpu_mem_gb * dcfg.gpu_memory_utilization - model_size_gb

        if available_mem_gb <= 0:
            logger.warning("GPU 显存不足! model=%.1fGB, avail=%.1fGB", model_size_gb, available_mem_gb)
            return 1, 1

        head_dim = model.embedding_dim // model.num_q_heads
        # KV cache per request: 2(K+V) × layers × kv_heads × head_dim × avg_seq_len × 2(FP16)
        avg_seq_len = 512 + 128
        kv_per_req_bytes = 2 * model.num_layers * model.num_kv_heads * head_dim * avg_seq_len * 2

        avg_prefill_tokens = self.config.request.prefill_length
        kv_per_prefill_req = 2 * model.num_layers * model.num_kv_heads * head_dim * avg_prefill_tokens * 2

        max_prefill_bs = max(1, int(available_mem_gb * 1e9 / max(1, kv_per_prefill_req)))
        max_decode_bs = max(1, int(available_mem_gb * 1e9 / max(1, kv_per_req_bytes)))

        prefill_bs = min(dcfg.prefill_batch_size, max_prefill_bs)
        decode_bs = min(dcfg.decode_batch_size, max_decode_bs)

        logger.info("动态 batch size: prefill=%d (max=%d), decode=%d (max=%d) "
                     "[model=%.1fGB, avail=%.1fGB, kv/req=%.1fMB]",
                     prefill_bs, max_prefill_bs, decode_bs, max_decode_bs,
                     model_size_gb, available_mem_gb, kv_per_req_bytes / 1e6)
        return prefill_bs, decode_bs

    def _compute_kv_transfer_time(self, kv_cache_bytes: int, concurrent_transfers: int = 1) -> float:
        """计算 KV Cache 传输时间 (ms)，根据策略选择传输方式。

        Args:
            kv_cache_bytes: KV Cache 数据量 (bytes)
            concurrent_transfers: 同时进行的传输流数量 (用于拥塞建模)
        """
        from distlmsim.types import KVCacheTransferStrategy
        strategy = self.config.disaggregated.kv_cache_transfer_strategy

        if strategy == KVCacheTransferStrategy.DIRECT:
            return self.ctx.rdma_model.get_transfer_time(
                kv_cache_bytes, concurrent_transfers=concurrent_transfers
            )
        elif strategy == KVCacheTransferStrategy.STORE_FORWARD:
            return self._compute_store_forward_time(kv_cache_bytes)
        else:
            # PIPELINED 不在此处计算，在 run() 中按 chunk 处理
            return self.ctx.rdma_model.get_transfer_time(
                kv_cache_bytes, concurrent_transfers=concurrent_transfers
            )

    def _compute_store_forward_time(self, kv_cache_bytes: int) -> float:
        """计算 Store-and-Forward 传输时间 (ms)。

        流程: prefill 节点写入中间存储 → decode 节点从中间存储读取
        时间 = write_time + storage_latency + read_time
        """
        dcfg = self.config.disaggregated
        write_bw = dcfg.store_forward_write_bw_gbps * 1e9 / 8  # GB/s → B/s
        read_bw = dcfg.store_forward_read_bw_gbps * 1e9 / 8
        latency_ms = dcfg.store_forward_latency_us / 1000.0

        write_time = kv_cache_bytes / write_bw * 1000  # ms
        read_time = kv_cache_bytes / read_bw * 1000

        return write_time + latency_ms + read_time

    def _compute_pipelined_kv_ready_time(
        self,
        batch_requests: List[Request],
        prefill_start: float,
    ) -> float:
        """计算 PIPELINED 策略下 KV Cache 就绪时间。

        在 chunked prefill 中，每个 chunk 完成后立即开始传输该 chunk 的 KV cache，
        传输与下一个 chunk 的计算重叠。最终就绪时间取决于最后一个 chunk 的传输完成。

        Returns:
            KV Cache 就绪时间 (ms, 绝对时间)
        """
        total_tokens = sum(r.prefill_tokens for r in batch_requests)
        chunk_times = self._compute_prefill_time_per_chunk(batch_requests)
        dcfg = self.config.disaggregated

        if dcfg.enable_chunked_prefill and dcfg.prefill_chunk_size > 0:
            chunk_size = dcfg.prefill_chunk_size
        else:
            chunk_size = total_tokens

        # 按 chunk 逐步计算
        compute_cursor = prefill_start  # 计算进度游标
        transfer_end = prefill_start    # 传输完成时间
        processed_tokens = 0

        for chunk_time in chunk_times:
            # 当前 chunk 的计算完成时间
            compute_cursor += chunk_time
            processed_tokens = min(processed_tokens + chunk_size, total_tokens)

            # 当前 chunk 对应的 KV cache 大小
            chunk_kv_bytes = sum(
                self._compute_kv_cache_size_for_tokens(r, min(chunk_size, r.prefill_tokens - (processed_tokens - chunk_size)))
                for r in batch_requests
            )

            # 传输可以在计算完成后立即开始
            # batch 中所有请求同时传输该 chunk 的 KV cache
            chunk_transfer_time = self.ctx.rdma_model.get_transfer_time(
                max(1, chunk_kv_bytes),
                concurrent_transfers=len(batch_requests),
            )
            chunk_transfer_end = compute_cursor + chunk_transfer_time

            # 更新最晚传输完成时间
            transfer_end = max(transfer_end, chunk_transfer_end)

        return transfer_end

    def _compute_kv_cache_size_for_tokens(self, request: Request, num_tokens: int) -> int:
        """计算指定 token 数的 KV Cache 大小 (bytes)。"""
        model = self.ctx.model_config
        head_dim = model.embedding_dim // model.num_q_heads
        kv_per_token = 2 * model.num_layers * model.num_kv_heads * head_dim * 2
        return num_tokens * kv_per_token

    def _select_from_queue(
        self,
        waiting_queue: List[Request],
        batch_size: int,
        policy: str,
        kv_ready_time: Optional[Dict[int, float]] = None,
        current_time: float = 0.0,
    ) -> List[Request]:
        """从等待队列中按策略选取最多 batch_size 个请求。

        只在已到达的请求中做选择，不做全局重排序。
        """
        if len(waiting_queue) <= batch_size:
            selected = list(waiting_queue)
        else:
            if policy == "fcfs":
                # 按到达时间升序 (最早到达优先)
                candidates = sorted(waiting_queue, key=lambda r: r.arrival_time)
                selected = candidates[:batch_size]
            elif policy == "sjf":
                # 按 prefill tokens 升序 (最短作业优先，减少平均等待)
                candidates = sorted(waiting_queue, key=lambda r: (r.prefill_tokens, r.arrival_time))
                selected = candidates[:batch_size]
            elif policy == "ljf":
                # 按 prefill tokens 降序 (最长作业优先)
                candidates = sorted(waiting_queue, key=lambda r: (-r.prefill_tokens, r.arrival_time))
                selected = candidates[:batch_size]
            elif policy == "srtf":
                # 按 decode tokens 升序 (最短 decode 优先，优化 E2E)
                candidates = sorted(waiting_queue, key=lambda r: (r.decode_tokens, r.arrival_time))
                selected = candidates[:batch_size]
            elif policy == "random":
                indices = self._rng.choice(len(waiting_queue), size=batch_size, replace=False)
                selected = [waiting_queue[i] for i in indices]
            elif policy == "mlfq":
                # 多级反馈队列
                selected = self._advanced_schedulers.select_mlfq(
                    waiting_queue, batch_size, current_time
                )
            elif policy == "po":
                # 优先级排序 (短作业 FCFS + 长作业 SJF)
                selected = self._advanced_schedulers.select_po(
                    waiting_queue, batch_size
                )
            elif policy == "opt":
                # 最优调度 (Score = remaining_tokens × noise_factor)
                selected = self._advanced_schedulers.select_opt(
                    waiting_queue, batch_size, current_time
                )
            elif policy == "lightllm":
                # LightLLM (分离 prefill/decode batch)
                # 根据是否已有 generated tokens 判断是 prefill 还是 decode
                has_prefilled = any(r.num_generated_tokens > 0 for r in waiting_queue)
                if has_prefilled:
                    selected = self._advanced_schedulers.select_lightllm_decode(
                        waiting_queue, batch_size
                    )
                else:
                    selected = self._advanced_schedulers.select_lightllm_prefill(
                        waiting_queue, batch_size, current_time
                    )
            else:
                selected = waiting_queue[:batch_size]

        return selected

    def run(self) -> MetricsStore:
        """运行完整的存算分离模拟 (continuous batching)。

        调度逻辑:
        1. 所有请求按到达时间排序
        2. 维护 prefill 等待队列: 请求到达后入队，prefill 节点空闲时按策略选取
        3. Prefill 完成后 KV Cache 通过 RDMA 传输，进入 decode 等待队列
        4. Decode 阶段采用 continuous batching (iteration-level scheduling):
           - 维护 active_decoding 持久字典，跨迭代保持
           - 每步迭代: admit 新 KV-ready 请求 → 运行 ONE step → 移除完成请求
           - batch 组成每步动态变化，与 vLLM/Orca 行为对齐

        调度策略只在"已形成队列"的请求中做选择，不做全局重排序。
        这是真实调度器的行为: 无法预知未来到达的请求。

        Returns:
            MetricsStore 包含所有请求的性能指标
        """
        ms = self.ctx.metrics_store
        all_requests = self._generate_requests()
        all_requests.sort(key=lambda r: r.arrival_time)
        total_requests = len(all_requests)

        logger.info(f"生成 {total_requests} 个请求")
        logger.info(f"集群: 2 节点, 每节点 {self.ctx.num_gpus_per_node} GPU (A800)")
        logger.info(f"  Prefill 节点: {self.ctx.prefill_node_id} (TP={self.ctx.tp_size})")
        logger.info(f"  Decode 节点:  {self.ctx.decode_node_id} (TP={self.ctx.tp_size})")
        logger.info(f"  RDMA: {self.ctx.network_config.rdma.protocol.name} "
                     f"{self.ctx.network_config.rdma.bandwidth_gbps} Gbps")
        logger.info(f"  NVLink: {self.ctx.network_config.nvlink.bandwidth_gbps} GB/s")
        logger.info(f"Prefill 调度策略: {self._prefill_policy}")
        logger.info(f"Decode 调度策略:  {self._decode_policy}")

        # 动态 batch size: 基于 GPU 显存约束
        prefill_bs, decode_bs = self._compute_dynamic_batch_sizes()

        # 记录到达信息
        for req in all_requests:
            ms.record_request_arrival(req.id, req.arrival_time)
            ms.set_request_tokens(req.id, req.prefill_tokens, req.decode_tokens)

        # ─── Prefill 阶段: 实时队列调度 ───────────────────────────────────
        arrival_idx = 0                      # 下一个待入队的请求索引
        prefill_free_time = 0.0              # prefill 节点最早可用时间
        prefill_waiting: List[Request] = []  # prefill 等待队列
        kv_ready_time: Dict[int, float] = {} # req.id -> KV Cache 就绪时间

        while arrival_idx < total_requests or prefill_waiting:
            # 确定当前时间
            if prefill_waiting:
                current_time = prefill_free_time
            else:
                # 队列空，跳到下一个请求到达时间
                current_time = max(prefill_free_time, all_requests[arrival_idx].arrival_time)

            # 将所有已到达的请求加入等待队列
            while arrival_idx < total_requests and all_requests[arrival_idx].arrival_time <= current_time:
                prefill_waiting.append(all_requests[arrival_idx])
                arrival_idx += 1

            if not prefill_waiting:
                continue

            # 按调度策略从队列中选取
            batch = self._select_from_queue(
                prefill_waiting, prefill_bs, self._prefill_policy,
                current_time=current_time
            )
            for req in batch:
                prefill_waiting.remove(req)

            # ─── 执行 Prefill ─────────────────────────────────────────────
            prefill_start = max(prefill_free_time, current_time)
            prefill_time = self._compute_prefill_time(batch)
            prefill_end = prefill_start + prefill_time
            prefill_free_time = prefill_end

            for req in batch:
                req.status = RequestStatus.PREFILLING
                req.prefill_node_id = self.ctx.prefill_node_id
                req.prefill_start_time = prefill_start
                req.prefill_end_time = prefill_end
                ms.record_prefill_start(req.id, prefill_start, self.ctx.prefill_node_id)
                ms.record_prefill_end(req.id, prefill_end)

            # ─── KV Cache 传输 (根据策略) ─────────────────────────────────
            from distlmsim.types import KVCacheTransferStrategy
            strategy = self.config.disaggregated.kv_cache_transfer_strategy

            if strategy == KVCacheTransferStrategy.PIPELINED:
                # PIPELINED: chunked prefill 中边算边传
                pipelined_end = self._compute_pipelined_kv_ready_time(batch, prefill_start)
                for req in batch:
                    kv_size = self._compute_kv_cache_size(req)
                    req.kv_cache_size_bytes = kv_size
                    kv_ready_time[req.id] = pipelined_end
                    req.status = RequestStatus.KV_CACHE_TRANSFERRING
                    ms.record_kv_cache_transfer_start(req.id, prefill_end)
                    ms.record_kv_cache_transfer_end(req.id, pipelined_end)
            else:
                # DIRECT / STORE_FORWARD: prefill 完成后一次性传输
                # 所有请求同时传输 KV cache，共享 RDMA 链路带宽
                batch_size = len(batch)
                for req in batch:
                    kv_size = self._compute_kv_cache_size(req)
                    req.kv_cache_size_bytes = kv_size
                    transfer_time = self._compute_kv_transfer_time(
                        kv_size, concurrent_transfers=batch_size
                    )
                    transfer_end = prefill_end + transfer_time
                    kv_ready_time[req.id] = transfer_end
                    req.status = RequestStatus.KV_CACHE_TRANSFERRING
                    ms.record_kv_cache_transfer_start(req.id, prefill_end)
                    ms.record_kv_cache_transfer_end(req.id, transfer_end)

            logger.debug(
                f"Prefill batch: size={len(batch)}, "
                f"start={prefill_start:.1f}ms, time={prefill_time:.1f}ms, "
                f"queue_depth={len(prefill_waiting)}"
            )

        # ─── Decode 阶段: continuous batching (iteration-level) ─────────────
        # 每步迭代: admit 新 KV-ready 请求 → 运行 ONE step → 移除完成请求
        # 与 vLLM/Orca 的 continuous batching 对齐: batch 组成每步动态变化
        decode_candidates = sorted(all_requests, key=lambda r: kv_ready_time[r.id])
        decode_arrival_idx = 0
        decode_free_time = 0.0
        decode_waiting: List[Request] = []
        active_decoding: Dict[int, Request] = {}  # 跨迭代持久的活跃 decode 请求
        spec_engine = self._get_spec_engine()

        while decode_arrival_idx < total_requests or decode_waiting or active_decoding:
            # 确定当前时间
            if active_decoding:
                current_time = decode_free_time
            elif decode_waiting:
                current_time = decode_free_time
            else:
                # 无活跃/等待请求，跳到下一个 KV-ready 时间
                current_time = max(
                    decode_free_time,
                    kv_ready_time[decode_candidates[decode_arrival_idx].id]
                )

            # 将所有 KV Cache 已就绪的请求加入 decode 等待队列
            while (decode_arrival_idx < total_requests
                   and kv_ready_time[decode_candidates[decode_arrival_idx].id] <= current_time):
                decode_waiting.append(decode_candidates[decode_arrival_idx])
                decode_arrival_idx += 1

            # Admit: 按调度策略从等待队列选取，填充到 decode_bs 容量
            available_slots = decode_bs - len(active_decoding)
            if available_slots > 0 and decode_waiting:
                admitted = self._select_from_queue(
                    decode_waiting, available_slots, self._decode_policy,
                    kv_ready_time=kv_ready_time,
                    current_time=current_time
                )
                for req in admitted:
                    decode_waiting.remove(req)
                    req.status = RequestStatus.DECODING
                    req.decode_node_id = self.ctx.decode_node_id
                    req.decode_start_time = current_time
                    ms.record_decode_start(req.id, current_time, self.ctx.decode_node_id)
                    active_decoding[req.id] = req

            if not active_decoding:
                continue

            # Execute ONE decode step for all active requests
            active_list = list(active_decoding.values())
            result = spec_engine.compute_cycle(
                active_list, current_time,
                compute_standard_step_fn=self._compute_decode_step_time,
            )
            step_end = current_time + result.cycle_time_ms

            # Update each request and remove completed ones
            completed_ids = []
            for i, req in enumerate(active_list):
                if result.per_request_accepted is not None:
                    req_accepted = result.per_request_accepted[i]
                else:
                    req_accepted = result.accepted_tokens
                req.num_generated_tokens += req_accepted
                req.accepted_tokens_last_cycle = req_accepted
                if result.is_speculative:
                    req.total_spec_cycles += 1
                    req.total_spec_accepted += req_accepted
                if req.num_generated_tokens >= req.decode_tokens:
                    req.decode_end_time = step_end
                    req.status = RequestStatus.COMPLETED
                    ms.record_decode_end(req.id, step_end)
                    completed_ids.append(req.id)

            for req_id in completed_ids:
                del active_decoding[req_id]

            decode_free_time = step_end

            logger.debug(
                f"Decode iter: active={len(active_decoding)}, "
                f"waiting={len(decode_waiting)}, step_time={result.cycle_time_ms:.2f}ms"
            )

        ms.finalize()
        return ms


# ─── 通用分布式模拟器 (事件驱动) ───────────────────────────────────────────────


class DistributedInferenceSimulator:
    """通用分布式推理模拟器 (事件驱动框架)。

    基于 heapq 的离散事件循环。适用于 TP+PP 场景的通用模拟，
    支持多副本部署和请求路由。

    事件循环:
    1. 生成请求序列 (Poisson 到达)
    2. 创建 RequestArrivalEvent 入堆
    3. 弹出最早事件 → handle_event() → 生成后续事件入堆
    4. 重复直到时间上限或无剩余事件
    """

    def __init__(self, config: SimulationConfig):
        self._config = config
        self._rng = np.random.default_rng(config.seed)
        self._request_counter = 0

        # 从 SimulationConfig 提取子配置
        replica_cfg = config.cluster.replica
        tp_size = replica_cfg.tensor_parallel_size

        # 初始化 SimContext
        self.ctx = SimContext(
            model_config=replica_cfg.model,
            device_config=replica_cfg.device_sku,
            network_config=config.cluster.network,
            metrics_store=MetricsStore(config.metrics),
            tp_size=tp_size,
            num_gpus_per_node=tp_size,
        )

        # 全局调度器 (简化版: 单副本 round-robin)
        from distlmsim.scheduling.global_scheduler import RoundRobinGlobalScheduler

        class _SingleReplicaCluster:
            """Minimal ClusterView for single-replica event-driven sim."""
            replicas = {0: None}
            nodes = {}

        self._scheduler = RoundRobinGlobalScheduler(_SingleReplicaCluster())
        logger.info("DistributedInferenceSimulator 初始化")

    def _create_request(self, arrival_time: float) -> Request:
        req_config = self._config.request
        prefill_tokens = self._sample_length(req_config.prefill_length)
        decode_tokens = self._sample_length(req_config.decode_length)
        req = Request(
            id=self._request_counter,
            arrival_time=arrival_time,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
        )
        self._request_counter += 1
        self.ctx.requests[req.id] = req
        return req

    def _sample_length(self, mean: int) -> int:
        dist = self._config.request.length_distribution
        if dist == "fixed":
            return mean
        cv = self._config.request.length_cv
        std = mean * cv
        return max(1, int(self._rng.normal(mean, std)))

    def run(self) -> MetricsStore:
        """运行事件驱动模拟。"""
        from distlmsim.events import (
            RequestArrivalEvent,
            PrefillCompleteEvent,
            DecodeStartEvent,
        )

        time_limit_ms = self._config.time_limit_s * 1e3
        metrics = self.ctx.metrics_store

        # 内部 decode 事件 (简化版，不依赖 BatchEndEvent 的 batch_id 语义)
        class _DecodeStepEvent:
            """内部 decode step 事件。"""
            __slots__ = ("_time", "_request_id")
            def __init__(self, time: float, request_id: int):
                self._time = time
                self._request_id = request_id

        # 1. Generate all requests
        requests: List[Request] = []
        current_time = 0.0
        qps = self._config.request.qps
        while current_time < time_limit_ms:
            req = self._create_request(current_time)
            requests.append(req)
            interval_s = self._rng.exponential(1.0 / qps) if qps > 0 else float("inf")
            current_time += interval_s * 1e3

        logger.info("生成 %d 个请求", len(requests))

        # 2. Build event heap
        event_heap: list = []
        event_id = 0
        for req in requests:
            heapq.heappush(event_heap, (req.arrival_time, event_id, RequestArrivalEvent(req.arrival_time, req.id)))
            event_id += 1

        # 3. Event loop
        while event_heap:
            t, _, event = heapq.heappop(event_heap)
            if t > time_limit_ms:
                break

            if isinstance(event, RequestArrivalEvent):
                req = self.ctx.requests.get(event._request_id)
                if req is None:
                    continue
                metrics.record_request_arrival(req.id, t)
                req.status = RequestStatus.PREFILLING
                metrics.record_request_scheduled(req.id, t)
                req.prefill_start_time = t

                # Compute prefill time
                et = self.ctx.time_predictor.get_execution_time(
                    req.prefill_tokens, 1, 0, is_prefill=True
                )
                prefill_time = et.total_time
                prefill_end = t + prefill_time
                req.prefill_end_time = prefill_end

                heapq.heappush(event_heap, (prefill_end, event_id,
                    PrefillCompleteEvent(prefill_end, req.id, 0, 0)))
                event_id += 1

            elif isinstance(event, PrefillCompleteEvent):
                req = self.ctx.requests.get(event._request_id)
                if req is None:
                    continue
                req.status = RequestStatus.DECODING
                req.decode_start_time = t
                metrics.record_decode_start(req.id, t, 0)
                req.num_generated_tokens = 0

                et = self.ctx.time_predictor.get_execution_time(
                    1, 1, req.prefill_tokens, is_prefill=False
                )
                heapq.heappush(event_heap, (t + et.total_time, event_id,
                    _DecodeStepEvent(t + et.total_time, req.id)))
                event_id += 1

            elif isinstance(event, _DecodeStepEvent):
                req = self.ctx.requests.get(event._request_id)
                if req is None:
                    continue
                req.num_generated_tokens += 1
                if req.num_generated_tokens >= req.decode_tokens:
                    req.status = RequestStatus.COMPLETED
                    req.completion_time = t
                    metrics.record_decode_end(req.id, t)
                    metrics.set_request_tokens(req.id, req.prefill_tokens, req.decode_tokens)
                else:
                    et = self.ctx.time_predictor.get_execution_time(
                        1, 1, req.prefill_tokens + req.num_generated_tokens, is_prefill=False
                    )
                    heapq.heappush(event_heap, (t + et.total_time, event_id,
                        _DecodeStepEvent(t + et.total_time, req.id)))
                    event_id += 1

        metrics.finalize()
        metrics.print_summary()
        return metrics


# ─── Colocated 模拟器 ──────────────────────────────────────────────────────────


class ColocatedSimulator:
    """Colocated (非分离) 推理模拟器。

    Prefill 和 Decode 共享同一组 GPU。每个 iteration 中，
    scheduler 从等待队列中选取请求，组成包含 prefill 和 decode
    请求的混合 batch。

    关键区别:
    - 无 KV Cache 传输延迟 (无 RDMA)
    - Prefill 和 Decode 竞争同一组 GPU 资源
    - Iteration 时间由混合 batch 中最慢的请求决定
    - TTFT = 第一个 decode token 的生成时间 - 请求到达时间
    """

    SUPPORTED_SCHEDULERS = DisaggregatedSimulator.SUPPORTED_SCHEDULERS

    def __init__(
        self,
        ctx: SimContext,
        config: SimulationConfig,
        schedule_policy: str = "fcfs",
    ):
        self.ctx = ctx
        self.config = config
        self._request_counter = 0
        self._rng = np.random.default_rng(config.seed)
        self._policy = schedule_policy
        self._advanced_schedulers = AdvancedSchedulers(seed=config.seed)

    def _compute_moe_imbalance_factor(self, num_tokens: int) -> float:
        """计算 MoE 专家负载不均衡因子 (同 DisaggregatedSimulator)。"""
        model = self.ctx.model_config
        if model.num_experts <= 0:
            return 1.0

        alpha = self.config.disaggregated.moe_expert_load_zipf_alpha
        num_experts = model.num_experts
        top_k = model.top_k_experts

        rng = self._rng
        ranks = np.arange(1, num_experts + 1, dtype=float)
        weights = 1.0 / (ranks ** alpha)
        weights /= weights.sum()

        expert_loads = np.zeros(num_experts, dtype=np.int64)
        for _ in range(num_tokens):
            chosen = rng.choice(num_experts, size=top_k, replace=False, p=weights)
            expert_loads[chosen] += 1

        ep_size = min(num_experts, self.ctx.tp_size)
        experts_per_gpu = num_experts // ep_size

        # Per-expert latency (Roofline 模型)
        ep_model = ExpertParallelModel(model, ep_size, device_config=self.ctx.device_config)
        per_expert_latencies = ep_model._compute_per_expert_latencies(expert_loads)

        # GPU 级延迟
        gpu_latencies = np.zeros(ep_size, dtype=np.float64)
        for gpu_id in range(ep_size):
            start = gpu_id * experts_per_gpu
            end = start + experts_per_gpu
            gpu_latencies[gpu_id] = per_expert_latencies[start:end].sum()

        max_latency = float(gpu_latencies.max())
        avg_latency = float(gpu_latencies[gpu_latencies > 0].mean())
        return max_latency / avg_latency if avg_latency > 0 else 1.0

    def _apply_tp_overlap(self, compute_ms: float, comm_ms: float) -> float:
        """应用通信-计算重叠模型 (同 DisaggregatedSimulator)。"""
        if comm_ms <= 0:
            return compute_ms

        proc = self.ctx.overlap_processor
        pair = proc.make_compute_comm_pair(compute_ms, comm_ms)
        pair = proc.apply_ratio_slowdown(pair)
        return max(pair.adjusted_a_ms, pair.adjusted_b_ms)

    def _create_request(self, arrival_time: float) -> Request:
        req_config = self.config.request
        prefill_tokens = self._sample_length(req_config.prefill_length)
        decode_tokens = self._sample_length(req_config.decode_length)
        req = Request(
            id=self._request_counter,
            arrival_time=arrival_time,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
        )
        self._request_counter += 1
        return req

    def _sample_length(self, mean: int) -> int:
        dist = self.config.request.length_distribution
        if dist == "fixed":
            return mean
        cv = self.config.request.length_cv
        std = mean * cv
        return max(1, int(self._rng.normal(mean, std)))

    def _generate_requests(self) -> List[Request]:
        requests = []
        current_time = 0.0
        qps = self.config.request.qps
        time_limit_ms = self.config.time_limit_s * 1e3

        while current_time < time_limit_ms:
            req = self._create_request(current_time)
            requests.append(req)
            self.ctx.requests[req.id] = req
            interval_s = self._rng.exponential(1.0 / qps) if qps > 0 else float("inf")
            current_time += interval_s * 1e3

        return requests

    def _compute_tp_allreduce_time(self, num_tokens: int) -> float:
        tp = self.ctx.tp_size
        if tp <= 1:
            return 0.0
        data_size = num_tokens * self.ctx.model_config.embedding_dim * 2
        return self.ctx.nvlink_model.get_allreduce_time(tp, data_size) * 2

    def _compute_dynamic_batch_sizes(self) -> tuple:
        """动态 batch size (同 DisaggregatedSimulator)。"""
        model = self.ctx.model_config
        device = self.ctx.device_config
        dcfg = self.config.disaggregated
        num_gpus = self.ctx.tp_size

        if model.num_experts > 0:
            active_params = model.num_layers * model.embedding_dim * (4 * model.embedding_dim) * 2
        else:
            active_params = model.num_layers * model.embedding_dim * (4 * model.embedding_dim) * 2
        model_size_gb = active_params / 1e9 / num_gpus
        available_mem_gb = device.memory_gb * dcfg.gpu_memory_utilization - model_size_gb
        if available_mem_gb <= 0:
            return 1, 1

        head_dim = model.embedding_dim // model.num_q_heads
        avg_seq_len = 512 + 128
        kv_per_req = 2 * model.num_layers * model.num_kv_heads * head_dim * avg_seq_len * 2
        avg_pf_tokens = self.config.request.prefill_length
        kv_per_pf = 2 * model.num_layers * model.num_kv_heads * head_dim * avg_pf_tokens * 2

        max_pf = max(1, int(available_mem_gb * 1e9 / max(1, kv_per_pf)))
        max_dec = max(1, int(available_mem_gb * 1e9 / max(1, kv_per_req)))
        return min(dcfg.prefill_batch_size, max_pf), min(dcfg.decode_batch_size, max_dec)

    def _compute_prefill_time(self, batch_requests: List[Request]) -> float:
        """Colocated prefill (支持 chunked prefill，同 DisaggregatedSimulator)。"""
        total_tokens = sum(r.prefill_tokens for r in batch_requests)
        batch_size = len(batch_requests)

        dcfg = self.config.disaggregated
        if dcfg.enable_chunked_prefill and dcfg.prefill_chunk_size > 0:
            chunk_size = dcfg.prefill_chunk_size
        else:
            chunk_size = total_tokens

        if total_tokens <= chunk_size:
            return self._compute_prefill_chunk(
                batch_requests, total_tokens, batch_size, kv_cache_size=0
            )

        total_time = 0.0
        processed_tokens = 0
        while processed_tokens < total_tokens:
            this_chunk = min(chunk_size, total_tokens - processed_tokens)
            chunk_time = self._compute_prefill_chunk(
                batch_requests, this_chunk, batch_size, kv_cache_size=processed_tokens
            )
            total_time += chunk_time
            processed_tokens += this_chunk

        return total_time

    def _compute_prefill_chunk(
        self,
        batch_requests: List[Request],
        num_tokens: int,
        batch_size: int,
        kv_cache_size: int,
    ) -> float:
        """计算单个 prefill chunk 的执行时间 (同 DisaggregatedSimulator)。"""
        model = self.ctx.model_config

        exec_time = self.ctx.time_predictor.get_execution_time(
            num_tokens=num_tokens,
            batch_size=batch_size,
            kv_cache_size=kv_cache_size,
            is_prefill=True,
        )

        if exec_time.expert_mlp_time > 0:
            imbalance_factor = self._compute_moe_imbalance_factor(num_tokens)
            adjusted_total = exec_time.total_time + exec_time.expert_mlp_time * (imbalance_factor - 1.0)
        else:
            adjusted_total = exec_time.total_time

        per_layer_time = adjusted_total
        tp_comm = self._compute_tp_allreduce_time(num_tokens)
        total_per_layer = self._apply_tp_overlap(per_layer_time, tp_comm)

        return total_per_layer * model.num_layers

    def _compute_decode_step_time(self, batch_requests: List[Request]) -> float:
        batch_size = len(batch_requests)
        num_tokens = batch_size
        model = self.ctx.model_config

        avg_kv_cache = sum(
            r.prefill_tokens + r.num_generated_tokens for r in batch_requests
        ) // max(1, batch_size)

        exec_time = self.ctx.time_predictor.get_execution_time(
            num_tokens=num_tokens,
            batch_size=batch_size,
            kv_cache_size=avg_kv_cache,
            is_prefill=False,
        )

        # MoE 负载不均衡调整
        if exec_time.expert_mlp_time > 0:
            imbalance_factor = self._compute_moe_imbalance_factor(num_tokens)
            adjusted_total = exec_time.total_time + exec_time.expert_mlp_time * (imbalance_factor - 1.0)
        else:
            adjusted_total = exec_time.total_time

        per_layer_time = adjusted_total
        tp_comm = self._compute_tp_allreduce_time(num_tokens)
        total_per_layer = self._apply_tp_overlap(per_layer_time, tp_comm)

        return total_per_layer * model.num_layers

    def _compute_mixed_batch_time(
        self,
        prefill_reqs: List[Request],
        decode_reqs: List[Request],
    ) -> float:
        """计算混合 batch (prefill + decode) 的执行时间。

        在 colocated 模式下，prefill 和 decode 请求在同一 iteration 中执行。
        采用 Sarathi 风格的 chunked prefill: prefill 和 decode 合并为一个
        forward pass，总 token 数 = prefill tokens + decode tokens (每请求 1)。
        """
        if not prefill_reqs and not decode_reqs:
            return 0.0

        prefill_tokens = sum(r.prefill_tokens for r in prefill_reqs)
        decode_tokens = len(decode_reqs)  # 每 decode 请求 1 token
        total_tokens = prefill_tokens + decode_tokens
        total_batch_size = len(prefill_reqs) + len(decode_reqs)

        if total_tokens == 0:
            return 0.0

        model = self.ctx.model_config

        # 如果有 prefill tokens (compute-bound)，用 prefill 模型
        # 否则用 decode 模型 (memory-bound)
        if prefill_tokens > 0:
            avg_kv = 0
            is_prefill = True
        else:
            avg_kv = sum(
                r.prefill_tokens + r.num_generated_tokens for r in decode_reqs
            ) // max(1, len(decode_reqs))
            is_prefill = False

        exec_time = self.ctx.time_predictor.get_execution_time(
            num_tokens=total_tokens,
            batch_size=total_batch_size,
            kv_cache_size=avg_kv,
            is_prefill=is_prefill,
        )

        # MoE 负载不均衡调整
        if exec_time.expert_mlp_time > 0:
            imbalance_factor = self._compute_moe_imbalance_factor(total_tokens)
            adjusted_total = exec_time.total_time + exec_time.expert_mlp_time * (imbalance_factor - 1.0)
        else:
            adjusted_total = exec_time.total_time

        per_layer_time = adjusted_total
        tp_comm = self._compute_tp_allreduce_time(total_tokens)
        total_per_layer = self._apply_tp_overlap(per_layer_time, tp_comm)

        return total_per_layer * model.num_layers

    def _compute_chunked_mixed_batch_time(
        self,
        chunked_prefill_tokens: int,
        prefill_reqs: List[Request],
        decode_reqs: List[Request],
    ) -> float:
        """计算 chunked prefill + decode 混合 batch 的执行时间。

        与 _compute_mixed_batch_time 的区别：prefill tokens 已经被 chunked，
        只计算本 iteration 需要处理的 prefill chunk tokens。
        """
        if chunked_prefill_tokens == 0 and not decode_reqs:
            return 0.0

        decode_tokens = len(decode_reqs)
        total_tokens = chunked_prefill_tokens + decode_tokens
        total_batch_size = len(prefill_reqs) + len(decode_reqs)

        if total_tokens == 0:
            return 0.0

        model = self.ctx.model_config

        if chunked_prefill_tokens > 0:
            avg_kv = 0
            is_prefill = True
        else:
            avg_kv = sum(
                r.prefill_tokens + r.num_generated_tokens for r in decode_reqs
            ) // max(1, len(decode_reqs))
            is_prefill = False

        exec_time = self.ctx.time_predictor.get_execution_time(
            num_tokens=total_tokens,
            batch_size=total_batch_size,
            kv_cache_size=avg_kv,
            is_prefill=is_prefill,
        )

        if exec_time.expert_mlp_time > 0:
            imbalance_factor = self._compute_moe_imbalance_factor(total_tokens)
            adjusted_total = exec_time.total_time + exec_time.expert_mlp_time * (imbalance_factor - 1.0)
        else:
            adjusted_total = exec_time.total_time

        per_layer_time = adjusted_total
        tp_comm = self._compute_tp_allreduce_time(total_tokens)
        total_per_layer = self._apply_tp_overlap(per_layer_time, tp_comm)

        return total_per_layer * model.num_layers

    def _select_from_queue(
        self,
        waiting_queue: List[Request],
        batch_size: int,
        policy: str,
        current_time: float = 0.0,
    ) -> List[Request]:
        if len(waiting_queue) <= batch_size:
            return list(waiting_queue)

        if policy == "fcfs":
            candidates = sorted(waiting_queue, key=lambda r: r.arrival_time)
            return candidates[:batch_size]
        elif policy == "sjf":
            candidates = sorted(waiting_queue, key=lambda r: (r.prefill_tokens, r.arrival_time))
            return candidates[:batch_size]
        elif policy == "ljf":
            candidates = sorted(waiting_queue, key=lambda r: (-r.prefill_tokens, r.arrival_time))
            return candidates[:batch_size]
        elif policy == "srtf":
            candidates = sorted(waiting_queue, key=lambda r: (r.decode_tokens, r.arrival_time))
            return candidates[:batch_size]
        elif policy == "random":
            indices = self._rng.choice(len(waiting_queue), size=batch_size, replace=False)
            return [waiting_queue[i] for i in indices]
        elif policy == "mlfq":
            return self._advanced_schedulers.select_mlfq(waiting_queue, batch_size, current_time)
        elif policy == "po":
            return self._advanced_schedulers.select_po(waiting_queue, batch_size)
        elif policy == "opt":
            return self._advanced_schedulers.select_opt(waiting_queue, batch_size, current_time)
        elif policy == "lightllm":
            has_prefilled = any(r.num_generated_tokens > 0 for r in waiting_queue)
            if has_prefilled:
                return self._advanced_schedulers.select_lightllm_decode(waiting_queue, batch_size)
            else:
                return self._advanced_schedulers.select_lightllm_prefill(waiting_queue, batch_size, current_time)
        else:
            return waiting_queue[:batch_size]

    def run(self) -> MetricsStore:
        """运行 Colocated 模式模拟。

        调度逻辑:
        1. 维护统一等待队列: 请求到达后入队
        2. 每次 iteration:
           a. 从队列中选取新请求做 prefill (受 max_prefill_per_iter 限制)
           b. 将所有已 prefill 完成的请求加入 decode batch
           c. 执行混合 batch (prefill 新请求 + decode 进行中的请求)
        3. Prefill 完成 → 请求进入 decode 阶段
        4. Decode 完成 → 请求结束
        """
        ms = self.ctx.metrics_store
        all_requests = self._generate_requests()
        all_requests.sort(key=lambda r: r.arrival_time)
        total_requests = len(all_requests)

        logger.info(f"Colocated 模式: 生成 {total_requests} 个请求")
        logger.info(f"集群: 1 节点, {self.ctx.num_gpus_per_node} GPU (A800), TP={self.ctx.tp_size}")
        logger.info(f"调度策略: {self._policy}")

        max_prefill_per_iter, max_batch_size = self._compute_dynamic_batch_sizes()

        # 记录到达信息
        for req in all_requests:
            ms.record_request_arrival(req.id, req.arrival_time)
            ms.set_request_tokens(req.id, req.prefill_tokens, req.decode_tokens)

        arrival_idx = 0
        waiting_queue: List[Request] = []
        prefilling: Dict[int, Request] = {}   # req.id -> Request (正在 prefill)
        decoding: Dict[int, Request] = {}     # req.id -> Request (正在 decode)
        prefill_progress: Dict[int, int] = {}  # req.id -> tokens processed so far
        current_time = 0.0

        dcfg = self.config.disaggregated
        chunked = dcfg.enable_chunked_prefill and dcfg.prefill_chunk_size > 0
        chunk_size = dcfg.prefill_chunk_size if chunked else float('inf')

        while arrival_idx < total_requests or waiting_queue or prefilling or decoding:
            # 将所有已到达的请求加入等待队列
            while arrival_idx < total_requests and all_requests[arrival_idx].arrival_time <= current_time:
                waiting_queue.append(all_requests[arrival_idx])
                arrival_idx += 1

            # 选取新请求做 prefill (仅当没有正在 chunked prefill 的请求时)
            prefill_batch = []
            has_ongoing_prefill = any(
                prefill_progress.get(r.id, 0) < r.prefill_tokens
                for r in prefilling.values()
            )
            if waiting_queue and not has_ongoing_prefill:
                prefill_candidates = self._select_from_queue(
                    waiting_queue, max_prefill_per_iter, self._policy, current_time
                )
                for req in prefill_candidates:
                    waiting_queue.remove(req)
                    req.status = RequestStatus.PREFILLING
                    req.prefill_node_id = 0
                    req.prefill_start_time = current_time
                    ms.record_prefill_start(req.id, current_time, 0)
                    prefill_batch.append(req)
                    prefilling[req.id] = req
                    prefill_progress[req.id] = 0

            # 计算本 iteration 的 chunked prefill tokens
            chunked_prefill_tokens = 0
            completed_prefill = []
            for req in list(prefilling.values()):
                done = prefill_progress.get(req.id, 0)
                remaining = req.prefill_tokens - done
                this_chunk = min(remaining, int(chunk_size))
                chunked_prefill_tokens += this_chunk
                if done + this_chunk >= req.prefill_tokens:
                    completed_prefill.append(req)

            # Decode batch: 所有已 prefill 完成但 decode 未完成的请求
            decode_batch = list(decoding.values())

            # 如果没有 chunked prefill 也没有 decode
            if not chunked_prefill_tokens and not decode_batch:
                # 无活跃请求，跳到下一个到达时间
                if arrival_idx < total_requests:
                    current_time = all_requests[arrival_idx].arrival_time
                    continue
                else:
                    break

            # 执行混合 batch (chunked prefill tokens + decode)
            step_time = self._compute_chunked_mixed_batch_time(
                chunked_prefill_tokens, prefill_batch, decode_batch
            )
            step_end = current_time + step_time

            # 更新 prefill 进度
            for req in prefilling.values():
                done = prefill_progress.get(req.id, 0)
                remaining = req.prefill_tokens - done
                this_chunk = min(remaining, int(chunk_size))
                prefill_progress[req.id] = done + this_chunk

            # Prefill 完成的请求 → 进入 decode
            for req in completed_prefill:
                req.prefill_end_time = step_end
                ms.record_prefill_end(req.id, step_end)
                req.status = RequestStatus.DECODING
                req.decode_node_id = 0
                req.decode_start_time = step_end
                ms.record_decode_start(req.id, step_end, 0)
                ms.record_kv_cache_transfer_start(req.id, step_end)
                ms.record_kv_cache_transfer_end(req.id, step_end)  # colocated: 0 delay
                decoding[req.id] = req
                del prefilling[req.id]
                del prefill_progress[req.id]

            # Decode step: 每个活跃请求生成 1 个 token
            completed_ids = []
            for req_id, req in decoding.items():
                req.num_generated_tokens += 1
                if req.num_generated_tokens >= req.decode_tokens:
                    req.decode_end_time = step_end
                    req.status = RequestStatus.COMPLETED
                    ms.record_decode_end(req.id, step_end)
                    completed_ids.append(req_id)

            for req_id in completed_ids:
                del decoding[req_id]

            current_time = step_end

            logger.debug(
                f"Iteration: prefill={len(prefill_batch)}, decode={len(decode_batch)}, "
                f"step_time={step_time:.2f}ms, queue={len(waiting_queue)}"
            )

        ms.finalize()
        return ms


def create_colocated_simulator(
    num_gpus_per_node: int = 4,
    qps: float = 10.0,
    prefill_length: int = 512,
    decode_length: int = 128,
    prefill_batch_size: int = 4,
    decode_batch_size: int = 32,
    tp_size: int = 4,
    nvlink_bandwidth_gbps: float = 600.0,
    model_name: str = "Qwen3-30B-A3B",
    num_layers: int = 48,
    time_limit_s: float = 60.0,
    seed: int = 42,
    schedule_policy: str = "fcfs",
    length_distribution: str = "fixed",
    length_cv: float = 0.3,
    prefill_length_cv: float = -1.0,
    decode_length_cv: float = -1.0,
    workload: Optional[str] = None,
    profiling_dir: Optional[str] = None,
    predictor_type: str = "auto",
) -> ColocatedSimulator:
    """快速创建 Colocated (非分离) 模拟器。

    集群拓扑: 1 节点 num_gpus_per_node 张 A800, prefill + decode 共享。

    Args:
        length_cv: 通用变异系数 (normal/lognormal 分布)
        prefill_length_cv: prefill 专用 CV (-1 = 使用 length_cv)
        decode_length_cv: decode 专用 CV (-1 = 使用 length_cv)
        workload: Vidur workload 名称 ("chat-1m", "arxiv-4k", "bwb-4k")
        profiling_dir: profiling 数据目录路径（可选）
        predictor_type: 预测器类型 ("auto", "analytical", "profiled", "random_forest", "high_fidelity")
    """
    from distlmsim.types import RDMAProtocolType

    device = DeviceSKUConfig()
    model = ModelConfig(
        model_name=model_name,
        num_layers=num_layers,
        num_q_heads=32,
        num_kv_heads=4,
        embedding_dim=2048,
        num_experts=128,
        top_k_experts=8,
    )
    nvlink = NVLinkConfig(bandwidth_gbps=nvlink_bandwidth_gbps)
    rdma = RDMAConfig(protocol=RDMAProtocolType.ROCE_V2, bandwidth_gbps=200.0)
    network = NetworkTopologyConfig(nvlink=nvlink, rdma=rdma)

    metrics_config = MetricsConfig()
    ms = MetricsStore(metrics_config)

    ctx = SimContext(
        model_config=model,
        device_config=device,
        network_config=network,
        num_gpus_per_node=num_gpus_per_node,
        tp_size=tp_size,
        metrics_store=ms,
        profiling_dir=profiling_dir,
        predictor_type=predictor_type,
    )

    config = SimulationConfig(
        seed=seed,
        time_limit_s=time_limit_s,
        disaggregated=DisaggregatedConfig(
            enabled=False,
            num_prefill_nodes=0,
            num_decode_nodes=0,
            prefill_batch_size=prefill_batch_size,
            decode_batch_size=decode_batch_size,
        ),
        request=RequestGeneratorConfig(
            qps=qps,
            prefill_length=prefill_length,
            decode_length=decode_length,
            length_distribution=length_distribution,
            length_cv=length_cv,
            prefill_length_cv=prefill_length_cv,
            decode_length_cv=decode_length_cv,
        ),
        metrics=metrics_config,
    )

    # Apply Vidur workload if specified
    if workload:
        from distlmsim.workloads import apply_workload
        apply_workload(config, workload)

    return ColocatedSimulator(ctx, config, schedule_policy=schedule_policy)


# ─── 便捷工厂函数 ────────────────────────────────────────────────────────────


def create_disaggregated_simulator(
    num_gpus_per_node: int = 4,
    num_requests: int = 50,
    qps: float = 10.0,
    prefill_length: int = 512,
    decode_length: int = 128,
    prefill_batch_size: int = 8,
    decode_batch_size: int = 32,
    tp_size: int = 4,
    rdma_bandwidth_gbps: float = 200.0,
    nvlink_bandwidth_gbps: float = 600.0,
    model_name: str = "Qwen3-30B-A3B",
    num_layers: int = 48,
    time_limit_s: float = 60.0,
    seed: int = 42,
    prefill_schedule_policy: str = "fcfs",
    decode_schedule_policy: str = "fcfs",
    length_distribution: str = "fixed",
    length_cv: float = 0.3,
    prefill_length_cv: float = -1.0,
    decode_length_cv: float = -1.0,
    workload: Optional[str] = None,
    profiling_dir: Optional[str] = None,
    predictor_type: str = "auto",
) -> DisaggregatedSimulator:
    """快速创建存算分离模拟器。

    集群拓扑: 1 Prefill 节点 + 1 Decode 节点，每节点 num_gpus_per_node 张 A800。

    Args:
        length_cv: 通用变异系数 (normal/lognormal 分布)
        prefill_length_cv: prefill 专用 CV (-1 = 使用 length_cv)
        decode_length_cv: decode 专用 CV (-1 = 使用 length_cv)
        workload: Vidur workload 名称 ("chat-1m", "arxiv-4k", "bwb-4k")，
                  设置后覆盖 prefill_length/decode_length/CV 参数
        profiling_dir: profiling 数据目录路径（可选）
        predictor_type: 预测器类型 ("auto", "analytical", "profiled", "random_forest", "high_fidelity")
    """
    from distlmsim.types import RDMAProtocolType

    device = DeviceSKUConfig()
    model = ModelConfig(
        model_name=model_name,
        num_layers=num_layers,
        num_q_heads=32,
        num_kv_heads=4,
        embedding_dim=2048,
        num_experts=128,
        top_k_experts=8,
    )
    nvlink = NVLinkConfig(bandwidth_gbps=nvlink_bandwidth_gbps)
    rdma = RDMAConfig(protocol=RDMAProtocolType.ROCE_V2, bandwidth_gbps=rdma_bandwidth_gbps)
    network = NetworkTopologyConfig(nvlink=nvlink, rdma=rdma)

    metrics_config = MetricsConfig()
    ms = MetricsStore(metrics_config)

    ctx = SimContext(
        model_config=model,
        device_config=device,
        network_config=network,
        num_gpus_per_node=num_gpus_per_node,
        tp_size=tp_size,
        metrics_store=ms,
        profiling_dir=profiling_dir,
        predictor_type=predictor_type,
    )

    config = SimulationConfig(
        seed=seed,
        time_limit_s=time_limit_s,
        disaggregated=DisaggregatedConfig(
            enabled=True,
            num_prefill_nodes=1,
            num_decode_nodes=1,
            prefill_batch_size=prefill_batch_size,
            decode_batch_size=decode_batch_size,
        ),
        request=RequestGeneratorConfig(
            qps=qps,
            prefill_length=prefill_length,
            decode_length=decode_length,
            length_distribution=length_distribution,
            length_cv=length_cv,
            prefill_length_cv=prefill_length_cv,
            decode_length_cv=decode_length_cv,
        ),
        metrics=metrics_config,
    )

    # Apply Vidur workload if specified (overrides prefill/decode length and CV)
    if workload:
        from distlmsim.workloads import apply_workload
        apply_workload(config, workload)

    return DisaggregatedSimulator(
        ctx, config,
        prefill_schedule_policy=prefill_schedule_policy,
        decode_schedule_policy=decode_schedule_policy,
    )


def main():
    import sys
    logging.basicConfig(level=logging.INFO)

    if "--demo" in sys.argv:
        sim = create_disaggregated_simulator()
        metrics = sim.run()
        metrics.print_summary()
    else:
        print("用法:")
        print("  python main.py --demo    运行存算分离演示")
        print("  python examples/demo_disaggregated.py    运行详细演示脚本")


if __name__ == "__main__":
    main()
