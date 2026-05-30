"""DistLMSim 入口点：分布式推理模拟器

提供两种运行方式：
1. DisaggregatedSimulator: 存算分离模式的完整模拟 (推荐入门)
2. DistributedInferenceSimulator: 通用分布式模拟 (事件驱动)
"""

from __future__ import annotations

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
from distlmsim.entities import Request, RequestStatus, ExecutionTime
from distlmsim.execution.execution_time_predictor import AnalyticalPredictor
from distlmsim.topology.nvlink_model import NVLinkModel
from distlmsim.topology.rdma_model import RDMAModel
from distlmsim.metrics.metrics_store import MetricsStore
from distlmsim.scheduling.advanced_schedulers import AdvancedSchedulers

logger = logging.getLogger(__name__)


# ─── 模拟上下文 ──────────────────────────────────────────────────────────────


@dataclass
class SimContext:
    """模拟运行时上下文，持有所有共享状态。"""
    model_config: ModelConfig
    device_config: DeviceSKUConfig
    network_config: NetworkTopologyConfig
    # 通信模型
    nvlink_model: NVLinkModel = field(default=None)
    rdma_model: RDMAModel = field(default=None)
    # 执行时间预测
    time_predictor: AnalyticalPredictor = field(default=None)
    # 请求池
    requests: Dict[int, Request] = field(default_factory=dict)
    # 指标
    metrics_store: MetricsStore = field(default=None)
    # 集群参数
    num_gpus_per_node: int = 4
    prefill_node_id: int = 0
    decode_node_id: int = 1
    tp_size: int = 4

    def __post_init__(self):
        if self.nvlink_model is None:
            self.nvlink_model = NVLinkModel(
                self.network_config.nvlink, self.num_gpus_per_node
            )
        if self.rdma_model is None:
            self.rdma_model = RDMAModel(self.network_config.rdma)
        if self.time_predictor is None:
            self.time_predictor = AnalyticalPredictor(
                self.model_config, self.device_config
            )


# ─── 存算分离模拟器 ────────────────────────────────────────────────────────────


class DisaggregatedSimulator:
    """存算分离 (Disaggregated Prefill/Decode) 模拟器。

    模拟 1 个 Prefill 节点 + 1 个 Decode 节点的推理服务：
    1. 请求到达 → Prefill 节点执行 prefill (批量)
    2. Prefill 完成 → KV Cache 通过 RDMA 传输到 Decode 节点
    3. Decode 节点执行 decode (逐 token 迭代)
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

    def _create_request(self, arrival_time: float) -> Request:
        """创建一个合成请求。"""
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

        时间 = sum(每层计算时间 + 每层 TP 通信时间) * num_layers
        """
        total_tokens = sum(r.prefill_tokens for r in batch_requests)
        batch_size = len(batch_requests)
        model = self.ctx.model_config

        exec_time = self.ctx.time_predictor.get_execution_time(
            num_tokens=total_tokens,
            batch_size=batch_size,
            kv_cache_size=0,
            is_prefill=True,
        )

        per_layer_time = exec_time.total_time
        tp_comm = self._compute_tp_allreduce_time(total_tokens)
        total_per_layer = per_layer_time + tp_comm

        return total_per_layer * model.num_layers

    def _compute_decode_step_time(self, batch_requests: List[Request]) -> float:
        """计算一步 decode 的执行时间 (ms)。

        Decode 每步生成 1 个 token per request。
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

        per_layer_time = exec_time.total_time
        tp_comm = self._compute_tp_allreduce_time(num_tokens)
        total_per_layer = per_layer_time + tp_comm

        return total_per_layer * model.num_layers

    def _compute_kv_transfer_time(self, kv_cache_bytes: int) -> float:
        """计算 KV Cache RDMA 传输时间 (ms)。"""
        return self.ctx.rdma_model.get_transfer_time(kv_cache_bytes)

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
        """运行完整的存算分离模拟 (实时队列调度)。

        调度逻辑:
        1. 所有请求按到达时间排序
        2. 维护 prefill 等待队列: 请求到达后入队，prefill 节点空闲时按策略选取
        3. Prefill 完成后 KV Cache 通过 RDMA 传输，进入 decode 等待队列
        4. Decode 节点空闲时按策略从 decode 队列中选取请求

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

        prefill_bs = self.config.disaggregated.prefill_batch_size
        decode_bs = self.config.disaggregated.decode_batch_size

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

            # ─── KV Cache 传输 (RDMA) ─────────────────────────────────────
            for req in batch:
                kv_size = self._compute_kv_cache_size(req)
                req.kv_cache_size_bytes = kv_size
                transfer_time = self._compute_kv_transfer_time(kv_size)
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

        # ─── Decode 阶段: 实时队列调度 ────────────────────────────────────
        # 按 KV Cache 就绪时间排序所有请求，模拟到达 decode 队列
        decode_candidates = sorted(all_requests, key=lambda r: kv_ready_time[r.id])
        decode_arrival_idx = 0
        decode_free_time = 0.0
        decode_waiting: List[Request] = []

        while decode_arrival_idx < total_requests or decode_waiting:
            # 确定当前时间
            if decode_waiting:
                current_time = decode_free_time
            else:
                current_time = max(
                    decode_free_time,
                    kv_ready_time[decode_candidates[decode_arrival_idx].id]
                )

            # 将所有 KV Cache 已就绪的请求加入 decode 等待队列
            while (decode_arrival_idx < total_requests
                   and kv_ready_time[decode_candidates[decode_arrival_idx].id] <= current_time):
                decode_waiting.append(decode_candidates[decode_arrival_idx])
                decode_arrival_idx += 1

            if not decode_waiting:
                continue

            # 按调度策略从队列中选取
            decode_batch = self._select_from_queue(
                decode_waiting, decode_bs, self._decode_policy,
                kv_ready_time=kv_ready_time,
                current_time=current_time
            )
            for req in decode_batch:
                decode_waiting.remove(req)
                req.status = RequestStatus.DECODING
                req.decode_node_id = self.ctx.decode_node_id

            # ─── 执行 Decode ──────────────────────────────────────────────
            batch_ready = max(kv_ready_time[r.id] for r in decode_batch)
            current_time = max(decode_free_time, batch_ready)

            max_decode_len = max(r.decode_tokens for r in decode_batch)
            for step in range(max_decode_len):
                active = [r for r in decode_batch if r.num_generated_tokens < r.decode_tokens]
                if not active:
                    break

                step_time = self._compute_decode_step_time(active)
                step_end = current_time + step_time

                for req in active:
                    req.num_generated_tokens += 1
                    if req.decode_start_time is None:
                        req.decode_start_time = current_time
                        ms.record_decode_start(req.id, current_time, self.ctx.decode_node_id)
                    if req.num_generated_tokens >= req.decode_tokens:
                        req.decode_end_time = step_end
                        req.status = RequestStatus.COMPLETED
                        ms.record_decode_end(req.id, step_end)

                current_time = step_end

            decode_free_time = current_time

            logger.debug(
                f"Decode batch: size={len(decode_batch)}, "
                f"queue_depth={len(decode_waiting)}"
            )

        ms.finalize()
        return ms


# ─── 通用分布式模拟器 (事件驱动) ───────────────────────────────────────────────


class DistributedInferenceSimulator:
    """通用分布式推理模拟器 (事件驱动框架)。

    基于 heapq 的离散事件循环。
    """

    def __init__(self, config: SimulationConfig):
        self._config = config
        logger.info("DistributedInferenceSimulator 初始化")

    def run(self) -> MetricsStore:
        """运行模拟。"""
        raise NotImplementedError(
            "通用事件驱动模拟器尚未完整实现。"
            "请使用 DisaggregatedSimulator 进行存算分离模拟。"
        )


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
) -> DisaggregatedSimulator:
    """快速创建存算分离模拟器。

    集群拓扑: 1 Prefill 节点 + 1 Decode 节点，每节点 num_gpus_per_node 张 A800。
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
        ),
        metrics=metrics_config,
    )

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
