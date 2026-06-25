"""专家并行 (Expert Parallelism) 模型

EP 将 MoE 模型的专家分布到多个 GPU/节点，
通过 All-to-All 通信进行 token 分发和收集。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from distlmsim.config import ModelConfig, DeviceSKUConfig


@dataclass
class ExpertPlacement:
    """专家放置方案。"""
    expert_id: int
    gpu_id: int
    node_id: int
    is_redundant: bool = False  # 是否为冗余副本


@dataclass
class ExpertRoutingResult:
    """专家路由结果。"""
    token_expert_assignment: np.ndarray   # shape: [num_tokens, top_k]
    expert_loads: np.ndarray             # shape: [num_experts] 每个专家的 token 数
    gpu_loads: np.ndarray                # shape: [num_gpus] 每个 GPU 的 token 数
    max_gpu_load: int                    # 最大 GPU 负载
    load_imbalance: float                # 负载不均衡度 (std/mean)
    per_expert_latencies: Optional[np.ndarray] = None  # shape: [num_experts] 每个专家的计算延迟 (ms)


class ExpertParallelModel:
    """专家并行模型。

    负责:
    1. 专家放置策略 (均匀分布 + 冗余 + 节点亲和性)
    2. Token 路由 (Top-K + 负载均衡)
    3. All-to-All 通信数据量计算
    4. EPLB 负载均衡调度
    """

    def __init__(
        self,
        model_config: ModelConfig,
        ep_size: int,
        num_gpus_per_node: int = 8,
        redundant_experts: int = 0,
        device_config: Optional[DeviceSKUConfig] = None,
    ):
        self._model = model_config
        self._ep_size = ep_size
        self._num_gpus_per_node = num_gpus_per_node
        self._redundant_experts = redundant_experts
        self._placement: List[ExpertPlacement] = []
        self._device = device_config or DeviceSKUConfig()

    def create_expert_placement(
        self,
        gpu_node_mapping: Dict[int, int],
    ) -> List[ExpertPlacement]:
        """创建专家放置方案。

        策略:
        1. 均匀分布专家到所有 GPU
        2. 冗余专家优先放在不同节点 (容错 + 负载均衡)
        3. 同节点内专家连续放置 (减少跨节点通信)

        Args:
            gpu_node_mapping: gpu_id -> node_id

        Returns:
            ExpertPlacement 列表
        """
        num_experts = self._model.num_experts
        placements = []

        # 基础放置: 均匀分配
        for expert_id in range(num_experts):
            gpu_id = expert_id % self._ep_size
            node_id = gpu_node_mapping.get(gpu_id, 0)
            placements.append(ExpertPlacement(
                expert_id=expert_id,
                gpu_id=gpu_id,
                node_id=node_id,
            ))

        # 冗余专家
        for i in range(self._redundant_experts):
            expert_id = i % num_experts
            gpu_id = (expert_id + self._ep_size // 2) % self._ep_size  # 放在不同位置
            node_id = gpu_node_mapping.get(gpu_id, 0)
            placements.append(ExpertPlacement(
                expert_id=expert_id,
                gpu_id=gpu_id,
                node_id=node_id,
                is_redundant=True,
            ))

        self._placement = placements
        return placements

    def route_tokens(
        self,
        expert_distribution: np.ndarray,
        top_k: int,
    ) -> ExpertRoutingResult:
        """执行 token 路由。

        根据专家分布矩阵，为每个 token 选择 top_k 个专家。

        Args:
            expert_distribution: shape [num_tokens, num_experts]，路由权重
            top_k: 每 token 选择的专家数

        Returns:
            ExpertRoutingResult
        """
        num_tokens = expert_distribution.shape[0]

        # Top-K 选择
        top_k_indices = np.argsort(expert_distribution, axis=1)[:, -top_k:]

        # 计算专家负载
        expert_loads = np.zeros(self._model.num_experts, dtype=np.int64)
        for token_idx in range(num_tokens):
            for k in range(top_k):
                expert_id = top_k_indices[token_idx, k]
                expert_loads[expert_id] += 1

        # Per-expert 计算延迟 (Roofline 模型)
        per_expert_latencies = self._compute_per_expert_latencies(expert_loads)

        # 计算 GPU 负载 (基于专家放置)
        gpu_loads = np.zeros(self._ep_size, dtype=np.int64)
        gpu_latencies = np.zeros(self._ep_size, dtype=np.float64)
        for placement in self._placement:
            if placement.gpu_id < self._ep_size:
                gpu_loads[placement.gpu_id] += expert_loads[placement.expert_id]
                gpu_latencies[placement.gpu_id] += per_expert_latencies[placement.expert_id]

        max_gpu_load = int(np.max(gpu_loads))
        mean_load = np.mean(gpu_loads[gpu_loads > 0])
        std_load = np.std(gpu_loads[gpu_loads > 0])
        imbalance = float(std_load / mean_load) if mean_load > 0 else 0.0

        return ExpertRoutingResult(
            token_expert_assignment=top_k_indices,
            expert_loads=expert_loads,
            gpu_loads=gpu_loads,
            max_gpu_load=max_gpu_load,
            load_imbalance=imbalance,
            per_expert_latencies=per_expert_latencies,
        )

    def _compute_per_expert_latencies(self, expert_loads: np.ndarray) -> np.ndarray:
        """基于 Roofline 模型计算每个专家的计算延迟 (ms)。

        Expert MLP: gate_up_proj + SiLU*up + down_proj
        FLOPs = 2 * tokens * h * (2*expert_dim + expert_dim) = 6 * tokens * h * expert_dim
        """
        h = self._model.embedding_dim
        expert_dim = (self._model.expert_intermediate_dim
                      or int(h * 0.375))  # 默认 Qwen3-30B-A3B: 768
        peak_flops = self._device.fp16_tflops * 1e12  # TFLOPS → FLOPS
        mem_bw = self._device.memory_bandwidth_gbps * 1e9  # GB/s → B/s
        eta_c, eta_m = 0.85, 0.90  # 效率因子

        latencies = np.zeros(self._model.num_experts, dtype=np.float64)
        for i in range(self._model.num_experts):
            tokens_i = expert_loads[i]
            if tokens_i == 0:
                continue
            # gate_up: [tokens, h] → [tokens, 2*expert_dim]
            gate_up_flops = 2 * tokens_i * h * 2 * expert_dim
            gate_up_mem = (tokens_i * h + 2 * tokens_i * expert_dim + h * 2 * expert_dim) * 2
            # activation: SiLU + element-wise mul
            act_flops = 2 * tokens_i * expert_dim
            act_mem = 3 * tokens_i * expert_dim * 2
            # down: [tokens, expert_dim] → [tokens, h]
            down_flops = 2 * tokens_i * expert_dim * h
            down_mem = (tokens_i * expert_dim + tokens_i * h + expert_dim * h) * 2

            total_flops = gate_up_flops + act_flops + down_flops
            total_mem = gate_up_mem + act_mem + down_mem

            compute_ms = total_flops / (peak_flops * eta_c) * 1e3
            memory_ms = total_mem / (mem_bw * eta_m) * 1e3
            latencies[i] = max(compute_ms, memory_ms)

        return latencies

    def get_alltoall_data_size(
        self,
        num_tokens: int,
        top_k: int,
    ) -> int:
        """计算 All-to-All 通信的数据量 (bytes)。

        Dispatch: 每 token 发送 top_k 份到目标专家
        Combine: 收集结果

        Args:
            num_tokens: batch 中的 token 数
            top_k: Top-K 路由

        Returns:
            每 GPU 的 All-to-All 数据量 (bytes)
        """
        if self._ep_size <= 1:
            return 0

        # 每 token: hidden_dim * 2 bytes (float16) * top_k
        per_token_bytes = self._model.embedding_dim * 2 * top_k
        # 均匀分布到 ep_size 个 GPU
        return int(num_tokens * per_token_bytes / self._ep_size)


# ─── MoE 专家负载均衡调度器 (迁移自 TRADIOS) ─────────────────────────────────


@dataclass
class MoELoadResult:
    """MoE 负载均衡结果。"""
    max_load: int             # 最大 GPU 负载
    avg_load: int             # 平均 GPU 负载
    deviation: float          # 负载偏差或迁移数 (取决于策略)
    num_migrations: int = 0   # 迁移次数 (Realistic EPLB / OmniPlacement)


class BaseMoEScheduler:
    """MoE 专家负载均衡基类。"""

    def __init__(self, num_experts: int, expert_parallel_size: int, top_k_experts: int):
        self.num_experts = num_experts
        self.expert_parallel_size = expert_parallel_size
        self.top_k_experts = top_k_experts
        self.experts_per_device = max(1, num_experts // expert_parallel_size)

    def _get_raw_expert_counts(self, layer_expert_demand: np.ndarray) -> np.ndarray:
        """从逐层专家需求计算平均专家负载。

        Args:
            layer_expert_demand: shape [num_layers, num_experts]

        Returns:
            shape (num_experts,) 的平均专家调用次数
        """
        return np.mean(layer_expert_demand, axis=0)

    def _calculate_device_loads(self, expert_counts: np.ndarray) -> np.ndarray:
        """专家负载映射到设备负载。"""
        device_loads = np.zeros(self.expert_parallel_size)
        for exp_id in range(self.num_experts):
            device_id = exp_id // self.experts_per_device
            if device_id < self.expert_parallel_size:
                device_loads[device_id] += expert_counts[exp_id]
        return device_loads

    def compute_load_distribution(
        self, layer_expert_demand: np.ndarray
    ) -> MoELoadResult:
        """计算负载分布。"""
        raise NotImplementedError


class DefaultRoutingScheduler(BaseMoEScheduler):
    """标准 Top-K 路由，无负载均衡。"""

    def compute_load_distribution(
        self, layer_expert_demand: np.ndarray
    ) -> MoELoadResult:
        expert_counts = self._get_raw_expert_counts(layer_expert_demand)
        device_loads = self._calculate_device_loads(expert_counts)
        max_load = int(np.ceil(np.max(device_loads)))
        avg_load = int(np.ceil(np.mean(device_loads)))
        deviation = float(max_load - avg_load)
        return MoELoadResult(max_load=max_load, avg_load=avg_load, deviation=deviation)


class EPLBScheduler(BaseMoEScheduler):
    """EPLB 简化版: capacity_factor=1.1 截断。"""

    def compute_load_distribution(
        self, layer_expert_demand: np.ndarray
    ) -> MoELoadResult:
        expert_counts = self._get_raw_expert_counts(layer_expert_demand)
        device_loads = self._calculate_device_loads(expert_counts)
        raw_max = int(np.ceil(np.max(device_loads)))
        raw_avg = int(np.ceil(np.mean(device_loads)))
        raw_deviation = float(raw_max - raw_avg)

        # 容量因子截断
        capacity_factor = 1.1
        max_capacity = int(np.ceil(raw_avg * capacity_factor))
        balanced_max = min(raw_max, max_capacity)

        return MoELoadResult(
            max_load=balanced_max, avg_load=raw_avg, deviation=raw_deviation
        )


class RealisticEPLBScheduler(BaseMoEScheduler):
    """真实 EPLB: 注水路由 + 冗余专家 + 周期重平衡 + 节点亲和性。

    从 TRADIOS 迁移。
    """

    def __init__(
        self,
        num_experts: int,
        expert_parallel_size: int,
        top_k_experts: int,
        num_nodes: int = 1,
        gpus_per_node: int = 8,
        redundant_experts: int = 16,
        rebalance_interval: int = 10,
    ):
        super().__init__(num_experts, expert_parallel_size, top_k_experts)
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.num_gpus = num_nodes * gpus_per_node
        self.redundant_experts = redundant_experts
        self.rebalance_interval = rebalance_interval

        self.total_slots = num_experts + redundant_experts
        self.slots_per_gpu = max(1, self.total_slots // self.num_gpus)
        self.batch_count = 0
        self.historical_load: Optional[np.ndarray] = None  # [num_layers, num_experts]
        # placement[layer_id][expert_id] = [gpu_ids]
        self.placement: List[Dict[int, List[int]]] = []

    def _init_placement_if_needed(self, num_layers: int) -> None:
        """冷启动: 初始专家放置。"""
        if self.placement:
            return

        self.placement = []
        for _ in range(num_layers):
            layer_placement: Dict[int, List[int]] = {}
            gpu_slots = [self.slots_per_gpu] * self.num_gpus

            # 基础放置: 每个专家 1 个副本
            for exp_id in range(self.num_experts):
                # 找剩余 slot 最多的 GPU
                best_gpu = int(np.argmax(gpu_slots))
                layer_placement[exp_id] = [best_gpu]
                gpu_slots[best_gpu] -= 1

            # 冗余放置
            for r in range(self.redundant_experts):
                exp_id = r % self.num_experts
                existing_gpus = set(layer_placement.get(exp_id, []))
                # 找不在现有副本中且 slot > 0 的 GPU
                candidates = [
                    g for g in range(self.num_gpus)
                    if g not in existing_gpus and gpu_slots[g] > 0
                ]
                if candidates:
                    best_gpu = max(candidates, key=lambda g: gpu_slots[g])
                    layer_placement[exp_id].append(best_gpu)
                    gpu_slots[best_gpu] -= 1

            self.placement.append(layer_placement)

        self.historical_load = np.zeros((num_layers, self.num_experts))

    def _waterfill_route(
        self, total_demand: float, current_loads: np.ndarray
    ) -> np.ndarray:
        """注水算法: 将 token 需求分配到多个副本 GPU。

        从最低水位开始填充，逐层齐平。
        """
        loads = current_loads.copy()
        added = np.zeros_like(loads)
        demand_left = total_demand

        while demand_left > 0.01:
            min_load = np.min(loads)
            min_indices = np.where(loads <= min_load + 0.01)[0]
            higher_loads = loads[loads > min_load + 0.01]

            if len(higher_loads) == 0:
                # 所有 GPU 已齐平: 平分剩余需求
                per_gpu = demand_left / len(min_indices)
                added[min_indices] += per_gpu
                break

            next_min = np.min(higher_loads)
            diff = next_min - min_load
            capacity = diff * len(min_indices)

            if demand_left <= capacity:
                per_gpu = demand_left / len(min_indices)
                added[min_indices] += per_gpu
                break
            else:
                added[min_indices] += diff
                loads[min_indices] += diff
                demand_left -= capacity

        return added

    def _route_batch(self, layer_expert_counts: np.ndarray) -> np.ndarray:
        """批量路由: 对每层每个专家的 token 需求进行注水分配。"""
        num_layers = layer_expert_counts.shape[0]
        batch_gpu_loads = np.zeros((num_layers, self.num_gpus))

        for l in range(num_layers):
            # 按需求从大到小排序 (Greedy 装箱)
            sorted_experts = np.argsort(-layer_expert_counts[l])
            for e in sorted_experts:
                demand = layer_expert_counts[l, e]
                if demand <= 0:
                    continue
                target_gpus = self.placement[l].get(e, [])
                if not target_gpus:
                    continue

                if len(target_gpus) == 1:
                    batch_gpu_loads[l, target_gpus[0]] += demand
                else:
                    current = batch_gpu_loads[l, target_gpus]
                    routed = self._waterfill_route(demand, current)
                    for i, gpu_id in enumerate(target_gpus):
                        batch_gpu_loads[l, gpu_id] += routed[i]

        return batch_gpu_loads

    def _rebalance(self) -> int:
        """周期重平衡: 根据历史负载重新分配冗余副本。"""
        num_layers = self.historical_load.shape[0]
        total_migrations = 0

        for l in range(num_layers):
            # 步骤 1: 基于历史负载分配副本数
            replicas = np.ones(self.num_experts, dtype=int)
            temp_load = self.historical_load[l].copy()
            for _ in range(self.redundant_experts):
                hot_e = int(np.argmax(temp_load))
                replicas[hot_e] += 1
                temp_load[hot_e] = self.historical_load[l, hot_e] / replicas[hot_e]

            # 步骤 2: 带节点亲和性的放置
            sorted_experts = np.argsort(-replicas)
            new_placement: Dict[int, List[int]] = {}
            gpu_slots = [self.slots_per_gpu] * self.num_gpus

            for e in sorted_experts:
                num_replicas = replicas[e]
                placed_gpus = []
                for _ in range(num_replicas):
                    # 优先同节点
                    best_node = -1
                    best_gpu = -1
                    best_slots = -1

                    for node_id in range(self.num_nodes):
                        node_start = node_id * self.gpus_per_node
                        node_end = node_start + self.gpus_per_node
                        for gpu_id in range(node_start, node_end):
                            if gpu_id in placed_gpus:
                                continue
                            if gpu_slots[gpu_id] > best_slots:
                                best_slots = gpu_slots[gpu_id]
                                best_gpu = gpu_id
                                best_node = node_id
                        if best_gpu >= 0:
                            break

                    if best_gpu < 0:
                        # 无可用 slot，找任意可用 GPU
                        candidates = [g for g in range(self.num_gpus) if gpu_slots[g] > 0]
                        if candidates:
                            best_gpu = max(candidates, key=lambda g: gpu_slots[g])

                    if best_gpu >= 0:
                        placed_gpus.append(best_gpu)
                        gpu_slots[best_gpu] -= 1

                new_placement[e] = placed_gpus

            # 步骤 3: 计算迁移数
            old_placement = self.placement[l]
            for e in range(self.num_experts):
                old_gpus = set(old_placement.get(e, []))
                new_gpus = set(new_placement.get(e, []))
                total_migrations += len(new_gpus - old_gpus)

            self.placement[l] = new_placement

        # 重置历史负载
        self.historical_load[:] = 0
        return total_migrations

    def compute_load_distribution(
        self, layer_expert_demand: np.ndarray
    ) -> MoELoadResult:
        num_layers = layer_expert_demand.shape[0]
        self._init_placement_if_needed(num_layers)

        # 累加历史负载
        self.historical_load += layer_expert_demand
        self.batch_count += 1

        # 周期重平衡
        migrations = 0
        if self.batch_count % self.rebalance_interval == 0:
            migrations = self._rebalance()

        # 执行路由
        gpu_loads = self._route_batch(layer_expert_demand)

        # 跨层统计
        max_loads_per_layer = np.max(gpu_loads, axis=1)
        avg_loads_per_layer = np.mean(gpu_loads, axis=1)
        final_max = int(np.ceil(np.mean(max_loads_per_layer)))
        final_avg = int(np.ceil(np.mean(avg_loads_per_layer)))

        return MoELoadResult(
            max_load=final_max,
            avg_load=final_avg,
            deviation=float(migrations),
            num_migrations=migrations,
        )


class OmniPlacementScheduler(BaseMoEScheduler):
    """OmniPlacement: 贪心交换优化专家放置。

    从 TRADIOS 迁移。budget_N 控制每步最大迁移次数。
    """

    def __init__(
        self,
        num_experts: int,
        expert_parallel_size: int,
        top_k_experts: int,
        budget_N: int = 4,
    ):
        super().__init__(num_experts, expert_parallel_size, top_k_experts)
        self.budget_N = budget_N
        # placement_P[l, e] = device_id
        self.placement_P: Optional[np.ndarray] = None

    def _init_placement_if_needed(self, num_layers: int) -> None:
        if self.placement_P is not None:
            return
        self.placement_P = np.zeros((num_layers, self.num_experts), dtype=int)
        for l in range(num_layers):
            for e in range(self.num_experts):
                self.placement_P[l, e] = e // self.experts_per_device

    def _greedy_swap_optimization(
        self,
        layer_activations: np.ndarray,
        current_placement: np.ndarray,
        budget: int,
    ) -> Tuple[np.ndarray, int]:
        """贪心交换优化: 在最大/最小负载设备间交换专家。"""
        placement = current_placement.copy()
        used_budget = 0

        while used_budget + 2 <= budget:
            # 计算当前设备负载
            device_loads = np.zeros(self.expert_parallel_size)
            for e in range(self.num_experts):
                device_loads[placement[e]] += layer_activations[e]

            max_device = int(np.argmax(device_loads))
            min_device = int(np.argmin(device_loads))

            if max_device == min_device:
                break

            current_max_load = device_loads[max_device]
            best_swap = None
            best_reduction = 0

            # 尝试交换 max_device 和 min_device 上的专家
            experts_on_max = [e for e in range(self.num_experts) if placement[e] == max_device]
            experts_on_min = [e for e in range(self.num_experts) if placement[e] == min_device]

            for e_max in experts_on_max:
                for e_min in experts_on_min:
                    # 模拟交换
                    new_max_load = device_loads[max_device] - layer_activations[e_max] + layer_activations[e_min]
                    new_min_load = device_loads[min_device] - layer_activations[e_min] + layer_activations[e_max]
                    simulated_max = max(new_max_load, new_min_load)

                    if simulated_max < current_max_load:
                        reduction = current_max_load - simulated_max
                        if reduction > best_reduction:
                            best_reduction = reduction
                            best_swap = (e_max, e_min)

            if best_swap is None:
                break

            e_max, e_min = best_swap
            placement[e_max] = min_device
            placement[e_min] = max_device
            used_budget += 2

        return placement, used_budget

    def compute_load_distribution(
        self, layer_expert_demand: np.ndarray
    ) -> MoELoadResult:
        num_layers = layer_expert_demand.shape[0]
        self._init_placement_if_needed(num_layers)

        n_remain = self.budget_N
        total_migrations = 0
        all_max_loads = []
        all_avg_loads = []

        for l in range(num_layers):
            layer_acts = layer_expert_demand[l]

            if n_remain > 0:
                self.placement_P[l], used = self._greedy_swap_optimization(
                    layer_acts, self.placement_P[l], n_remain
                )
                n_remain -= used
                total_migrations += used

            # 计算设备负载
            device_loads = np.zeros(self.expert_parallel_size)
            for e in range(self.num_experts):
                device_loads[self.placement_P[l, e]] += layer_acts[e]

            all_max_loads.append(np.max(device_loads))
            all_avg_loads.append(np.mean(device_loads))

        final_max = int(np.ceil(np.mean(all_max_loads)))
        final_avg = int(np.ceil(np.mean(all_avg_loads)))

        return MoELoadResult(
            max_load=final_max,
            avg_load=final_avg,
            deviation=float(total_migrations),
            num_migrations=total_migrations,
        )
