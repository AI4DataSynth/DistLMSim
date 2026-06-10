"""通信-计算重叠处理器

当计算与通信并行执行时，由于资源竞争，双方延迟都会增加。
从 Charon 项目移植，适配 DistLMSim 的推理场景。

支持两种重叠模型:

1. Ratio-Based (基于比例): 对重叠部分应用减速因子
   - compute_slowdown: 计算变慢倍数 (默认 1.15)
   - comm_slowdown: 通信变慢倍数 (默认 1.20)
   - comm_comm_slowdown: 通信-通信重叠变慢倍数 (默认 1.30)

2. Bandwidth-Aware (带宽感知): 根据链路带宽竞争计算减速
   - slowdown = concurrent_count * congestion_penalty
   - 适用于多路通信共享链路的场景

推理场景中的重叠:
- TP all-reduce 与下一层计算重叠
- PP send/recv 与当前 stage 计算重叠
- EP all-to-all 与 expert 计算重叠
- KV Cache 传输与 decode 计算重叠 (存算分离)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from distlmsim.entities import ExecutionTime


@dataclass
class OverlapConfig:
    """重叠模型配置"""
    # 基于比例的减速因子
    compute_slowdown: float = 1.15    # 计算在重叠时变慢 15%
    comm_slowdown: float = 1.20       # 通信在重叠时变慢 20%
    comm_comm_slowdown: float = 1.30  # 通信-通信重叠变慢 30%
    # 带宽感知参数
    congestion_alpha: float = 0.1     # 拥塞惩罚系数
    # 默认重叠比例
    default_overlap_ratio: float = 0.9  # TP all-reduce 与计算的重叠比例


@dataclass
class OverlapPair:
    """一对重叠操作"""
    op_a_name: str = ""
    op_a_latency_ms: float = 0.0
    op_a_is_comm: bool = False
    op_b_name: str = ""
    op_b_latency_ms: float = 0.0
    op_b_is_comm: bool = False
    overlap_ratio: float = 0.0         # 重叠比例 (0-1)
    overlap_type: str = "compute_comm"  # compute_comm / comm_comm
    adjusted_a_ms: float = 0.0
    adjusted_b_ms: float = 0.0

    @property
    def overlap_duration_ms(self) -> float:
        """重叠持续时间 = min(op_a, op_b) * ratio"""
        return min(self.op_a_latency_ms, self.op_b_latency_ms) * self.overlap_ratio

    @property
    def time_saved_ms(self) -> float:
        """节省的墙钟时间 = 重叠时间 - 额外开销"""
        overlap = self.overlap_duration_ms
        if overlap <= 0:
            return 0.0
        # 不重叠时: max(a, b); 重叠时: max(adjusted_a, adjusted_b)
        no_overlap = max(self.op_a_latency_ms, self.op_b_latency_ms)
        with_overlap = max(self.adjusted_a_ms, self.adjusted_b_ms)
        return max(0.0, no_overlap - with_overlap)


@dataclass
class OverlapResult:
    """重叠处理结果"""
    pairs: List[OverlapPair] = field(default_factory=list)
    num_compute_comm: int = 0
    num_comm_comm: int = 0
    total_overlap_ms: float = 0.0
    total_time_saved_ms: float = 0.0
    avg_compute_slowdown: float = 1.0
    avg_comm_slowdown: float = 1.0

    def summary(self) -> Dict:
        return {
            "num_compute_comm_pairs": self.num_compute_comm,
            "num_comm_comm_pairs": self.num_comm_comm,
            "total_overlap_ms": round(self.total_overlap_ms, 4),
            "total_time_saved_ms": round(self.total_time_saved_ms, 4),
            "avg_compute_slowdown": round(self.avg_compute_slowdown, 4),
            "avg_comm_slowdown": round(self.avg_comm_slowdown, 4),
        }


class OverlapProcessor:
    """通信-计算重叠处理器

    处理推理过程中的计算-通信重叠，调整执行时间。

    使用方法:
        processor = OverlapProcessor()
        # 方式1: 处理单个重叠对
        pair = processor.make_compute_comm_pair(compute_ms=5.0, comm_ms=2.0, ratio=0.9)
        adjusted = processor.apply_ratio_slowdown(pair)
        # 方式2: 处理 ExecutionTime
        adjusted_et = processor.adjust_execution_time(et)
        # 方式3: 批量处理
        result = processor.process_pairs(pairs)
    """

    def __init__(self, config: Optional[OverlapConfig] = None):
        self._config = config or OverlapConfig()

    # --------------------------------------------------------
    # 创建重叠对
    # --------------------------------------------------------

    def make_compute_comm_pair(
        self,
        compute_ms: float,
        comm_ms: float,
        overlap_ratio: Optional[float] = None,
        compute_name: str = "compute",
        comm_name: str = "communication",
    ) -> OverlapPair:
        """创建计算-通信重叠对。"""
        ratio = overlap_ratio if overlap_ratio is not None else self._config.default_overlap_ratio
        return OverlapPair(
            op_a_name=compute_name,
            op_a_latency_ms=compute_ms,
            op_a_is_comm=False,
            op_b_name=comm_name,
            op_b_latency_ms=comm_ms,
            op_b_is_comm=True,
            overlap_ratio=ratio,
            overlap_type="compute_comm",
        )

    def make_comm_comm_pair(
        self,
        comm_a_ms: float,
        comm_b_ms: float,
        overlap_ratio: float = 0.5,
        name_a: str = "comm_a",
        name_b: str = "comm_b",
    ) -> OverlapPair:
        """创建通信-通信重叠对。"""
        return OverlapPair(
            op_a_name=name_a,
            op_a_latency_ms=comm_a_ms,
            op_a_is_comm=True,
            op_b_name=name_b,
            op_b_latency_ms=comm_b_ms,
            op_b_is_comm=True,
            overlap_ratio=overlap_ratio,
            overlap_type="comm_comm",
        )

    # --------------------------------------------------------
    # 减速模型
    # --------------------------------------------------------

    def apply_ratio_slowdown(self, pair: OverlapPair) -> OverlapPair:
        """基于比例的减速模型。

        重叠部分乘以减速因子，非重叠部分不变。
        adjusted = overlap_part * slowdown + non_overlap_part
        """
        r = pair.overlap_ratio
        if r <= 0:
            pair.adjusted_a_ms = pair.op_a_latency_ms
            pair.adjusted_b_ms = pair.op_b_latency_ms
            return pair

        if pair.overlap_type == "compute_comm":
            if not pair.op_a_is_comm:
                compute_lat = pair.op_a_latency_ms
                comm_lat = pair.op_b_latency_ms
            else:
                compute_lat = pair.op_b_latency_ms
                comm_lat = pair.op_a_latency_ms

            adj_compute = compute_lat * r * self._config.compute_slowdown + compute_lat * (1 - r)
            adj_comm = comm_lat * r * self._config.comm_slowdown + comm_lat * (1 - r)

            if not pair.op_a_is_comm:
                pair.adjusted_a_ms = adj_compute
                pair.adjusted_b_ms = adj_comm
            else:
                pair.adjusted_a_ms = adj_comm
                pair.adjusted_b_ms = adj_compute

        elif pair.overlap_type == "comm_comm":
            a = pair.op_a_latency_ms
            b = pair.op_b_latency_ms
            pair.adjusted_a_ms = a * r * self._config.comm_comm_slowdown + a * (1 - r)
            pair.adjusted_b_ms = b * r * self._config.comm_comm_slowdown + b * (1 - r)

        return pair

    def apply_bandwidth_aware_slowdown(
        self,
        pair: OverlapPair,
        concurrent_count: int = 2,
        link_capacity: int = 8,
    ) -> OverlapPair:
        """带宽感知减速模型 (仅用于 comm-comm 重叠)。

        slowdown = concurrent_count * (1 + alpha * max(0, concurrent - capacity))
        """
        base_slowdown = float(concurrent_count)
        congestion_penalty = 1.0
        if concurrent_count > link_capacity:
            congestion_penalty = 1.0 + self._config.congestion_alpha * (
                concurrent_count - link_capacity
            )
        final_slowdown = base_slowdown * congestion_penalty

        r = pair.overlap_ratio
        a = pair.op_a_latency_ms
        b = pair.op_b_latency_ms
        pair.adjusted_a_ms = a * r * final_slowdown + a * (1 - r)
        pair.adjusted_b_ms = b * r * final_slowdown + b * (1 - r)
        return pair

    # --------------------------------------------------------
    # ExecutionTime 调整
    # --------------------------------------------------------

    def adjust_execution_time(
        self,
        et: ExecutionTime,
        tp_overlap_ratio: Optional[float] = None,
        pp_overlap_ratio: float = 0.0,
        ep_overlap_ratio: float = 0.0,
    ) -> ExecutionTime:
        """调整 ExecutionTime 中的通信时间以反映重叠。

        在推理中:
        - TP all-reduce 与下一层计算重叠 (高重叠比)
        - PP send/recv 可能与计算部分重叠
        - EP all-to-all 与 expert 计算重叠

        Args:
            et: 原始执行时间
            tp_overlap_ratio: TP 通信与计算的重叠比例
            pp_overlap_ratio: PP 通信与计算的重叠比例
            ep_overlap_ratio: EP 通信与计算的重叠比例

        Returns:
            调整后的 ExecutionTime (通信时间因重叠而增加)
        """
        tp_ratio = tp_overlap_ratio if tp_overlap_ratio is not None else self._config.default_overlap_ratio
        result = ExecutionTime()

        # 复制计算时间
        result.attn_pre_proj_time = et.attn_pre_proj_time
        result.attn_rope_time = et.attn_rope_time
        result.attn_kv_cache_save_time = et.attn_kv_cache_save_time
        result.attn_prefill_time = et.attn_prefill_time
        result.attn_decode_time = et.attn_decode_time
        result.attn_post_proj_time = et.attn_post_proj_time
        result.mlp_up_proj_time = et.mlp_up_proj_time
        result.mlp_act_time = et.mlp_act_time
        result.mlp_down_proj_time = et.mlp_down_proj_time
        result.input_layernorm_time = et.input_layernorm_time
        result.post_attention_layernorm_time = et.post_attention_layernorm_time
        result.add_time = et.add_time
        result.expert_mlp_time = et.expert_mlp_time
        result.eplb_overhead_time = et.eplb_overhead_time
        result.cpu_overhead_time = et.cpu_overhead_time

        # TP 通信调整: 重叠部分变慢
        if et.tensor_parallel_comm_time > 0 and tp_ratio > 0:
            result.tensor_parallel_comm_time = (
                et.tensor_parallel_comm_time * tp_ratio * self._config.comm_slowdown
                + et.tensor_parallel_comm_time * (1 - tp_ratio)
            )
        else:
            result.tensor_parallel_comm_time = et.tensor_parallel_comm_time

        # PP 通信调整
        if et.pipeline_parallel_comm_time > 0 and pp_overlap_ratio > 0:
            result.pipeline_parallel_comm_time = (
                et.pipeline_parallel_comm_time * pp_overlap_ratio * self._config.comm_slowdown
                + et.pipeline_parallel_comm_time * (1 - pp_overlap_ratio)
            )
        else:
            result.pipeline_parallel_comm_time = et.pipeline_parallel_comm_time

        # EP 通信调整
        if et.expert_parallel_comm_time > 0 and ep_overlap_ratio > 0:
            result.expert_parallel_comm_time = (
                et.expert_parallel_comm_time * ep_overlap_ratio * self._config.comm_slowdown
                + et.expert_parallel_comm_time * (1 - ep_overlap_ratio)
            )
        else:
            result.expert_parallel_comm_time = et.expert_parallel_comm_time

        return result

    # --------------------------------------------------------
    # 批量处理
    # --------------------------------------------------------

    def process_pairs(self, pairs: List[OverlapPair]) -> OverlapResult:
        """处理一批重叠对，返回汇总结果。"""
        result = OverlapResult()

        for pair in pairs:
            adjusted = self.apply_ratio_slowdown(pair)
            result.pairs.append(adjusted)
            result.total_overlap_ms += adjusted.overlap_duration_ms
            result.total_time_saved_ms += adjusted.time_saved_ms

            if adjusted.overlap_type == "compute_comm":
                result.num_compute_comm += 1
            elif adjusted.overlap_type == "comm_comm":
                result.num_comm_comm += 1

        # 平均减速
        cc_pairs = [p for p in result.pairs if p.overlap_type == "compute_comm"]
        if cc_pairs:
            total_compute_slow = sum(
                p.adjusted_a_ms / max(p.op_a_latency_ms, 1e-9)
                if not p.op_a_is_comm
                else p.adjusted_b_ms / max(p.op_b_latency_ms, 1e-9)
                for p in cc_pairs
            )
            result.avg_compute_slowdown = total_compute_slow / len(cc_pairs)

        return result
