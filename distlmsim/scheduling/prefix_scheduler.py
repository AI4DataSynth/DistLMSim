"""Hardware-Aware Prefix Scheduler (DSpark Algorithm 1)

负载感知前缀调度器：根据 confidence scores 和硬件吞吐量曲线，
贪心选择每个 request 的 verify prefix 长度，最大化整体吞吐。

依赖层次: Layer 6
  输入: entities (Request), execution/draft_model (load_sps_curve)
  输出: HardwareAwarePrefixScheduler (被 SpeculativeDecodingEngine 消费)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from distlmsim.entities import Request


class HardwareAwarePrefixScheduler:
    """DSpark 的负载感知前缀调度器。

    输入:
      - 每个 request 的 confidence scores [c_1, ..., c_γ]
      - SPS(B) 曲线: batch_size → steps_per_second

    输出:
      - 每个 request 的 scheduled prefix length [l_1, ..., l_R]

    核心逻辑 (论文 Algorithm 1):
    1. 计算 prefix survival probabilities:
       a_{r,j} = Π_{i=1}^{j} c_{r,i}
    2. 收集所有 valid (r, j) pairs with a_{r,j} > 0
    3. 按 a_{r,j} 降序全局排序
    4. 贪心 admit tokens:
       - 每次取最高 a_{r,j} 的 token
       - 计算当前总 accept 数 τ*
       - 查 SPS(B) 得到 throughput Θ = τ* × SPS(B)
       - 如果 Θ < best_Θ, break (early stopping)
    5. 返回每个 request 的 admitted prefix length

    退化模式:
      - 无 SPS 曲线时: 使用静态阈值调度 (confidence < threshold → 截断)
      - 无 confidence scores 时: 返回 full block_size
    """

    def __init__(
        self,
        sps_curve: Optional[Dict[int, float]] = None,
        confidence_threshold: float = 0.0,
    ):
        self._sps_curve = sps_curve or {}
        self._confidence_threshold = confidence_threshold

    def _lookup_sps(self, batch_size: int) -> float:
        """查找 SPS(B) 值，支持插值。"""
        if not self._sps_curve:
            return 0.0
        sizes = sorted(self._sps_curve.keys())
        if batch_size <= sizes[0]:
            return self._sps_curve[sizes[0]]
        if batch_size >= sizes[-1]:
            return self._sps_curve[sizes[-1]]
        # 线性插值
        for i in range(len(sizes) - 1):
            if sizes[i] <= batch_size <= sizes[i + 1]:
                lo, hi = sizes[i], sizes[i + 1]
                t = (batch_size - lo) / (hi - lo)
                return self._sps_curve[lo] * (1 - t) + self._sps_curve[hi] * t
        return self._sps_curve[sizes[-1]]

    def schedule(
        self,
        requests: List[Request],
        confidence_scores: List[List[float]],
        block_size: int,
    ) -> List[int]:
        """计算每个 request 的 scheduled prefix length。

        Args:
            requests: 当前 batch 中的请求列表
            confidence_scores: 每个 request 的 confidence scores
                confidence_scores[r] = [c_{r,1}, ..., c_{r,γ}]
            block_size: draft block 大小 (γ)

        Returns:
            List[int]: 每个 request 的 scheduled prefix length
        """
        R = len(requests)
        if R == 0:
            return []

        # 退化: 无 confidence scores → 返回 full block
        if not confidence_scores or all(not cs for cs in confidence_scores):
            return [block_size] * R

        # 退化: 无 SPS 曲线 → 静态阈值调度
        if not self._sps_curve:
            return self._static_threshold_schedule(confidence_scores, block_size)

        # Algorithm 1: Hardware-Aware Prefix Scheduler
        return self._greedy_schedule(confidence_scores, block_size)

    def _static_threshold_schedule(
        self,
        confidence_scores: List[List[float]],
        block_size: int,
    ) -> List[int]:
        """静态阈值调度: 截断到第一个低于阈值的位置。"""
        lengths = []
        threshold = self._confidence_threshold

        for cs in confidence_scores:
            prefix_len = 0
            for j, c in enumerate(cs[:block_size]):
                if c < threshold:
                    break
                prefix_len = j + 1
            lengths.append(max(1, prefix_len))  # 至少 verify 1 token

        return lengths

    def _greedy_schedule(
        self,
        confidence_scores: List[List[float]],
        block_size: int,
    ) -> List[int]:
        """贪心调度 (DSpark Algorithm 1)。"""
        R = len(confidence_scores)

        # Step 1: 计算 prefix survival probabilities
        # a[r][j] = Π_{i=0}^{j} c_{r,i}
        survival: List[List[float]] = []
        for r in range(R):
            cs = confidence_scores[r]
            a = []
            cumprod = 1.0
            for j in range(min(len(cs), block_size)):
                cumprod *= cs[j]
                a.append(cumprod)
            survival.append(a)

        # Step 2: 收集所有 valid (r, j) pairs
        candidates: List[Tuple[float, int, int]] = []  # (survival_prob, r, j)
        for r in range(R):
            for j in range(len(survival[r])):
                if survival[r][j] > 0:
                    candidates.append((survival[r][j], r, j))

        # Step 3: 按 survival probability 降序排序
        candidates.sort(key=lambda x: -x[0])

        # Step 4: 贪心 admit tokens
        admitted: List[int] = [0] * R  # admitted prefix length per request
        # admitted 需要是前缀连续的: 如果 admit (r, j), 必须已 admit (r, 0..j-1)
        # 所以我们用一个 set 跟踪每个 r 已 admit 的最大连续 prefix
        admitted_set: List[int] = [0] * R  # 已 admit 的最大连续位置+1

        best_throughput = 0.0
        best_admitted = [0] * R

        for prob, r, j in candidates:
            # 只有当 j 是当前 r 的下一个连续位置时才 admit
            if j != admitted_set[r]:
                continue

            # Tentatively admit this token
            admitted_set[r] = j + 1

            # 计算当前总 accept 和 throughput
            total_accepted = sum(admitted_set)
            batch_size = R
            sps = self._lookup_sps(batch_size)
            throughput = total_accepted * sps

            if throughput >= best_throughput:
                best_throughput = throughput
                best_admitted = list(admitted_set)
            else:
                # Throughput 下降 → early stop
                break

        # 确保至少 verify 1 token per request
        result = [max(1, a) for a in best_admitted]
        return result


def generate_confidence_scores(
    block_size: int,
    acceptance_rate: float,
    position_decay: float = 0.05,
) -> List[float]:
    """生成模拟的 confidence scores。

    根据 DSpark 论文 Figure 2，位置越靠后，条件接受率越低。
    使用指数衰减模型: c_k = α × (1 - decay × k)

    Args:
        block_size: draft block 大小
        acceptance_rate: 基础接受率 α
        position_decay: 位置衰减系数 (默认 0.05)

    Returns:
        List[float]: 每位置的 confidence score
    """
    scores = []
    for k in range(block_size):
        c = acceptance_rate * (1.0 - position_decay * k)
        scores.append(max(0.1, min(1.0, c)))
    return scores
