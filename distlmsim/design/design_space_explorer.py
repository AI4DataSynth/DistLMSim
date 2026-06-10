"""设计空间探索模块

枚举并行策略和调度策略的组合，通过规则剪枝排除不可行配置，
对剩余配置运行仿真以找到最优方案。
从 Charon 项目移植，适配 DistLMSim 的推理场景。

搜索维度:
- TP size (张量并行度)
- EP size (专家并行度)
- 全局调度策略 (GlobalSchedulerType)
- 副本调度策略 (ReplicaSchedulerType)

分析:
- Pareto 前沿 (TTFT vs Throughput)
- SLO 约束 (max_ttft_ms, max_tbt_ms)
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from distlmsim.types import GlobalSchedulerType, ReplicaSchedulerType

logger = logging.getLogger(__name__)


@dataclass
class DesignSpaceConfig:
    """设计空间搜索配置"""
    tp_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    ep_sizes: List[int] = field(default_factory=lambda: [1])
    global_schedulers: List[GlobalSchedulerType] = field(
        default_factory=lambda: [GlobalSchedulerType.ROUND_ROBIN]
    )
    replica_schedulers: List[ReplicaSchedulerType] = field(
        default_factory=lambda: [ReplicaSchedulerType.SARATHI]
    )
    # SLO 约束
    max_ttft_ms: Optional[float] = None
    max_tbt_ms: Optional[float] = None


@dataclass
class DesignPoint:
    """设计空间中的一个配置点"""
    tp_size: int = 1
    ep_size: int = 1
    global_scheduler: GlobalSchedulerType = GlobalSchedulerType.ROUND_ROBIN
    replica_scheduler: ReplicaSchedulerType = ReplicaSchedulerType.SARATHI
    is_feasible: bool = True
    feasibility_reason: str = ""

    def config_key(self) -> str:
        return (
            f"tp{self.tp_size}_ep{self.ep_size}_"
            f"gs{self.global_scheduler.name}_"
            f"rs{self.replica_scheduler.name}"
        )


@dataclass
class DesignResult:
    """单个设计点的仿真结果"""
    ttft_p50_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    tbt_p50_ms: float = 0.0
    tbt_p99_ms: float = 0.0
    e2e_p50_ms: float = 0.0
    throughput_tokens_per_s: float = 0.0
    completed_requests: int = 0
    total_time_ms: float = 0.0
    memory_utilization: float = 0.0


@dataclass
class PruningRule:
    """剪枝规则"""
    name: str
    condition: str     # 条件表达式 (可使用 tp, ep 变量)
    reason: str

    def check(self, point: DesignPoint) -> bool:
        local_vars = {
            "tp": point.tp_size,
            "ep": point.ep_size,
            "world_size": point.tp_size * point.ep_size,
        }
        try:
            return bool(eval(self.condition, {"__builtins__": {}}, local_vars))
        except Exception:
            return False


DEFAULT_PRUNING_RULES = [
    PruningRule(
        name="TP不能超过8",
        condition="tp > 8",
        reason="TP通常限制在单节点内 (最大8 GPU)",
    ),
    PruningRule(
        name="EP不能超过8",
        condition="ep > 8",
        reason="EP通常限制在单节点内",
    ),
    PruningRule(
        name="TP*EP不超过节点GPU数",
        condition="world_size > 8",
        reason="TP*EP超过单节点GPU数量",
    ),
]


class DesignSpaceExplorer:
    """设计空间探索器

    枚举并行策略和调度策略组合，通过剪枝和仿真找到最优配置。

    使用方法:
        config = DesignSpaceConfig(tp_sizes=[1,2,4], ...)
        explorer = DesignSpaceExplorer(config, simulator_fn=my_sim)
        summary = explorer.search()
    """

    def __init__(
        self,
        config: DesignSpaceConfig,
        simulator_fn: Optional[Callable[[DesignPoint], DesignResult]] = None,
    ):
        self._config = config
        self._simulator_fn = simulator_fn or self._default_simulator
        self._pruning_rules: List[PruningRule] = list(DEFAULT_PRUNING_RULES)
        self._results: List[Tuple[DesignPoint, Optional[DesignResult]]] = []
        self._pruned_count = 0

    def add_pruning_rule(self, rule: PruningRule) -> None:
        self._pruning_rules.append(rule)

    # --------------------------------------------------------
    # 生成与剪枝
    # --------------------------------------------------------

    def generate_design_points(self) -> List[DesignPoint]:
        """枚举所有设计点 (TP × EP × GlobalSched × ReplicaSched 笛卡尔积)"""
        points = []
        for tp, ep, gs, rs in itertools.product(
            self._config.tp_sizes,
            self._config.ep_sizes,
            self._config.global_schedulers,
            self._config.replica_schedulers,
        ):
            points.append(DesignPoint(
                tp_size=tp, ep_size=ep,
                global_scheduler=gs, replica_scheduler=rs,
            ))
        logger.info("生成 %d 个候选设计点", len(points))
        return points

    def prune(
        self, points: List[DesignPoint]
    ) -> Tuple[List[DesignPoint], List[DesignPoint]]:
        """应用剪枝规则，返回 (保留点, 被剪枝点)"""
        kept, pruned = [], []
        for point in points:
            should_prune = False
            for rule in self._pruning_rules:
                if rule.check(point):
                    point.is_feasible = False
                    point.feasibility_reason = rule.reason
                    should_prune = True
                    break
            if should_prune:
                pruned.append(point)
            else:
                kept.append(point)
        self._pruned_count = len(pruned)
        logger.info("剪枝: 保留 %d, 剪除 %d", len(kept), len(pruned))
        return kept, pruned

    # --------------------------------------------------------
    # 仿真评估
    # --------------------------------------------------------

    def evaluate(
        self, points: List[DesignPoint]
    ) -> List[Tuple[DesignPoint, DesignResult]]:
        """对每个设计点运行仿真"""
        results = []
        start = time.time()
        for point in points:
            try:
                result = self._simulator_fn(point)
                results.append((point, result))
            except Exception as e:
                logger.warning("仿真失败 %s: %s", point.config_key(), e)
                point.is_feasible = False
                point.feasibility_reason = str(e)
        elapsed = time.time() - start
        logger.info("仿真完成: %d 点, 耗时 %.2fs", len(results), elapsed)
        return results

    # --------------------------------------------------------
    # Pareto 分析
    # --------------------------------------------------------

    def find_pareto_frontier(
        self,
        results: List[Tuple[DesignPoint, DesignResult]],
        objective_x: str = "ttft_p50_ms",
        objective_y: str = "throughput_tokens_per_s",
        minimize_x: bool = True,
        maximize_y: bool = True,
    ) -> List[DesignPoint]:
        """找出 Pareto 最优前沿。

        默认: 最小化 TTFT (X) 且最大化吞吐量 (Y)。

        Args:
            results: 仿真结果列表
            objective_x: X 轴目标属性名
            objective_y: Y 轴目标属性名
            minimize_x: 是否最小化 X
            maximize_y: 是否最大化 Y

        Returns:
            Pareto 最优设计点列表
        """
        valid = [
            (dp, r) for dp, r in results
            if dp.is_feasible and r is not None
        ]
        if not valid:
            return []

        # 应用 SLO 约束过滤
        filtered = []
        for dp, r in valid:
            if self._config.max_ttft_ms and r.ttft_p99_ms > self._config.max_ttft_ms:
                continue
            if self._config.max_tbt_ms and r.tbt_p99_ms > self._config.max_tbt_ms:
                continue
            filtered.append((dp, r))

        if not filtered:
            return []

        # 排序: X 从小到大
        def x_val(r):
            return getattr(r, objective_x, 0)

        def y_val(r):
            return getattr(r, objective_y, 0)

        sorted_points = sorted(
            filtered,
            key=lambda x: x_val(x[1]),
            reverse=not minimize_x,
        )

        # 贪心找 Pareto 前沿
        pareto: List[DesignPoint] = []
        best_y = float("-inf") if maximize_y else float("inf")

        for dp, r in sorted_points:
            y = y_val(r)
            if maximize_y and y > best_y:
                pareto.append(dp)
                best_y = y
            elif not maximize_y and y < best_y:
                pareto.append(dp)
                best_y = y

        return pareto

    # --------------------------------------------------------
    # 完整搜索
    # --------------------------------------------------------

    def search(self) -> Dict[str, Any]:
        """执行完整的设计空间搜索流程"""
        logger.info("=" * 50)
        logger.info("开始设计空间探索")

        all_points = self.generate_design_points()
        kept, pruned = self.prune(all_points)
        results = self.evaluate(kept)
        pareto = self.find_pareto_frontier(results)

        self._results = results

        summary = {
            "total_candidates": len(all_points),
            "pruned": len(pruned),
            "simulated": len(results),
            "feasible": len([dp for dp, _ in results if dp.is_feasible]),
            "pareto_points": len(pareto),
        }

        if pareto:
            summary["best_config"] = pareto[-1].config_key()

        logger.info("探索完成: %s", json.dumps(summary, indent=2))
        return summary

    # --------------------------------------------------------
    # 保存结果
    # --------------------------------------------------------

    def save_results(self, output_path: str) -> None:
        """保存搜索结果到 JSON"""
        data = []
        for dp, result in self._results:
            entry = {
                "config": dp.config_key(),
                "tp": dp.tp_size,
                "ep": dp.ep_size,
                "global_scheduler": dp.global_scheduler.name,
                "replica_scheduler": dp.replica_scheduler.name,
                "feasible": dp.is_feasible,
            }
            if result:
                entry.update({
                    "ttft_p50_ms": round(result.ttft_p50_ms, 3),
                    "ttft_p99_ms": round(result.ttft_p99_ms, 3),
                    "tbt_p50_ms": round(result.tbt_p50_ms, 3),
                    "throughput_tokens_per_s": round(result.throughput_tokens_per_s, 1),
                    "completed_requests": result.completed_requests,
                })
            data.append(entry)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("结果已保存至 %s", output_path)

    # --------------------------------------------------------
    # 默认仿真器
    # --------------------------------------------------------

    @staticmethod
    def _default_simulator(point: DesignPoint) -> DesignResult:
        """默认空仿真器 (返回零值), 用于测试"""
        return DesignResult()
