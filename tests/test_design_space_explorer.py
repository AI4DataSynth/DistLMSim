"""Tests for distlmsim.design.design_space_explorer module."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, '.')

from distlmsim.types import GlobalSchedulerType, ReplicaSchedulerType
from distlmsim.design.design_space_explorer import (
    DesignSpaceExplorer,
    DesignSpaceConfig,
    DesignPoint,
    DesignResult,
    PruningRule,
    DEFAULT_PRUNING_RULES,
)


# ─── DesignPoint 测试 ────────────────────────────────────────────────────────

class TestDesignPoint(unittest.TestCase):

    def test_config_key(self):
        dp = DesignPoint(tp_size=4, ep_size=2)
        key = dp.config_key()
        self.assertIn("tp4", key)
        self.assertIn("ep2", key)

    def test_default_values(self):
        dp = DesignPoint()
        self.assertEqual(dp.tp_size, 1)
        self.assertEqual(dp.ep_size, 1)
        self.assertTrue(dp.is_feasible)


# ─── PruningRule 测试 ────────────────────────────────────────────────────────

class TestPruningRule(unittest.TestCase):

    def test_tp_exceeds_limit(self):
        rule = PruningRule("test", "tp > 8", "too many")
        dp = DesignPoint(tp_size=16)
        self.assertTrue(rule.check(dp))

    def test_tp_within_limit(self):
        rule = PruningRule("test", "tp > 8", "too many")
        dp = DesignPoint(tp_size=4)
        self.assertFalse(rule.check(dp))

    def test_world_size_check(self):
        rule = PruningRule("test", "world_size > 8", "too big")
        dp = DesignPoint(tp_size=4, ep_size=4)  # 16 > 8
        self.assertTrue(rule.check(dp))

    def test_invalid_condition(self):
        """无效条件应返回 False (不剪枝)"""
        rule = PruningRule("test", "invalid_var > 5", "bad")
        dp = DesignPoint(tp_size=1)
        self.assertFalse(rule.check(dp))


# ─── 生成与剪枝测试 ──────────────────────────────────────────────────────────

class TestGenerateAndPrune(unittest.TestCase):

    def test_generate_cartesian_product(self):
        """验证笛卡尔积生成"""
        config = DesignSpaceConfig(
            tp_sizes=[1, 2, 4],
            ep_sizes=[1],
            global_schedulers=[GlobalSchedulerType.ROUND_ROBIN],
            replica_schedulers=[ReplicaSchedulerType.SARATHI],
        )
        explorer = DesignSpaceExplorer(config)
        points = explorer.generate_design_points()
        self.assertEqual(len(points), 3)

    def test_generate_multi_dimension(self):
        """验证多维度笛卡尔积"""
        config = DesignSpaceConfig(
            tp_sizes=[1, 2],
            ep_sizes=[1, 2],
            global_schedulers=[GlobalSchedulerType.ROUND_ROBIN, GlobalSchedulerType.RANDOM],
            replica_schedulers=[ReplicaSchedulerType.SARATHI],
        )
        explorer = DesignSpaceExplorer(config)
        points = explorer.generate_design_points()
        # 2 × 2 × 2 × 1 = 8
        self.assertEqual(len(points), 8)

    def test_prune_tp_too_large(self):
        """验证 TP > 8 被剪枝"""
        config = DesignSpaceConfig(tp_sizes=[1, 4, 16])
        explorer = DesignSpaceExplorer(config)
        points = explorer.generate_design_points()
        kept, pruned = explorer.prune(points)
        # TP=16 应被剪枝
        pruned_keys = [p.config_key() for p in pruned]
        self.assertTrue(any("tp16" in k for k in pruned_keys))

    def test_prune_world_size(self):
        """验证 TP*EP > 8 被剪枝"""
        config = DesignSpaceConfig(tp_sizes=[4], ep_sizes=[1, 4])
        explorer = DesignSpaceExplorer(config)
        points = explorer.generate_design_points()
        kept, pruned = explorer.prune(points)
        # TP=4, EP=4 → world_size=16 > 8 应被剪枝
        self.assertGreater(len(pruned), 0)


# ─── 仿真评估测试 ────────────────────────────────────────────────────────────

class TestEvaluation(unittest.TestCase):

    def test_evaluate_with_custom_simulator(self):
        """验证使用自定义仿真器"""
        def my_sim(point: DesignPoint) -> DesignResult:
            return DesignResult(
                ttft_p50_ms=10.0 / point.tp_size,
                throughput_tokens_per_s=1000.0 * point.tp_size,
                completed_requests=100,
            )

        config = DesignSpaceConfig(tp_sizes=[1, 2, 4])
        explorer = DesignSpaceExplorer(config, simulator_fn=my_sim)
        points = explorer.generate_design_points()
        kept, _ = explorer.prune(points)
        results = explorer.evaluate(kept)

        self.assertEqual(len(results), 3)
        # TP=4 应有更高吞吐
        tp4_result = [r for dp, r in results if dp.tp_size == 4][0]
        tp1_result = [r for dp, r in results if dp.tp_size == 1][0]
        self.assertGreater(
            tp4_result.throughput_tokens_per_s,
            tp1_result.throughput_tokens_per_s,
        )

    def test_evaluate_with_failing_simulator(self):
        """验证仿真失败时不崩溃"""
        def bad_sim(point: DesignPoint) -> DesignResult:
            if point.tp_size == 4:
                raise RuntimeError("OOM")
            return DesignResult()

        config = DesignSpaceConfig(tp_sizes=[1, 2, 4])
        explorer = DesignSpaceExplorer(config, simulator_fn=bad_sim)
        points = explorer.generate_design_points()
        results = explorer.evaluate(points)
        # TP=4 失败, 只有 2 个结果
        self.assertEqual(len(results), 2)


# ─── Pareto 分析测试 ─────────────────────────────────────────────────────────

class TestParetoAnalysis(unittest.TestCase):

    def test_pareto_frontier(self):
        """验证 Pareto 前沿正确识别"""
        results = [
            (DesignPoint(tp_size=1), DesignResult(
                ttft_p50_ms=100.0, throughput_tokens_per_s=100.0,
            )),
            (DesignPoint(tp_size=2), DesignResult(
                ttft_p50_ms=50.0, throughput_tokens_per_s=200.0,
            )),
            (DesignPoint(tp_size=4), DesignResult(
                ttft_p50_ms=30.0, throughput_tokens_per_s=150.0,
            )),
        ]
        config = DesignSpaceConfig()
        explorer = DesignSpaceExplorer(config)
        pareto = explorer.find_pareto_frontier(results)
        # TP=1 (worst TTFT, worst throughput) → not on frontier
        # TP=2 (better TTFT, best throughput) → on frontier
        # TP=4 (best TTFT, medium throughput) → on frontier
        pareto_tps = [p.tp_size for p in pareto]
        self.assertIn(2, pareto_tps)
        self.assertIn(4, pareto_tps)
        self.assertNotIn(1, pareto_tps)

    def test_pareto_with_slo_constraint(self):
        """验证 SLO 约束过滤"""
        results = [
            (DesignPoint(tp_size=1), DesignResult(
                ttft_p50_ms=10.0, ttft_p99_ms=500.0,
                throughput_tokens_per_s=1000.0,
            )),
            (DesignPoint(tp_size=2), DesignResult(
                ttft_p50_ms=5.0, ttft_p99_ms=50.0,
                throughput_tokens_per_s=500.0,
            )),
        ]
        config = DesignSpaceConfig(max_ttft_ms=100.0)
        explorer = DesignSpaceExplorer(config)
        pareto = explorer.find_pareto_frontier(results)
        # TP=1 的 ttft_p99=500 > max_ttft=100 → 被过滤
        pareto_tps = [p.tp_size for p in pareto]
        self.assertNotIn(1, pareto_tps)
        self.assertIn(2, pareto_tps)

    def test_pareto_empty_results(self):
        """验证空结果返回空 Pareto"""
        explorer = DesignSpaceExplorer(DesignSpaceConfig())
        pareto = explorer.find_pareto_frontier([])
        self.assertEqual(pareto, [])


# ─── 完整搜索测试 ────────────────────────────────────────────────────────────

class TestFullSearch(unittest.TestCase):

    def test_search_end_to_end(self):
        """验证完整搜索流程"""
        counter = {"n": 0}

        def sim(point: DesignPoint) -> DesignResult:
            counter["n"] += 1
            return DesignResult(
                ttft_p50_ms=50.0 / point.tp_size,
                throughput_tokens_per_s=500.0 * point.tp_size,
                completed_requests=50,
            )

        config = DesignSpaceConfig(
            tp_sizes=[1, 2, 4],
            ep_sizes=[1],
            global_schedulers=[GlobalSchedulerType.ROUND_ROBIN],
            replica_schedulers=[ReplicaSchedulerType.SARATHI],
        )
        explorer = DesignSpaceExplorer(config, simulator_fn=sim)
        summary = explorer.search()

        self.assertEqual(summary["total_candidates"], 3)
        self.assertGreater(summary["simulated"], 0)
        self.assertIn("best_config", summary)
        self.assertGreater(counter["n"], 0)

    def test_save_results(self):
        """验证保存结果到 JSON"""
        def sim(point: DesignPoint) -> DesignResult:
            return DesignResult(ttft_p50_ms=10.0, throughput_tokens_per_s=100.0)

        config = DesignSpaceConfig(tp_sizes=[1, 2])
        explorer = DesignSpaceExplorer(config, simulator_fn=sim)
        explorer.search()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            explorer.save_results(path)
            self.assertTrue(os.path.isfile(path))
            with open(path) as f:
                data = json.load(f)
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)
        finally:
            os.unlink(path)

    def test_add_custom_pruning_rule(self):
        """验证添加自定义剪枝规则"""
        config = DesignSpaceConfig(tp_sizes=[1, 2, 4, 8])
        explorer = DesignSpaceExplorer(config)
        explorer.add_pruning_rule(PruningRule(
            "no tp=8", "tp == 8", "too expensive",
        ))
        points = explorer.generate_design_points()
        kept, pruned = explorer.prune(points)
        pruned_tps = [p.tp_size for p in pruned]
        self.assertIn(8, pruned_tps)


if __name__ == '__main__':
    unittest.main()
