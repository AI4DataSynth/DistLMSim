"""Tests for distlmsim.topology.overlap_processor module."""

import sys
import unittest

sys.path.insert(0, '.')

from distlmsim.entities import ExecutionTime
from distlmsim.topology.overlap_processor import (
    OverlapProcessor,
    OverlapConfig,
    OverlapPair,
    OverlapResult,
)


# ─── OverlapPair 测试 ────────────────────────────────────────────────────────

class TestOverlapPair(unittest.TestCase):

    def test_overlap_duration(self):
        """验证重叠持续时间 = min(a, b) * ratio"""
        pair = OverlapPair(
            op_a_latency_ms=10.0,
            op_b_latency_ms=5.0,
            overlap_ratio=0.8,
        )
        # min(10, 5) * 0.8 = 4.0
        self.assertAlmostEqual(pair.overlap_duration_ms, 4.0)

    def test_overlap_duration_zero_ratio(self):
        pair = OverlapPair(
            op_a_latency_ms=10.0,
            op_b_latency_ms=5.0,
            overlap_ratio=0.0,
        )
        self.assertAlmostEqual(pair.overlap_duration_ms, 0.0)

    def test_time_saved_positive(self):
        """验证节省时间为正"""
        pair = OverlapPair(
            op_a_latency_ms=10.0,
            op_b_latency_ms=5.0,
            overlap_ratio=0.9,
            adjusted_a_ms=10.35,
            adjusted_b_ms=5.4,
        )
        # no_overlap = max(10, 5) = 10
        # with_overlap = max(10.35, 5.4) = 10.35
        # saved = 10 - 10.35 = -0.35 → clamp to 0
        # 实际上如果 adjusted 大于原始, 则没有节省
        self.assertGreaterEqual(pair.time_saved_ms, 0.0)


# ─── Ratio-Based 减速测试 ────────────────────────────────────────────────────

class TestRatioSlowdown(unittest.TestCase):

    def setUp(self):
        self.processor = OverlapProcessor(OverlapConfig(
            compute_slowdown=1.15,
            comm_slowdown=1.20,
            comm_comm_slowdown=1.30,
        ))

    def test_compute_comm_slowdown(self):
        """验证计算-通信重叠后两者都变慢"""
        pair = self.processor.make_compute_comm_pair(
            compute_ms=10.0, comm_ms=5.0, overlap_ratio=1.0,
        )
        adjusted = self.processor.apply_ratio_slowdown(pair)
        # compute: 10.0 * 1.0 * 1.15 + 10.0 * 0 = 11.5
        self.assertAlmostEqual(adjusted.adjusted_a_ms, 11.5)
        # comm: 5.0 * 1.0 * 1.20 + 5.0 * 0 = 6.0
        self.assertAlmostEqual(adjusted.adjusted_b_ms, 6.0)

    def test_partial_overlap(self):
        """验证部分重叠: 只有重叠部分变慢"""
        pair = self.processor.make_compute_comm_pair(
            compute_ms=10.0, comm_ms=5.0, overlap_ratio=0.5,
        )
        adjusted = self.processor.apply_ratio_slowdown(pair)
        # compute: 10 * 0.5 * 1.15 + 10 * 0.5 = 5.75 + 5.0 = 10.75
        self.assertAlmostEqual(adjusted.adjusted_a_ms, 10.75)
        # comm: 5 * 0.5 * 1.20 + 5 * 0.5 = 3.0 + 2.5 = 5.5
        self.assertAlmostEqual(adjusted.adjusted_b_ms, 5.5)

    def test_zero_overlap_no_change(self):
        """验证零重叠时无变化"""
        pair = self.processor.make_compute_comm_pair(
            compute_ms=10.0, comm_ms=5.0, overlap_ratio=0.0,
        )
        adjusted = self.processor.apply_ratio_slowdown(pair)
        self.assertAlmostEqual(adjusted.adjusted_a_ms, 10.0)
        self.assertAlmostEqual(adjusted.adjusted_b_ms, 5.0)

    def test_comm_comm_slowdown(self):
        """验证通信-通信重叠"""
        pair = self.processor.make_comm_comm_pair(
            comm_a_ms=3.0, comm_b_ms=4.0, overlap_ratio=0.8,
        )
        adjusted = self.processor.apply_ratio_slowdown(pair)
        # a: 3 * 0.8 * 1.30 + 3 * 0.2 = 3.12 + 0.6 = 3.72
        self.assertAlmostEqual(adjusted.adjusted_a_ms, 3.72)
        # b: 4 * 0.8 * 1.30 + 4 * 0.2 = 4.16 + 0.8 = 4.96
        self.assertAlmostEqual(adjusted.adjusted_b_ms, 4.96)

    def test_adjusted_greater_than_original(self):
        """验证调整后的时间 >= 原始时间"""
        pair = self.processor.make_compute_comm_pair(
            compute_ms=10.0, comm_ms=5.0, overlap_ratio=0.9,
        )
        adjusted = self.processor.apply_ratio_slowdown(pair)
        self.assertGreaterEqual(adjusted.adjusted_a_ms, 10.0)
        self.assertGreaterEqual(adjusted.adjusted_b_ms, 5.0)


# ─── Bandwidth-Aware 减速测试 ────────────────────────────────────────────────

class TestBandwidthAwareSlowdown(unittest.TestCase):

    def setUp(self):
        self.processor = OverlapProcessor(OverlapConfig(congestion_alpha=0.1))

    def test_basic_bandwidth_slowdown(self):
        """验证带宽感知减速"""
        pair = self.processor.make_comm_comm_pair(
            comm_a_ms=2.0, comm_b_ms=3.0, overlap_ratio=1.0,
        )
        # 2 个并发通信, 容量 8
        adjusted = self.processor.apply_bandwidth_aware_slowdown(
            pair, concurrent_count=2, link_capacity=8,
        )
        # slowdown = 2 * 1.0 = 2.0
        # a: 2 * 1.0 * 2.0 + 2 * 0 = 4.0
        self.assertAlmostEqual(adjusted.adjusted_a_ms, 4.0)

    def test_congestion_penalty(self):
        """验证超过链路容量时有额外惩罚"""
        pair = self.processor.make_comm_comm_pair(
            comm_a_ms=2.0, comm_b_ms=3.0, overlap_ratio=1.0,
        )
        # 10 并发 > 容量 8
        adjusted = self.processor.apply_bandwidth_aware_slowdown(
            pair, concurrent_count=10, link_capacity=8,
        )
        # base = 10, penalty = 1 + 0.1 * (10 - 8) = 1.2
        # final = 10 * 1.2 = 12.0
        # a: 2 * 1.0 * 12.0 = 24.0
        self.assertAlmostEqual(adjusted.adjusted_a_ms, 24.0)

    def test_no_congestion_within_capacity(self):
        """验证不超容量时无拥塞惩罚"""
        pair = self.processor.make_comm_comm_pair(
            comm_a_ms=2.0, comm_b_ms=3.0, overlap_ratio=1.0,
        )
        adj1 = self.processor.apply_bandwidth_aware_slowdown(
            OverlapPair(
                op_a_latency_ms=2.0, op_b_latency_ms=3.0,
                overlap_ratio=1.0, overlap_type="comm_comm",
            ),
            concurrent_count=4, link_capacity=8,
        )
        # slowdown = 4 * 1.0 = 4.0 (无惩罚)
        self.assertAlmostEqual(adj1.adjusted_a_ms, 8.0)


# ─── ExecutionTime 调整测试 ──────────────────────────────────────────────────

class TestExecutionTimeAdjustment(unittest.TestCase):

    def setUp(self):
        self.processor = OverlapProcessor(OverlapConfig(
            compute_slowdown=1.15,
            comm_slowdown=1.20,
        ))

    def test_tp_comm_adjustment(self):
        """验证 TP 通信时间调整"""
        et = ExecutionTime(
            attn_prefill_time=5.0,
            mlp_up_proj_time=3.0,
            tensor_parallel_comm_time=2.0,
        )
        adjusted = self.processor.adjust_execution_time(et, tp_overlap_ratio=0.9)
        # TP comm: 2.0 * 0.9 * 1.20 + 2.0 * 0.1 = 2.16 + 0.2 = 2.36
        self.assertAlmostEqual(adjusted.tensor_parallel_comm_time, 2.36)
        # 计算时间不变
        self.assertAlmostEqual(adjusted.attn_prefill_time, 5.0)

    def test_no_overlap_no_change(self):
        """验证无重叠时执行时间不变"""
        et = ExecutionTime(
            attn_prefill_time=5.0,
            tensor_parallel_comm_time=2.0,
        )
        adjusted = self.processor.adjust_execution_time(et, tp_overlap_ratio=0.0)
        self.assertAlmostEqual(adjusted.tensor_parallel_comm_time, 2.0)
        self.assertAlmostEqual(adjusted.attn_prefill_time, 5.0)

    def test_pp_comm_adjustment(self):
        """验证 PP 通信时间调整"""
        et = ExecutionTime(
            pipeline_parallel_comm_time=3.0,
        )
        adjusted = self.processor.adjust_execution_time(et, pp_overlap_ratio=0.5)
        # 3.0 * 0.5 * 1.20 + 3.0 * 0.5 = 1.8 + 1.5 = 3.3
        self.assertAlmostEqual(adjusted.pipeline_parallel_comm_time, 3.3)

    def test_ep_comm_adjustment(self):
        """验证 EP 通信时间调整"""
        et = ExecutionTime(
            expert_parallel_comm_time=4.0,
        )
        adjusted = self.processor.adjust_execution_time(et, ep_overlap_ratio=0.7)
        # 4.0 * 0.7 * 1.20 + 4.0 * 0.3 = 3.36 + 1.2 = 4.56
        self.assertAlmostEqual(adjusted.expert_parallel_comm_time, 4.56)

    def test_zero_comm_not_affected(self):
        """验证零通信不受影响"""
        et = ExecutionTime(attn_prefill_time=5.0)
        adjusted = self.processor.adjust_execution_time(et, tp_overlap_ratio=0.9)
        self.assertEqual(adjusted.tensor_parallel_comm_time, 0.0)


# ─── 批量处理测试 ────────────────────────────────────────────────────────────

class TestBatchProcessing(unittest.TestCase):

    def setUp(self):
        self.processor = OverlapProcessor()

    def test_process_multiple_pairs(self):
        """验证批量处理多个重叠对"""
        pairs = [
            self.processor.make_compute_comm_pair(10.0, 3.0, 0.9),
            self.processor.make_compute_comm_pair(8.0, 2.0, 0.8),
            self.processor.make_comm_comm_pair(4.0, 5.0, 0.6),
        ]
        result = self.processor.process_pairs(pairs)
        self.assertEqual(result.num_compute_comm, 2)
        self.assertEqual(result.num_comm_comm, 1)
        self.assertEqual(len(result.pairs), 3)
        self.assertGreater(result.total_overlap_ms, 0)

    def test_process_empty(self):
        """验证空列表不崩溃"""
        result = self.processor.process_pairs([])
        self.assertEqual(result.num_compute_comm, 0)
        self.assertEqual(result.total_overlap_ms, 0.0)

    def test_summary_format(self):
        """验证 summary 输出格式"""
        pairs = [self.processor.make_compute_comm_pair(10.0, 3.0)]
        result = self.processor.process_pairs(pairs)
        s = result.summary()
        self.assertIn("num_compute_comm_pairs", s)
        self.assertIn("total_time_saved_ms", s)


if __name__ == '__main__':
    unittest.main()
