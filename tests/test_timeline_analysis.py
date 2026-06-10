"""Tests for distlmsim.analysis.timeline_analysis module."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, '.')

from distlmsim.config import MetricsConfig
from distlmsim.metrics.metrics_store import MetricsStore, RequestMetrics
from distlmsim.analysis.timeline_analysis import (
    TimelineAnalyzer,
    TraceEvent,
    TID_PREFILL,
    TID_DECODE,
    TID_KV_TRANSFER,
)


def _make_store_with_requests(num_requests: int = 5, disaggregated: bool = False):
    """创建含完成请求的 MetricsStore"""
    store = MetricsStore(MetricsConfig())
    for i in range(num_requests):
        store.record_request_arrival(i, time=float(i * 100))
        store.record_request_scheduled(i, time=float(i * 100 + 5))
        store.record_prefill_start(i, time=float(i * 100 + 10), node_id=0)
        store.record_prefill_end(i, time=float(i * 100 + 30))
        store.set_request_tokens(i, prefill_tokens=1024, decode_tokens=256)
        if disaggregated:
            store.record_kv_cache_transfer_start(i, time=float(i * 100 + 30))
            store.record_kv_cache_transfer_end(i, time=float(i * 100 + 35))
            store.record_decode_start(i, time=float(i * 100 + 35), node_id=1)
        else:
            store.record_decode_start(i, time=float(i * 100 + 30), node_id=0)
        store.record_decode_end(i, time=float(i * 100 + 80))
    return store


# ─── TraceEvent 测试 ─────────────────────────────────────────────────────────

class TestTraceEvent(unittest.TestCase):

    def test_to_dict_complete_event(self):
        """验证完整事件的 dict 格式"""
        e = TraceEvent(
            name="prefill_r0", cat="compute", ph="X",
            ts=1000.0, dur=500.0, pid=0, tid=0,
            args={"request_id": 0},
        )
        d = e.to_dict()
        self.assertEqual(d["name"], "prefill_r0")
        self.assertEqual(d["cat"], "compute")
        self.assertEqual(d["ph"], "X")
        self.assertAlmostEqual(d["ts"], 1000.0)
        self.assertAlmostEqual(d["dur"], 500.0)
        self.assertIn("args", d)

    def test_to_dict_metadata_event(self):
        """验证 metadata 事件无 dur 字段"""
        e = TraceEvent(
            name="process_name", ph="M", ts=0, pid=0, tid=0,
            args={"name": "Node 0"},
        )
        d = e.to_dict()
        self.assertEqual(d["ph"], "M")
        self.assertNotIn("dur", d)

    def test_minimum_duration(self):
        """验证最小持续时间为 0.001 μs"""
        e = TraceEvent(name="test", cat="x", ph="X", ts=0, dur=0.0)
        d = e.to_dict()
        self.assertAlmostEqual(d["dur"], 0.001)


# ─── 基本生成测试 ────────────────────────────────────────────────────────────

class TestTraceGeneration(unittest.TestCase):

    def setUp(self):
        self.analyzer = TimelineAnalyzer()

    def test_empty_store(self):
        """空 store 返回空列表"""
        store = MetricsStore(MetricsConfig())
        events = self.analyzer.generate_trace(store)
        self.assertEqual(len(events), 0)

    def test_basic_generation(self):
        """验证基本 trace 生成"""
        store = _make_store_with_requests(3)
        events = self.analyzer.generate_trace(store)
        self.assertGreater(len(events), 0)
        # 应包含 metadata 和请求事件
        meta_events = [e for e in events if e["ph"] == "M"]
        self.assertGreater(len(meta_events), 0)
        complete_events = [e for e in events if e["ph"] == "X"]
        self.assertGreater(len(complete_events), 0)

    def test_event_count(self):
        """验证事件数量: 每个请求至少 3 个事件 (sched+prefill+decode)"""
        store = _make_store_with_requests(5)
        events = self.analyzer.generate_trace(store)
        complete = [e for e in events if e["ph"] == "X"]
        # 5 requests × at least 3 events = 15
        self.assertGreaterEqual(len(complete), 15)

    def test_prefill_events_present(self):
        """验证包含 prefill 事件"""
        store = _make_store_with_requests(2)
        events = self.analyzer.generate_trace(store)
        prefill = [e for e in events if "prefill" in e.get("name", "")]
        self.assertEqual(len(prefill), 2)

    def test_decode_events_present(self):
        """验证包含 decode 事件"""
        store = _make_store_with_requests(2)
        events = self.analyzer.generate_trace(store)
        decode = [e for e in events if "decode" in e.get("name", "")]
        self.assertEqual(len(decode), 2)

    def test_time_units_microseconds(self):
        """验证时间单位为微秒"""
        store = _make_store_with_requests(1)
        events = self.analyzer.generate_trace(store)
        complete = [e for e in events if e["ph"] == "X"]
        # arrival_time = 0ms, scheduled = 5ms → sched ts = 0μs
        # prefill_start = 10ms → 10000μs
        for e in complete:
            self.assertGreaterEqual(e["ts"], 0)


# ─── 存算分离模式测试 ────────────────────────────────────────────────────────

class TestDisaggregatedTrace(unittest.TestCase):

    def setUp(self):
        self.analyzer = TimelineAnalyzer()

    def test_kv_transfer_events(self):
        """验证存算分离模式有 KV Cache 传输事件"""
        store = _make_store_with_requests(3, disaggregated=True)
        events = self.analyzer.generate_trace(store, disaggregated=True)
        kv_events = [e for e in events if "kv_xfer" in e.get("name", "")]
        self.assertEqual(len(kv_events), 3)

    def test_multi_node_pid(self):
        """验证多节点有不同的 pid"""
        store = _make_store_with_requests(2, disaggregated=True)
        events = self.analyzer.generate_trace(store, disaggregated=True)
        pids = set(e["pid"] for e in events if e["ph"] == "X")
        self.assertGreaterEqual(len(pids), 2)


# ─── 保存和统计测试 ─────────────────────────────────────────────────────────

class TestSaveAndStats(unittest.TestCase):

    def setUp(self):
        self.analyzer = TimelineAnalyzer()

    def test_save_trace(self):
        """验证保存 JSON 文件"""
        store = _make_store_with_requests(2)
        events = self.analyzer.generate_trace(store)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            self.analyzer.save_trace(events, path)
            self.assertTrue(os.path.isfile(path))
            with open(path) as f:
                data = json.load(f)
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)
        finally:
            os.unlink(path)

    def test_compute_stats(self):
        """验证统计计算"""
        store = _make_store_with_requests(5)
        events = self.analyzer.generate_trace(store)
        stats = self.analyzer.compute_stats(events)
        self.assertIn("total_time_us", stats)
        self.assertIn("total_time_ms", stats)
        self.assertIn("num_events", stats)
        self.assertIn("by_category", stats)
        self.assertIn("by_node", stats)
        self.assertGreater(stats["total_time_us"], 0)
        self.assertGreater(stats["num_events"], 0)

    def test_compute_stats_empty(self):
        """验证空事件的统计"""
        stats = self.analyzer.compute_stats([])
        self.assertEqual(stats, {})

    def test_stats_categories(self):
        """验证统计包含各事件类别"""
        store = _make_store_with_requests(3, disaggregated=True)
        events = self.analyzer.generate_trace(store, disaggregated=True)
        stats = self.analyzer.compute_stats(events)
        cats = stats["by_category"]
        self.assertIn("compute", cats)
        self.assertIn("scheduling", cats)
        self.assertIn("communication", cats)


if __name__ == '__main__':
    unittest.main()
