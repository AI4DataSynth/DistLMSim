"""Tests for distlmsim.metrics module."""

import sys
import unittest
from io import StringIO

sys.path.insert(0, '.')

from distlmsim.config import MetricsConfig
from distlmsim.metrics.metrics_store import MetricsStore, RequestMetrics


class TestRequestMetrics(unittest.TestCase):
    """Test RequestMetrics properties."""

    def test_ttft_calculation(self):
        """Verify TTFT is calculated correctly."""
        m = RequestMetrics(
            request_id=0,
            arrival_time=10.0,
            prefill_start_time=12.0,
            prefill_end_time=25.0,
        )
        self.assertAlmostEqual(m.ttft, 15.0)  # 25 - 10

    def test_ttft_zero_when_no_prefill(self):
        """Verify TTFT is 0 when prefill hasn't ended."""
        m = RequestMetrics(
            request_id=0,
            arrival_time=10.0,
            prefill_end_time=0.0,
        )
        self.assertEqual(m.ttft, 0.0)

    def test_e2e_latency(self):
        """Verify E2E latency is calculated correctly."""
        m = RequestMetrics(
            request_id=0,
            arrival_time=10.0,
            decode_end_time=110.0,
        )
        self.assertAlmostEqual(m.e2e_latency, 100.0)

    def test_e2e_latency_zero_when_incomplete(self):
        """Verify E2E latency is 0 when decode hasn't ended."""
        m = RequestMetrics(
            request_id=0,
            arrival_time=10.0,
            decode_end_time=0.0,
        )
        self.assertEqual(m.e2e_latency, 0.0)

    def test_tbt_calculation(self):
        """Verify TBT is calculated correctly."""
        m = RequestMetrics(
            request_id=0,
            decode_start_time=30.0,
            decode_end_time=80.0,
            decode_tokens=51,  # 50 intervals
        )
        # (80 - 30) / (51 - 1) = 1.0
        self.assertAlmostEqual(m.tbt, 1.0)

    def test_tbt_zero_for_single_token(self):
        """Verify TBT is 0 for single decode token."""
        m = RequestMetrics(
            request_id=0,
            decode_start_time=30.0,
            decode_end_time=31.0,
            decode_tokens=1,
        )
        self.assertEqual(m.tbt, 0.0)

    def test_scheduling_delay(self):
        """Verify scheduling delay is calculated correctly."""
        m = RequestMetrics(
            request_id=0,
            arrival_time=10.0,
            scheduled_time=15.0,
        )
        self.assertAlmostEqual(m.scheduling_delay, 5.0)

    def test_prefill_time(self):
        """Verify prefill time calculation."""
        m = RequestMetrics(
            request_id=0,
            prefill_start_time=12.0,
            prefill_end_time=25.0,
        )
        self.assertAlmostEqual(m.prefill_time, 13.0)

    def test_decode_time(self):
        """Verify decode time calculation."""
        m = RequestMetrics(
            request_id=0,
            decode_start_time=30.0,
            decode_end_time=80.0,
        )
        self.assertAlmostEqual(m.decode_time, 50.0)

    def test_kv_cache_transfer_time(self):
        """Verify KV cache transfer time calculation."""
        m = RequestMetrics(
            request_id=0,
            kv_cache_transfer_start=25.0,
            kv_cache_transfer_end=28.0,
        )
        self.assertAlmostEqual(m.kv_cache_transfer_time, 3.0)


class TestMetricsStore(unittest.TestCase):
    """Test MetricsStore operations."""

    def setUp(self):
        self.config = MetricsConfig()
        self.store = MetricsStore(self.config)

    def test_record_request_arrival(self):
        """Verify recording request arrival."""
        self.store.record_request_arrival(0, time=10.0)
        self.assertEqual(self.store.get_completed_count(), 0)

    def test_record_prefill_start_end(self):
        """Verify recording prefill start and end."""
        self.store.record_request_arrival(0, time=10.0)
        self.store.record_prefill_start(0, time=12.0, node_id=0)
        self.store.record_prefill_end(0, time=25.0)
        # Not complete yet (no decode_end)
        self.assertEqual(self.store.get_completed_count(), 0)

    def test_record_decode_start_end(self):
        """Verify recording decode start and end."""
        self.store.record_request_arrival(0, time=10.0)
        self.store.record_prefill_start(0, time=12.0, node_id=0)
        self.store.record_prefill_end(0, time=25.0)
        self.store.record_decode_start(0, time=28.0, node_id=1)
        self.store.record_decode_end(0, time=80.0)
        self.assertEqual(self.store.get_completed_count(), 1)

    def test_full_request_lifecycle(self):
        """Verify complete request lifecycle recording."""
        self.store.record_request_arrival(0, time=10.0)
        self.store.set_request_tokens(0, prefill_tokens=100, decode_tokens=50)
        self.store.record_request_scheduled(0, time=11.0)
        self.store.record_prefill_start(0, time=12.0, node_id=0)
        self.store.record_prefill_end(0, time=25.0)
        self.store.record_kv_cache_transfer_start(0, time=25.0)
        self.store.record_kv_cache_transfer_end(0, time=28.0)
        self.store.record_decode_start(0, time=28.0, node_id=1)
        self.store.record_decode_end(0, time=80.0)

        self.assertEqual(self.store.get_completed_count(), 1)

    def test_multiple_requests(self):
        """Verify multiple requests can be tracked."""
        for i in range(10):
            self.store.record_request_arrival(i, time=float(i * 10))
            self.store.record_prefill_start(i, time=float(i * 10 + 1), node_id=0)
            self.store.record_prefill_end(i, time=float(i * 10 + 5))
            self.store.record_decode_start(i, time=float(i * 10 + 6), node_id=1)
            self.store.record_decode_end(i, time=float(i * 10 + 20))
            self.store.set_request_tokens(i, prefill_tokens=100, decode_tokens=50)

        self.assertEqual(self.store.get_completed_count(), 10)

    def test_print_summary_no_crash(self):
        """Verify print_summary doesn't crash with completed requests."""
        for i in range(5):
            self.store.record_request_arrival(i, time=float(i * 10))
            self.store.record_prefill_start(i, time=float(i * 10 + 1), node_id=0)
            self.store.record_prefill_end(i, time=float(i * 10 + 5))
            self.store.record_decode_start(i, time=float(i * 10 + 6), node_id=1)
            self.store.record_decode_end(i, time=float(i * 10 + 20))
            self.store.set_request_tokens(i, prefill_tokens=100, decode_tokens=50)

        # Capture stdout to verify it doesn't crash
        captured = StringIO()
        sys.stdout = captured
        try:
            self.store.print_summary()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("完成请求数", output)

    def test_print_summary_empty(self):
        """Verify print_summary handles empty store."""
        captured = StringIO()
        sys.stdout = captured
        try:
            self.store.print_summary()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("没有完成的请求", output)

    def test_finalize_no_crash(self):
        """Verify finalize doesn't crash."""
        self.store.finalize()


if __name__ == '__main__':
    unittest.main()
