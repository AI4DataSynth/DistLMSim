"""End-to-end tests for DisaggregatedSimulator."""

import sys
import unittest

sys.path.insert(0, '.')

from main import create_disaggregated_simulator
from distlmsim.metrics.metrics_store import MetricsStore


class TestDisaggregatedSimulatorE2E(unittest.TestCase):
    """Test DisaggregatedSimulator end-to-end."""

    def test_create_and_run(self):
        """Verify create_disaggregated_simulator runs successfully."""
        sim = create_disaggregated_simulator(
            num_requests=10,
            qps=20.0,
            prefill_length=128,
            decode_length=32,
            prefill_batch_size=4,
            decode_batch_size=8,
            time_limit_s=10.0,
            seed=42,
            length_distribution="fixed",
        )
        metrics = sim.run()
        self.assertIsInstance(metrics, MetricsStore)

    def test_completed_requests(self):
        """Verify simulator produces completed requests."""
        sim = create_disaggregated_simulator(
            qps=20.0,
            prefill_length=128,
            decode_length=32,
            prefill_batch_size=4,
            decode_batch_size=8,
            time_limit_s=10.0,
            seed=42,
            length_distribution="fixed",
        )
        metrics = sim.run()
        completed = metrics.get_completed_count()
        self.assertGreater(completed, 0)

    def test_positive_ttft(self):
        """Verify all completed requests have positive TTFT."""
        sim = create_disaggregated_simulator(
            qps=20.0,
            prefill_length=128,
            decode_length=32,
            prefill_batch_size=4,
            decode_batch_size=8,
            time_limit_s=10.0,
            seed=42,
            length_distribution="fixed",
        )
        metrics = sim.run()
        completed = [
            m for m in metrics._request_metrics.values()
            if m.decode_end_time > 0
        ]
        self.assertGreater(len(completed), 0)
        for m in completed:
            self.assertGreater(m.ttft, 0.0, f"Request {m.request_id} has non-positive TTFT")

    def test_positive_e2e_latency(self):
        """Verify all completed requests have positive E2E latency."""
        sim = create_disaggregated_simulator(
            qps=20.0,
            prefill_length=128,
            decode_length=32,
            prefill_batch_size=4,
            decode_batch_size=8,
            time_limit_s=10.0,
            seed=42,
            length_distribution="fixed",
        )
        metrics = sim.run()
        completed = [
            m for m in metrics._request_metrics.values()
            if m.decode_end_time > 0
        ]
        for m in completed:
            self.assertGreater(m.e2e_latency, 0.0)

    def test_positive_throughput(self):
        """Verify throughput is positive."""
        sim = create_disaggregated_simulator(
            qps=20.0,
            prefill_length=128,
            decode_length=32,
            prefill_batch_size=4,
            decode_batch_size=8,
            time_limit_s=10.0,
            seed=42,
            length_distribution="fixed",
        )
        metrics = sim.run()
        completed = [
            m for m in metrics._request_metrics.values()
            if m.decode_end_time > 0 and m.decode_tokens > 0
        ]
        total_decode_tokens = sum(m.decode_tokens for m in completed)
        wall_time = max(m.decode_end_time for m in completed)
        first_arrival = min(m.arrival_time for m in completed)
        effective_time_s = (wall_time - first_arrival) / 1000.0
        throughput = total_decode_tokens / effective_time_s
        self.assertGreater(throughput, 0.0)


class TestDisaggregatedSimulatorSchedulers(unittest.TestCase):
    """Test DisaggregatedSimulator with different schedulers."""

    def _run_with_scheduler(self, policy, seed=42):
        """Helper to run simulator with a given scheduler policy."""
        sim = create_disaggregated_simulator(
            qps=20.0,
            prefill_length=128,
            decode_length=32,
            prefill_batch_size=4,
            decode_batch_size=8,
            time_limit_s=10.0,
            seed=seed,
            prefill_schedule_policy=policy,
            decode_schedule_policy=policy,
            length_distribution="fixed",
        )
        return sim.run()

    def test_fcfs_scheduler(self):
        """Verify FCFS scheduler completes requests."""
        metrics = self._run_with_scheduler("fcfs")
        self.assertGreater(metrics.get_completed_count(), 0)

    def test_sjf_scheduler(self):
        """Verify SJF scheduler completes requests."""
        metrics = self._run_with_scheduler("sjf")
        self.assertGreater(metrics.get_completed_count(), 0)

    def test_mlfq_scheduler(self):
        """Verify MLFQ scheduler completes requests."""
        metrics = self._run_with_scheduler("mlfq")
        self.assertGreater(metrics.get_completed_count(), 0)

    def test_ljf_scheduler(self):
        """Verify LJF scheduler completes requests."""
        metrics = self._run_with_scheduler("ljf")
        self.assertGreater(metrics.get_completed_count(), 0)

    def test_srtf_scheduler(self):
        """Verify SRTF scheduler completes requests."""
        metrics = self._run_with_scheduler("srtf")
        self.assertGreater(metrics.get_completed_count(), 0)

    def test_random_scheduler(self):
        """Verify Random scheduler completes requests."""
        metrics = self._run_with_scheduler("random")
        self.assertGreater(metrics.get_completed_count(), 0)

    def test_po_scheduler(self):
        """Verify PO scheduler completes requests."""
        metrics = self._run_with_scheduler("po")
        self.assertGreater(metrics.get_completed_count(), 0)

    def test_opt_scheduler(self):
        """Verify OPT scheduler completes requests."""
        metrics = self._run_with_scheduler("opt")
        self.assertGreater(metrics.get_completed_count(), 0)

    def test_lightllm_scheduler(self):
        """Verify LightLLM scheduler completes requests."""
        metrics = self._run_with_scheduler("lightllm")
        self.assertGreater(metrics.get_completed_count(), 0)


class TestDisaggregatedSimulatorDeterminism(unittest.TestCase):
    """Test DisaggregatedSimulator determinism."""

    def test_deterministic_with_same_seed(self):
        """Verify same seed produces same results."""
        results = []
        for _ in range(2):
            sim = create_disaggregated_simulator(
                qps=20.0,
                prefill_length=128,
                decode_length=32,
                prefill_batch_size=4,
                decode_batch_size=8,
                time_limit_s=5.0,
                seed=42,
                length_distribution="fixed",
            )
            metrics = sim.run()
            completed = [
                m for m in metrics._request_metrics.values()
                if m.decode_end_time > 0
            ]
            results.append((
                len(completed),
                sum(m.e2e_latency for m in completed),
            ))

        self.assertEqual(results[0][0], results[1][0])
        self.assertAlmostEqual(results[0][1], results[1][1], places=2)


if __name__ == '__main__':
    unittest.main()
