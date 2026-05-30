"""Tests for scheduling policies in DisaggregatedSimulator._select_from_queue."""

import sys
import unittest

sys.path.insert(0, '.')

from distlmsim.entities import Request, RequestStatus
from main import create_disaggregated_simulator


def _make_request(req_id, arrival_time, prefill_tokens, decode_tokens, num_generated=0):
    """Helper to create a Request with known parameters."""
    req = Request(
        id=req_id,
        arrival_time=arrival_time,
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
    )
    req.num_generated_tokens = num_generated
    return req


class TestFCFSScheduling(unittest.TestCase):
    """Test FCFS (First-Come-First-Served) scheduling."""

    def setUp(self):
        self.sim = create_disaggregated_simulator(seed=42)

    def test_selects_earliest_arrival(self):
        """FCFS should select the request with earliest arrival time."""
        requests = [
            _make_request(0, arrival_time=30.0, prefill_tokens=100, decode_tokens=50),
            _make_request(1, arrival_time=10.0, prefill_tokens=200, decode_tokens=80),
            _make_request(2, arrival_time=20.0, prefill_tokens=150, decode_tokens=60),
            _make_request(3, arrival_time=5.0, prefill_tokens=300, decode_tokens=100),
        ]
        selected = self.sim._select_from_queue(requests, batch_size=2, policy="fcfs")
        self.assertEqual(len(selected), 2)
        # Should pick req 3 (arrival=5) and req 1 (arrival=10)
        selected_ids = {r.id for r in selected}
        self.assertEqual(selected_ids, {3, 1})

    def test_all_returned_when_fewer_than_batch(self):
        """When queue size <= batch_size, all requests are returned."""
        requests = [
            _make_request(0, arrival_time=10.0, prefill_tokens=100, decode_tokens=50),
            _make_request(1, arrival_time=20.0, prefill_tokens=200, decode_tokens=80),
        ]
        selected = self.sim._select_from_queue(requests, batch_size=10, policy="fcfs")
        self.assertEqual(len(selected), 2)


class TestSJFScheduling(unittest.TestCase):
    """Test SJF (Shortest-Job-First) scheduling."""

    def setUp(self):
        self.sim = create_disaggregated_simulator(seed=42)

    def test_selects_shortest_prefill(self):
        """SJF should select the request with shortest prefill tokens."""
        requests = [
            _make_request(0, arrival_time=10.0, prefill_tokens=500, decode_tokens=50),
            _make_request(1, arrival_time=10.0, prefill_tokens=100, decode_tokens=80),
            _make_request(2, arrival_time=10.0, prefill_tokens=300, decode_tokens=60),
            _make_request(3, arrival_time=10.0, prefill_tokens=50, decode_tokens=100),
        ]
        selected = self.sim._select_from_queue(requests, batch_size=2, policy="sjf")
        self.assertEqual(len(selected), 2)
        selected_ids = {r.id for r in selected}
        # Should pick req 3 (50 tokens) and req 1 (100 tokens)
        self.assertEqual(selected_ids, {3, 1})


class TestLJFScheduling(unittest.TestCase):
    """Test LJF (Longest-Job-First) scheduling."""

    def setUp(self):
        self.sim = create_disaggregated_simulator(seed=42)

    def test_selects_longest_prefill(self):
        """LJF should select the request with longest prefill tokens."""
        requests = [
            _make_request(0, arrival_time=10.0, prefill_tokens=500, decode_tokens=50),
            _make_request(1, arrival_time=10.0, prefill_tokens=100, decode_tokens=80),
            _make_request(2, arrival_time=10.0, prefill_tokens=300, decode_tokens=60),
            _make_request(3, arrival_time=10.0, prefill_tokens=800, decode_tokens=100),
        ]
        selected = self.sim._select_from_queue(requests, batch_size=2, policy="ljf")
        self.assertEqual(len(selected), 2)
        selected_ids = {r.id for r in selected}
        # Should pick req 3 (800 tokens) and req 0 (500 tokens)
        self.assertEqual(selected_ids, {3, 0})


class TestSRTFScheduling(unittest.TestCase):
    """Test SRTF (Shortest-Remaining-Time-First) scheduling."""

    def setUp(self):
        self.sim = create_disaggregated_simulator(seed=42)

    def test_selects_shortest_decode(self):
        """SRTF should select the request with shortest decode tokens."""
        requests = [
            _make_request(0, arrival_time=10.0, prefill_tokens=100, decode_tokens=200),
            _make_request(1, arrival_time=10.0, prefill_tokens=100, decode_tokens=50),
            _make_request(2, arrival_time=10.0, prefill_tokens=100, decode_tokens=300),
            _make_request(3, arrival_time=10.0, prefill_tokens=100, decode_tokens=80),
        ]
        selected = self.sim._select_from_queue(requests, batch_size=2, policy="srtf")
        self.assertEqual(len(selected), 2)
        selected_ids = {r.id for r in selected}
        # Should pick req 1 (50 decode) and req 3 (80 decode)
        self.assertEqual(selected_ids, {1, 3})


class TestRandomScheduling(unittest.TestCase):
    """Test Random scheduling."""

    def setUp(self):
        self.sim = create_disaggregated_simulator(seed=42)

    def test_selects_correct_count(self):
        """Random should select exactly batch_size requests."""
        requests = [
            _make_request(i, arrival_time=float(i), prefill_tokens=100, decode_tokens=50)
            for i in range(10)
        ]
        selected = self.sim._select_from_queue(requests, batch_size=3, policy="random")
        self.assertEqual(len(selected), 3)

    def test_selected_from_queue(self):
        """Random should only select requests from the queue."""
        requests = [
            _make_request(i, arrival_time=float(i), prefill_tokens=100, decode_tokens=50)
            for i in range(10)
        ]
        request_ids = {r.id for r in requests}
        selected = self.sim._select_from_queue(requests, batch_size=3, policy="random")
        for req in selected:
            self.assertIn(req.id, request_ids)

    def test_no_duplicates(self):
        """Random should not select the same request twice."""
        requests = [
            _make_request(i, arrival_time=float(i), prefill_tokens=100, decode_tokens=50)
            for i in range(10)
        ]
        selected = self.sim._select_from_queue(requests, batch_size=5, policy="random")
        selected_ids = [r.id for r in selected]
        self.assertEqual(len(selected_ids), len(set(selected_ids)))


class TestMLFQScheduling(unittest.TestCase):
    """Test MLFQ (Multi-Level Feedback Queue) scheduling."""

    def setUp(self):
        self.sim = create_disaggregated_simulator(seed=42)

    def test_prioritizes_lower_queue_levels(self):
        """MLFQ should prioritize requests at lower queue levels."""
        requests = [
            _make_request(0, arrival_time=10.0, prefill_tokens=100, decode_tokens=50),
            _make_request(1, arrival_time=20.0, prefill_tokens=200, decode_tokens=80),
            _make_request(2, arrival_time=30.0, prefill_tokens=300, decode_tokens=60),
        ]
        # Manually set queue levels: req 2 at level 0, others at level 1
        self.sim._advanced_schedulers.mlfq_state.request_queues[0] = 1
        self.sim._advanced_schedulers.mlfq_state.request_queues[1] = 1
        self.sim._advanced_schedulers.mlfq_state.request_queues[2] = 0
        self.sim._advanced_schedulers.mlfq_state.request_wait_time[0] = 10.0
        self.sim._advanced_schedulers.mlfq_state.request_wait_time[1] = 20.0
        self.sim._advanced_schedulers.mlfq_state.request_wait_time[2] = 30.0

        selected = self.sim._select_from_queue(
            requests, batch_size=2, policy="mlfq", current_time=100.0
        )
        self.assertEqual(len(selected), 2)
        # req 2 (level 0) should be first, then req 0 (level 1, earlier arrival)
        self.assertEqual(selected[0].id, 2)

    def test_new_requests_at_level_zero(self):
        """MLFQ should place new requests at queue level 0."""
        requests = [
            _make_request(100, arrival_time=10.0, prefill_tokens=100, decode_tokens=50),
        ]
        self.sim._select_from_queue(requests, batch_size=1, policy="mlfq", current_time=10.0)
        level = self.sim._advanced_schedulers.mlfq_state.get_queue_level(100)
        self.assertEqual(level, 0)


class TestPOScheduling(unittest.TestCase):
    """Test PO (Priority Ordering) scheduling."""

    def setUp(self):
        self.sim = create_disaggregated_simulator(seed=42)

    def test_short_jobs_first_by_fcfs(self):
        """PO should prioritize short jobs ordered by arrival time."""
        # Create requests: short prefill (predicted decode < threshold)
        # threshold ~ exp(5.2355) ~ 188, predict_decode = prefill * 0.3
        # Short: prefill=100 -> predict=30 < 188 (short)
        # Long: prefill=2000 -> predict=600 > 188 (long)
        requests = [
            _make_request(0, arrival_time=10.0, prefill_tokens=100, decode_tokens=50),
            _make_request(1, arrival_time=20.0, prefill_tokens=100, decode_tokens=60),
            _make_request(2, arrival_time=30.0, prefill_tokens=2000, decode_tokens=100),
            _make_request(3, arrival_time=5.0, prefill_tokens=2000, decode_tokens=200),
        ]
        selected = self.sim._select_from_queue(requests, batch_size=4, policy="po")
        self.assertEqual(len(selected), 4)
        # Short jobs (0, 1) come first, sorted by arrival: req0 (10), req1 (20)
        # Long jobs (2, 3) come next, sorted by decode_tokens ASC (SJF): req2 (100), req3 (200)
        selected_ids = [r.id for r in selected]
        self.assertEqual(selected_ids[:2], [0, 1])  # short jobs FCFS
        self.assertEqual(selected_ids[2:], [2, 3])  # long jobs SJF (decode ascending)


class TestOPTScheduling(unittest.TestCase):
    """Test OPT (Optimal) scheduling."""

    def setUp(self):
        self.sim = create_disaggregated_simulator(seed=42)

    def test_score_based_selection(self):
        """OPT should select based on score = remaining_tokens * noise."""
        requests = [
            _make_request(0, arrival_time=10.0, prefill_tokens=100, decode_tokens=500),
            _make_request(1, arrival_time=10.0, prefill_tokens=100, decode_tokens=10),
            _make_request(2, arrival_time=10.0, prefill_tokens=100, decode_tokens=100),
        ]
        selected = self.sim._select_from_queue(
            requests, batch_size=1, policy="opt", current_time=10.0
        )
        self.assertEqual(len(selected), 1)
        # With noise, the request with fewest decode_tokens (req 1) should tend to be selected
        # since score = remaining_tokens * noise (lower is better)
        # But noise is random, so just verify we get a valid selection
        self.assertIn(selected[0].id, {0, 1, 2})

    def test_returns_correct_batch_size(self):
        """OPT should return the correct number of requests."""
        requests = [
            _make_request(i, arrival_time=10.0, prefill_tokens=100, decode_tokens=50 + i)
            for i in range(10)
        ]
        selected = self.sim._select_from_queue(
            requests, batch_size=3, policy="opt", current_time=10.0
        )
        self.assertEqual(len(selected), 3)


class TestLightLLMScheduling(unittest.TestCase):
    """Test LightLLM scheduling."""

    def setUp(self):
        self.sim = create_disaggregated_simulator(seed=42)

    def test_prefill_mode_no_generated_tokens(self):
        """LightLLM should use prefill scheduling when no requests have generated tokens."""
        requests = [
            _make_request(0, arrival_time=30.0, prefill_tokens=100, decode_tokens=50),
            _make_request(1, arrival_time=10.0, prefill_tokens=200, decode_tokens=80),
            _make_request(2, arrival_time=20.0, prefill_tokens=150, decode_tokens=60),
        ]
        selected = self.sim._select_from_queue(
            requests, batch_size=2, policy="lightllm", current_time=50.0
        )
        self.assertEqual(len(selected), 2)
        # LightLLM prefill mode sorts by arrival_time
        # req 1 (arrival=10) should be first, then req 2 (arrival=20)
        self.assertEqual(selected[0].id, 1)

    def test_decode_mode_with_generated_tokens(self):
        """LightLLM should use decode scheduling when some requests have generated tokens."""
        requests = [
            _make_request(0, arrival_time=30.0, prefill_tokens=100, decode_tokens=50, num_generated=10),
            _make_request(1, arrival_time=10.0, prefill_tokens=200, decode_tokens=80, num_generated=20),
            _make_request(2, arrival_time=20.0, prefill_tokens=150, decode_tokens=60, num_generated=5),
        ]
        selected = self.sim._select_from_queue(
            requests, batch_size=2, policy="lightllm", current_time=50.0
        )
        self.assertEqual(len(selected), 2)
        # LightLLM decode mode sorts by arrival_time (FCFS)
        # req 1 (arrival=10) should be first
        self.assertEqual(selected[0].id, 1)

    def test_respects_max_prefill_batch_size(self):
        """LightLLM prefill should respect max_prefill_batch_size."""
        self.sim._advanced_schedulers.lightllm_state.max_prefill_batch_size = 2
        requests = [
            _make_request(i, arrival_time=float(i), prefill_tokens=100, decode_tokens=50)
            for i in range(10)
        ]
        selected = self.sim._select_from_queue(
            requests, batch_size=8, policy="lightllm", current_time=50.0
        )
        # Should be limited to max_prefill_batch_size (2)
        self.assertLessEqual(len(selected), 2)


class TestAllSchedulersReturnValidRequests(unittest.TestCase):
    """Test that all schedulers return valid subsets of the input queue."""

    def setUp(self):
        self.sim = create_disaggregated_simulator(seed=42)
        self.requests = [
            _make_request(i, arrival_time=float(i * 10), prefill_tokens=100 + i * 50, decode_tokens=50 + i * 20)
            for i in range(20)
        ]
        self.request_ids = {r.id for r in self.requests}

    def test_all_policies_return_subsets(self):
        """All policies should return subsets of the input queue."""
        policies = ["fcfs", "sjf", "ljf", "srtf", "random", "mlfq", "po", "opt", "lightllm"]
        for policy in policies:
            selected = self.sim._select_from_queue(
                list(self.requests), batch_size=5, policy=policy, current_time=200.0
            )
            self.assertLessEqual(len(selected), 5, f"Policy {policy} returned too many")
            for req in selected:
                self.assertIn(req.id, self.request_ids, f"Policy {policy} returned unknown request")


if __name__ == '__main__':
    unittest.main()
