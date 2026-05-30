"""Tests for distlmsim.entities module."""

import sys
import unittest

import numpy as np

sys.path.insert(0, '.')

from distlmsim.entities import (
    Request,
    RequestStatus,
    Batch,
    BatchStage,
    ExecutionTime,
    Node,
    Replica,
)


class TestRequestCreation(unittest.TestCase):
    """Test Request creation and basic properties."""

    def test_basic_creation(self):
        """Verify basic Request creation."""
        req = Request(id=1, arrival_time=10.0, prefill_tokens=100, decode_tokens=50)
        self.assertEqual(req.id, 1)
        self.assertEqual(req.arrival_time, 10.0)
        self.assertEqual(req.prefill_tokens, 100)
        self.assertEqual(req.decode_tokens, 50)
        self.assertEqual(req.status, RequestStatus.WAITING)
        self.assertIsNone(req.scheduled_time)
        self.assertIsNone(req.prefill_start_time)
        self.assertIsNone(req.prefill_end_time)
        self.assertIsNone(req.decode_start_time)
        self.assertIsNone(req.decode_end_time)

    def test_is_complete_false_initially(self):
        """Verify request is not complete initially."""
        req = Request(id=1, arrival_time=0.0, prefill_tokens=100, decode_tokens=50)
        self.assertFalse(req.is_complete)

    def test_is_complete_true_when_completed(self):
        """Verify is_complete is True when status is COMPLETED."""
        req = Request(id=1, arrival_time=0.0, prefill_tokens=100, decode_tokens=50)
        req.status = RequestStatus.COMPLETED
        self.assertTrue(req.is_complete)

    def test_is_prefill_complete_false_initially(self):
        """Verify is_prefill_complete is False when prefill_end_time is None."""
        req = Request(id=1, arrival_time=0.0, prefill_tokens=100, decode_tokens=50)
        self.assertFalse(req.is_prefill_complete)

    def test_is_prefill_complete_true(self):
        """Verify is_prefill_complete is True when prefill_end_time is set."""
        req = Request(id=1, arrival_time=0.0, prefill_tokens=100, decode_tokens=50)
        req.prefill_end_time = 15.0
        self.assertTrue(req.is_prefill_complete)


class TestRequestLatency(unittest.TestCase):
    """Test Request latency properties."""

    def test_e2e_latency_zero_when_incomplete(self):
        """Verify e2e_latency is 0.0 when decode_end_time is None."""
        req = Request(id=1, arrival_time=10.0, prefill_tokens=100, decode_tokens=50)
        self.assertEqual(req.e2e_latency, 0.0)

    def test_e2e_latency_calculation(self):
        """Verify e2e_latency is calculated correctly."""
        req = Request(id=1, arrival_time=10.0, prefill_tokens=100, decode_tokens=50)
        req.decode_end_time = 60.0
        self.assertEqual(req.e2e_latency, 50.0)

    def test_ttft_zero_when_incomplete(self):
        """Verify ttft is 0.0 when prefill_end_time is None."""
        req = Request(id=1, arrival_time=10.0, prefill_tokens=100, decode_tokens=50)
        self.assertEqual(req.ttft, 0.0)

    def test_ttft_calculation(self):
        """Verify ttft is calculated correctly."""
        req = Request(id=1, arrival_time=10.0, prefill_tokens=100, decode_tokens=50)
        req.prefill_end_time = 25.0
        self.assertEqual(req.ttft, 15.0)


class TestBatchOperations(unittest.TestCase):
    """Test Batch operations."""

    def test_batch_creation(self):
        """Verify basic Batch creation."""
        batch = Batch(id=1, replica_id=0)
        self.assertEqual(batch.id, 1)
        self.assertEqual(batch.replica_id, 0)
        self.assertEqual(batch.batch_size, 0)
        self.assertEqual(batch.total_tokens, 0)
        self.assertTrue(batch.is_prefill_batch)

    def test_add_request(self):
        """Verify add_request adds request and tokens correctly."""
        batch = Batch(id=1, replica_id=0)
        req1 = Request(id=1, arrival_time=0.0, prefill_tokens=100, decode_tokens=50)
        req2 = Request(id=2, arrival_time=1.0, prefill_tokens=200, decode_tokens=100)

        batch.add_request(req1, 100)
        self.assertEqual(batch.batch_size, 1)
        self.assertEqual(batch.total_tokens, 100)

        batch.add_request(req2, 200)
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.total_tokens, 300)

    def test_remove_request(self):
        """Verify remove_request removes request and tokens correctly."""
        batch = Batch(id=1, replica_id=0)
        req1 = Request(id=1, arrival_time=0.0, prefill_tokens=100, decode_tokens=50)
        req2 = Request(id=2, arrival_time=1.0, prefill_tokens=200, decode_tokens=100)

        batch.add_request(req1, 100)
        batch.add_request(req2, 200)
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.total_tokens, 300)

        batch.remove_request(req1)
        self.assertEqual(batch.batch_size, 1)
        self.assertEqual(batch.total_tokens, 200)
        self.assertEqual(batch.requests[0].id, 2)

    def test_total_tokens_sum(self):
        """Verify total_tokens sums all num_tokens."""
        batch = Batch(id=1, replica_id=0)
        for i in range(5):
            req = Request(id=i, arrival_time=float(i), prefill_tokens=10, decode_tokens=5)
            batch.add_request(req, 10 * (i + 1))

        # 10 + 20 + 30 + 40 + 50 = 150
        self.assertEqual(batch.total_tokens, 150)
        self.assertEqual(batch.batch_size, 5)


class TestBatchStage(unittest.TestCase):
    """Test BatchStage."""

    def test_duration_without_execution_time(self):
        """Verify duration is 0.0 when execution_time is None."""
        batch = Batch(id=1, replica_id=0)
        stage = BatchStage(id=1, batch=batch, stage_id=0)
        self.assertEqual(stage.duration, 0.0)

    def test_duration_with_execution_time(self):
        """Verify duration uses execution_time.total_time."""
        batch = Batch(id=1, replica_id=0)
        exec_time = ExecutionTime(
            attn_prefill_time=1.0,
            mlp_up_proj_time=2.0,
            mlp_down_proj_time=1.0,
        )
        stage = BatchStage(id=1, batch=batch, stage_id=0, execution_time=exec_time)
        self.assertGreater(stage.duration, 0.0)
        self.assertEqual(stage.duration, exec_time.total_time)


class TestExecutionTime(unittest.TestCase):
    """Test ExecutionTime properties."""

    def test_default_all_zeros(self):
        """Verify all fields default to 0.0."""
        et = ExecutionTime()
        self.assertEqual(et.attn_pre_proj_time, 0.0)
        self.assertEqual(et.mlp_up_proj_time, 0.0)
        self.assertEqual(et.tensor_parallel_comm_time, 0.0)
        self.assertEqual(et.total_time, 0.0)

    def test_attention_time_property(self):
        """Verify attention_time sums all attention sub-fields."""
        et = ExecutionTime(
            attn_pre_proj_time=1.0,
            attn_rope_time=0.5,
            attn_kv_cache_save_time=0.3,
            attn_prefill_time=2.0,
            attn_decode_time=0.0,
            attn_post_proj_time=0.8,
        )
        expected = 1.0 + 0.5 + 0.3 + 2.0 + 0.0 + 0.8
        self.assertAlmostEqual(et.attention_time, expected)

    def test_mlp_time_property(self):
        """Verify mlp_time sums all MLP sub-fields."""
        et = ExecutionTime(
            mlp_up_proj_time=3.0,
            mlp_act_time=0.5,
            mlp_down_proj_time=2.5,
        )
        self.assertAlmostEqual(et.mlp_time, 6.0)

    def test_comm_time_property(self):
        """Verify comm_time sums all communication sub-fields."""
        et = ExecutionTime(
            tensor_parallel_comm_time=1.0,
            pipeline_parallel_comm_time=2.0,
            expert_parallel_comm_time=0.5,
        )
        self.assertAlmostEqual(et.comm_time, 3.5)

    def test_total_time_property(self):
        """Verify total_time includes layer_time + comm + overhead."""
        et = ExecutionTime(
            attn_prefill_time=1.0,
            mlp_up_proj_time=2.0,
            input_layernorm_time=0.1,
            post_attention_layernorm_time=0.1,
            add_time=0.05,
            tensor_parallel_comm_time=0.5,
            eplb_overhead_time=0.2,
            cpu_overhead_time=0.1,
        )
        expected = (
            et.attention_time + et.mlp_time + et.expert_mlp_time
            + et.input_layernorm_time + et.post_attention_layernorm_time
            + et.add_time
            + et.comm_time
            + et.eplb_overhead_time
            + et.cpu_overhead_time
        )
        self.assertAlmostEqual(et.total_time, expected)

    def test_total_time_positive(self):
        """Verify total_time is positive when components are set."""
        et = ExecutionTime(
            attn_prefill_time=5.0,
            mlp_up_proj_time=3.0,
            mlp_down_proj_time=3.0,
        )
        self.assertGreater(et.total_time, 0.0)

    def test_layer_time_property(self):
        """Verify layer_time sums compute sub-fields."""
        et = ExecutionTime(
            attn_prefill_time=2.0,
            mlp_up_proj_time=3.0,
            expert_mlp_time=1.0,
            input_layernorm_time=0.1,
            post_attention_layernorm_time=0.1,
            add_time=0.05,
        )
        expected = 2.0 + 3.0 + 1.0 + 0.1 + 0.1 + 0.05
        self.assertAlmostEqual(et.layer_time, expected)


class TestNode(unittest.TestCase):
    """Test Node entity."""

    def test_basic_creation(self):
        """Verify basic Node creation."""
        node = Node(id=0, node_sku_name="A800_DGX")
        self.assertEqual(node.id, 0)
        self.assertEqual(node.node_sku_name, "A800_DGX")
        self.assertEqual(node.num_gpus, 8)
        self.assertEqual(node.role, "mixed")
        self.assertEqual(len(node.gpu_ids), 0)

    def test_gpu_management(self):
        """Verify GPU IDs can be set."""
        node = Node(id=0, node_sku_name="A800_DGX", gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(len(node.gpu_ids), 8)
        self.assertEqual(node.gpu_ids[0], 0)
        self.assertEqual(node.gpu_ids[-1], 7)

    def test_replica_ids(self):
        """Verify current_replica_ids is empty by default."""
        node = Node(id=0, node_sku_name="A800_DGX")
        self.assertEqual(node.current_replica_ids, [])


class TestReplica(unittest.TestCase):
    """Test Replica entity."""

    def test_basic_creation(self):
        """Verify basic Replica creation."""
        replica = Replica(id=0, model_name="Qwen3-30B-A3B")
        self.assertEqual(replica.id, 0)
        self.assertEqual(replica.model_name, "Qwen3-30B-A3B")
        self.assertEqual(replica.tensor_parallel_size, 1)
        self.assertEqual(replica.num_pipeline_stages, 1)
        self.assertEqual(replica.expert_parallel_size, 1)

    def test_with_parallelism(self):
        """Verify Replica with parallelism settings."""
        replica = Replica(
            id=0,
            model_name="Qwen3-30B-A3B",
            tensor_parallel_size=4,
            num_pipeline_stages=2,
            expert_parallel_size=8,
            node_ids=[0, 1],
            gpu_ids=[0, 1, 2, 3, 8, 9, 10, 11],
        )
        self.assertEqual(replica.tensor_parallel_size, 4)
        self.assertEqual(replica.num_pipeline_stages, 2)
        self.assertEqual(len(replica.node_ids), 2)
        self.assertEqual(len(replica.gpu_ids), 8)


if __name__ == '__main__':
    unittest.main()
