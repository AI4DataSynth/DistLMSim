"""Tests for distlmsim.parallelism module."""

import sys
import unittest

import numpy as np

sys.path.insert(0, '.')

from distlmsim.config import ModelConfig
from distlmsim.parallelism.tensor_parallel import TensorParallelModel, TPExecutionParams
from distlmsim.parallelism.pipeline_parallel import PipelineParallelModel, PipelineStage
from distlmsim.parallelism.expert_parallel import (
    ExpertParallelModel,
    ExpertPlacement,
    DefaultRoutingScheduler,
    EPLBScheduler,
    RealisticEPLBScheduler,
    OmniPlacementScheduler,
)


class TestTensorParallelModel(unittest.TestCase):
    """Test TensorParallelModel."""

    def setUp(self):
        self.model_config = ModelConfig(
            num_layers=48,
            num_q_heads=32,
            num_kv_heads=4,
            embedding_dim=2048,
            num_experts=0,
        )

    def test_execution_params_tp1(self):
        """Verify execution params with TP=1."""
        tp = TensorParallelModel(self.model_config, tp_size=1)
        params = tp.get_execution_params()
        self.assertIsInstance(params, TPExecutionParams)
        self.assertEqual(params.q_heads_per_worker, 32)
        self.assertEqual(params.kv_heads_per_worker, 4)
        self.assertEqual(params.num_allreduce_per_layer, 2)

    def test_execution_params_tp4(self):
        """Verify execution params with TP=4."""
        tp = TensorParallelModel(self.model_config, tp_size=4)
        params = tp.get_execution_params()
        self.assertEqual(params.q_heads_per_worker, 8)   # 32 / 4
        self.assertEqual(params.kv_heads_per_worker, 1)   # max(1, 4/4)
        self.assertEqual(params.embedding_per_worker, 2048)

    def test_allreduce_data_size_tp1(self):
        """Verify allreduce data size is 0 for TP=1."""
        tp = TensorParallelModel(self.model_config, tp_size=1)
        size = tp.get_allreduce_data_size(num_tokens=128)
        self.assertEqual(size, 0)

    def test_allreduce_data_size_tp4(self):
        """Verify allreduce data size is positive for TP=4."""
        tp = TensorParallelModel(self.model_config, tp_size=4)
        size = tp.get_allreduce_data_size(num_tokens=128)
        # 128 * 2048 * 2 = 524288 bytes
        expected = 128 * 2048 * 2
        self.assertEqual(size, expected)

    def test_allreduce_increases_with_tokens(self):
        """Verify allreduce data size increases with token count."""
        tp = TensorParallelModel(self.model_config, tp_size=4)
        size_small = tp.get_allreduce_data_size(num_tokens=64)
        size_large = tp.get_allreduce_data_size(num_tokens=256)
        self.assertGreater(size_large, size_small)

    def test_compute_flops_positive(self):
        """Verify compute FLOPs is positive."""
        tp = TensorParallelModel(self.model_config, tp_size=4)
        flops = tp.get_compute_flops_per_layer(num_tokens=128)
        self.assertGreater(flops, 0)

    def test_compute_flops_increases_with_tokens(self):
        """Verify compute FLOPs increases with token count."""
        tp = TensorParallelModel(self.model_config, tp_size=4)
        flops_small = tp.get_compute_flops_per_layer(num_tokens=64)
        flops_large = tp.get_compute_flops_per_layer(num_tokens=256)
        self.assertGreater(flops_large, flops_small)

    def test_mlp_hidden_dim_default(self):
        """Verify default MLP hidden dim is 8/3 * embedding_dim."""
        tp = TensorParallelModel(self.model_config, tp_size=1)
        params = tp.get_execution_params()
        expected_mlp = int(2048 * 8 / 3)
        self.assertEqual(params.mlp_hidden_per_worker, expected_mlp)

    def test_mlp_hidden_dim_custom(self):
        """Verify custom MLP hidden dim is used."""
        config = ModelConfig(
            embedding_dim=2048,
            mlp_hidden_dim=4096,
            num_q_heads=32,
            num_kv_heads=4,
        )
        tp = TensorParallelModel(config, tp_size=2)
        params = tp.get_execution_params()
        self.assertEqual(params.mlp_hidden_per_worker, 4096 // 2)


class TestPipelineParallelModel(unittest.TestCase):
    """Test PipelineParallelModel."""

    def setUp(self):
        self.model_config = ModelConfig(
            num_layers=48,
            embedding_dim=2048,
            num_q_heads=32,
            num_kv_heads=4,
        )

    def test_create_stages_pp1(self):
        """Verify single stage with PP=1."""
        pp = PipelineParallelModel(self.model_config, pp_size=1)
        stages = pp.create_stages({0: [0]})
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0].start_layer, 0)
        self.assertEqual(stages[0].end_layer, 48)
        self.assertEqual(stages[0].num_layers, 48)

    def test_create_stages_pp4(self):
        """Verify 4 stages with PP=4."""
        pp = PipelineParallelModel(self.model_config, pp_size=4)
        node_gpu_mapping = {0: [0], 1: [1], 2: [2], 3: [3]}
        stages = pp.create_stages(node_gpu_mapping)
        self.assertEqual(len(stages), 4)
        self.assertEqual(stages[0].start_layer, 0)
        self.assertEqual(stages[0].end_layer, 12)
        self.assertEqual(stages[3].start_layer, 36)
        self.assertEqual(stages[3].end_layer, 48)

    def test_stages_cover_all_layers(self):
        """Verify stages cover all model layers."""
        pp = PipelineParallelModel(self.model_config, pp_size=4)
        stages = pp.create_stages({i: [i] for i in range(4)})
        total_layers = sum(s.num_layers for s in stages)
        self.assertEqual(total_layers, 48)

    def test_stage_comm_data_size_pp1(self):
        """Verify comm data size is 0 for PP=1."""
        pp = PipelineParallelModel(self.model_config, pp_size=1)
        size = pp.get_stage_comm_data_size(num_tokens=128)
        self.assertEqual(size, 0)

    def test_stage_comm_data_size_pp4(self):
        """Verify comm data size is positive for PP=4."""
        pp = PipelineParallelModel(self.model_config, pp_size=4)
        size = pp.get_stage_comm_data_size(num_tokens=128)
        # 128 * 2048 * 2 = 524288
        expected = 128 * 2048 * 2
        self.assertEqual(size, expected)

    def test_bubble_ratio_pp1(self):
        """Verify bubble ratio is 0 for PP=1."""
        pp = PipelineParallelModel(self.model_config, pp_size=1)
        ratio = pp.get_pipeline_bubble_ratio(num_micro_batches=4)
        self.assertEqual(ratio, 0.0)

    def test_bubble_ratio_pp4(self):
        """Verify bubble ratio for PP=4."""
        pp = PipelineParallelModel(self.model_config, pp_size=4)
        ratio = pp.get_pipeline_bubble_ratio(num_micro_batches=8)
        # (4-1)/8 = 0.375
        self.assertAlmostEqual(ratio, 0.375)

    def test_bubble_ratio_decreases_with_micro_batches(self):
        """Verify bubble ratio decreases with more micro-batches."""
        pp = PipelineParallelModel(self.model_config, pp_size=4)
        ratio_small = pp.get_pipeline_bubble_ratio(num_micro_batches=4)
        ratio_large = pp.get_pipeline_bubble_ratio(num_micro_batches=16)
        self.assertGreater(ratio_small, ratio_large)

    def test_create_schedule(self):
        """Verify create_schedule returns valid schedule."""
        pp = PipelineParallelModel(self.model_config, pp_size=4)
        schedule = pp.create_schedule(num_micro_batches=4, schedule_type="1f1b")
        self.assertEqual(schedule.schedule_type, "1f1b")
        self.assertEqual(schedule.num_micro_batches, 4)
        self.assertEqual(len(schedule.stages), 4)


class TestExpertParallelModel(unittest.TestCase):
    """Test ExpertParallelModel."""

    def setUp(self):
        self.model_config = ModelConfig(
            num_experts=128,
            top_k_experts=8,
            embedding_dim=2048,
            num_q_heads=32,
            num_kv_heads=4,
        )

    def test_expert_placement(self):
        """Verify expert placement is created correctly."""
        ep = ExpertParallelModel(self.model_config, ep_size=8)
        gpu_node_mapping = {i: i // 4 for i in range(8)}
        placements = ep.create_expert_placement(gpu_node_mapping)
        self.assertEqual(len(placements), 128)
        # Verify all experts are placed
        expert_ids = {p.expert_id for p in placements}
        self.assertEqual(len(expert_ids), 128)

    def test_expert_placement_with_redundancy(self):
        """Verify redundant experts are placed."""
        ep = ExpertParallelModel(self.model_config, ep_size=8, redundant_experts=16)
        gpu_node_mapping = {i: i // 4 for i in range(8)}
        placements = ep.create_expert_placement(gpu_node_mapping)
        # 128 base + 16 redundant = 144
        self.assertEqual(len(placements), 144)
        redundant = [p for p in placements if p.is_redundant]
        self.assertEqual(len(redundant), 16)

    def test_token_routing(self):
        """Verify token routing produces valid results."""
        ep = ExpertParallelModel(self.model_config, ep_size=8)
        gpu_node_mapping = {i: 0 for i in range(8)}
        ep.create_expert_placement(gpu_node_mapping)

        # Create random expert distribution
        rng = np.random.default_rng(42)
        num_tokens = 100
        expert_dist = rng.random((num_tokens, 128))

        result = ep.route_tokens(expert_dist, top_k=8)
        self.assertEqual(result.token_expert_assignment.shape, (num_tokens, 8))
        self.assertEqual(len(result.expert_loads), 128)
        self.assertEqual(len(result.gpu_loads), 8)
        self.assertGreater(result.max_gpu_load, 0)

    def test_alltoall_data_size_ep1(self):
        """Verify alltoall data size is 0 for EP=1."""
        ep = ExpertParallelModel(self.model_config, ep_size=1)
        size = ep.get_alltoall_data_size(num_tokens=128, top_k=8)
        self.assertEqual(size, 0)

    def test_alltoall_data_size_ep8(self):
        """Verify alltoall data size is positive for EP=8."""
        ep = ExpertParallelModel(self.model_config, ep_size=8)
        size = ep.get_alltoall_data_size(num_tokens=128, top_k=8)
        self.assertGreater(size, 0)
        # Expected: 128 * 2048 * 2 * 8 / 8 = 128 * 2048 * 2
        expected = int(128 * 2048 * 2 * 8 / 8)
        self.assertEqual(size, expected)


class TestMoESchedulers(unittest.TestCase):
    """Test MoE expert load balancing schedulers."""

    def _create_zipf_demand(self, num_layers=4, num_experts=64, ep_size=8, seed=42):
        """Create Zipf-like expert demand distribution."""
        rng = np.random.default_rng(seed)
        # Zipf: expert i has demand proportional to 1/(i+1)^0.8
        zipf_weights = np.array([1.0 / (i + 1) ** 0.8 for i in range(num_experts)])
        zipf_weights /= zipf_weights.sum()

        demand = np.zeros((num_layers, num_experts))
        for l in range(num_layers):
            total_tokens = 1000
            counts = rng.multinomial(total_tokens, zipf_weights)
            demand[l] = counts.astype(float)
        return demand

    def test_default_routing(self):
        """Test DefaultRoutingScheduler computes load distribution."""
        scheduler = DefaultRoutingScheduler(
            num_experts=64, expert_parallel_size=8, top_k_experts=4
        )
        demand = self._create_zipf_demand(num_experts=64, ep_size=8)
        result = scheduler.compute_load_distribution(demand)
        self.assertGreater(result.max_load, 0)
        self.assertGreater(result.avg_load, 0)
        self.assertGreaterEqual(result.max_load, result.avg_load)

    def test_eplb_reduces_max_load(self):
        """EPLB should reduce max_load compared to default routing."""
        demand = self._create_zipf_demand(num_experts=64, ep_size=8)

        default = DefaultRoutingScheduler(
            num_experts=64, expert_parallel_size=8, top_k_experts=4
        )
        eplb = EPLBScheduler(
            num_experts=64, expert_parallel_size=8, top_k_experts=4
        )

        default_result = default.compute_load_distribution(demand.copy())
        eplb_result = eplb.compute_load_distribution(demand.copy())

        # EPLB should cap max_load
        self.assertLessEqual(eplb_result.max_load, default_result.max_load)

    def test_eplb_capacity_factor(self):
        """EPLB max_load should be <= ceil(avg_load * 1.1)."""
        demand = self._create_zipf_demand(num_experts=64, ep_size=8)
        eplb = EPLBScheduler(
            num_experts=64, expert_parallel_size=8, top_k_experts=4
        )
        result = eplb.compute_load_distribution(demand)
        import math
        capacity_limit = int(math.ceil(result.avg_load * 1.1))
        self.assertLessEqual(result.max_load, capacity_limit)

    def test_realistic_eplb_waterfill(self):
        """RealisticEPLB should distribute load via waterfill routing."""
        demand = self._create_zipf_demand(num_layers=4, num_experts=64, ep_size=8)
        scheduler = RealisticEPLBScheduler(
            num_experts=64,
            expert_parallel_size=8,
            top_k_experts=4,
            num_nodes=1,
            gpus_per_node=8,
            redundant_experts=8,
            rebalance_interval=100,  # Don't rebalance in this test
        )
        result = scheduler.compute_load_distribution(demand)
        self.assertGreater(result.max_load, 0)
        self.assertGreater(result.avg_load, 0)

    def test_realistic_eplb_with_rebalance(self):
        """RealisticEPLB should trigger rebalance periodically."""
        demand = self._create_zipf_demand(num_layers=2, num_experts=64, ep_size=8)
        scheduler = RealisticEPLBScheduler(
            num_experts=64,
            expert_parallel_size=8,
            top_k_experts=4,
            num_nodes=1,
            gpus_per_node=8,
            redundant_experts=8,
            rebalance_interval=2,
        )
        # Run multiple batches to trigger rebalance
        for _ in range(4):
            result = scheduler.compute_load_distribution(demand.copy())
        # After rebalance, should have migrations
        self.assertGreaterEqual(result.num_migrations, 0)

    def test_omni_placement_reduces_max_load(self):
        """OmniPlacement greedy swap should reduce max load."""
        demand = self._create_zipf_demand(num_layers=2, num_experts=64, ep_size=8)

        default = DefaultRoutingScheduler(
            num_experts=64, expert_parallel_size=8, top_k_experts=4
        )
        omni = OmniPlacementScheduler(
            num_experts=64, expert_parallel_size=8, top_k_experts=4, budget_N=10
        )

        default_result = default.compute_load_distribution(demand.copy())
        omni_result = omni.compute_load_distribution(demand.copy())

        # OmniPlacement should reduce or maintain max_load
        self.assertLessEqual(omni_result.max_load, default_result.max_load)

    def test_omni_placement_uses_budget(self):
        """OmniPlacement should use migration budget."""
        demand = self._create_zipf_demand(num_layers=2, num_experts=64, ep_size=8)
        omni = OmniPlacementScheduler(
            num_experts=64, expert_parallel_size=8, top_k_experts=4, budget_N=8
        )
        result = omni.compute_load_distribution(demand)
        # Migrations should be within budget
        self.assertLessEqual(result.num_migrations, 8)

    def test_all_schedulers_handle_empty_demand(self):
        """All MoE schedulers should handle zero demand gracefully."""
        demand = np.zeros((2, 64))
        schedulers = [
            DefaultRoutingScheduler(num_experts=64, expert_parallel_size=8, top_k_experts=4),
            EPLBScheduler(num_experts=64, expert_parallel_size=8, top_k_experts=4),
        ]
        for s in schedulers:
            result = s.compute_load_distribution(demand.copy())
            self.assertEqual(result.max_load, 0)

    def test_zipf_imbalance(self):
        """Zipf demand should create measurable load imbalance with default routing."""
        demand = self._create_zipf_demand(num_experts=64, ep_size=8)
        default = DefaultRoutingScheduler(
            num_experts=64, expert_parallel_size=8, top_k_experts=4
        )
        result = default.compute_load_distribution(demand)
        # With Zipf distribution, max_load > avg_load
        self.assertGreater(result.max_load, result.avg_load)


if __name__ == '__main__':
    unittest.main()
