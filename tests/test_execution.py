"""Tests for distlmsim.execution module."""

import sys
import unittest

sys.path.insert(0, '.')

from distlmsim.config import ModelConfig, DeviceSKUConfig
from distlmsim.entities import ExecutionTime
from distlmsim.execution.execution_time_predictor import (
    AnalyticalPredictor,
    RandomForestPredictor,
)


class TestAnalyticalPredictor(unittest.TestCase):
    """Test AnalyticalPredictor."""

    def setUp(self):
        self.model_config = ModelConfig(
            model_name="test-model",
            num_layers=48,
            num_q_heads=32,
            num_kv_heads=4,
            embedding_dim=2048,
            mlp_hidden_dim=0,
            num_experts=0,
        )
        self.device_config = DeviceSKUConfig()
        self.predictor = AnalyticalPredictor(self.model_config, self.device_config)

    def test_returns_execution_time(self):
        """Verify predictor returns an ExecutionTime object."""
        result = self.predictor.get_execution_time(
            num_tokens=128,
            batch_size=4,
            kv_cache_size=0,
            is_prefill=True,
        )
        self.assertIsInstance(result, ExecutionTime)

    def test_positive_values_prefill(self):
        """Verify all execution time components are positive for prefill."""
        result = self.predictor.get_execution_time(
            num_tokens=128,
            batch_size=4,
            kv_cache_size=0,
            is_prefill=True,
        )
        self.assertGreater(result.total_time, 0.0)
        self.assertGreater(result.attention_time, 0.0)
        self.assertGreater(result.mlp_time, 0.0)
        self.assertGreater(result.attn_pre_proj_time, 0.0)
        self.assertGreater(result.mlp_up_proj_time, 0.0)

    def test_positive_values_decode(self):
        """Verify all execution time components are positive for decode."""
        result = self.predictor.get_execution_time(
            num_tokens=4,
            batch_size=4,
            kv_cache_size=1024,
            is_prefill=False,
        )
        self.assertGreater(result.total_time, 0.0)
        self.assertGreater(result.attention_time, 0.0)
        self.assertGreater(result.mlp_time, 0.0)

    def test_prefill_uses_prefill_time_not_decode(self):
        """Verify prefill sets attn_prefill_time and not attn_decode_time."""
        result = self.predictor.get_execution_time(
            num_tokens=128,
            batch_size=4,
            kv_cache_size=0,
            is_prefill=True,
        )
        self.assertGreater(result.attn_prefill_time, 0.0)
        self.assertEqual(result.attn_decode_time, 0.0)

    def test_decode_uses_decode_time_not_prefill(self):
        """Verify decode sets attn_decode_time and not attn_prefill_time."""
        result = self.predictor.get_execution_time(
            num_tokens=4,
            batch_size=4,
            kv_cache_size=1024,
            is_prefill=False,
        )
        self.assertGreater(result.attn_decode_time, 0.0)
        self.assertEqual(result.attn_prefill_time, 0.0)

    def test_time_increases_with_tokens(self):
        """Verify execution time increases with more tokens."""
        result_small = self.predictor.get_execution_time(
            num_tokens=32,
            batch_size=4,
            kv_cache_size=0,
            is_prefill=True,
        )
        result_large = self.predictor.get_execution_time(
            num_tokens=512,
            batch_size=4,
            kv_cache_size=0,
            is_prefill=True,
        )
        self.assertGreater(result_large.total_time, result_small.total_time)

    def test_total_time_sums_correctly(self):
        """Verify total_time = layer_time + comm_time + overhead."""
        result = self.predictor.get_execution_time(
            num_tokens=128,
            batch_size=4,
            kv_cache_size=0,
            is_prefill=True,
        )
        expected_total = (
            result.layer_time
            + result.comm_time
            + result.eplb_overhead_time
            + result.cpu_overhead_time
        )
        self.assertAlmostEqual(result.total_time, expected_total, places=6)


class TestRandomForestPredictor(unittest.TestCase):
    """Test RandomForestPredictor (fallback to Analytical)."""

    def test_fallback_to_analytical(self):
        """Verify RF predictor falls back to analytical model."""
        model_config = ModelConfig()
        device_config = DeviceSKUConfig()
        predictor = RandomForestPredictor(model_config, device_config, profiling_dir="data/profiling")
        result = predictor.get_execution_time(
            num_tokens=128,
            batch_size=4,
            kv_cache_size=0,
            is_prefill=True,
        )
        self.assertIsInstance(result, ExecutionTime)
        self.assertGreater(result.total_time, 0.0)


if __name__ == '__main__':
    unittest.main()
