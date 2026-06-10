"""Tests for Roofline model enhancement in AnalyticalPredictor."""

import sys
import unittest

sys.path.insert(0, '.')

from distlmsim.config import ModelConfig, DeviceSKUConfig
from distlmsim.entities import ExecutionTime
from distlmsim.execution.execution_time_predictor import AnalyticalPredictor


def _make_model(
    embedding_dim: int = 4096,
    num_layers: int = 32,
    num_q_heads: int = 32,
    num_kv_heads: int = 8,
) -> ModelConfig:
    return ModelConfig(
        model_name="test",
        num_layers=num_layers,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        embedding_dim=embedding_dim,
        mlp_hidden_dim=int(embedding_dim * 8 / 3),
        num_experts=0,
        vocab_size=32000,
    )


def _make_device(
    fp16_tflops: float = 25.0,
    memory_bandwidth_gbps: float = 2039.0,
) -> DeviceSKUConfig:
    return DeviceSKUConfig(
        fp16_tflops=fp16_tflops,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
    )


class TestRooflineModel(unittest.TestCase):

    def test_roofline_time_returns_positive(self):
        """验证 Roofline 返回正数"""
        pred = AnalyticalPredictor(_make_model(), _make_device())
        t = pred._roofline_time_ms(flops=int(1e9), memory_bytes=int(1e6))
        self.assertGreater(t, 0)

    def test_compute_bound(self):
        """验证 compute-bound 场景: 大 FLOPS, 小内存"""
        pred = AnalyticalPredictor(_make_model(), _make_device())
        # 大 FLOPS, 小 bytes → compute-bound
        t1 = pred._roofline_time_ms(flops=int(50e12), memory_bytes=int(1e6))
        t2 = pred._roofline_time_ms(flops=int(100e12), memory_bytes=int(1e6))
        # 2x FLOPS → 约 2x 时间
        self.assertAlmostEqual(t2 / t1, 2.0, places=1)

    def test_memory_bound(self):
        """验证 memory-bound 场景: 小 FLOPS, 大内存"""
        pred = AnalyticalPredictor(_make_model(), _make_device())
        # 小 FLOPS, 大 bytes → memory-bound
        t1 = pred._roofline_time_ms(flops=1, memory_bytes=int(10e9))
        t2 = pred._roofline_time_ms(flops=1, memory_bytes=int(20e9))
        # 2x bytes → 约 2x 时间
        self.assertAlmostEqual(t2 / t1, 2.0, places=1)

    def test_decode_memory_bound(self):
        """验证 decode (batch=1) 是 memory-bound"""
        model = _make_model()
        pred = AnalyticalPredictor(model, _make_device())
        et = pred.get_execution_time(
            num_tokens=1, batch_size=1, kv_cache_size=2048, is_prefill=False,
        )
        # decode 时总时间应大于零
        self.assertGreater(et.total_time, 0)
        # decode 阶段 attention 有值
        self.assertGreater(et.attn_decode_time, 0)
        self.assertEqual(et.attn_prefill_time, 0.0)

    def test_prefill_compute_bound(self):
        """验证 prefill (大 batch) 是 compute-bound"""
        model = _make_model()
        pred = AnalyticalPredictor(model, _make_device())
        et = pred.get_execution_time(
            num_tokens=2048, batch_size=32, kv_cache_size=0, is_prefill=True,
        )
        self.assertGreater(et.total_time, 0)
        self.assertGreater(et.attn_prefill_time, 0)
        self.assertEqual(et.attn_decode_time, 0.0)

    def test_decode_slower_per_token_than_prefill(self):
        """验证 decode 每 token 时间比 prefill 每 token 更长 (memory-bound)"""
        model = _make_model()
        pred = AnalyticalPredictor(model, _make_device())

        prefill_et = pred.get_execution_time(
            num_tokens=2048, batch_size=32, kv_cache_size=0, is_prefill=True,
        )
        decode_et = pred.get_execution_time(
            num_tokens=1, batch_size=1, kv_cache_size=2048, is_prefill=False,
        )

        # decode per-token layer time > prefill per-token layer time
        prefill_per_token = prefill_et.layer_time / 2048
        decode_per_token = decode_et.layer_time / 1
        self.assertGreater(decode_per_token, prefill_per_token * 0.01)

    def test_larger_model_slower(self):
        """验证更大模型执行更慢"""
        small_model = _make_model(embedding_dim=1024, num_q_heads=16, num_kv_heads=4)
        large_model = _make_model(embedding_dim=8192, num_q_heads=64, num_kv_heads=8)
        device = _make_device()

        small_pred = AnalyticalPredictor(small_model, device)
        large_pred = AnalyticalPredictor(large_model, device)

        small_et = small_pred.get_execution_time(
            num_tokens=1024, batch_size=8, kv_cache_size=0, is_prefill=True,
        )
        large_et = large_pred.get_execution_time(
            num_tokens=1024, batch_size=8, kv_cache_size=0, is_prefill=True,
        )
        self.assertGreater(large_et.total_time, small_et.total_time)

    def test_faster_gpu_faster(self):
        """验证更快 GPU 执行更快"""
        model = _make_model()
        slow_gpu = _make_device(fp16_tflops=10.0, memory_bandwidth_gbps=1000.0)
        fast_gpu = _make_device(fp16_tflops=50.0, memory_bandwidth_gbps=3000.0)

        slow_pred = AnalyticalPredictor(model, slow_gpu)
        fast_pred = AnalyticalPredictor(model, fast_gpu)

        slow_et = slow_pred.get_execution_time(
            num_tokens=2048, batch_size=32, kv_cache_size=0, is_prefill=True,
        )
        fast_et = fast_pred.get_execution_time(
            num_tokens=2048, batch_size=32, kv_cache_size=0, is_prefill=True,
        )
        self.assertLess(fast_et.total_time, slow_et.total_time)

    def test_all_components_positive_prefill(self):
        """验证 prefill 所有子阶段时间为正"""
        pred = AnalyticalPredictor(_make_model(), _make_device())
        et = pred.get_execution_time(
            num_tokens=1024, batch_size=8, kv_cache_size=0, is_prefill=True,
        )
        self.assertGreater(et.attn_pre_proj_time, 0)
        self.assertGreater(et.attn_rope_time, 0)
        self.assertGreater(et.attn_kv_cache_save_time, 0)
        self.assertGreater(et.attn_prefill_time, 0)
        self.assertGreater(et.attn_post_proj_time, 0)
        self.assertGreater(et.mlp_up_proj_time, 0)
        self.assertGreater(et.mlp_act_time, 0)
        self.assertGreater(et.mlp_down_proj_time, 0)
        self.assertGreater(et.input_layernorm_time, 0)
        self.assertGreater(et.post_attention_layernorm_time, 0)
        self.assertGreater(et.add_time, 0)

    def test_all_components_positive_decode(self):
        """验证 decode 所有子阶段时间为正"""
        pred = AnalyticalPredictor(_make_model(), _make_device())
        et = pred.get_execution_time(
            num_tokens=1, batch_size=1, kv_cache_size=2048, is_prefill=False,
        )
        self.assertGreater(et.attn_pre_proj_time, 0)
        self.assertGreater(et.attn_decode_time, 0)
        self.assertGreater(et.mlp_up_proj_time, 0)
        self.assertGreater(et.mlp_down_proj_time, 0)

    def test_kv_cache_size_affects_decode(self):
        """验证更大的 KV cache 使 decode 更慢"""
        pred = AnalyticalPredictor(_make_model(), _make_device())
        et_small = pred.get_execution_time(
            num_tokens=1, batch_size=1, kv_cache_size=256, is_prefill=False,
        )
        et_large = pred.get_execution_time(
            num_tokens=1, batch_size=1, kv_cache_size=4096, is_prefill=False,
        )
        self.assertGreater(et_large.attn_decode_time, et_small.attn_decode_time)


if __name__ == '__main__':
    unittest.main()
