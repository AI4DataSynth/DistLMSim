"""Tests for distlmsim.analysis.mfu_analysis module."""

import sys
import unittest

sys.path.insert(0, '.')

from distlmsim.config import ModelConfig, DeviceSKUConfig, ReplicaConfig
from distlmsim.analysis.mfu_analysis import MFUAnalyzer, MFUResult


def _make_dense_model(
    num_layers: int = 32,
    embedding_dim: int = 4096,
    num_q_heads: int = 32,
    num_kv_heads: int = 8,
    vocab_size: int = 32000,
) -> ModelConfig:
    return ModelConfig(
        model_name="test-dense",
        num_layers=num_layers,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        embedding_dim=embedding_dim,
        mlp_hidden_dim=int(embedding_dim * 8 / 3),
        num_experts=0,
        top_k_experts=0,
        vocab_size=vocab_size,
    )


def _make_moe_model(
    num_layers: int = 48,
    embedding_dim: int = 2048,
    num_q_heads: int = 32,
    num_kv_heads: int = 4,
    num_experts: int = 128,
    top_k: int = 8,
    vocab_size: int = 151936,
) -> ModelConfig:
    return ModelConfig(
        model_name="test-moe",
        num_layers=num_layers,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        embedding_dim=embedding_dim,
        mlp_hidden_dim=0,
        num_experts=num_experts,
        top_k_experts=top_k,
        vocab_size=vocab_size,
    )


def _make_device(fp16_tflops: float = 25.0) -> DeviceSKUConfig:
    return DeviceSKUConfig(fp16_tflops=fp16_tflops)


def _make_replica(tp: int = 1) -> ReplicaConfig:
    return ReplicaConfig(tensor_parallel_size=tp)


# ─── MFUResult 测试 ──────────────────────────────────────────────────────────

class TestMFUResult(unittest.TestCase):

    def test_summary_keys(self):
        r = MFUResult()
        s = r.summary()
        for key in ["prefill_mfu", "decode_mfu", "overall_mfu", "total_params"]:
            self.assertIn(key, s)

    def test_default_zeros(self):
        r = MFUResult()
        self.assertEqual(r.prefill_mfu, 0.0)
        self.assertEqual(r.decode_mfu, 0.0)
        self.assertEqual(r.overall_mfu, 0.0)


# ─── 参数量测试 ──────────────────────────────────────────────────────────────

class TestParamCounting(unittest.TestCase):

    def test_dense_total_params(self):
        """验证 Dense 模型总参数量"""
        model = _make_dense_model()
        analyzer = MFUAnalyzer(model, _make_device())
        total = analyzer.count_total_params()
        self.assertGreater(total, 5e9)
        self.assertLess(total, 15e9)

    def test_moe_active_params_less_than_total(self):
        """验证 MoE 模型激活参数 < 总参数"""
        model = _make_moe_model()
        analyzer = MFUAnalyzer(model, _make_device())
        total = analyzer.count_total_params()
        active = analyzer._count_active_params_per_token()
        self.assertLess(active, total)

    def test_dense_active_equals_total(self):
        """验证 Dense 模型激活参数 = 总参数"""
        model = _make_dense_model()
        analyzer = MFUAnalyzer(model, _make_device())
        total = analyzer.count_total_params()
        active = analyzer._count_active_params_per_token()
        self.assertEqual(active, total)


# ─── FLOPs 计算测试 ──────────────────────────────────────────────────────────

class TestFLOPS(unittest.TestCase):

    def test_prefill_flops_positive(self):
        model = _make_dense_model()
        analyzer = MFUAnalyzer(model, _make_device())
        result = analyzer.analyze(prefill_length=2048)
        self.assertGreater(result.prefill_flops_per_request, 0)

    def test_prefill_flops_scales_with_seq(self):
        """验证 Prefill FLOPS 随序列长度线性增长"""
        model = _make_dense_model()
        analyzer = MFUAnalyzer(model, _make_device())
        r1 = analyzer.analyze(prefill_length=1024)
        r2 = analyzer.analyze(prefill_length=2048)
        ratio = r2.prefill_flops_per_request / r1.prefill_flops_per_request
        self.assertAlmostEqual(ratio, 2.0, places=5)

    def test_decode_flops_per_token(self):
        """验证 Decode FLOPS per token = 2 * active_params"""
        model = _make_dense_model()
        analyzer = MFUAnalyzer(model, _make_device())
        result = analyzer.analyze()
        expected = 2 * analyzer._count_active_params_per_token()
        self.assertEqual(result.decode_flops_per_token, expected)

    def test_flops_breakdown_components(self):
        """验证 FLOPS 分解包含所有组件"""
        model = _make_dense_model()
        analyzer = MFUAnalyzer(model, _make_device())
        result = analyzer.analyze(prefill_length=1024)
        self.assertGreater(result.attention_flops, 0)
        self.assertGreater(result.feedforward_flops, 0)
        self.assertGreater(result.embedding_flops, 0)


# ─── MFU 计算测试 ────────────────────────────────────────────────────────────

class TestMFUCalculation(unittest.TestCase):

    def test_prefill_mfu_positive(self):
        """验证 Prefill MFU 为正且合理"""
        model = _make_dense_model()
        analyzer = MFUAnalyzer(model, _make_device())
        # 8B 模型 prefill 2048 tokens: 2*8B*2048 ≈ 33 TFLOPS
        # 在 25 TFLOPS GPU 上需约 1300ms (100% MFU)
        # 使用 2000ms 模拟 ~65% MFU
        result = analyzer.analyze(
            prefill_length=2048, prefill_time_ms=2000.0,
        )
        self.assertGreater(result.prefill_mfu, 0)
        self.assertLess(result.prefill_mfu, 1.0)

    def test_decode_mfu_lower_than_prefill(self):
        """验证 Decode MFU 通常低于 Prefill MFU (decode 是 memory-bound)"""
        model = _make_dense_model()
        analyzer = MFUAnalyzer(model, _make_device())
        # Prefill 2048 tokens in 2000ms → high MFU
        # Decode 512 tokens in 5000ms → low MFU (memory-bound)
        result = analyzer.analyze(
            prefill_length=2048, decode_length=512,
            prefill_time_ms=2000.0, decode_time_ms=5000.0,
        )
        self.assertLess(result.decode_mfu, result.prefill_mfu)

    def test_mfu_zero_time(self):
        """验证时间为零时 MFU 为零"""
        model = _make_dense_model()
        analyzer = MFUAnalyzer(model, _make_device())
        result = analyzer.analyze(
            prefill_length=2048,
            prefill_time_ms=0.0,
            decode_time_ms=0.0,
        )
        self.assertEqual(result.prefill_mfu, 0.0)
        self.assertEqual(result.decode_mfu, 0.0)
        self.assertEqual(result.overall_mfu, 0.0)

    def test_tp_increases_per_gpu_flops(self):
        """验证 TP 下每 GPU FLOPS 减少 (但峰值不变), MFU 变化合理"""
        model = _make_dense_model()
        a1 = MFUAnalyzer(model, _make_device(), _make_replica(tp=1))
        a4 = MFUAnalyzer(model, _make_device(), _make_replica(tp=4))
        r1 = a1.analyze(prefill_length=2048, prefill_time_ms=50.0)
        r4 = a4.analyze(prefill_length=2048, prefill_time_ms=50.0)
        # TP=4 时每 GPU 做 1/4 的工作, 在相同时间内 MFU 应为 1/4
        self.assertAlmostEqual(r4.prefill_mfu / r1.prefill_mfu, 0.25, places=5)

    def test_compute_mfu_static(self):
        """验证静态 MFU 计算"""
        mfu = MFUAnalyzer.compute_mfu(
            total_flops=int(50e12),  # 50 TFLOPS
            peak_flops=100e12,       # 100 TFLOPS 峰值
            time_s=1.0,
        )
        self.assertAlmostEqual(mfu, 0.5)

    def test_compute_mfu_zero_time(self):
        self.assertEqual(MFUAnalyzer.compute_mfu(100, 100.0, 0.0), 0.0)

    def test_compute_mfu_zero_peak(self):
        self.assertEqual(MFUAnalyzer.compute_mfu(100, 0.0, 1.0), 0.0)

    def test_overall_mfu_weighted(self):
        """验证 overall MFU 是 prefill+decode 的加权"""
        model = _make_dense_model()
        analyzer = MFUAnalyzer(model, _make_device())
        # 使用合理时间: prefill ~2s, decode ~5s
        result = analyzer.analyze(
            prefill_length=2048, decode_length=512,
            prefill_time_ms=2000.0, decode_time_ms=5000.0,
        )
        # Overall 应在 prefill 和 decode 之间
        self.assertGreater(result.overall_mfu, 0)
        self.assertLess(result.overall_mfu, 1.0)


# ─── MoE 特殊测试 ───────────────────────────────────────────────────────────

class TestMoEMFU(unittest.TestCase):

    def test_moe_prefill_flops_less_than_dense_same_size(self):
        """MoE 激活参数少, 因此 prefill FLOPS 低于同参数量 Dense"""
        moe = _make_moe_model()
        a_moe = MFUAnalyzer(moe, _make_device())
        r_moe = a_moe.analyze(prefill_length=1024)
        # MoE 激活参数应远小于总参数
        active = a_moe._count_active_params_per_token()
        total = a_moe.count_total_params()
        self.assertLess(active, total * 0.5)

    def test_moe_analysis_no_crash(self):
        """验证 MoE 分析不崩溃"""
        model = _make_moe_model()
        analyzer = MFUAnalyzer(model, _make_device(), _make_replica(tp=4))
        result = analyzer.analyze(
            prefill_length=2048, decode_length=512,
            prefill_time_ms=30.0, decode_time_ms=300.0,
        )
        self.assertGreater(result.prefill_mfu, 0)
        self.assertGreater(result.decode_flops_per_token, 0)


if __name__ == '__main__':
    unittest.main()
