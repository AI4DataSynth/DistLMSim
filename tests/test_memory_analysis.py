"""Tests for distlmsim.analysis.memory_analysis module."""

import sys
import unittest

sys.path.insert(0, '.')

from distlmsim.config import ModelConfig, DeviceSKUConfig, ReplicaConfig
from distlmsim.analysis.memory_analysis import MemoryAnalyzer, MemoryBreakdown


def _make_dense_model(
    num_layers: int = 32,
    embedding_dim: int = 4096,
    num_q_heads: int = 32,
    num_kv_heads: int = 8,
    vocab_size: int = 32000,
) -> ModelConfig:
    """创建 Dense 模型配置 (类似 LLaMA3-8B)"""
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
    """创建 MoE 模型配置 (类似 Qwen3-30B-A3B)"""
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


def _make_device(memory_gb: float = 80.0) -> DeviceSKUConfig:
    return DeviceSKUConfig(memory_gb=memory_gb)


def _make_replica(tp: int = 1, ep: int = 1) -> ReplicaConfig:
    return ReplicaConfig(tensor_parallel_size=tp, expert_parallel_size=ep)


# ─── MemoryBreakdown 测试 ────────────────────────────────────────────────────

class TestMemoryBreakdown(unittest.TestCase):

    def test_to_gb(self):
        """验证字节到 GB 转换"""
        self.assertAlmostEqual(MemoryBreakdown.to_gb(1024 ** 3), 1.0)
        self.assertAlmostEqual(MemoryBreakdown.to_gb(0), 0.0)
        self.assertAlmostEqual(MemoryBreakdown.to_gb(2 * 1024 ** 3), 2.0)

    def test_summary_keys(self):
        """验证 summary 包含所有必要字段"""
        bd = MemoryBreakdown()
        s = bd.summary()
        for key in [
            "params_gb", "kv_cache_gb", "activations_gb",
            "peak_allocated_gb", "gpu_capacity_gb",
            "memory_utilization", "is_oom",
        ]:
            self.assertIn(key, s)


# ─── Dense 模型参数计算测试 ──────────────────────────────────────────────────

class TestDenseModelParams(unittest.TestCase):

    def test_total_params_llama_like(self):
        """验证 Dense 模型参数量计算 (LLaMA-8B 类似)"""
        model = _make_dense_model(
            num_layers=32, embedding_dim=4096,
            num_q_heads=32, num_kv_heads=8, vocab_size=32000,
        )
        analyzer = MemoryAnalyzer(model, _make_device())
        total = analyzer.count_total_params()
        # LLaMA-8B 约 8B 参数, 精确值取决于配置
        self.assertGreater(total, 5e9)
        self.assertLess(total, 15e9)

    def test_params_scale_with_layers(self):
        """验证参数量随层数线性增长"""
        model1 = _make_dense_model(num_layers=16)
        model2 = _make_dense_model(num_layers=32)
        a1 = MemoryAnalyzer(model1, _make_device())
        a2 = MemoryAnalyzer(model2, _make_device())
        p1 = a1.count_total_params()
        p2 = a2.count_total_params()
        # 32 层 ≈ 2x 16 层 (共享参数不变, 层参数翻倍)
        self.assertGreater(p2, p1 * 1.8)
        self.assertLess(p2, p1 * 2.2)

    def test_tp_reduces_params_per_gpu(self):
        """验证 TP 减少每 GPU 参数量"""
        model = _make_dense_model()
        a1 = MemoryAnalyzer(model, _make_device(), _make_replica(tp=1))
        a4 = MemoryAnalyzer(model, _make_device(), _make_replica(tp=4))
        bd1 = a1.analyze(batch_size=1)
        bd4 = a4.analyze(batch_size=1)
        self.assertGreater(bd1.params_bytes, bd4.params_bytes)
        # TP=4 应该约为 TP=1 的 1/4
        ratio = bd1.params_bytes / bd4.params_bytes
        self.assertAlmostEqual(ratio, 4.0, places=0)


# ─── MoE 模型参数计算测试 ────────────────────────────────────────────────────

class TestMoEModelParams(unittest.TestCase):

    def test_moe_has_more_params(self):
        """验证 MoE 模型参数多于同规模 Dense"""
        dense = _make_dense_model(
            num_layers=48, embedding_dim=2048,
            num_q_heads=32, num_kv_heads=4, vocab_size=151936,
        )
        moe = _make_moe_model()
        a_dense = MemoryAnalyzer(dense, _make_device())
        a_moe = MemoryAnalyzer(moe, _make_device())
        self.assertGreater(a_moe.count_total_params(), a_dense.count_total_params())

    def test_ep_reduces_expert_params(self):
        """验证 EP 减少每 GPU 专家参数量"""
        model = _make_moe_model()
        a1 = MemoryAnalyzer(model, _make_device(), _make_replica(ep=1))
        a4 = MemoryAnalyzer(model, _make_device(), _make_replica(ep=4))
        bd1 = a1.analyze(batch_size=1)
        bd4 = a4.analyze(batch_size=1)
        # EP=4 专家参数应约为 EP=1 的 1/4
        if bd1.expert_params_bytes > 0:
            ratio = bd1.expert_params_bytes / bd4.expert_params_bytes
            self.assertAlmostEqual(ratio, 4.0, places=0)


# ─── KV Cache 测试 ───────────────────────────────────────────────────────────

class TestKVCache(unittest.TestCase):

    def test_kv_cache_scales_with_batch(self):
        """验证 KV Cache 随 batch 线性增长"""
        model = _make_dense_model()
        analyzer = MemoryAnalyzer(model, _make_device())
        bd1 = analyzer.analyze(batch_size=1, prefill_length=1024, decode_length=256)
        bd4 = analyzer.analyze(batch_size=4, prefill_length=1024, decode_length=256)
        self.assertAlmostEqual(
            bd4.kv_cache_bytes / bd1.kv_cache_bytes, 4.0, places=5
        )

    def test_kv_cache_scales_with_seq_len(self):
        """验证 KV Cache 随序列长度线性增长"""
        model = _make_dense_model()
        analyzer = MemoryAnalyzer(model, _make_device())
        bd1 = analyzer.analyze(batch_size=1, prefill_length=512, decode_length=128)
        bd2 = analyzer.analyze(batch_size=1, prefill_length=1024, decode_length=256)
        # 2x seq len → 2x KV cache
        ratio = bd2.kv_cache_bytes / bd1.kv_cache_bytes
        self.assertAlmostEqual(ratio, 2.0, places=5)

    def test_kv_cache_tp_sharding(self):
        """验证 KV Cache 在 TP 间分片"""
        model = _make_dense_model()
        a1 = MemoryAnalyzer(model, _make_device(), _make_replica(tp=1))
        a4 = MemoryAnalyzer(model, _make_device(), _make_replica(tp=4))
        bd1 = a1.analyze(batch_size=1, prefill_length=1024, decode_length=256)
        bd4 = a4.analyze(batch_size=1, prefill_length=1024, decode_length=256)
        ratio = bd1.kv_cache_bytes / bd4.kv_cache_bytes
        self.assertAlmostEqual(ratio, 4.0, places=0)

    def test_no_kv_cache(self):
        """验证禁用 KV Cache 时为零"""
        model = _make_dense_model()
        analyzer = MemoryAnalyzer(model, _make_device())
        bd = analyzer.analyze(batch_size=4, kv_cache_enabled=False)
        self.assertEqual(bd.kv_cache_bytes, 0)

    def test_kv_cache_gqa_smaller(self):
        """验证 GQA (num_kv_heads < num_q_heads) 的 KV Cache 更小"""
        model_gqa = _make_dense_model(num_q_heads=32, num_kv_heads=4)
        model_mha = _make_dense_model(num_q_heads=32, num_kv_heads=32)
        a_gqa = MemoryAnalyzer(model_gqa, _make_device())
        a_mha = MemoryAnalyzer(model_mha, _make_device())
        bd_gqa = a_gqa.analyze(batch_size=1, prefill_length=1024, decode_length=256)
        bd_mha = a_mha.analyze(batch_size=1, prefill_length=1024, decode_length=256)
        self.assertLess(bd_gqa.kv_cache_bytes, bd_mha.kv_cache_bytes)


# ─── 内存分析综合测试 ─────────────────────────────────────────────────────────

class TestMemoryAnalysis(unittest.TestCase):

    def test_dense_analysis_basic(self):
        """验证 Dense 模型基本内存分析"""
        model = _make_dense_model()
        device = _make_device(memory_gb=80.0)
        analyzer = MemoryAnalyzer(model, device)
        bd = analyzer.analyze(batch_size=32, prefill_length=2048, decode_length=512)

        self.assertGreater(bd.params_bytes, 0)
        self.assertGreater(bd.kv_cache_bytes, 0)
        self.assertGreater(bd.peak_allocated_bytes, 0)
        self.assertGreater(bd.gpu_capacity_bytes, 0)
        self.assertGreater(bd.memory_utilization, 0)
        self.assertIsInstance(bd.is_oom, bool)
        self.assertGreater(bd.max_batch_before_oom, 0)

    def test_moe_analysis_basic(self):
        """验证 MoE 模型基本内存分析"""
        model = _make_moe_model()
        device = _make_device(memory_gb=80.0)
        analyzer = MemoryAnalyzer(model, device, _make_replica(ep=8))
        bd = analyzer.analyze(batch_size=16, prefill_length=2048, decode_length=512)

        self.assertGreater(bd.params_bytes, 0)
        self.assertGreater(bd.expert_params_bytes, 0)
        self.assertGreater(bd.routing_buffers_bytes, 0)
        self.assertGreater(bd.peak_allocated_bytes, 0)

    def test_oom_detection(self):
        """验证 OOM 检测"""
        model = _make_dense_model(
            num_layers=80, embedding_dim=8192,
            num_q_heads=64, num_kv_heads=8, vocab_size=128000,
        )
        device = _make_device(memory_gb=16.0)  # 小显存
        analyzer = MemoryAnalyzer(model, device)
        bd = analyzer.analyze(batch_size=256, prefill_length=4096, decode_length=1024)
        # 大模型 + 小显存 + 大 batch → 应该 OOM
        self.assertTrue(bd.is_oom)
        self.assertGreater(bd.memory_utilization, 1.0)

    def test_no_oom_small_model(self):
        """验证小模型不 OOM"""
        model = _make_dense_model(
            num_layers=4, embedding_dim=512,
            num_q_heads=8, num_kv_heads=2, vocab_size=8000,
        )
        device = _make_device(memory_gb=80.0)
        analyzer = MemoryAnalyzer(model, device)
        bd = analyzer.analyze(batch_size=8, prefill_length=256, decode_length=64)
        self.assertFalse(bd.is_oom)
        self.assertLess(bd.memory_utilization, 1.0)

    def test_peak_reserved_includes_fragmentation(self):
        """验证 peak_reserved > peak_allocated (含碎片)"""
        model = _make_dense_model()
        analyzer = MemoryAnalyzer(model, _make_device())
        bd = analyzer.analyze()
        self.assertGreater(bd.peak_reserved_bytes, bd.peak_allocated_bytes)
        # 约 10% 碎片
        ratio = bd.peak_reserved_bytes / bd.peak_allocated_bytes
        self.assertAlmostEqual(ratio, 1.1, places=1)

    def test_breakdown_pct_sums_to_100(self):
        """验证分解比例总和约 100%"""
        model = _make_dense_model()
        analyzer = MemoryAnalyzer(model, _make_device())
        bd = analyzer.analyze(batch_size=32)
        total_pct = sum(bd.breakdown_pct.values())
        self.assertAlmostEqual(total_pct, 100.0, places=0)

    def test_summary_dict(self):
        """验证 summary 输出格式"""
        model = _make_dense_model()
        analyzer = MemoryAnalyzer(model, _make_device())
        bd = analyzer.analyze(batch_size=8)
        s = bd.summary()
        self.assertIn("params_gb", s)
        self.assertIn("kv_cache_gb", s)
        self.assertIn("peak_allocated_gb", s)
        self.assertIn("is_oom", s)
        self.assertIn("max_batch_before_oom", s)

    def test_max_batch_before_oom(self):
        """验证最大 batch 估算"""
        model = _make_dense_model()
        device = _make_device(memory_gb=80.0)
        analyzer = MemoryAnalyzer(model, device)
        bd = analyzer.analyze(batch_size=32, prefill_length=2048, decode_length=512)
        # 80GB 的 GPU 跑 8B 模型, max batch 应该很大
        self.assertGreater(bd.max_batch_before_oom, 32)

    def test_tp4_analysis(self):
        """验证 TP=4 的完整分析"""
        model = _make_dense_model()
        device = _make_device()
        replica = _make_replica(tp=4)
        analyzer = MemoryAnalyzer(model, device, replica)
        bd = analyzer.analyze(batch_size=32)
        self.assertGreater(bd.params_bytes, 0)
        self.assertGreater(bd.kv_cache_bytes, 0)
        self.assertFalse(bd.is_oom)


if __name__ == '__main__':
    unittest.main()
