#!/usr/bin/env python3
"""E2E Validation: Batch Size > 1 Continuous Batching (A800, Qwen3-30B-A3B)

在高 QPS 下验证 ColocatedSimulator 的连续批处理实现：
- QPS 从低到高 (5, 10, 20, 50)，使 decode batch size > 1
- 使用 chunked prefill + correction factors
- 对比 vLLM 实测的 TTFT/TBT/E2E 和 batch size 分布

用法:
  python3 examples/experiment_e2e_batch_scaling.py
"""

import json
import os
import sys
import copy

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distlmsim.config import (
    ModelConfig, DeviceSKUConfig, NVLinkConfig, RDMAConfig,
    NetworkTopologyConfig, SimulationConfig, MetricsConfig,
    RequestGeneratorConfig, DisaggregatedConfig,
)
from distlmsim.context import SimContext
from distlmsim.metrics import MetricsStore
from distlmsim.types import RDMAProtocolType
from distlmsim.execution.execution_time_predictor import FusedKernelPredictor
from main import ColocatedSimulator

PROFILING_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "profiling")

# Correction factors (from single-request calibration)
# CUDA graph eliminates a fixed fraction of kernel launch overhead
# This ratio is approximately constant across batch sizes
DECODE_CORRECTION = 0.308  # correction = actual_time / profiling_time
PREFILL_CORRECTION_MAP = {256: 0.130, 512: 0.118, 1024: 0.111}


class _CorrectedPredictor:
    """Wraps a predictor to apply phase-specific correction factors."""

    def __init__(self, base_predictor, prefill_corr_map):
        self._base = base_predictor
        self._pf_map = prefill_corr_map

    def _get_pf_corr(self, num_tokens):
        known = sorted(self._pf_map.keys())
        if num_tokens <= known[0]:
            return self._pf_map[known[0]]
        if num_tokens >= known[-1]:
            return self._pf_map[known[-1]]
        for i in range(len(known) - 1):
            if known[i] <= num_tokens <= known[i + 1]:
                t = (num_tokens - known[i]) / (known[i + 1] - known[i])
                return self._pf_map[known[i]] * (1 - t) + self._pf_map[known[i + 1]] * t
        return 0.119

    def get_execution_time(self, num_tokens, batch_size, kv_cache_size, is_prefill=False):
        et = self._base.get_execution_time(num_tokens, batch_size, kv_cache_size, is_prefill)
        et = copy.deepcopy(et)
        c = self._get_pf_corr(num_tokens) if is_prefill else DECODE_CORRECTION
        et.attn_prefill_time *= c
        et.attn_decode_time *= c
        et.attn_pre_proj_time *= c
        et.attn_post_proj_time *= c
        et.attn_rope_time *= c
        et.attn_input_reshape_time *= c
        et.attn_kv_cache_save_time *= c
        et.attn_output_reshape_time *= c
        et.mlp_up_proj_time *= c
        et.mlp_act_time *= c
        et.mlp_down_proj_time *= c
        et.input_layernorm_time *= c
        et.post_attention_layernorm_time *= c
        et.add_time *= c
        et.expert_mlp_time *= c
        et.eplb_overhead_time *= c
        et.cpu_overhead_time *= c
        et.tensor_parallel_comm_time *= c
        et.pipeline_parallel_comm_time *= c
        et.expert_parallel_comm_time *= c
        return et


def run_config(pf, qps, decode_length=128, time_limit_s=30.0, seed=42,
               chunk_size=128):
    """运行单个高 QPS 配置，返回指标和 batch size 统计。"""
    model = ModelConfig(
        model_name="Qwen3-30B-A3B",
        num_layers=48, num_q_heads=32, num_kv_heads=4,
        embedding_dim=2048, num_experts=128, top_k_experts=8,
    )
    device = DeviceSKUConfig()
    nvlink = NVLinkConfig(bandwidth_gbps=600.0)
    rdma = RDMAConfig(protocol=RDMAProtocolType.ROCE_V2, bandwidth_gbps=200.0)
    network = NetworkTopologyConfig(nvlink=nvlink, rdma=rdma)

    base_predictor = FusedKernelPredictor(model, device, PROFILING_DIR)
    predictor = _CorrectedPredictor(base_predictor, PREFILL_CORRECTION_MAP)

    ms = MetricsStore(MetricsConfig(enable_detailed_logging=False))
    ctx = SimContext(
        model_config=model, device_config=device, network_config=network,
        num_gpus_per_node=4, tp_size=4,
        time_predictor=predictor, metrics_store=ms,
        profiling_dir=PROFILING_DIR, predictor_type="fused_kernel",
    )

    config = SimulationConfig(
        seed=seed, time_limit_s=time_limit_s,
        request=RequestGeneratorConfig(
            qps=qps, prefill_length=pf, decode_length=decode_length,
            length_distribution="fixed",
        ),
        disaggregated=DisaggregatedConfig(
            enabled=False,
            prefill_batch_size=4, decode_batch_size=64,
            gpu_memory_utilization=0.9,
            enable_chunked_prefill=True,
            prefill_chunk_size=chunk_size,
        ),
        metrics=MetricsConfig(enable_detailed_logging=False),
    )

    sim = ColocatedSimulator(ctx, config, "fcfs")
    ms = sim.run()

    completed = [m for m in ms._request_metrics.values() if m.decode_end_time > 0]
    if not completed:
        return {"completed": 0}

    ttfts = [m.ttft for m in completed if m.ttft > 0]
    tbts = [m.tbt for m in completed if m.tbt > 0]
    e2es = [m.e2e_latency for m in completed if m.e2e_latency > 0]

    # Batch size statistics from decode steps
    # We need to infer from the simulation - check per-request concurrency
    arrival_times = [m.arrival_time for m in completed]
    decode_starts = [m.decode_start_time for m in completed if m.decode_start_time > 0]
    decode_ends = [m.decode_end_time for m in completed if m.decode_end_time > 0]

    # Estimate average concurrent decode requests (batch size)
    if decode_starts and decode_ends:
        # Sample at decode step boundaries
        all_events = []
        for m in completed:
            if m.decode_start_time > 0 and m.decode_end_time > 0:
                all_events.append((m.decode_start_time, 1))
                all_events.append((m.decode_end_time, -1))
        all_events.sort()
        concurrent = 0
        max_concurrent = 0
        concurrent_samples = []
        for t, delta in all_events:
            concurrent += delta
            max_concurrent = max(max_concurrent, concurrent)
            concurrent_samples.append(concurrent)
        avg_batch = np.mean(concurrent_samples) if concurrent_samples else 1.0
    else:
        avg_batch = 1.0
        max_concurrent = 1

    return {
        "completed": len(completed),
        "ttft_p50": float(np.percentile(ttfts, 50)) if ttfts else 0,
        "tbt_p50": float(np.percentile(tbts, 50)) if tbts else 0,
        "e2e_p50": float(np.percentile(e2es, 50)) if e2es else 0,
        "ttft_p99": float(np.percentile(ttfts, 99)) if ttfts else 0,
        "tbt_p99": float(np.percentile(tbts, 99)) if tbts else 0,
        "avg_decode_batch_size": round(avg_batch, 1),
        "max_decode_batch_size": max_concurrent,
    }


def main():
    print()
    print("=" * 85)
    print("  E2E Batch Size Scaling: ColocatedSimulator (chunked prefill)")
    print("  A800 × 4 (TP=4), Qwen3-30B-A3B, correction factors applied")
    print("=" * 85)
    print()
    print(f"  Correction: decode={DECODE_CORRECTION}, prefill={PREFILL_CORRECTION_MAP}")
    print(f"  Chunked prefill: chunk_size=128")
    print()

    configs = [
        # (prefill_length, qps, time_limit_s, label)
        (512, 5,   30.0, "pf=512, qps=5"),
        (512, 10,  30.0, "pf=512, qps=10"),
        (512, 20,  30.0, "pf=512, qps=20"),
        (512, 50,  30.0, "pf=512, qps=50"),
        (512, 100, 20.0, "pf=512, qps=100"),
    ]

    results = []

    for pf, qps, tlim, label in configs:
        print(f"  {label:<25} ...", end=" ", flush=True)
        r = run_config(pf=pf, qps=qps, time_limit_s=tlim)
        if r["completed"] == 0:
            print("FAILED (no completed requests)")
            continue
        results.append({"config": label, "pf": pf, "qps": qps, **r})
        print(f"done={r['completed']:>3}, "
              f"TTFT_P50={r['ttft_p50']:>8.1f}ms, "
              f"TBT_P50={r['tbt_p50']:>6.1f}ms, "
              f"E2E_P50={r['e2e_p50']:>9.1f}ms, "
              f"avg_BS={r['avg_decode_batch_size']:.1f}, "
              f"max_BS={r['max_decode_batch_size']}")

    # Summary table
    print()
    print("=" * 85)
    print(f"  {'Config':<25} {'Done':>5} {'TTFT_P50':>10} {'TBT_P50':>10} "
          f"{'E2E_P50':>10} {'Avg BS':>7} {'Max BS':>7}")
    print("  " + "-" * 80)
    for r in results:
        print(f"  {r['config']:<25} {r['completed']:>5} "
              f"{r['ttft_p50']:>9.1f} {r['tbt_p50']:>9.1f} "
              f"{r['e2e_p50']:>9.1f} "
              f"{r['avg_decode_batch_size']:>6.1f} "
              f"{r['max_decode_batch_size']:>6}")
    print("=" * 85)

    # Save JSON
    out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                            "e2e_batch_scaling.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "E2E Batch Size Scaling (A800, Qwen3 MoE, colocated)",
            "correction_factors": {
                "decode": DECODE_CORRECTION,
                "prefill_map": PREFILL_CORRECTION_MAP,
            },
            "chunk_size": 128,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
