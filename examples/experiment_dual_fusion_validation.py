#!/usr/bin/env python3
"""实验: Dual Fusion Factor 验证 (DistLMSim vs vLLM on H100)

使用 prefill_fusion_factor 和 decode_fusion_factor 分别校正 prefill 和 decode 阶段，
同时测试 colocated 和 disaggregated 两种部署模式。

vLLM benchmark 结果 (GPU5 H100, Llama-2-13B, QPS=5, prefill=512, decode=128):
  TTFT P50: 39.28 ms
  TBT P50:  16.36 ms
  E2E P50:  2110.04 ms
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distlmsim.config import (
    ModelConfig, DeviceSKUConfig, NVLinkConfig, RDMAConfig,
    NetworkTopologyConfig, SimulationConfig, MetricsConfig,
)
from distlmsim.types import DeviceSKUType, RDMAProtocolType
from distlmsim.context import SimContext
from distlmsim.execution.execution_time_predictor import HighFidelityPredictor
from distlmsim.metrics.metrics_store import MetricsStore
from main import DisaggregatedSimulator, ColocatedSimulator

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PROFILING_BASE = os.path.join(os.path.dirname(__file__), "..", "data", "profiling", "compute")

VLLM_GROUND_TRUTH = {
    "ttft_p50_ms": 39.28,
    "tbt_p50_ms": 16.36,
    "e2e_p50_ms": 2110.04,
}


def run_simulator(prefill_fusion: float, decode_fusion: float, mode: str = "colocated"):
    """Run DistLMSim with dual fusion factors."""
    device = DeviceSKUConfig(
        device_type=DeviceSKUType.H100,
        fp16_tflops=990.0, memory_gb=81.0, memory_bandwidth_gbps=3350.0,
    )
    model = ModelConfig(
        model_name="Llama-2-13b-chat-hf", num_layers=40,
        num_q_heads=40, num_kv_heads=40, embedding_dim=5120,
    )

    base = os.path.join(PROFILING_BASE, "h100", "Meta", "Llama-2-13b-chat-hf")
    hifi = HighFidelityPredictor(
        model, device,
        os.path.join(os.path.dirname(__file__), "..", "data", "profiling"),
        prefill_fusion_factor=prefill_fusion,
        decode_fusion_factor=decode_fusion,
    )
    hifi._attn_df = pd.read_csv(os.path.join(base, "attention.csv"))
    if "num_tensor_parallel_workers" in hifi._attn_df.columns:
        hifi._attn_df = hifi._attn_df[hifi._attn_df["num_tensor_parallel_workers"] == 1]
    hifi._mlp_df = pd.read_csv(os.path.join(base, "mlp.csv"))
    hifi._expert_df = None
    hifi._cache = {}

    nvlink = NVLinkConfig(bandwidth_gbps=900.0)
    rdma = RDMAConfig(protocol=RDMAProtocolType.ROCE_V2, bandwidth_gbps=400.0)
    network = NetworkTopologyConfig(nvlink=nvlink, rdma=rdma)

    config = SimulationConfig(seed=42, time_limit_s=30)
    config.disaggregated.enabled = (mode == "disaggregated")
    config.disaggregated.prefill_batch_size = 4
    config.disaggregated.decode_batch_size = 32
    config.disaggregated.gpu_memory_utilization = 0.90
    config.disaggregated.enable_chunked_prefill = True
    config.disaggregated.prefill_chunk_size = 2048
    config.request.qps = 5
    config.request.prefill_length = 512
    config.request.decode_length = 128
    config.request.length_distribution = "fixed"

    ms = MetricsStore(config.metrics)
    ctx = SimContext(
        model_config=model, device_config=device, network_config=network,
        time_predictor=hifi, metrics_store=ms,
        num_gpus_per_node=1, tp_size=1,
        profiling_dir=os.path.join(os.path.dirname(__file__), "..", "data", "profiling"),
        predictor_type="high_fidelity",
    )

    if mode == "colocated":
        sim = ColocatedSimulator(ctx, config, "fcfs")
    else:
        sim = DisaggregatedSimulator(ctx, config, "fcfs", "fcfs")

    ms = sim.run()

    completed = [m for m in ms._request_metrics.values() if m.decode_end_time > 0]
    ttfts = [m.ttft for m in completed if m.ttft > 0]
    tbts = [m.tbt for m in completed if m.tbt > 0]
    e2es = [m.e2e_latency for m in completed if m.e2e_latency > 0]

    return {
        "ttft_p50_ms": float(np.median(ttfts)) if ttfts else 0,
        "tbt_p50_ms": float(np.median(tbts)) if tbts else 0,
        "e2e_p50_ms": float(np.median(e2es)) if e2es else 0,
        "num_completed": len(completed),
    }


def main():
    print("=" * 80)
    print("Dual Fusion Factor 验证: DistLMSim vs vLLM (H100)")
    print("=" * 80)
    print(f"vLLM ground truth: TTFT={VLLM_GROUND_TRUTH['ttft_p50_ms']:.2f}ms, "
          f"TBT={VLLM_GROUND_TRUTH['tbt_p50_ms']:.2f}ms, "
          f"E2E={VLLM_GROUND_TRUTH['e2e_p50_ms']:.2f}ms")

    all_results = []

    for mode in ["colocated", "disaggregated"]:
        print(f"\n{'='*40}")
        print(f"Mode: {mode}")
        print(f"{'='*40}")

        for pf in [0.80, 0.85, 0.90, 0.95, 1.00]:
            for df in [0.40, 0.50, 0.55, 0.60, 0.70]:
                result = run_simulator(pf, df, mode)
                ttft_err = abs(result["ttft_p50_ms"] - VLLM_GROUND_TRUTH["ttft_p50_ms"]) / VLLM_GROUND_TRUTH["ttft_p50_ms"] * 100
                tbt_err = abs(result["tbt_p50_ms"] - VLLM_GROUND_TRUTH["tbt_p50_ms"]) / VLLM_GROUND_TRUTH["tbt_p50_ms"] * 100
                e2e_err = abs(result["e2e_p50_ms"] - VLLM_GROUND_TRUTH["e2e_p50_ms"]) / VLLM_GROUND_TRUTH["e2e_p50_ms"] * 100
                total_err = ttft_err + tbt_err + e2e_err

                entry = {
                    "mode": mode,
                    "prefill_fusion": pf,
                    "decode_fusion": df,
                    "ttft_p50_ms": result["ttft_p50_ms"],
                    "tbt_p50_ms": result["tbt_p50_ms"],
                    "e2e_p50_ms": result["e2e_p50_ms"],
                    "ttft_error_pct": ttft_err,
                    "tbt_error_pct": tbt_err,
                    "e2e_error_pct": e2e_err,
                    "total_error_pct": total_err,
                }
                all_results.append(entry)

                if total_err < 50:
                    print(f"  pf={pf:.2f}, df={df:.2f}: "
                          f"TTFT={result['ttft_p50_ms']:.1f}ms ({ttft_err:.1f}%), "
                          f"TBT={result['tbt_p50_ms']:.1f}ms ({tbt_err:.1f}%), "
                          f"E2E={result['e2e_p50_ms']:.1f}ms ({e2e_err:.1f}%) "
                          f"[total={total_err:.1f}%]")

    # Find best per mode
    for mode in ["colocated", "disaggregated"]:
        mode_results = [r for r in all_results if r["mode"] == mode]
        best = min(mode_results, key=lambda r: r["total_error_pct"])
        print(f"\n=== Best {mode} configuration ===")
        print(f"  prefill_fusion={best['prefill_fusion']:.2f}, decode_fusion={best['decode_fusion']:.2f}")
        print(f"  TTFT: {best['ttft_p50_ms']:.2f}ms (error: {best['ttft_error_pct']:.1f}%)")
        print(f"  TBT:  {best['tbt_p50_ms']:.2f}ms (error: {best['tbt_error_pct']:.1f}%)")
        print(f"  E2E:  {best['e2e_p50_ms']:.2f}ms (error: {best['e2e_error_pct']:.1f}%)")
        print(f"  Total error: {best['total_error_pct']:.1f}%")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {"vllm_ground_truth": VLLM_GROUND_TRUTH, "results": all_results}
    json_path = os.path.join(RESULTS_DIR, "dual_fusion_validation.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {json_path}")


if __name__ == "__main__":
    main()
