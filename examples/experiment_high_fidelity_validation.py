#!/usr/bin/env python3
"""实验: High-Fidelity 模式端到端验证 (DistLMSim vs vLLM on H100)

在 H100 GPU 上部署 vLLM + Llama-2-13B，测量 ground truth TTFT/TBT/E2E，
然后使用 DistLMSim HighFidelity 模式（H100 profiling 数据 + kernel fusion 校正）
进行模拟，对比两者结果。

vLLM benchmark 结果 (GPU5 H100, Llama-2-13B, QPS=5, prefill=512, decode=128):
  TTFT P50: 39.28 ms
  TBT P50:  16.36 ms
  E2E P50:  2110.04 ms

输出: results/high_fidelity_validation.json + high_fidelity_validation.pdf
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distlmsim.config import (
    ModelConfig, DeviceSKUConfig, NVLinkConfig, RDMAConfig,
    NetworkTopologyConfig, DisaggregatedConfig, SimulationConfig,
    MetricsConfig,
)
from distlmsim.types import DeviceSKUType, RDMAProtocolType
from distlmsim.context import SimContext
from distlmsim.execution.execution_time_predictor import HighFidelityPredictor
from distlmsim.metrics.metrics_store import MetricsStore
from main import DisaggregatedSimulator

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PROFILING_BASE = os.path.join(os.path.dirname(__file__), "..", "data", "profiling", "compute")

# vLLM ground truth (GPU5 H100, Llama-2-13B, QPS=5)
VLLM_GROUND_TRUTH = {
    "ttft_p50_ms": 39.28,
    "tbt_p50_ms": 16.36,
    "e2e_p50_ms": 2110.04,
}


def run_simulator(fusion_factor: float, device_name: str = "h100"):
    """Run DistLMSim with given fusion_factor and return metrics."""
    device = DeviceSKUConfig(
        device_type=DeviceSKUType.H100,
        fp16_tflops=990.0, memory_gb=81.0, memory_bandwidth_gbps=3350.0,
    )
    model = ModelConfig(
        model_name="Llama-2-13b-chat-hf", num_layers=40,
        num_q_heads=40, num_kv_heads=40, embedding_dim=5120,
    )

    base = os.path.join(PROFILING_BASE, device_name, "Meta", "Llama-2-13b-chat-hf")
    hifi = HighFidelityPredictor(model, device, os.path.join(os.path.dirname(__file__), "..", "data", "profiling"),
                                  fusion_factor=fusion_factor)
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
    config.disaggregated.enabled = True
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

    sim = DisaggregatedSimulator(ctx, config, "fcfs", "fcfs")
    ms = sim.run()

    completed = [m for m in ms._request_metrics.values() if m.decode_end_time > 0]
    ttfts = [m.ttft for m in completed if m.ttft > 0]
    tbts = [m.tbt for m in completed if m.tbt > 0]
    e2es = [m.e2e_latency for m in completed if m.e2e_latency > 0]

    return {
        "fusion_factor": fusion_factor,
        "ttft_p50_ms": float(np.median(ttfts)),
        "tbt_p50_ms": float(np.median(tbts)),
        "e2e_p50_ms": float(np.median(e2es)),
        "num_completed": len(completed),
    }


def main():
    print("=" * 70)
    print("High-Fidelity 模式端到端验证: DistLMSim vs vLLM (H100)")
    print("=" * 70)
    print(f"vLLM ground truth: TTFT={VLLM_GROUND_TRUTH['ttft_p50_ms']:.2f}ms, "
          f"TBT={VLLM_GROUND_TRUTH['tbt_p50_ms']:.2f}ms, "
          f"E2E={VLLM_GROUND_TRUTH['e2e_p50_ms']:.2f}ms")
    print()

    # Sweep fusion factors
    fusion_factors = [0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00]
    all_results = []

    for ff in fusion_factors:
        result = run_simulator(ff)
        ttft_err = abs(result["ttft_p50_ms"] - VLLM_GROUND_TRUTH["ttft_p50_ms"]) / VLLM_GROUND_TRUTH["ttft_p50_ms"] * 100
        tbt_err = abs(result["tbt_p50_ms"] - VLLM_GROUND_TRUTH["tbt_p50_ms"]) / VLLM_GROUND_TRUTH["tbt_p50_ms"] * 100
        e2e_err = abs(result["e2e_p50_ms"] - VLLM_GROUND_TRUTH["e2e_p50_ms"]) / VLLM_GROUND_TRUTH["e2e_p50_ms"] * 100
        result["ttft_error_pct"] = ttft_err
        result["tbt_error_pct"] = tbt_err
        result["e2e_error_pct"] = e2e_err
        all_results.append(result)
        print(f"  fusion={ff:.2f}: TTFT={result['ttft_p50_ms']:.1f}ms ({ttft_err:.1f}%), "
              f"TBT={result['tbt_p50_ms']:.1f}ms ({tbt_err:.1f}%), "
              f"E2E={result['e2e_p50_ms']:.1f}ms ({e2e_err:.1f}%)")

    # Find optimal fusion factor (minimizes total error)
    for r in all_results:
        r["total_error"] = r["ttft_error_pct"] + r["tbt_error_pct"] + r["e2e_error_pct"]
    optimal = min(all_results, key=lambda r: r["total_error"])

    print(f"\n{'='*70}")
    print(f"Optimal fusion_factor: {optimal['fusion_factor']:.2f}")
    print(f"  TTFT: {optimal['ttft_p50_ms']:.2f}ms (vLLM: {VLLM_GROUND_TRUTH['ttft_p50_ms']:.2f}ms, error: {optimal['ttft_error_pct']:.1f}%)")
    print(f"  TBT:  {optimal['tbt_p50_ms']:.2f}ms (vLLM: {VLLM_GROUND_TRUTH['tbt_p50_ms']:.2f}ms, error: {optimal['tbt_error_pct']:.1f}%)")
    print(f"  E2E:  {optimal['e2e_p50_ms']:.2f}ms (vLLM: {VLLM_GROUND_TRUTH['e2e_p50_ms']:.2f}ms, error: {optimal['e2e_error_pct']:.1f}%)")

    # Save JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "vllm_ground_truth": VLLM_GROUND_TRUTH,
        "simulator_results": all_results,
        "optimal_fusion_factor": optimal["fusion_factor"],
    }
    json_path = os.path.join(RESULTS_DIR, "high_fidelity_validation.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {json_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Error vs fusion_factor
    ax = axes[0]
    ffs = [r["fusion_factor"] for r in all_results]
    ax.plot(ffs, [r["ttft_error_pct"] for r in all_results], "o-", label="TTFT", color="#E74C3C")
    ax.plot(ffs, [r["tbt_error_pct"] for r in all_results], "s-", label="TBT", color="#3498DB")
    ax.plot(ffs, [r["e2e_error_pct"] for r in all_results], "^-", label="E2E", color="#2ECC71")
    ax.axvline(x=optimal["fusion_factor"], color="gray", linestyle="--", alpha=0.5,
               label=f"Optimal ({optimal['fusion_factor']:.2f})")
    ax.set_xlabel("Kernel Fusion Factor")
    ax.set_ylabel("Error vs vLLM (%)")
    ax.set_title("Prediction Error vs Fusion Factor")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Predicted vs Measured bar chart (optimal fusion)
    ax = axes[1]
    metrics = ["TTFT P50", "TBT P50", "E2E P50"]
    sim_vals = [optimal["ttft_p50_ms"], optimal["tbt_p50_ms"], optimal["e2e_p50_ms"]]
    vllm_vals = [VLLM_GROUND_TRUTH["ttft_p50_ms"], VLLM_GROUND_TRUTH["tbt_p50_ms"], VLLM_GROUND_TRUTH["e2e_p50_ms"]]

    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x - width/2, sim_vals, width, label=f"DistLMSim (fusion={optimal['fusion_factor']:.2f})", color="#3498DB", alpha=0.7)
    bars2 = ax.bar(x + width/2, vllm_vals, width, label="vLLM (H100 ground truth)", color="#E74C3C", alpha=0.7)

    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"DistLMSim vs vLLM (fusion={optimal['fusion_factor']:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "high_fidelity_validation.pdf")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.savefig(fig_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存: {fig_path}")


if __name__ == "__main__":
    main()
