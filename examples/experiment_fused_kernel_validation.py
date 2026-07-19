#!/usr/bin/env python3
"""实验: FusedKernelPredictor 验证

验证 FusedKernelPredictor 直接从 fused_kernels_a800.csv 查表，
不需要任何 fusion factor 调参，即可精确匹配 profiling 测量值。

对比:
  - AnalyticalPredictor (Roofline): 高误差
  - HighFidelityPredictor (sub-op + fusion factor): 需要调参
  - FusedKernelPredictor (fused kernel 直接查表): 零调参，零误差

输出: results/fused_kernel_validation.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distlmsim.config import ModelConfig, DeviceSKUConfig
from distlmsim.types import DeviceSKUType
from distlmsim.execution.execution_time_predictor import (
    AnalyticalPredictor,
    HighFidelityPredictor,
    FusedKernelPredictor,
)

PROFILING_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "profiling")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

MODEL_NAME = "Llama-2-13b-chat-hf"


def main():
    print("=" * 70)
    print("FusedKernelPredictor Validation (A800 + Llama-2-13B)")
    print("=" * 70)

    model = ModelConfig(
        model_name=MODEL_NAME, num_layers=40,
        num_q_heads=40, num_kv_heads=40, embedding_dim=5120,
    )
    device = DeviceSKUConfig(
        device_type=DeviceSKUType.A800,
        fp16_tflops=312.0, memory_gb=80.0, memory_bandwidth_gbps=2039.0,
    )

    # Load fused kernel profiling data as ground truth
    csv_path = os.path.join(PROFILING_DIR, "fused_kernels_a800.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded fused_kernels_a800.csv: {len(df)} rows")

    # Create predictors
    analytical = AnalyticalPredictor(model, device)
    high_fidelity = HighFidelityPredictor(model, device, PROFILING_DIR)
    fused_kernel = FusedKernelPredictor(model, device, PROFILING_DIR)

    # Test configurations
    configs = [
        {"num_tokens": 64, "batch_size": 1, "kv_cache_size": 0, "is_prefill": True, "label": "Prefill nt=64"},
        {"num_tokens": 256, "batch_size": 1, "kv_cache_size": 0, "is_prefill": True, "label": "Prefill nt=256"},
        {"num_tokens": 512, "batch_size": 1, "kv_cache_size": 0, "is_prefill": True, "label": "Prefill nt=512"},
        {"num_tokens": 1024, "batch_size": 1, "kv_cache_size": 0, "is_prefill": True, "label": "Prefill nt=1024"},
        {"num_tokens": 2048, "batch_size": 1, "kv_cache_size": 0, "is_prefill": True, "label": "Prefill nt=2048"},
        {"num_tokens": 4096, "batch_size": 1, "kv_cache_size": 0, "is_prefill": True, "label": "Prefill nt=4096"},
        {"num_tokens": 1, "batch_size": 1, "kv_cache_size": 256, "is_prefill": False, "label": "Decode bs=1"},
        {"num_tokens": 1, "batch_size": 4, "kv_cache_size": 256, "is_prefill": False, "label": "Decode bs=4"},
        {"num_tokens": 1, "batch_size": 16, "kv_cache_size": 256, "is_prefill": False, "label": "Decode bs=16"},
        {"num_tokens": 1, "batch_size": 64, "kv_cache_size": 256, "is_prefill": False, "label": "Decode bs=64"},
    ]

    results = []
    print(f"\n{'Config':<20} {'Measured':>10} {'Analytical':>12} {'HighFidelity':>14} {'FusedKernel':>13} {'FK Error':>10}")
    print("-" * 85)

    for cfg in configs:
        # Ground truth from fused_kernels CSV (average across layers)
        phase = "prefill" if cfg["is_prefill"] else "decode"
        mask = df["phase"] == phase
        if cfg["is_prefill"]:
            mask = mask & (df["num_tokens"] == cfg["num_tokens"])
        else:
            mask = mask & (df["batch_size"] == cfg["batch_size"])
        gt_rows = df[mask]
        if len(gt_rows) == 0:
            continue
        measured_layer = float(gt_rows["layer_mean"].mean())

        # Analytical predictor
        et_anal = analytical.get_execution_time(
            cfg["num_tokens"], cfg["batch_size"], cfg["kv_cache_size"], cfg["is_prefill"])
        predicted_anal = et_anal.attention_time + et_anal.mlp_time + et_anal.input_layernorm_time + et_anal.post_attention_layernorm_time

        # HighFidelity predictor (with default fusion factors)
        et_hf = high_fidelity.get_execution_time(
            cfg["num_tokens"], cfg["batch_size"], cfg["kv_cache_size"], cfg["is_prefill"])
        predicted_hf = et_hf.attention_time + et_hf.mlp_time + et_hf.input_layernorm_time + et_hf.post_attention_layernorm_time

        # FusedKernel predictor (zero tuning)
        et_fk = fused_kernel.get_execution_time(
            cfg["num_tokens"], cfg["batch_size"], cfg["kv_cache_size"], cfg["is_prefill"])
        predicted_fk = (et_fk.attention_time + et_fk.mlp_time +
                        et_fk.input_layernorm_time + et_fk.post_attention_layernorm_time +
                        et_fk.add_time)

        # Errors
        err_anal = abs(predicted_anal - measured_layer) / measured_layer * 100
        err_hf = abs(predicted_hf - measured_layer) / measured_layer * 100
        err_fk = abs(predicted_fk - measured_layer) / measured_layer * 100

        print(f"{cfg['label']:<20} {measured_layer:>9.3f}ms {predicted_anal:>11.3f}ms {predicted_hf:>13.3f}ms {predicted_fk:>12.3f}ms {err_fk:>9.1f}%")

        results.append({
            "config": cfg["label"],
            "phase": phase,
            "num_tokens": cfg["num_tokens"],
            "batch_size": cfg["batch_size"],
            "measured_ms": measured_layer,
            "analytical_predicted_ms": predicted_anal,
            "analytical_error_pct": err_anal,
            "highfidelity_predicted_ms": predicted_hf,
            "highfidelity_error_pct": err_hf,
            "fused_kernel_predicted_ms": predicted_fk,
            "fused_kernel_error_pct": err_fk,
        })

    # Summary
    avg_err_anal = np.mean([r["analytical_error_pct"] for r in results])
    avg_err_hf = np.mean([r["highfidelity_error_pct"] for r in results])
    avg_err_fk = np.mean([r["fused_kernel_error_pct"] for r in results])
    print(f"\n{'Average':<20} {'':>10} {'':>12} {avg_err_anal:>13.1f}% {avg_err_hf:>13.1f}% {avg_err_fk:>9.1f}%")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "model": MODEL_NAME,
        "gpu": "NVIDIA A800",
        "method": "fused_kernel_direct_lookup_vs_measured",
        "summary": {
            "analytical_avg_error_pct": float(avg_err_anal),
            "highfidelity_avg_error_pct": float(avg_err_hf),
            "fused_kernel_avg_error_pct": float(avg_err_fk),
        },
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, "fused_kernel_validation.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {path}")


if __name__ == "__main__":
    main()
