#!/usr/bin/env python3
"""Hybrid Backend 精度对比实验

对比 3 种预测后端的精度:
  1. Analytical (Roofline-only)
  2. ProfilingBased (线性回归 + profiling CSV)
  3. RandomForest (sklearn RF + profiling CSV)

论文 §5.1 声称 hybrid backend 精度显著优于纯 Roofline，本实验验证此声称。

用法:
  python3 examples/experiment_hybrid_accuracy.py
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "examples")

from distlmsim.config import ModelConfig, DeviceSKUConfig
from distlmsim.execution.execution_time_predictor import (
    AnalyticalPredictor,
    ProfilingBasedPredictor,
    RandomForestPredictor,
)

# 复用 experiment_accuracy 中的测量函数
from experiment_accuracy import compute_measured_per_layer_time


def run_validation_for_predictor(predictor, attn_df, mlp_df, expert_df, model, name):
    """对单个 predictor 运行精度验证。"""
    results = []

    # Decode 验证
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        for kv_cache_size in [32, 64, 128, 256, 512, 1024, 2048, 4032]:
            num_tokens = batch_size
            measured = compute_measured_per_layer_time(
                attn_df, mlp_df, expert_df,
                batch_size=batch_size, num_tokens=num_tokens,
                kv_cache_size=kv_cache_size, is_prefill=False
            )
            if measured is None or measured["total_measured"] <= 0:
                continue

            exec_time = predictor.get_execution_time(
                num_tokens=num_tokens, batch_size=batch_size,
                kv_cache_size=kv_cache_size, is_prefill=False,
            )
            predicted = exec_time.total_time
            error_pct = abs(predicted - measured["total_measured"]) / measured["total_measured"] * 100

            results.append({
                "predictor": name,
                "phase": "decode",
                "batch_size": batch_size,
                "kv_cache_size": kv_cache_size,
                "measured_ms": measured["total_measured"],
                "predicted_ms": predicted,
                "error_pct": error_pct,
            })

    # Prefill 验证
    for num_tokens in [64, 128, 256, 512, 1024, 2048, 4096]:
        measured = compute_measured_per_layer_time(
            attn_df, mlp_df, expert_df,
            batch_size=1, num_tokens=num_tokens, kv_cache_size=0, is_prefill=True
        )
        if measured is None or measured["total_measured"] <= 0:
            continue

        exec_time = predictor.get_execution_time(
            num_tokens=num_tokens, batch_size=1,
            kv_cache_size=0, is_prefill=True,
        )
        predicted = exec_time.total_time
        error_pct = abs(predicted - measured["total_measured"]) / measured["total_measured"] * 100

        results.append({
            "predictor": name,
            "phase": "prefill",
            "batch_size": 1,
            "kv_cache_size": 0,
            "num_tokens": num_tokens,
            "measured_ms": measured["total_measured"],
            "predicted_ms": predicted,
            "error_pct": error_pct,
        })

    return results


def main():
    logging.basicConfig(level=logging.WARNING)

    print("=" * 70)
    print("  Hybrid Backend Accuracy Comparison")
    print("  Roofline vs ProfilingBased vs RandomForest")
    print("=" * 70)

    base_dir = Path("data/profiling/compute/a800/Qwen/Qwen3-30B-A3B")
    attn_df = pd.read_csv(base_dir / "attention.csv")
    mlp_df = pd.read_csv(base_dir / "mlp.csv")
    expert_df = pd.read_csv(base_dir / "expert.csv") if (base_dir / "expert.csv").exists() else None

    model = ModelConfig(
        model_name="Qwen3-30B-A3B",
        num_layers=48, num_q_heads=32, num_kv_heads=4,
        embedding_dim=2048, num_experts=128, top_k_experts=8,
    )
    device = DeviceSKUConfig()

    profiling_dir = str(Path("data/profiling/compute/a800/Qwen/Qwen3-30B-A3B"))

    # 创建 3 种 predictor
    predictors = {}
    try:
        predictors["Analytical"] = AnalyticalPredictor(model, device)
        print("  ✅ AnalyticalPredictor (Roofline-only)")
    except Exception as e:
        print(f"  ❌ AnalyticalPredictor: {e}")

    try:
        predictors["ProfilingBased"] = ProfilingBasedPredictor(model, device, profiling_dir)
        print("  ✅ ProfilingBasedPredictor (linear regression)")
    except Exception as e:
        print(f"  ⚠️  ProfilingBasedPredictor: {e}")

    try:
        predictors["RandomForest"] = RandomForestPredictor(model, device, profiling_dir)
        print("  ✅ RandomForestPredictor (sklearn RF)")
    except Exception as e:
        print(f"  ⚠️  RandomForestPredictor: {e}")

    # 运行验证
    all_results = []
    for name, predictor in predictors.items():
        print(f"\n  运行 {name} 验证...")
        results = run_validation_for_predictor(
            predictor, attn_df, mlp_df, expert_df, model, name
        )
        all_results.extend(results)
        df = pd.DataFrame(results)
        print(f"    样本数: {len(df)}, MAPE: {df['error_pct'].mean():.1f}%, "
              f"Median: {df['error_pct'].median():.1f}%")

    if not all_results:
        print("无法执行验证!")
        return

    results_df = pd.DataFrame(all_results)

    # 总结表格
    print(f"\n{'='*70}")
    print(f"  Hybrid Backend 精度对比总结")
    print(f"{'='*70}")
    print(f"  {'Backend':<20} {'Samples':>8} {'MAPE':>10} {'Median':>10} {'P90':>10}")
    print(f"  {'─'*60}")

    for name in predictors.keys():
        subset = results_df[results_df["predictor"] == name]
        if len(subset) > 0:
            print(f"  {name:<20} {len(subset):>8} "
                  f"{subset['error_pct'].mean():>9.1f}% "
                  f"{subset['error_pct'].median():>9.1f}% "
                  f"{subset['error_pct'].quantile(0.9):>9.1f}%")

    # Decode vs Prefill 分组
    print(f"\n  Decode phase:")
    for name in predictors.keys():
        subset = results_df[(results_df["predictor"] == name) & (results_df["phase"] == "decode")]
        if len(subset) > 0:
            print(f"    {name:<20} MAPE: {subset['error_pct'].mean():.1f}%")

    print(f"\n  Prefill phase:")
    for name in predictors.keys():
        subset = results_df[(results_df["predictor"] == name) & (results_df["phase"] == "prefill")]
        if len(subset) > 0:
            print(f"    {name:<20} MAPE: {subset['error_pct'].mean():.1f}%")

    print(f"{'='*70}")

    # 保存
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    results_df.to_json(output_dir / "hybrid_accuracy.json", orient="records", indent=2)
    print(f"\n结果已保存: {output_dir / 'hybrid_accuracy.json'}")

    # 绘图
    plot_hybrid_comparison(results_df, str(output_dir))


def plot_hybrid_comparison(results_df, output_dir):
    """绘制 Hybrid Backend 精度对比图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    predictor_names = results_df["predictor"].unique()
    colors = {"Analytical": "#E74C3C", "ProfilingBased": "#3498DB", "RandomForest": "#2ECC71"}

    # 1. Bar chart: MAPE by predictor
    ax = axes[0]
    mape_values = []
    median_values = []
    names = []
    for name in predictor_names:
        subset = results_df[results_df["predictor"] == name]
        mape_values.append(subset["error_pct"].mean())
        median_values.append(subset["error_pct"].median())
        names.append(name)

    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width/2, mape_values, width, label="MAPE", color=[colors.get(n, "#999") for n in names], alpha=0.7)
    bars2 = ax.bar(x + width/2, median_values, width, label="Median", color=[colors.get(n, "#999") for n in names], alpha=0.4)
    ax.set_xlabel("Prediction Backend")
    ax.set_ylabel("Error (%)")
    ax.set_title("Prediction Accuracy: MAPE by Backend")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

    # 2. Scatter: predicted vs measured (all predictors)
    ax = axes[1]
    for name in predictor_names:
        subset = results_df[results_df["predictor"] == name]
        ax.scatter(subset["measured_ms"], subset["predicted_ms"],
                   alpha=0.4, s=15, c=colors.get(name, "#999"), label=name)

    max_val = max(results_df["measured_ms"].max(), results_df["predicted_ms"].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Ideal (y=x)")
    ax.set_xlabel("Measured Per-Layer Time (ms)")
    ax.set_ylabel("Predicted Per-Layer Time (ms)")
    ax.set_title("Predicted vs Measured (All Backends)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/hybrid_accuracy.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_dir}/hybrid_accuracy.pdf", bbox_inches="tight")
    plt.close()
    print(f"\n图表已保存: {output_dir}/hybrid_accuracy.pdf")


if __name__ == "__main__":
    main()
