#!/usr/bin/env python3
"""实验: High-Fidelity 模式精度验证

对比四种预测器的精度:
1. Analytical (Roofline) - 解析模型
2. ProfilingBased (Piecewise) - 分段线性回归
3. RandomForest - 随机森林回归
4. HighFidelity - 精确查表 (无回归)

输出: results/high_fidelity_comparison.json + high_fidelity_comparison.pdf
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

from distlmsim.config import ModelConfig, DeviceSKUConfig
from distlmsim.execution.execution_time_predictor import (
    AnalyticalPredictor,
    ProfilingBasedPredictor,
    RandomForestPredictor,
    HighFidelityPredictor,
)

# 从 experiment_accuracy 导入 ground truth 计算函数
sys.path.insert(0, os.path.dirname(__file__))
from experiment_accuracy import compute_measured_per_layer_time

PROFILING_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "profiling")
MODEL_NAME = "Qwen3-30B-A3B"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    model = ModelConfig(
        model_name=MODEL_NAME,
        num_layers=48,
        num_q_heads=32,
        num_kv_heads=4,
        embedding_dim=2048,
        num_experts=128,
        top_k_experts=8,
    )
    device = DeviceSKUConfig()

    base = os.path.join(PROFILING_DIR, "compute", "a800", "Qwen", MODEL_NAME)
    attn_df = pd.read_csv(os.path.join(base, "attention.csv"))
    mlp_df = pd.read_csv(os.path.join(base, "mlp.csv"))
    expert_df = pd.read_csv(os.path.join(base, "expert.csv"))

    # 创建四种预测器
    predictors = {
        "Analytical": AnalyticalPredictor(model, device),
        "ProfilingBased": ProfilingBasedPredictor(model, device, base),
        "RandomForest": RandomForestPredictor(model, device, base),
        "HighFidelity": HighFidelityPredictor(model, device, PROFILING_DIR),
    }

    # 收集结果
    all_results = []

    # Decode 样本
    for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
        for kv in [64, 128, 256, 512, 1024, 2048, 4096]:
            measured = compute_measured_per_layer_time(
                attn_df, mlp_df, expert_df, bs, bs, kv, False
            )
            if measured is None or measured["total_measured"] <= 0:
                continue

            for name, pred in predictors.items():
                et = pred.get_execution_time(bs, bs, kv, False)
                error = abs(et.total_time - measured["total_measured"]) / measured["total_measured"] * 100
                all_results.append({
                    "phase": "decode",
                    "batch_size": bs,
                    "kv_cache_size": kv,
                    "num_tokens": bs,
                    "predictor": name,
                    "measured_ms": measured["total_measured"],
                    "predicted_ms": et.total_time,
                    "error_pct": error,
                })

    # Prefill 样本
    for nt in [64, 128, 256, 512, 1024, 2048, 4096]:
        measured = compute_measured_per_layer_time(
            attn_df, mlp_df, expert_df, 1, nt, 0, True
        )
        if measured is None or measured["total_measured"] <= 0:
            continue

        for name, pred in predictors.items():
            et = pred.get_execution_time(nt, 1, 0, True)
            error = abs(et.total_time - measured["total_measured"]) / measured["total_measured"] * 100
            all_results.append({
                "phase": "prefill",
                "batch_size": 1,
                "kv_cache_size": 0,
                "num_tokens": nt,
                "predictor": name,
                "measured_ms": measured["total_measured"],
                "predicted_ms": et.total_time,
                "error_pct": error,
            })

    df = pd.DataFrame(all_results)

    # 打印汇总
    print("=" * 80)
    print("High-Fidelity 模式精度对比")
    print("=" * 80)

    for name in predictors:
        subset = df[df["predictor"] == name]
        decode = subset[subset["phase"] == "decode"]
        prefill = subset[subset["phase"] == "prefill"]

        mape = subset["error_pct"].mean()
        median = subset["error_pct"].median()
        p90 = subset["error_pct"].quantile(0.9)
        max_err = subset["error_pct"].max()

        decode_mape = decode["error_pct"].mean() if len(decode) > 0 else 0
        prefill_mape = prefill["error_pct"].mean() if len(prefill) > 0 else 0

        print(f"\n{name}:")
        print(f"  Overall: MAPE={mape:.2f}%, Median={median:.2f}%, P90={p90:.2f}%, Max={max_err:.2f}%")
        print(f"  Decode:  MAPE={decode_mape:.2f}% ({len(decode)} samples)")
        print(f"  Prefill: MAPE={prefill_mape:.2f}% ({len(prefill)} samples)")

    # 保存 JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "high_fidelity_comparison.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存: {json_path}")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: MAPE 柱状图
    ax = axes[0]
    names = list(predictors.keys())
    mapes = [df[df["predictor"] == n]["error_pct"].mean() for n in names]
    medians = [df[df["predictor"] == n]["error_pct"].median() for n in names]
    x = np.arange(len(names))
    width = 0.35
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]

    ax.bar(x - width/2, mapes, width, label="MAPE", alpha=0.7, color=colors)
    ax.bar(x + width/2, medians, width, label="Median", alpha=0.4, color=colors)
    ax.set_xlabel("Prediction Backend")
    ax.set_ylabel("Error (%)")
    ax.set_title("MAPE by Backend")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()

    # 在柱子上标注数值
    for i, (mape, median) in enumerate(zip(mapes, medians)):
        ax.text(i - width/2, mape + 1, f"{mape:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.text(i + width/2, median + 1, f"{median:.1f}%", ha="center", va="bottom", fontsize=8)

    # 右: Predicted vs Measured 散点图
    ax = axes[1]
    markers = ["o", "s", "^", "D"]
    for i, name in enumerate(names):
        subset = df[df["predictor"] == name]
        ax.scatter(
            subset["measured_ms"],
            subset["predicted_ms"],
            label=name,
            alpha=0.4,
            s=15,
            marker=markers[i],
            color=colors[i],
        )

    max_val = max(df["measured_ms"].max(), df["predicted_ms"].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="Ideal (y=x)")
    ax.set_xlabel("Measured Per-Layer Time (ms)")
    ax.set_ylabel("Predicted Time (ms)")
    ax.set_title("Predicted vs Measured")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "high_fidelity_comparison.pdf")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.savefig(fig_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存: {fig_path}")


if __name__ == "__main__":
    main()
