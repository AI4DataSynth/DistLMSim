#!/usr/bin/env python3
"""实验: HighFidelity Predictor 论文级结果生成

整合三部分结果:
1. 算子级精度对比 (4 predictors × aggregate + per-operator-type)
2. E2E 泛化验证 (7 configs vs vLLM on H100)
3. Per-operator-type accuracy breakdown

输出:
  - results/highfidelity_paper.json (整合数据)
  - results/highfidelity_operator_accuracy.pdf (论文图表)
  - results/highfidelity_e2e_generalization.pdf (论文图表)
  - results/highfidelity_operator_breakdown.pdf (算子分类 breakdown)
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

sys.path.insert(0, os.path.dirname(__file__))
from experiment_accuracy import compute_measured_per_layer_time

PROFILING_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "profiling")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_NAME = "Qwen3-30B-A3B"


def run_operator_level_comparison():
    """算子级精度: 4 种 predictor 的 aggregate + per-type 误差。"""
    model = ModelConfig(
        model_name=MODEL_NAME,
        num_layers=48, num_q_heads=32, num_kv_heads=4,
        embedding_dim=2048, num_experts=128, top_k_experts=8,
    )
    device = DeviceSKUConfig()

    base = os.path.join(PROFILING_DIR, "compute", "a800", "Qwen", MODEL_NAME)
    attn_df = pd.read_csv(os.path.join(base, "attention.csv"))
    mlp_df = pd.read_csv(os.path.join(base, "mlp.csv"))
    expert_df = pd.read_csv(os.path.join(base, "expert.csv"))

    # fusion=1.0 for HighFidelity: raw profiling lookup without kernel fusion
    # correction, for direct comparison against measured profiling values.
    # Fusion factors are applied in E2E mode (vs vLLM), not operator-level.
    predictors = {
        "Analytical": AnalyticalPredictor(model, device),
        "ProfilingBased": ProfilingBasedPredictor(model, device, base),
        "RandomForest": RandomForestPredictor(model, device, base),
        "HighFidelity": HighFidelityPredictor(
            model, device, PROFILING_DIR,
            prefill_fusion_factor=1.0, decode_fusion_factor=1.0,
        ),
    }

    # Verify ProfilingBased loaded correctly
    pb = predictors["ProfilingBased"]
    if not hasattr(pb, '_models') or not pb._models:
        print("  WARNING: ProfilingBased fell back to Analytical, re-creating with correct path...")
        # Re-create with the model-specific directory
        predictors["ProfilingBased"] = ProfilingBasedPredictor(model, device, base)
    rf = predictors["RandomForest"]
    if not hasattr(rf, '_models') or not rf._models:
        print("  WARNING: RandomForest fell back to Analytical, re-creating...")
        predictors["RandomForest"] = RandomForestPredictor(model, device, base)

    all_results = []

    # Decode samples
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

                # Per-operator-type errors
                attn_err = abs(et.attention_time - measured["attn_time"]) / max(measured["attn_time"], 1e-6) * 100
                mlp_err = abs(et.mlp_time - measured["mlp_time"]) / max(measured["mlp_time"], 1e-6) * 100
                expert_err = abs(et.expert_mlp_time - measured["expert_time"]) / max(measured["expert_time"], 1e-6) * 100 if measured["expert_time"] > 0 else 0.0

                all_results.append({
                    "phase": "decode",
                    "batch_size": bs,
                    "kv_cache_size": kv,
                    "predictor": name,
                    "measured_ms": measured["total_measured"],
                    "predicted_ms": et.total_time,
                    "error_pct": error,
                    "attn_measured": measured["attn_time"],
                    "attn_predicted": et.attention_time,
                    "attn_error_pct": attn_err,
                    "mlp_measured": measured["mlp_time"],
                    "mlp_predicted": et.mlp_time,
                    "mlp_error_pct": mlp_err,
                    "expert_measured": measured["expert_time"],
                    "expert_predicted": et.expert_mlp_time,
                    "expert_error_pct": expert_err,
                })

    # Prefill samples
    for nt in [64, 128, 256, 512, 1024, 2048, 4096]:
        measured = compute_measured_per_layer_time(
            attn_df, mlp_df, expert_df, 1, nt, 0, True
        )
        if measured is None or measured["total_measured"] <= 0:
            continue

        for name, pred in predictors.items():
            et = pred.get_execution_time(nt, 1, 0, True)
            error = abs(et.total_time - measured["total_measured"]) / measured["total_measured"] * 100

            attn_err = abs(et.attention_time - measured["attn_time"]) / max(measured["attn_time"], 1e-6) * 100
            mlp_err = abs(et.mlp_time - measured["mlp_time"]) / max(measured["mlp_time"], 1e-6) * 100
            expert_err = abs(et.expert_mlp_time - measured["expert_time"]) / max(measured["expert_time"], 1e-6) * 100 if measured["expert_time"] > 0 else 0.0

            all_results.append({
                "phase": "prefill",
                "batch_size": 1,
                "kv_cache_size": 0,
                "num_tokens": nt,
                "predictor": name,
                "measured_ms": measured["total_measured"],
                "predicted_ms": et.total_time,
                "error_pct": error,
                "attn_measured": measured["attn_time"],
                "attn_predicted": et.attention_time,
                "attn_error_pct": attn_err,
                "mlp_measured": measured["mlp_time"],
                "mlp_predicted": et.mlp_time,
                "mlp_error_pct": mlp_err,
                "expert_measured": measured["expert_time"],
                "expert_predicted": et.expert_mlp_time,
                "expert_error_pct": expert_err,
            })

    return all_results


def load_generalization_results():
    """加载 E2E 泛化验证结果。"""
    path = os.path.join(RESULTS_DIR, "generalization_comparison.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def print_summary(operator_results, gen_results):
    """打印论文级汇总表。"""
    df = pd.DataFrame(operator_results)

    print("\n" + "=" * 80)
    print("  HighFidelity Predictor: 论文级结果汇总")
    print("=" * 80)

    # Table 1: 4-way operator-level comparison
    print("\n--- Table 1: Operator-level accuracy (A800 + Qwen3-30B-A3B) ---")
    print(f"{'Backend':<20} {'MAPE':>8} {'Median':>8} {'P90':>8} {'Decode':>8} {'Prefill':>8}")
    print("-" * 60)

    for name in ["Analytical", "ProfilingBased", "RandomForest", "HighFidelity"]:
        sub = df[df["predictor"] == name]
        decode = sub[sub["phase"] == "decode"]
        prefill = sub[sub["phase"] == "prefill"]
        print(f"{name:<20} "
              f"{sub['error_pct'].mean():>7.1f}% "
              f"{sub['error_pct'].median():>7.1f}% "
              f"{sub['error_pct'].quantile(0.9):>7.1f}% "
              f"{decode['error_pct'].mean():>7.1f}% "
              f"{prefill['error_pct'].mean():>7.1f}%")

    # Table 2: Per-operator-type breakdown (HighFidelity)
    print("\n--- Table 2: Per-operator-type accuracy (HighFidelity) ---")
    hf = df[df["predictor"] == "HighFidelity"]
    for phase in ["decode", "prefill"]:
        sub = hf[hf["phase"] == phase]
        if len(sub) == 0:
            continue
        print(f"\n  {phase.capitalize()}:")
        for op_type, col in [("Attention", "attn_error_pct"), ("MLP", "mlp_error_pct"), ("Expert", "expert_error_pct")]:
            vals = sub[col]
            if vals.mean() > 0:
                print(f"    {op_type:<12}: MAPE={vals.mean():.2f}%, Median={vals.median():.2f}%, P90={vals.quantile(0.9):.2f}%")

    # Table 3: E2E generalization
    if gen_results:
        print("\n--- Table 3: E2E Generalization (H100 + Llama-2-13B vs vLLM) ---")
        print(f"{'Config':<25} {'TTFT err':>10} {'TBT err':>10} {'E2E err':>10}")
        print("-" * 58)
        for c in gen_results["comparisons"]:
            print(f"{c['config']:<25} {c['ttft_err']:>9.1f}% {c['tbt_err']:>9.1f}% {c['e2e_err']:>9.1f}%")

        avg_ttft = np.mean([c["ttft_err"] for c in gen_results["comparisons"]])
        avg_tbt = np.mean([c["tbt_err"] for c in gen_results["comparisons"]])
        avg_e2e = np.mean([c["e2e_err"] for c in gen_results["comparisons"]])
        print("-" * 58)
        print(f"{'Average':<25} {avg_ttft:>9.1f}% {avg_tbt:>9.1f}% {avg_e2e:>9.1f}%")


def plot_operator_accuracy(df, output_dir):
    """Figure: 4-way operator-level accuracy comparison."""
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))

    names = ["Analytical", "ProfilingBased", "RandomForest", "HighFidelity"]
    short_names = ["Roofline", "Profiled", "RF", "HighFidelity"]
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]

    # (a) Overall MAPE bar chart
    ax = axes[0]
    mapes = []
    medians = []
    for name in names:
        sub = df[df["predictor"] == name]
        mapes.append(sub["error_pct"].mean())
        medians.append(sub["error_pct"].median())

    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, mapes, width, label="MAPE", alpha=0.8, color=colors)
    ax.bar(x + width/2, medians, width, label="Median", alpha=0.4, color=colors)
    for i, (m, med) in enumerate(zip(mapes, medians)):
        ax.text(i - width/2, m + 1, f"{m:.1f}%", ha="center", va="bottom", fontsize=7)
        ax.text(i + width/2, med + 1, f"{med:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_ylabel("Error (%)")
    ax.set_title("(a) Overall Accuracy")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(mapes) * 1.3)

    # (b) Decode vs Prefill MAPE
    ax = axes[1]
    decode_mapes = [df[(df["predictor"] == n) & (df["phase"] == "decode")]["error_pct"].mean() for n in names]
    prefill_mapes = [df[(df["predictor"] == n) & (df["phase"] == "prefill")]["error_pct"].mean() for n in names]

    ax.bar(x - width/2, decode_mapes, width, label="Decode", alpha=0.8, color=colors)
    ax.bar(x + width/2, prefill_mapes, width, label="Prefill", alpha=0.8, color=[c + "80" for c in ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]])
    for i, (d, p) in enumerate(zip(decode_mapes, prefill_mapes)):
        ax.text(i - width/2, d + 0.5, f"{d:.1f}%", ha="center", va="bottom", fontsize=7)
        ax.text(i + width/2, p + 0.5, f"{p:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("(b) Decode vs Prefill")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(max(decode_mapes), max(prefill_mapes)) * 1.3)

    # (c) Predicted vs Measured scatter (HighFidelity)
    ax = axes[2]
    hf = df[df["predictor"] == "HighFidelity"]
    decode_pts = hf[hf["phase"] == "decode"]
    prefill_pts = hf[hf["phase"] == "prefill"]

    ax.scatter(decode_pts["measured_ms"], decode_pts["predicted_ms"],
               alpha=0.5, s=12, c="#E74C3C", label=f"Decode ({len(decode_pts)})")
    ax.scatter(prefill_pts["measured_ms"], prefill_pts["predicted_ms"],
               alpha=0.7, s=20, c="#3498DB", label=f"Prefill ({len(prefill_pts)})")
    max_val = max(hf["measured_ms"].max(), hf["predicted_ms"].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Measured (ms)")
    ax.set_ylabel("Predicted (ms)")
    ax.set_title("(c) HighFidelity: Pred vs Meas")
    ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, "highfidelity_operator_accuracy.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}")
    return path


def plot_operator_breakdown(df, output_dir):
    """Figure: Per-operator-type accuracy breakdown."""
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    names = ["Analytical", "ProfilingBased", "RandomForest", "HighFidelity"]
    short_names = ["Roofline", "Profiled", "RF", "HighFidelity"]
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]
    op_types = ["Attention", "MLP", "Expert"]
    op_colors = ["#E74C3C", "#3498DB", "#2ECC71"]

    for phase_idx, phase in enumerate(["decode", "prefill"]):
        ax = axes[phase_idx]
        x = np.arange(len(names))
        width = 0.25

        for op_idx, (op, col) in enumerate([("attn", "attn_error_pct"), ("mlp", "mlp_error_pct"), ("expert", "expert_error_pct")]):
            mapes = []
            for name in names:
                sub = df[(df["predictor"] == name) & (df["phase"] == phase)]
                if col in sub.columns:
                    mape = sub[col].mean()
                else:
                    mape = 0.0
                mapes.append(mape)

            bars = ax.bar(x + (op_idx - 1) * width, mapes, width,
                         label=op_types[op_idx], alpha=0.8, color=op_colors[op_idx])
            for i, m in enumerate(mapes):
                if m > 0:
                    ax.text(i + (op_idx - 1) * width, m + 0.3, f"{m:.0f}%",
                           ha="center", va="bottom", fontsize=6)

        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=30, ha="right")
        ax.set_ylabel("MAPE (%)")
        ax.set_title(f"({'a' if phase_idx == 0 else 'b'}) {phase.capitalize()}: Per-Operator MAPE")
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, "highfidelity_operator_breakdown.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}")
    return path


def plot_e2e_generalization(gen_results, output_dir):
    """Figure: E2E generalization heatmap."""
    if not gen_results:
        print("No generalization results to plot.")
        return None

    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
    })

    comparisons = gen_results["comparisons"]
    configs = [c["config"] for c in comparisons]
    metrics = ["TTFT", "TBT", "E2E"]
    errors = np.array([
        [c["ttft_err"], c["tbt_err"], c["e2e_err"]]
        for c in comparisons
    ])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # (a) Heatmap
    ax = axes[0]
    im = ax.imshow(errors, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=10)
    ax.set_xticks(range(3))
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=8)
    for i in range(len(configs)):
        for j in range(3):
            color = "white" if errors[i, j] > 5 else "black"
            ax.text(j, i, f"{errors[i, j]:.1f}%", ha="center", va="center",
                   fontsize=9, color=color, fontweight="bold")
    ax.set_title("(a) Prediction Error vs vLLM (%)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Error %")

    # (b) Simulated vs Measured bar chart
    ax = axes[1]
    x = np.arange(len(configs))
    width = 0.35
    sim_ttft = [c["sim_ttft"] for c in comparisons]
    vllm_ttft = [c["vllm_ttft"] for c in comparisons]

    short_configs = [c.replace(",", "\n") for c in configs]
    ax.bar(x - width/2, sim_ttft, width, label="DistLMSim", color="#3498DB", alpha=0.7)
    ax.bar(x + width/2, vllm_ttft, width, label="vLLM", color="#E74C3C", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(short_configs, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("TTFT P50 (ms)")
    ax.set_title("(b) TTFT: DistLMSim vs vLLM")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "highfidelity_e2e_generalization.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}")
    return path


def main():
    print("=" * 80)
    print("  HighFidelity Predictor: 论文级实验")
    print("=" * 80)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Operator-level comparison
    print("\n[1/3] Running operator-level comparison...")
    operator_results = run_operator_level_comparison()
    df = pd.DataFrame(operator_results)
    print(f"  {len(df)} samples collected")

    # 2. Load E2E generalization
    print("\n[2/3] Loading E2E generalization results...")
    gen_results = load_generalization_results()
    if gen_results:
        print(f"  {len(gen_results['comparisons'])} configs loaded")
    else:
        print("  No generalization results found")

    # 3. Print summary
    print("\n[3/3] Generating summary and figures...")
    print_summary(df, gen_results)

    # Generate figures
    plot_operator_accuracy(df, RESULTS_DIR)
    plot_operator_breakdown(df, RESULTS_DIR)
    plot_e2e_generalization(gen_results, RESULTS_DIR)

    # Save consolidated JSON
    output = {
        "operator_level": operator_results,
        "generalization": gen_results,
        "summary": {},
    }

    for name in ["Analytical", "ProfilingBased", "RandomForest", "HighFidelity"]:
        sub = df[df["predictor"] == name]
        output["summary"][name] = {
            "mape": round(sub["error_pct"].mean(), 2),
            "median": round(sub["error_pct"].median(), 2),
            "p90": round(sub["error_pct"].quantile(0.9), 2),
            "decode_mape": round(sub[sub["phase"] == "decode"]["error_pct"].mean(), 2),
            "prefill_mape": round(sub[sub["phase"] == "prefill"]["error_pct"].mean(), 2),
        }

    json_path = os.path.join(RESULTS_DIR, "highfidelity_paper.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nConsolidated results saved: {json_path}")


if __name__ == "__main__":
    main()
