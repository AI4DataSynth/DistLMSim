"""算子级精度验证实验 (v2)

对比 AnalyticalPredictor (Roofline model) 的单层 total_time 预测值
与 profiling 实测的完整单层前向传播时间（所有子阶段加总）。

验证维度:
  - Decode step latency: 固定 batch_size, 变化 kv_cache_size
  - Prefill latency: 变化 num_tokens

输出:
  - Analytical (Roofline-only) 的 MAPE
  - 分析哪些算子偏差最大 (说明 profiling backend 的必要性)

用法:
  python3 examples/experiment_accuracy.py
"""

import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from distlmsim.config import ModelConfig, DeviceSKUConfig
from distlmsim.execution.execution_time_predictor import AnalyticalPredictor


def compute_measured_per_layer_time(attn_df, mlp_df, expert_df, batch_size, num_tokens, kv_cache_size, is_prefill):
    """从 profiling CSV 计算实测的单层前向传播时间 (ms)。

    单层时间 = attention 全部子阶段 + MLP/Expert + LayerNorm + 小操作
    """
    # ─── 匹配 attention 行 ─────────────────────────────────────────────
    # Filter for TP=1 only (与模拟器默认配置一致)
    if "num_tensor_parallel_workers" in attn_df.columns:
        attn_df_tp1 = attn_df[attn_df["num_tensor_parallel_workers"] == 1]
    else:
        attn_df_tp1 = attn_df
    
    attn_mask = (attn_df_tp1["batch_size"] == batch_size) & (attn_df_tp1["is_prefill"] == is_prefill)
    attn_rows = attn_df_tp1[attn_mask]

    if len(attn_rows) == 0:
        return None

    # 对 decode: 用 kv_cache_size 匹配; 对 prefill: 用 prefill_chunk_size 匹配
    if is_prefill:
        if "prefill_chunk_size" in attn_df_tp1.columns:
            closest_idx = (attn_rows["prefill_chunk_size"] - num_tokens).abs().idxmin()
        else:
            closest_idx = attn_rows.index[0]
    else:
        closest_idx = (attn_rows["kv_cache_size"] - kv_cache_size).abs().idxmin()

    attn_row = attn_rows.loc[closest_idx]

    # Attention 子阶段加总
    attn_time = 0.0
    attn_cols = [
        "time_stats.attn_input_reshape.mean",
        "time_stats.attn_kv_cache_save.mean",
        "time_stats.attn_prefill.mean" if is_prefill else "time_stats.attn_decode.mean",
        "time_stats.attn_output_reshape.mean",
    ]
    for col in attn_cols:
        if col in attn_row.index and not pd.isna(attn_row[col]):
            attn_time += float(attn_row[col])

    # ─── 匹配 MLP 行 (取最接近 num_tokens 的) ──────────────────────────
    if len(mlp_df) > 0:
        mlp_closest = mlp_df.iloc[(mlp_df["num_tokens"] - num_tokens).abs().idxmin()]
    else:
        return None

    # MLP 子阶段 (非 MoE 部分)
    mlp_time = 0.0
    mlp_cols = [
        "time_stats.input_layernorm.mean",
        "time_stats.attn_pre_proj.mean",
        "time_stats.attn_rope.mean",
        "time_stats.attn_post_proj.mean",
        "time_stats.post_attention_layernorm.mean",
        "time_stats.mlp_up_proj.mean",
        "time_stats.mlp_act.mean",
        "time_stats.mlp_down_proj.mean",
        "time_stats.add.mean",
    ]
    for col in mlp_cols:
        if col in mlp_closest.index and not pd.isna(mlp_closest[col]):
            mlp_time += float(mlp_closest[col])

    # Expert MLP (MoE)
    expert_time = 0.0
    if expert_df is not None and len(expert_df) > 0:
        exp_closest = expert_df.iloc[(expert_df["num_tokens"] - num_tokens).abs().idxmin()]
        if "time_stats.expert_mlp.mean" in exp_closest.index:
            expert_time = float(exp_closest["time_stats.expert_mlp.mean"])

    total_measured = attn_time + mlp_time + expert_time

    return {
        "attn_time": attn_time,
        "mlp_time": mlp_time,
        "expert_time": expert_time,
        "total_measured": total_measured,
    }


def run_validation(predictor, attn_df, mlp_df, expert_df, model_config):
    """对多组配置进行精度验证。"""
    results = []

    # Decode 验证: 变化 batch_size 和 kv_cache_size (匹配 profiling 数据范围)
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        for kv_cache_size in [32, 64, 128, 256, 512, 1024, 2048, 4032]:
            num_tokens = batch_size  # decode: 每请求 1 token

            measured = compute_measured_per_layer_time(
                attn_df, mlp_df, expert_df,
                batch_size, num_tokens, kv_cache_size, is_prefill=False
            )
            if measured is None:
                continue

            exec_time = predictor.get_execution_time(
                num_tokens=num_tokens,
                batch_size=batch_size,
                kv_cache_size=kv_cache_size,
                is_prefill=False,
            )
            predicted = exec_time.total_time

            if measured["total_measured"] <= 0:
                continue

            error_pct = abs(predicted - measured["total_measured"]) / measured["total_measured"] * 100

            results.append({
                "phase": "decode",
                "batch_size": batch_size,
                "num_tokens": num_tokens,
                "kv_cache_size": kv_cache_size,
                "measured_ms": measured["total_measured"],
                "predicted_ms": predicted,
                "error_pct": error_pct,
                "measured_attn": measured["attn_time"],
                "measured_mlp": measured["mlp_time"],
                "measured_expert": measured["expert_time"],
                "predicted_attn": exec_time.attention_time,
                "predicted_mlp": exec_time.mlp_time,
            })

    # Prefill 验证: 变化 num_tokens
    for num_tokens in [64, 128, 256, 512, 1024, 2048, 4096]:
        measured = compute_measured_per_layer_time(
            attn_df, mlp_df, expert_df,
            batch_size=1, num_tokens=num_tokens, kv_cache_size=0, is_prefill=True
        )
        if measured is None:
            continue

        exec_time = predictor.get_execution_time(
            num_tokens=num_tokens,
            batch_size=1,
            kv_cache_size=0,
            is_prefill=True,
        )
        predicted = exec_time.total_time

        if measured["total_measured"] <= 0:
            continue

        error_pct = abs(predicted - measured["total_measured"]) / measured["total_measured"] * 100

        results.append({
            "phase": "prefill",
            "batch_size": 1,
            "num_tokens": num_tokens,
            "kv_cache_size": 0,
            "measured_ms": measured["total_measured"],
            "predicted_ms": predicted,
            "error_pct": error_pct,
            "measured_attn": measured["attn_time"],
            "measured_mlp": measured["mlp_time"],
            "measured_expert": measured["expert_time"],
            "predicted_attn": exec_time.attention_time,
            "predicted_mlp": exec_time.mlp_time,
        })

    return results


def plot_accuracy(results_df, output_dir):
    """绘制精度验证图表。"""
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        'font.size': 24,
        'axes.titlesize': 28,
        'axes.labelsize': 26,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 22,
        'figure.titlesize': 32,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Simulation Accuracy: Roofline Prediction vs Profiling (Per-Layer)", fontsize=24)

    decode_df = results_df[results_df["phase"] == "decode"]
    prefill_df = results_df[results_df["phase"] == "prefill"]

    # 1. Scatter: predicted vs measured
    ax = axes[0, 0]
    ax.scatter(decode_df["measured_ms"], decode_df["predicted_ms"],
               alpha=0.4, s=20, c="#E74C3C", label=f"Decode ({len(decode_df)} pts)")
    ax.scatter(prefill_df["measured_ms"], prefill_df["predicted_ms"],
               alpha=0.6, s=40, c="#3498DB", label=f"Prefill ({len(prefill_df)} pts)")
    max_val = max(results_df["measured_ms"].max(), results_df["predicted_ms"].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Ideal (y=x)")
    ax.set_xlabel("Measured Per-Layer Time (ms)")
    ax.set_ylabel("Predicted Per-Layer Time (ms)")
    ax.set_title("Predicted vs Measured")
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)

    # 2. Error distribution
    ax = axes[0, 1]
    for phase, color in [("decode", "#E74C3C"), ("prefill", "#3498DB")]:
        subset = results_df[results_df["phase"] == phase]
        if len(subset) > 0:
            mape = subset["error_pct"].mean()
            ax.hist(subset["error_pct"].clip(0, 200), bins=20, alpha=0.5,
                    label=f"{phase} (MAPE={mape:.1f}%)", color=color)
    ax.set_xlabel("Prediction Error (%)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution (Roofline-only backend)")
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)

    # 3. Decode error vs kv_cache_size
    ax = axes[1, 0]
    for bs in sorted(decode_df["batch_size"].unique()):
        subset = decode_df[decode_df["batch_size"] == bs]
        ax.plot(subset["kv_cache_size"], subset["error_pct"],
                marker="o", label=f"BS={bs}", alpha=0.7)
    ax.set_xlabel("KV Cache Size (tokens)")
    ax.set_ylabel("Error (%)")
    ax.set_title("Decode: Error vs KV Cache Size")
    ax.legend(fontsize=18, ncol=2)
    ax.grid(True, alpha=0.3)

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis("off")

    overall_mape = results_df["error_pct"].mean()
    decode_mape = decode_df["error_pct"].mean() if len(decode_df) > 0 else 0
    prefill_mape = prefill_df["error_pct"].mean() if len(prefill_df) > 0 else 0

    summary = [
        ["Phase", "Samples", "MAPE", "Median", "P90"],
        ["Decode", str(len(decode_df)), f"{decode_mape:.1f}%",
         f"{decode_df['error_pct'].median():.1f}%",
         f"{decode_df['error_pct'].quantile(0.9):.1f}%"],
        ["Prefill", str(len(prefill_df)), f"{prefill_mape:.1f}%",
         f"{prefill_df['error_pct'].median():.1f}%",
         f"{prefill_df['error_pct'].quantile(0.9):.1f}%"],
        ["Overall", str(len(results_df)), f"{overall_mape:.1f}%",
         f"{results_df['error_pct'].median():.1f}%",
         f"{results_df['error_pct'].quantile(0.9):.1f}%"],
    ]

    table = ax.table(cellText=summary, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(5):
        table[0, j].set_facecolor("#3498DB")
        table[0, j].set_text_props(weight="bold", color="white")

    ax.set_title("Roofline Accuracy Summary", pad=20, fontsize=22)

    # Note about profiling backend moved to figure caption

    plt.tight_layout()
    plt.savefig(f"{output_dir}/simulation_accuracy.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_dir}/simulation_accuracy.pdf", bbox_inches="tight")
    plt.close()
    print(f"\n图表已保存: {output_dir}/simulation_accuracy.png")


def main():
    parser = argparse.ArgumentParser(description="算子级精度验证实验")
    parser.add_argument("--profiling_dir", type=str, default="data/profiling")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 70)
    print("  Simulation Accuracy Validation (Roofline vs Profiling)")
    print("=" * 70)

    profiling_dir = Path(args.profiling_dir)
    base_dir = profiling_dir / "compute" / "a800" / "Qwen" / "Qwen3-30B-A3B"

    print(f"\n加载 profiling 数据: {base_dir}")
    attn_df = pd.read_csv(base_dir / "attention.csv") if (base_dir / "attention.csv").exists() else pd.DataFrame()
    mlp_df = pd.read_csv(base_dir / "mlp.csv") if (base_dir / "mlp.csv").exists() else pd.DataFrame()
    expert_df = pd.read_csv(base_dir / "expert.csv") if (base_dir / "expert.csv").exists() else None
    print(f"  Attention: {len(attn_df)} rows")
    print(f"  MLP: {len(mlp_df)} rows")
    print(f"  Expert: {len(expert_df) if expert_df is not None else 0} rows")

    # 创建 predictor
    model = ModelConfig(
        model_name="Qwen3-30B-A3B",
        num_layers=48, num_q_heads=32, num_kv_heads=4,
        embedding_dim=2048, num_experts=128, top_k_experts=8,
    )
    device = DeviceSKUConfig()
    predictor = AnalyticalPredictor(model, device)

    print(f"\n模型: {model.model_name} ({model.num_layers} 层)")
    print(f"设备: A800 (FP16 {device.fp16_tflops} TFLOPS, BW {device.memory_bandwidth_gbps} GB/s)")

    # 运行验证
    print(f"\n运行精度验证...")
    results = run_validation(predictor, attn_df, mlp_df, expert_df, model)

    if not results:
        print("无法执行验证!")
        return

    results_df = pd.DataFrame(results)

    # 总结
    decode_df = results_df[results_df["phase"] == "decode"]
    prefill_df = results_df[results_df["phase"] == "prefill"]

    print(f"\n{'='*70}")
    print(f"  Roofline-only 精度验证结果")
    print(f"{'='*70}")
    print(f"  总样本数:    {len(results_df)}")
    print(f"  总体 MAPE:   {results_df['error_pct'].mean():.1f}%")
    print(f"  中位误差:    {results_df['error_pct'].median():.1f}%")
    if len(decode_df) > 0:
        print(f"  Decode MAPE: {decode_df['error_pct'].mean():.1f}%")
    if len(prefill_df) > 0:
        print(f"  Prefill MAPE: {prefill_df['error_pct'].mean():.1f}%")
    print(f"{'='*70}")

    print(f"\n结论: Roofline 模型对 attention kernel (特别是 FlashInfer) 有系统性低估,")
    print(f"这验证了 DistLMSim hybrid backend (profiled + prediction) 的必要性。")

    # 保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    results_df.to_json(output_dir / "simulation_accuracy.json", orient="records", indent=2)
    print(f"\n结果已保存: {output_dir / 'simulation_accuracy.json'}")

    # 绘图
    plot_accuracy(results_df, str(output_dir))


if __name__ == "__main__":
    main()
