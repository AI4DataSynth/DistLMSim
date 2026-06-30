"""Speculative Decoding 配置调优实验

对比不同投机长度 K 和接受率 α 对 PD 分离推理性能的影响。

实验设计:
  - K (speculation length): 1, 2, 4, 6, 8, 10, 12
  - α (acceptance rate): 0.6, 0.7, 0.8, 0.9
  - 基线: K=1 (标准 decode, 无投机)
  - 测量: TBT P50, E2E P50, decode throughput

预期结果:
  - K 太小: draft 开销低但收益也低
  - K 太大: verify 开销增大, 低 α 时反而变慢
  - 最优 K 取决于 α: α=0.9 → K=6-8, α=0.6 → K=2-4

用法:
  python3 examples/experiment_speculative_decoding.py
  python3 examples/experiment_speculative_decoding.py --qps 10 --time_limit 30
"""

import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from main import create_disaggregated_simulator


def run_one(K, alpha, qps, time_limit, seed):
    """运行一组 speculative decoding 配置。"""
    sim = create_disaggregated_simulator(
        qps=qps,
        prefill_length=512,
        decode_length=128,
        prefill_batch_size=4,
        decode_batch_size=32,
        tp_size=4,
        time_limit_s=time_limit,
        seed=seed,
        length_distribution="normal",
    )

    # 修改 config 启用 speculative decoding
    sim.config.disaggregated.enable_speculative_decoding = (K > 1)
    sim.config.disaggregated.speculation_length = K
    sim.config.disaggregated.acceptance_rate = alpha

    ms = sim.run()

    completed = [m for m in ms._request_metrics.values() if m.decode_end_time > 0]
    if not completed:
        return None

    tbts = [m.tbt for m in completed if m.tbt > 0]
    e2es = [m.e2e_latency for m in completed]
    total_decode = sum(m.decode_tokens for m in completed)
    wall_time = max(m.decode_end_time for m in completed)
    first_arrival = min(m.arrival_time for m in completed)
    effective_s = (wall_time - first_arrival) / 1000.0

    return {
        "K": K,
        "alpha": alpha,
        "completed": len(completed),
        "tbt_p50": float(np.percentile(tbts, 50)) if tbts else 0.0,
        "tbt_p99": float(np.percentile(tbts, 99)) if tbts else 0.0,
        "e2e_p50": float(np.percentile(e2es, 50)),
        "e2e_p99": float(np.percentile(e2es, 99)),
        "decode_throughput": total_decode / effective_s if effective_s > 0 else 0.0,
    }


def plot_results(results, output_dir):
    """绘制实验结果图表。"""
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
    
    K_values = sorted(set(r["K"] for r in results))
    alpha_values = sorted(set(r["alpha"] for r in results))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Speculative Decoding: K vs α Performance", fontsize=28)

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]

    # 1. TBT P50
    ax = axes[0, 0]
    for i, alpha in enumerate(alpha_values):
        data = [r for r in results if r["alpha"] == alpha]
        ks = [r["K"] for r in data]
        tbts = [r["tbt_p50"] for r in data]
        ax.plot(ks, tbts, marker="o", label=f"α={alpha}", color=colors[i], linewidth=2)
    ax.set_xlabel("Speculation Length K")
    ax.set_ylabel("TBT P50 (ms)")
    ax.set_title("TBT (lower=better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. E2E P50
    ax = axes[0, 1]
    for i, alpha in enumerate(alpha_values):
        data = [r for r in results if r["alpha"] == alpha]
        ks = [r["K"] for r in data]
        e2es = [r["e2e_p50"] for r in data]
        ax.plot(ks, e2es, marker="s", label=f"α={alpha}", color=colors[i], linewidth=2)
    ax.set_xlabel("Speculation Length K")
    ax.set_ylabel("E2E P50 (ms)")
    ax.set_title("E2E Latency (lower=better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Decode Throughput
    ax = axes[1, 0]
    for i, alpha in enumerate(alpha_values):
        data = [r for r in results if r["alpha"] == alpha]
        ks = [r["K"] for r in data]
        tputs = [r["decode_throughput"] for r in data]
        ax.plot(ks, tputs, marker="^", label=f"α={alpha}", color=colors[i], linewidth=2)
    ax.set_xlabel("Speculation Length K")
    ax.set_ylabel("Decode Throughput (tok/s)")
    ax.set_title("Throughput (higher = better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Speedup heatmap (TBT speedup vs baseline)
    ax = axes[1, 1]
    baseline_tbt = None
    for r in results:
        if r["K"] == 1:
            baseline_tbt = r["tbt_p50"]
            break

    if baseline_tbt:
        speedups = []
        for alpha in alpha_values:
            row = []
            for K in K_values:
                r = next((r for r in results if r["K"] == K and r["alpha"] == alpha), None)
                if r and r["tbt_p50"] > 0:
                    row.append(baseline_tbt / r["tbt_p50"])
                else:
                    row.append(1.0)
            speedups.append(row)

        im = ax.imshow(speedups, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(K_values)))
        ax.set_xticklabels(K_values)
        ax.set_yticks(range(len(alpha_values)))
        ax.set_yticklabels([f"α={a}" for a in alpha_values])
        ax.yaxis.set_tick_params(pad=8)
        ax.set_xlabel("Speculation Length K")
        ax.set_title("TBT Speedup vs Baseline")

        # Skip first column annotations (baseline ≈ 1.00x, adds no info)
        for i in range(len(alpha_values)):
            for j in range(1, len(K_values)):
                ax.text(j, i, f"{speedups[i][j]:.2f}x",
                        ha="center", va="center", fontsize=10, weight="bold")

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/speculative_decoding.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_dir}/speculative_decoding.pdf", bbox_inches="tight")
    plt.close()
    print(f"\n图表已保存: {output_dir}/speculative_decoding.png")


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding 配置实验")
    parser.add_argument("--qps", type=float, default=10.0)
    parser.add_argument("--time_limit", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    K_values = [1, 2, 4, 6, 8, 10, 12]
    alpha_values = [0.6, 0.7, 0.8, 0.9]

    print("=" * 70)
    print("  Speculative Decoding Configuration Analysis")
    print("=" * 70)
    print(f"\n配置: QPS={args.qps}, 时长={args.time_limit}s")
    print(f"模型: Qwen3-30B-A3B (48 层)")
    print(f"集群: PD Disagg (2×4 A800, TP=4)")
    print(f"Draft model: 4 层, dim=512")
    print(f"K ∈ {K_values}, α ∈ {alpha_values}")

    all_results = []

    # 基线: K=1 (标准 decode)
    print(f"\n{'='*60}")
    print(f"基线: 标准 decode (K=1)")
    print(f"{'='*60}")
    baseline = run_one(K=1, alpha=1.0, qps=args.qps, time_limit=args.time_limit, seed=args.seed)
    all_results.append(baseline)
    print(f"  TBT P50: {baseline['tbt_p50']:.3f} ms")
    print(f"  E2E P50: {baseline['e2e_p50']:.1f} ms")
    print(f"  Decode tok/s: {baseline['decode_throughput']:.1f}")

    # 扫描 K × α
    for K in K_values[1:]:
        for alpha in alpha_values:
            print(f"\n{'='*60}")
            print(f"K={K}, α={alpha}")
            print(f"{'='*60}")

            r = run_one(K=K, alpha=alpha, qps=args.qps, time_limit=args.time_limit, seed=args.seed)
            if r is None:
                print("  无完成请求, 跳过")
                continue

            all_results.append(r)

            tbt_speedup = baseline["tbt_p50"] / r["tbt_p50"] if r["tbt_p50"] > 0 else 0
            e2e_speedup = baseline["e2e_p50"] / r["e2e_p50"] if r["e2e_p50"] > 0 else 0

            print(f"  TBT P50:  {r['tbt_p50']:.3f} ms ({tbt_speedup:.2f}x)")
            print(f"  E2E P50:  {r['e2e_p50']:.1f} ms ({e2e_speedup:.2f}x)")
            print(f"  Decode tok/s: {r['decode_throughput']:.1f}")

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "speculative_decoding.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存: {output_dir / 'speculative_decoding.json'}")

    # 绘图
    plot_results(all_results, str(output_dir))

    # 找最优配置
    best = min(
        [r for r in all_results if r["K"] > 1],
        key=lambda r: r["tbt_p50"],
        default=None
    )
    if best:
        print(f"\n[最优配置] K={best['K']}, α={best['alpha']}")
        print(f"  TBT P50: {best['tbt_p50']:.3f} ms (baseline × {baseline['tbt_p50']/best['tbt_p50']:.2f})")
        print(f"  E2E P50: {best['e2e_p50']:.1f} ms (baseline × {baseline['e2e_p50']/best['e2e_p50']:.2f})")

    print("\n实验完成!")


if __name__ == "__main__":
    main()
