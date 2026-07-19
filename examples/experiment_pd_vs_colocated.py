"""PD Disaggregated vs Colocated 部署对比实验

对比两种部署模式在不同 QPS 下的性能差异:
  1. PD Disaggregated: 1 Prefill 节点 + 1 Decode 节点 (各 4×A800)
  2. Colocated: 1 节点 (4×A800) prefill + decode 共享

实验设计:
  - 变化 QPS (5, 10, 15, 20, 25, 30)
  - 测量 TTFT, TBT, E2E, throughput
  - 分析在什么 QPS 下 PD 分离优于 Colocated

预期结果:
  - 低 QPS: Colocated 更优 (无 KV Cache 传输开销)
  - 高 QPS: PD Disaggregated 更优 (计算资源隔离，无 prefill-decode 竞争)

用法:
  python3 examples/experiment_pd_vs_colocated.py
  python3 examples/experiment_pd_vs_colocated.py --qps_list 5,10,20,30
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from main import create_disaggregated_simulator, create_colocated_simulator


def extract_metrics(ms) -> Dict:
    """从 MetricsStore 提取关键指标。"""
    completed = [m for m in ms._request_metrics.values() if m.decode_end_time > 0]
    if not completed:
        return {"completed": 0}

    ttfts = [m.ttft for m in completed]
    tbts = [m.tbt for m in completed if m.tbt > 0]
    e2es = [m.e2e_latency for m in completed]

    total_decode = sum(m.decode_tokens for m in completed if m.decode_tokens > 0)
    total_prefill = sum(m.prefill_tokens for m in completed if m.prefill_tokens > 0)
    wall_time = max(m.decode_end_time for m in completed)
    first_arrival = min(m.arrival_time for m in completed)
    effective_s = (wall_time - first_arrival) / 1000.0 if wall_time > first_arrival else 1.0

    return {
        "completed": len(completed),
        "ttft_p50": float(np.percentile(ttfts, 50)),
        "ttft_p90": float(np.percentile(ttfts, 90)),
        "ttft_p99": float(np.percentile(ttfts, 99)),
        "tbt_p50": float(np.percentile(tbts, 50)) if tbts else 0.0,
        "e2e_p50": float(np.percentile(e2es, 50)),
        "e2e_p90": float(np.percentile(e2es, 90)),
        "e2e_p99": float(np.percentile(e2es, 99)),
        "decode_throughput": total_decode / effective_s if effective_s > 0 else 0.0,
        "prefill_throughput": total_prefill / effective_s if effective_s > 0 else 0.0,
    }


def run_single_config(
    qps: float,
    prefill_length: int,
    decode_length: int,
    prefill_bs: int,
    decode_bs: int,
    tp_size: int,
    time_limit: float,
    seed: int,
    length_distribution: str,
    workload: str = None,
    length_cv: float = 0.5,
) -> Dict:
    """运行一组 PD Disagg + Colocated 对比。"""

    results = {"qps": qps, "disaggregated": {}, "colocated": {}}

    # PD Disaggregated
    sim_d = create_disaggregated_simulator(
        num_gpus_per_node=tp_size,
        qps=qps,
        prefill_length=prefill_length,
        decode_length=decode_length,
        prefill_batch_size=prefill_bs,
        decode_batch_size=decode_bs,
        tp_size=tp_size,
        time_limit_s=time_limit,
        seed=seed,
        length_distribution=length_distribution,
        length_cv=length_cv,
        workload=workload,
    )
    ms_d = sim_d.run()
    results["disaggregated"] = extract_metrics(ms_d)

    # Colocated (相同总 GPU 数: 4 GPU)
    sim_c = create_colocated_simulator(
        num_gpus_per_node=tp_size,
        qps=qps,
        prefill_length=prefill_length,
        decode_length=decode_length,
        prefill_batch_size=prefill_bs,
        decode_batch_size=decode_bs,
        tp_size=tp_size,
        time_limit_s=time_limit,
        seed=seed,
        length_distribution=length_distribution,
        length_cv=length_cv,
        workload=workload,
    )
    ms_c = sim_c.run()
    results["colocated"] = extract_metrics(ms_c)

    return results


def plot_results(results: List[Dict], output_dir: Path):
    """绘制对比图表。"""
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
    
    qps_list = [r["qps"] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Disaggregated vs Colocated Deployment", fontsize=24)

    metrics = [
        ("ttft_p50", "TTFT P50 (ms)", axes[0, 0]),
        ("ttft_p99", "TTFT P99 (ms)", axes[0, 1]),
        ("tbt_p50", "TBT P50 (ms)", axes[0, 2]),
        ("e2e_p50", "E2E P50 (ms)", axes[1, 0]),
        ("e2e_p99", "E2E P99 (ms)", axes[1, 1]),
        ("decode_throughput", "Decode Throughput (tok/s)", axes[1, 2]),
    ]

    for key, ylabel, ax in metrics:
        d_vals = [r["disaggregated"][key] for r in results]
        c_vals = [r["colocated"][key] for r in results]

        x = np.arange(len(qps_list))
        width = 0.35

        ax.bar(x - width/2, d_vals, width, label='Disagg', color='#E74C3C', alpha=0.8)
        ax.bar(x + width/2, c_vals, width, label='Colocated', color='#3498DB', alpha=0.8)

        ax.set_xlabel("QPS")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(qps_list, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "pd_vs_colocated.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "pd_vs_colocated.pdf", bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {output_dir / 'pd_vs_colocated.png'}")


def main():
    parser = argparse.ArgumentParser(description="PD Disagg vs Colocated 对比实验")
    parser.add_argument("--qps_list", type=str, default="5,10,15,20,25,30",
                        help="QPS 列表 (逗号分隔)")
    parser.add_argument("--prefill_length", type=int, default=512)
    parser.add_argument("--decode_length", type=int, default=128)
    parser.add_argument("--prefill_bs", type=int, default=4)
    parser.add_argument("--decode_bs", type=int, default=32)
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--time_limit", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length_distribution", type=str, default="normal",
                        help="长度分布: fixed 或 normal")
    parser.add_argument("--length_cv", type=float, default=0.5)
    parser.add_argument("--workload", type=str, default=None,
                        choices=["chat-1m", "arxiv-4k", "bwb-4k", "default"],
                        help="Vidur workload trace (覆盖 prefill/decode/CV 参数)")
    parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()
    qps_list = [float(x) for x in args.qps_list.split(",")]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 70)
    print("  PD Disaggregated vs Colocated Deployment Comparison")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  QPS:           {qps_list}")
    print(f"  Prefill:       {args.prefill_length} tokens (bs={args.prefill_bs})")
    print(f"  Decode:        {args.decode_length} tokens (bs={args.decode_bs})")
    print(f"  TP:            {args.tp_size}")
    print(f"  时长:          {args.time_limit}s")
    print(f"  长度分布:      {args.length_distribution} (CV={args.length_cv})")
    print()
    print(f"  PD Disagg:     2 节点 (Prefill {args.tp_size}×A800 + Decode {args.tp_size}×A800)")
    print(f"  Colocated:     1 节点 ({args.tp_size}×A800)")
    print()

    all_results = []

    for qps in qps_list:
        print(f"\n{'='*60}")
        print(f"QPS = {qps}")
        print(f"{'='*60}")

        r = run_single_config(
            qps=qps,
            prefill_length=args.prefill_length,
            decode_length=args.decode_length,
            prefill_bs=args.prefill_bs,
            decode_bs=args.decode_bs,
            tp_size=args.tp_size,
            time_limit=args.time_limit,
            seed=args.seed,
            length_distribution=args.length_distribution,
            workload=args.workload,
            length_cv=args.length_cv,
        )

        d = r["disaggregated"]
        c = r["colocated"]

        print(f"\n{'指标':<25} {'PD Disagg':>12} {'Colocated':>12} {'Winner':>10}")
        print(f"{'─'*60}")
        for key, label in [
            ("ttft_p50", "TTFT P50 (ms)"),
            ("ttft_p99", "TTFT P99 (ms)"),
            ("tbt_p50", "TBT P50 (ms)"),
            ("e2e_p50", "E2E P50 (ms)"),
            ("e2e_p99", "E2E P99 (ms)"),
            ("decode_throughput", "Decode tok/s"),
        ]:
            dv = d[key]
            cv = c[key]
            if "throughput" in key:
                winner = "Disagg" if dv > cv else "Colocated"
            else:
                winner = "Disagg" if dv < cv else "Colocated"
            print(f"  {label:<23} {dv:>12.1f} {cv:>12.1f} {winner:>10}")

        print(f"  {'Completed':<23} {d['completed']:>12} {c['completed']:>12}")

        all_results.append(r)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 保存 JSON
    json_path = output_dir / "pd_vs_colocated.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存: {json_path}")

    # 绘图
    plot_results(all_results, output_dir)

    print("\n实验完成!")


if __name__ == "__main__":
    main()
