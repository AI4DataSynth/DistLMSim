#!/usr/bin/env python3
"""Experiment 7: KV Cache Transfer Strategy Comparison

对比 3 种 KV Cache 传输策略: DIRECT / PIPELINED / STORE_FORWARD
论文 §3.4 "KV Cache Transfer" 和 Table 1 中声称支持这 3 种策略。

实验设计:
- 模型: Qwen3-30B-A3B (48 layers, MoE)
- 变化维度: prefill_length (影响 KV cache 大小) 和 QPS
- 测量: TTFT, E2E, transfer time, decode throughput
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, ".")

from main import create_disaggregated_simulator
from distlmsim.types import KVCacheTransferStrategy


@dataclass
class TransferResult:
    strategy: str = ""
    prefill_length: int = 0
    qps: float = 0.0
    chunk_size: int = 0
    ttft_p50: float = 0.0
    ttft_mean: float = 0.0
    e2e_p50: float = 0.0
    tbt_p50: float = 0.0
    completed: int = 0
    decode_tps: float = 0.0
    kv_size_mb: float = 0.0


def run_single(
    strategy: KVCacheTransferStrategy,
    prefill_length: int,
    qps: float,
    chunk_size: int = 0,
    decode_length: int = 128,
    time_limit: float = 30.0,
    seed: int = 42,
) -> TransferResult:
    sim = create_disaggregated_simulator(
        tp_size=4, qps=qps, time_limit_s=time_limit, seed=seed,
        prefill_length=prefill_length, decode_length=decode_length,
        prefill_batch_size=2, decode_batch_size=32,
    )
    sim.config.disaggregated.kv_cache_transfer_strategy = strategy

    if strategy == KVCacheTransferStrategy.PIPELINED:
        sim.config.disaggregated.enable_chunked_prefill = True
        sim.config.disaggregated.prefill_chunk_size = chunk_size or 1024

    ms = sim.run()
    completed = [m for m in ms._request_metrics.values() if m.decode_end_time > 0]

    if not completed:
        return TransferResult(strategy=strategy.name, prefill_length=prefill_length, qps=qps)

    ttfts = [m.ttft for m in completed]
    tbts = [m.tbt for m in completed if m.tbt > 0]
    e2es = [m.e2e_latency for m in completed]

    total_decode = sum(m.decode_tokens for m in completed)
    wall = max(m.decode_end_time for m in completed)
    first = min(m.arrival_time for m in completed)
    eff_s = (wall - first) / 1000.0 if wall > first else 1.0

    kv_bytes = sim._compute_kv_cache_size(completed[0])

    return TransferResult(
        strategy=strategy.name,
        prefill_length=prefill_length,
        qps=qps,
        chunk_size=chunk_size,
        ttft_p50=float(np.percentile(ttfts, 50)),
        ttft_mean=float(np.mean(ttfts)),
        e2e_p50=float(np.percentile(e2es, 50)),
        tbt_p50=float(np.percentile(tbts, 50)) if tbts else 0.0,
        completed=len(completed),
        decode_tps=total_decode / eff_s if eff_s > 0 else 0.0,
        kv_size_mb=kv_bytes / 1024 / 1024,
    )


def main():
    parser = argparse.ArgumentParser(description="KV Cache Transfer Strategy Experiment")
    parser.add_argument("--prefill_lengths", type=str, default="2048,4096,8192")
    parser.add_argument("--qps_values", type=str, default="5,10,20")
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    prefill_lengths = [int(x) for x in args.prefill_lengths.split(",")]
    qps_values = [float(x) for x in args.qps_values.split(",")]

    print()
    print("  DistLMSim KV Cache Transfer Strategy Experiment")
    print(f"  Model:       Qwen3-30B-A3B (48 layers, MoE)")
    print(f"  Cluster:     PD Disagg (2×4 A800, TP=4)")
    print(f"  Prefill:     {prefill_lengths}")
    print(f"  QPS:         {qps_values}")
    print(f"  Chunk size:  {args.chunk_size} (for PIPELINED)")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    strategies = [
        (KVCacheTransferStrategy.DIRECT, "DIRECT", 0),
        (KVCacheTransferStrategy.PIPELINED, "PIPELINED", args.chunk_size),
        (KVCacheTransferStrategy.STORE_FORWARD, "STORE_FORWARD", 0),
    ]

    all_results: List[TransferResult] = []

    for pf_len in prefill_lengths:
        for qps in qps_values:
            print(f"\n{'='*70}")
            print(f"  Prefill={pf_len}, QPS={qps}")
            print(f"{'='*70}")
            print(f"  {'Strategy':15s} {'TTFT P50':>12} {'E2E P50':>12} "
                  f"{'TBT P50':>10} {'Dec tok/s':>10} {'Done':>6}")
            print("  " + "─" * 70)

            for strat, name, cs in strategies:
                r = run_single(strat, pf_len, qps, chunk_size=cs)
                r.prefill_length = pf_len
                r.qps = qps
                all_results.append(r)
                print(f"  {name:15s} {r.ttft_p50:>12.1f} {r.e2e_p50:>12.1f} "
                      f"{r.tbt_p50:>10.4f} {r.decode_tps:>10.1f} {r.completed:>6}")

    # Save JSON
    json_data = [{
        "strategy": r.strategy, "prefill_length": r.prefill_length,
        "qps": r.qps, "chunk_size": r.chunk_size,
        "ttft_p50": r.ttft_p50, "ttft_mean": r.ttft_mean,
        "e2e_p50": r.e2e_p50, "tbt_p50": r.tbt_p50,
        "completed": r.completed, "decode_tps": r.decode_tps,
        "kv_size_mb": r.kv_size_mb,
    } for r in all_results]

    json_path = output_dir / "kv_transfer_strategies.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\n结果已保存: {json_path}")

    # Plot
    plot_results(all_results, prefill_lengths, qps_values, output_dir)


def plot_results(results, prefill_lengths, qps_values, output_dir):
    """生成 KV 传输策略对比图表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = {"DIRECT": "#3498db", "PIPELINED": "#2ecc71", "STORE_FORWARD": "#e74c3c"}
    markers = {"DIRECT": "o", "PIPELINED": "s", "STORE_FORWARD": "^"}

    for pf_len in prefill_lengths:
        # TTFT P50 vs QPS
        ax = axes[0]
        for strat_name in ["DIRECT", "PIPELINED", "STORE_FORWARD"]:
            strat_results = [r for r in results
                             if r.strategy == strat_name and r.prefill_length == pf_len]
            qs = [r.qps for r in strat_results]
            ttfts = [r.ttft_p50 for r in strat_results]
            ax.plot(qs, ttfts, f'-{markers[strat_name]}', color=colors[strat_name],
                    label=f"{strat_name} (pf={pf_len})" if pf_len == prefill_lengths[0] else None,
                    markersize=7)

    axes[0].set_xlabel("QPS (requests/sec)")
    axes[0].set_ylabel("TTFT P50 (ms)")
    axes[0].set_title("KV Transfer Strategy: TTFT vs QPS")
    axes[0].legend(fontsize=7, ncol=3)
    axes[0].grid(True, alpha=0.3)

    # E2E P50 vs prefill_length (at middle QPS)
    mid_qps = qps_values[len(qps_values) // 2]
    ax = axes[1]
    for strat_name in ["DIRECT", "PIPELINED", "STORE_FORWARD"]:
        strat_results = [r for r in results
                         if r.strategy == strat_name and r.qps == mid_qps]
        pf_lens = [r.prefill_length for r in strat_results]
        e2es = [r.e2e_p50 for r in strat_results]
        ax.plot(pf_lens, e2es, f'-{markers[strat_name]}', color=colors[strat_name],
                label=strat_name, markersize=8)

    axes[1].set_xlabel("Prefill Length (tokens)")
    axes[1].set_ylabel("E2E P50 (ms)")
    axes[1].set_title(f"KV Transfer Strategy: E2E vs Prefill Length (QPS={mid_qps:.0f})")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Decode throughput comparison
    ax = axes[2]
    for strat_name in ["DIRECT", "PIPELINED", "STORE_FORWARD"]:
        strat_results = [r for r in results
                         if r.strategy == strat_name and r.qps == mid_qps]
        pf_lens = [r.prefill_length for r in strat_results]
        tps = [r.decode_tps for r in strat_results]
        ax.plot(pf_lens, tps, f'-{markers[strat_name]}', color=colors[strat_name],
                label=strat_name, markersize=8)

    axes[2].set_xlabel("Prefill Length (tokens)")
    axes[2].set_ylabel("Decode Throughput (tok/s)")
    axes[2].set_title(f"KV Transfer Strategy: Throughput vs Prefill Length (QPS={mid_qps:.0f})")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "kv_transfer_strategies.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "kv_transfer_strategies.pdf", bbox_inches="tight")
    print(f"\n图表已保存: {output_dir / 'kv_transfer_strategies.pdf'}")


if __name__ == "__main__":
    main()
