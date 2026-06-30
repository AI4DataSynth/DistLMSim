#!/usr/bin/env python3
"""Experiment 6: Chunked Prefill Analysis

研究 chunked prefill 对 prefill 延迟和 TTFT 的影响。
论文 §3.4 声称 DistLMSim 支持 prefill chunking。

实验设计:
- 固定 prefill_length (长上下文场景: 4096, 8192)
- 变化 chunk_size: [256, 512, 1024, 2048, 4096, no_chunk]
- 测量: prefill 时间, TTFT P50, decode throughput
- 对比: chunked vs non-chunked

用法:
    python examples/experiment_chunked_prefill.py
    python examples/experiment_chunked_prefill.py --output_dir results
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, ".")

from main import create_disaggregated_simulator


@dataclass
class ChunkResult:
    prefill_length: int = 0
    chunk_size: int = 0  # 0 = no chunking
    num_chunks: int = 0
    # Metrics
    ttft_p50: float = 0.0
    ttft_mean: float = 0.0
    e2e_p50: float = 0.0
    tbt_p50: float = 0.0
    completed: int = 0
    decode_tps: float = 0.0


def run_single(
    prefill_length: int,
    chunk_size: int,  # 0 = disable chunking
    decode_length: int = 128,
    qps: float = 10.0,
    time_limit: float = 30.0,
    seed: int = 42,
) -> ChunkResult:
    sim = create_disaggregated_simulator(
        tp_size=4, qps=qps, time_limit_s=time_limit, seed=seed,
        prefill_length=prefill_length, decode_length=decode_length,
        prefill_batch_size=2, decode_batch_size=32,
    )

    if chunk_size > 0:
        sim.config.disaggregated.enable_chunked_prefill = True
        sim.config.disaggregated.prefill_chunk_size = chunk_size
        num_chunks = (prefill_length + chunk_size - 1) // chunk_size
    else:
        sim.config.disaggregated.enable_chunked_prefill = False
        num_chunks = 1

    ms = sim.run()
    completed = [m for m in ms._request_metrics.values() if m.decode_end_time > 0]

    if not completed:
        return ChunkResult(
            prefill_length=prefill_length, chunk_size=chunk_size,
            num_chunks=num_chunks,
        )

    ttfts = [m.ttft for m in completed]
    tbts = [m.tbt for m in completed if m.tbt > 0]
    e2es = [m.e2e_latency for m in completed]

    total_decode = sum(m.decode_tokens for m in completed if m.decode_tokens > 0)
    wall_time = max(m.decode_end_time for m in completed)
    first_arrival = min(m.arrival_time for m in completed)
    eff_s = (wall_time - first_arrival) / 1000.0 if wall_time > first_arrival else 1.0

    return ChunkResult(
        prefill_length=prefill_length,
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        ttft_p50=float(np.percentile(ttfts, 50)),
        ttft_mean=float(np.mean(ttfts)),
        e2e_p50=float(np.percentile(e2es, 50)),
        tbt_p50=float(np.percentile(tbts, 50)) if tbts else 0.0,
        completed=len(completed),
        decode_tps=total_decode / eff_s if eff_s > 0 else 0.0,
    )


def print_table(results: List[ChunkResult], title: str = ""):
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")

    print(f"\n{'Chunk':>8} {'Chunks':>6} {'TTFT P50':>12} {'TTFT Mean':>12} "
          f"{'E2E P50':>12} {'TBT P50':>10} {'Dec tok/s':>10} {'Done':>6}")
    print("─" * 80)

    for r in results:
        label = f"{r.chunk_size}" if r.chunk_size > 0 else "no_chunk"
        print(f"{label:>8} {r.num_chunks:>6} {r.ttft_p50:>12.1f} {r.ttft_mean:>12.1f} "
              f"{r.e2e_p50:>12.1f} {r.tbt_p50:>10.4f} {r.decode_tps:>10.1f} {r.completed:>6}")


def plot_results(all_results: Dict[int, List[ChunkResult]], output_dir: Path):
    """生成 chunked prefill 分析图表。"""
    import matplotlib
    matplotlib.use("Agg")
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    for idx, (pf_len, results) in enumerate(all_results.items()):
        chunk_sizes = []
        ttft_p50s = []
        tbt_p50s = []
        for r in results:
            label = r.chunk_size if r.chunk_size > 0 else pf_len  # no_chunk = full length
            chunk_sizes.append(label)
            ttft_p50s.append(r.ttft_p50)
            tbt_p50s.append(r.tbt_p50)

        color = colors[idx % len(colors)]

        # TTFT P50 vs chunk size
        ax = axes[0]
        ax.plot(chunk_sizes[:-1], ttft_p50s[:-1], 'o-', color=color,
                label=f"Prefill={pf_len}", markersize=8)
        # Mark no-chunk baseline
        ax.axhline(y=ttft_p50s[-1], color=color, linestyle="--", alpha=0.5)
        ax.scatter([pf_len], [ttft_p50s[-1]], marker="s", s=120,
                   color=color, edgecolors="black", zorder=5, label=f"No-chunk ({pf_len})")

        # TBT P50 vs chunk size
        ax = axes[1]
        ax.plot(chunk_sizes[:-1], tbt_p50s[:-1], 'o-', color=color,
                label=f"Prefill={pf_len}", markersize=8)
        ax.axhline(y=tbt_p50s[-1], color=color, linestyle="--", alpha=0.5)
        ax.scatter([pf_len], [tbt_p50s[-1]], marker="s", s=120,
                   color=color, edgecolors="black", zorder=5, label=f"No-chunk ({pf_len})")

    axes[0].set_xlabel("Chunk Size (tokens)")
    axes[0].set_ylabel("TTFT P50 (ms)")
    axes[0].set_title("TTFT vs Chunk Size")
    axes[0].set_xscale("log", base=2)
    axes[0].legend(fontsize=16)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Chunk Size (tokens)")
    axes[1].set_ylabel("TBT P50 (ms)")
    axes[1].set_title("TBT vs Chunk Size")
    axes[1].set_xscale("log", base=2)
    axes[1].legend(fontsize=16)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "chunked_prefill.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "chunked_prefill.pdf", bbox_inches="tight")
    print(f"\n图表已保存: {output_dir / 'chunked_prefill.pdf'}")


def main():
    parser = argparse.ArgumentParser(description="Chunked Prefill Analysis")
    parser.add_argument("--prefill_lengths", type=str, default="4096,8192",
                        help="Comma-separated prefill lengths")
    parser.add_argument("--chunk_sizes", type=str, default="256,512,1024,2048,4096",
                        help="Comma-separated chunk sizes (0=no chunk)")
    parser.add_argument("--qps", type=float, default=10.0)
    parser.add_argument("--time_limit", type=float, default=30.0)
    parser.add_argument("--decode_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    prefill_lengths = [int(x) for x in args.prefill_lengths.split(",")]
    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]

    print()
    print("  DistLMSim Chunked Prefill Experiment")
    print(f"  Model:         Qwen3-30B-A3B (48 layers, MoE)")
    print(f"  Cluster:       PD Disagg (2×4 A800, TP=4)")
    print(f"  Prefill lens:  {prefill_lengths}")
    print(f"  Chunk sizes:   {chunk_sizes} + [no_chunk]")
    print(f"  QPS={args.qps}  Decode={args.decode_length}  Time={args.time_limit}s")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    all_results: Dict[int, List[ChunkResult]] = {}

    for pf_len in prefill_lengths:
        results = []
        print(f"\n{'='*70}")
        print(f"  Prefill Length = {pf_len} tokens")
        print(f"{'='*70}")

        for cs in chunk_sizes:
            if cs >= pf_len:
                # Skip chunk sizes >= prefill length (equivalent to no chunk)
                continue
            label = f"chunk={cs}"
            print(f"\n  ── {label} ──", flush=True)
            r = run_single(
                prefill_length=pf_len, chunk_size=cs,
                decode_length=args.decode_length,
                qps=args.qps, time_limit=args.time_limit,
                seed=args.seed,
            )
            results.append(r)
            print(f"  TTFT P50={r.ttft_p50:.1f}ms, E2E P50={r.e2e_p50:.1f}ms, "
                  f"TBT P50={r.tbt_p50:.4f}ms, Dec tok/s={r.decode_tps:.1f}")

        # No-chunk baseline
        print(f"\n  ── no_chunk (full {pf_len} tokens) ──", flush=True)
        r_no = run_single(
            prefill_length=pf_len, chunk_size=0,
            decode_length=args.decode_length,
            qps=args.qps, time_limit=args.time_limit,
            seed=args.seed,
        )
        results.append(r_no)
        print(f"  TTFT P50={r_no.ttft_p50:.1f}ms, E2E P50={r_no.e2e_p50:.1f}ms, "
              f"TBT P50={r_no.tbt_p50:.4f}ms, Dec tok/s={r_no.decode_tps:.1f}")

        all_results[pf_len] = results
        print_table(results, title=f"Prefill={pf_len} tokens")

    # Save JSON
    json_data = []
    for pf_len, results in all_results.items():
        for r in results:
            json_data.append({
                "prefill_length": r.prefill_length,
                "chunk_size": r.chunk_size,
                "num_chunks": r.num_chunks,
                "ttft_p50": r.ttft_p50,
                "ttft_mean": r.ttft_mean,
                "e2e_p50": r.e2e_p50,
                "tbt_p50": r.tbt_p50,
                "completed": r.completed,
                "decode_tps": r.decode_tps,
            })

    json_path = output_dir / "chunked_prefill.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\n结果已保存: {json_path}")

    # Plot
    plot_results(all_results, output_dir)

    # Summary insight
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    for pf_len, results in all_results.items():
        no_chunk = results[-1]
        best_chunk = min(results[:-1], key=lambda r: r.ttft_p50)
        print(f"\n  Prefill={pf_len}:")
        print(f"    No-chunk TTFT:  {no_chunk.ttft_p50:.1f}ms")
        print(f"    Best chunk:     {best_chunk.chunk_size} → TTFT {best_chunk.ttft_p50:.1f}ms "
              f"({no_chunk.ttft_p50/best_chunk.ttft_p50:.2f}x speedup)")
        if no_chunk.ttft_p50 > 0:
            speedup = no_chunk.ttft_p50 / best_chunk.ttft_p50
            print(f"    Chunked prefill reduces TTFT by {(1-1/speedup)*100:.1f}%")

    print("\n实验完成!")


if __name__ == "__main__":
    main()
