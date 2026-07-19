#!/usr/bin/env python3
"""实验: Vidur 三种 Workload 对比 (Chat-1M, Arxiv-4K, BWB-4K)

基于 Vidur 论文 Section 7.1 的三种真实 workload 统计量，
在 DistLMSim 中用合成请求（匹配 mean/std）运行调度+PD 实验。

Workload 画像:
  Chat-1M:  短 prefill 高方差, 短 decode       (P:D=2.20, 类聊天)
  Arxiv-4K: 长 prefill 低方差, 短 decode       (P:D=8.88, 类摘要)
  BWB-4K:   中 prefill, 极长 decode 低方差     (P:D=0.67, 类翻译)

输出:
  results/workload_comparison.json
  results/workload_comparison.pdf
"""

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distlmsim.config import (
    ModelConfig, DeviceSKUConfig, NVLinkConfig, RDMAConfig,
    NetworkTopologyConfig, SimulationConfig, MetricsConfig,
    RequestGeneratorConfig, DisaggregatedConfig,
)
from distlmsim.context import SimContext
from distlmsim.metrics import MetricsStore
from distlmsim.types import RDMAProtocolType


# ─── Vidur Workload 定义 ──────────────────────────────────────────────────────

WORKLOADS = {
    "Chat-1M": {
        "prefill_mean": 462,
        "decode_mean": 210,
        "prefill_cv": 1.42,   # std/mean = 657/462
        "decode_cv": 0.86,    # std/mean = 180/210
        "description": "Short prefill (high variance), short decode — chat",
    },
    "Arxiv-4K": {
        "prefill_mean": 2588,
        "decode_mean": 291,
        "prefill_cv": 0.36,   # std/mean = 945/2588
        "decode_cv": 1.82,    # std/mean = 531/291
        "description": "Long prefill, short decode — summarization",
    },
    "BWB-4K": {
        "prefill_mean": 1072,
        "decode_mean": 1602,
        "prefill_cv": 0.43,   # std/mean = 462/1072
        "decode_cv": 0.17,    # std/mean = 267/1602
        "description": "Medium prefill, very long decode — translation",
    },
    "Default": {
        "prefill_mean": 512,
        "decode_mean": 128,
        "prefill_cv": 0.5,
        "decode_cv": 0.5,
        "description": "DistLMSim default synthetic workload",
    },
}

SCHEDULERS = ["fcfs", "sjf", "po", "srtf"]
QPS_LEVELS = [5, 10, 20]


# ─── 模拟器创建 ──────────────────────────────────────────────────────────────

def create_simulator(
    workload_name: str,
    scheduler: str = "fcfs",
    qps: float = 10.0,
    time_limit_s: float = 30.0,
    seed: int = 42,
):
    """创建 DisaggregatedSimulator，配置 Vidur workload 参数。"""
    from main import DisaggregatedSimulator

    wl = WORKLOADS[workload_name]

    device = DeviceSKUConfig()
    model = ModelConfig(
        model_name="Qwen3-30B-A3B",
        num_layers=48, num_q_heads=32, num_kv_heads=4,
        embedding_dim=2048, num_experts=128, top_k_experts=8,
    )
    nvlink = NVLinkConfig(bandwidth_gbps=600.0)
    rdma = RDMAConfig(protocol=RDMAProtocolType.ROCE_V2, bandwidth_gbps=200.0)
    network = NetworkTopologyConfig(nvlink=nvlink, rdma=rdma)

    request_config = RequestGeneratorConfig(
        qps=qps,
        prefill_length=wl["prefill_mean"],
        decode_length=wl["decode_mean"],
        prefill_length_cv=wl["prefill_cv"],
        decode_length_cv=wl["decode_cv"],
        length_distribution="normal",
    )

    disaggregated = DisaggregatedConfig(
        prefill_batch_size=2,
        decode_batch_size=32,
        gpu_memory_utilization=0.9,
    )

    metrics_config = MetricsConfig(enable_detailed_logging=False)
    ms = MetricsStore(metrics_config)

    ctx = SimContext(
        model_config=model,
        device_config=device,
        network_config=network,
        num_gpus_per_node=4,
        tp_size=4,
        metrics_store=ms,
    )

    config = SimulationConfig(
        seed=seed,
        time_limit_s=time_limit_s,
        request=request_config,
        disaggregated=disaggregated,
        metrics=metrics_config,
    )

    sim = DisaggregatedSimulator(ctx, config, prefill_schedule_policy=scheduler)
    return sim


# ─── 实验运行 ──────────────────────────────────────────────────────────────

def run_single(
    workload_name: str,
    scheduler: str,
    qps: float,
    time_limit_s: float = 30.0,
    seed: int = 42,
) -> Dict:
    """运行单个实验配置，返回指标字典。"""
    sim = create_simulator(workload_name, scheduler, qps, time_limit_s, seed)
    ms = sim.run()
    ms.finalize()

    completed = [m for m in ms._request_metrics.values() if m.decode_end_time > 0]
    if not completed:
        return {"completed": 0}

    ttft_vals = sorted([m.ttft for m in completed])
    tbt_vals = sorted([m.tbt for m in completed if m.tbt > 0])
    e2e_vals = sorted([m.e2e_latency for m in completed])

    def percentile(vals, p):
        if not vals:
            return 0
        idx = int(len(vals) * p / 100)
        return vals[min(idx, len(vals) - 1)]

    return {
        "workload": workload_name,
        "scheduler": scheduler,
        "qps": qps,
        "completed": len(completed),
        "ttft_mean": float(np.mean(ttft_vals)),
        "ttft_p50": percentile(ttft_vals, 50),
        "ttft_p90": percentile(ttft_vals, 90),
        "ttft_p99": percentile(ttft_vals, 99),
        "tbt_mean": float(np.mean(tbt_vals)),
        "tbt_p50": percentile(tbt_vals, 50),
        "tbt_p90": percentile(tbt_vals, 90),
        "e2e_mean": float(np.mean(e2e_vals)),
        "e2e_p50": percentile(e2e_vals, 50),
        "e2e_p90": percentile(e2e_vals, 90),
    }


def run_all(
    workloads: Optional[List[str]] = None,
    schedulers: Optional[List[str]] = None,
    qps_levels: Optional[List[float]] = None,
    time_limit_s: float = 30.0,
    seed: int = 42,
) -> List[Dict]:
    """运行所有 workload × scheduler × QPS 组合。"""
    workloads = workloads or list(WORKLOADS.keys())
    schedulers = schedulers or SCHEDULERS
    qps_levels = qps_levels or QPS_LEVELS

    results = []
    total = len(workloads) * len(schedulers) * len(qps_levels)
    idx = 0

    for wl in workloads:
        for sched in schedulers:
            for qps in qps_levels:
                idx += 1
                print(f"  [{idx}/{total}] {wl} + {sched} + QPS={qps} ...", end="", flush=True)
                r = run_single(wl, sched, qps, time_limit_s, seed)
                print(f" done={r['completed']}, TTFT P50={r.get('ttft_p50', 0):.1f}ms, "
                      f"TBT P50={r.get('tbt_p50', 0):.2f}ms")
                results.append(r)

    return results


# ─── 结果打印 ──────────────────────────────────────────────────────────────

def print_summary(results: List[Dict]):
    """打印论文风格的汇总表。"""
    print("\n" + "=" * 100)
    print("  Vidur Workload Comparison: DistLMSim Results")
    print("=" * 100)

    for qps in QPS_LEVELS:
        print(f"\n{'─' * 100}")
        print(f"  QPS = {qps}")
        print(f"{'─' * 100}")
        print(f"  {'Workload':<12} {'Scheduler':<10} {'TTFT P50':>10} {'TTFT Mean':>10} "
              f"{'TBT P50':>10} {'E2E P50':>10} {'Completed':>10}")
        print(f"  {'─' * 82}")

        for wl in WORKLOADS:
            for sched in SCHEDULERS:
                r = next((x for x in results
                         if x["workload"] == wl and x["scheduler"] == sched and x["qps"] == qps),
                        None)
                if r and r["completed"] > 0:
                    print(f"  {wl:<12} {sched:<10} {r['ttft_p50']:>10.1f} {r['ttft_mean']:>10.1f} "
                          f"{r['tbt_p50']:>10.2f} {r['e2e_p50']:>10.1f} {r['completed']:>10}")

    # SJF vs FCFS improvement table
    print(f"\n{'═' * 100}")
    print("  SJF vs FCFS TTFT P50 Improvement")
    print(f"{'═' * 100}")
    print(f"  {'Workload':<12} {'QPS':>5} {'FCFS TTFT P50':>15} {'SJF TTFT P50':>15} {'Improvement':>12}")
    print(f"  {'─' * 62}")

    for wl in WORKLOADS:
        for qps in QPS_LEVELS:
            fcfs = next((x for x in results
                        if x["workload"] == wl and x["scheduler"] == "fcfs" and x["qps"] == qps),
                       None)
            sjf = next((x for x in results
                       if x["workload"] == wl and x["scheduler"] == "sjf" and x["qps"] == qps),
                      None)
            if fcfs and sjf and fcfs["completed"] > 0 and sjf["completed"] > 0:
                ratio = fcfs["ttft_p50"] / max(sjf["ttft_p50"], 0.1)
                print(f"  {wl:<12} {qps:>5.0f} {fcfs['ttft_p50']:>14.1f}ms {sjf['ttft_p50']:>14.1f}ms "
                      f"{ratio:>10.1f}×")


# ─── 主函数 ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vidur workload comparison")
    parser.add_argument("--time_limit", type=float, default=30.0, help="Simulation time (s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workloads", nargs="+", default=list(WORKLOADS.keys()))
    parser.add_argument("--schedulers", nargs="+", default=SCHEDULERS)
    parser.add_argument("--qps", nargs="+", type=float, default=QPS_LEVELS)
    args = parser.parse_args()

    print("=" * 100)
    print("  Vidur Workload Comparison Experiment")
    print("=" * 100)
    print()
    for name, wl in WORKLOADS.items():
        if name in args.workloads:
            print(f"  {name:<12}: prefill={wl['prefill_mean']}±{int(wl['prefill_mean']*wl['prefill_cv'])}, "
                  f"decode={wl['decode_mean']}±{int(wl['decode_mean']*wl['decode_cv'])}, "
                  f"P:D={wl['prefill_mean']/wl['decode_mean']:.2f} — {wl['description']}")
    print()
    print(f"  Schedulers: {args.schedulers}")
    print(f"  QPS levels: {args.qps}")
    print(f"  Time limit: {args.time_limit}s, Seed: {args.seed}")
    print()

    results = run_all(args.workloads, args.schedulers, args.qps, args.time_limit, args.seed)
    print_summary(results)

    # Save JSON
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "workload_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
