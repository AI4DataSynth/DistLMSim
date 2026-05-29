"""调度器对比实验

验证不同请求调度策略在**排队条件**下对推理服务性能指标的影响。

实验设计:
  - 集群: 1 Prefill 节点 (4x A800, TP=4) + 1 Decode 节点 (4x A800, TP=4)
  - 互联: NVLink 600 GB/s (节点内) + RDMA RoCEv2 200 Gb/s (节点间)
  - 模型: Qwen3-30B-A3B (48 层, MoE 128 专家)
  - 请求: 变长 (normal 分布, CV=0.5), 使不同调度策略产生差异化结果

  关键: prefill_batch_size 必须小于 QPS * prefill_time,
  使请求在 prefill 节点处理当前 batch 期间持续积累, 调度器才有选择空间。

调度策略:
  1. FCFS  — First-Come-First-Served, 按到达时间选取 (基线)
  2. SJF   — Shortest-Job-First, 按 prefill tokens 升序 (优化平均 TTFT)
  3. LJF   — Longest-Job-First, 按 prefill tokens 降序 (反面参照)
  4. SRTF  — Shortest-Remaining-Time-First, 按 decode tokens 升序 (优化 E2E)
  5. Random — 随机选取 (无策略基线)

预期结果:
  - SJF/SRTF: 显著降低 TTFT/E2E 的 P50 (短请求跳队), 但 P99 升高 (长请求饥饿)
  - LJF: TTFT/E2E 全面劣化 (head-of-line blocking)
  - FCFS: 均衡但非最优, 尾延迟比 SJF 低
  - Random: 介于 FCFS 和 SJF 之间

用法:
  python3 examples/experiment_schedulers.py
  python3 examples/experiment_schedulers.py --qps 30 --prefill_batch_size 4
  python3 examples/experiment_schedulers.py --seeds 42,43,44  # 多种子取平均
"""

import sys
import logging
import argparse
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, ".")

from main import create_disaggregated_simulator

logger = logging.getLogger(__name__)


# ─── 实验配置 ─────────────────────────────────────────────────────────────────

SCHEDULERS = [
    ("fcfs",   "FCFS",   "先到先服务"),
    ("sjf",    "SJF",    "短 prefill 优先"),
    ("ljf",    "LJF",    "长 prefill 优先"),
    ("srtf",   "SRTF",   "短 decode 优先"),
    ("random", "Random", "随机"),
]


@dataclass
class ExperimentResult:
    scheduler_key: str
    scheduler_name: str
    description: str
    num_requests: int = 0
    # TTFT
    ttft_p50: float = 0.0
    ttft_p90: float = 0.0
    ttft_p99: float = 0.0
    ttft_mean: float = 0.0
    # TBT
    tbt_p50: float = 0.0
    tbt_p90: float = 0.0
    tbt_mean: float = 0.0
    # E2E
    e2e_p50: float = 0.0
    e2e_p90: float = 0.0
    e2e_p99: float = 0.0
    e2e_mean: float = 0.0
    # 吞吐
    decode_tps: float = 0.0
    prefill_tps: float = 0.0
    # 公平性
    ttft_tail_ratio: float = 0.0
    e2e_tail_ratio: float = 0.0


def run_one(
    scheduler_key: str,
    scheduler_name: str,
    description: str,
    args,
) -> ExperimentResult:
    """运行单个调度策略。"""
    sim = create_disaggregated_simulator(
        qps=args.qps,
        prefill_length=args.prefill_length,
        decode_length=args.decode_length,
        prefill_batch_size=args.prefill_batch_size,
        decode_batch_size=args.decode_batch_size,
        time_limit_s=args.time_limit,
        seed=args.seed,
        prefill_schedule_policy=scheduler_key,
        decode_schedule_policy=scheduler_key,
        length_distribution="normal",
    )
    metrics = sim.run()

    completed = [
        m for m in metrics._request_metrics.values()
        if m.decode_end_time > 0
    ]
    if not completed:
        return ExperimentResult(scheduler_key, scheduler_name, description)

    ttfts = np.array([m.ttft for m in completed])
    tbts = np.array([m.tbt for m in completed if m.tbt > 0])
    e2es = np.array([m.e2e_latency for m in completed])

    wall_time = max(m.decode_end_time for m in completed)
    first_arrival = min(m.arrival_time for m in completed)
    eff_s = max(0.001, (wall_time - first_arrival) / 1000.0)

    total_decode = sum(m.decode_tokens for m in completed)
    total_prefill = sum(m.prefill_tokens for m in completed)

    ttft_p50 = float(np.percentile(ttfts, 50))
    e2e_p50 = float(np.percentile(e2es, 50))

    return ExperimentResult(
        scheduler_key=scheduler_key,
        scheduler_name=scheduler_name,
        description=description,
        num_requests=len(completed),
        ttft_p50=ttft_p50,
        ttft_p90=float(np.percentile(ttfts, 90)),
        ttft_p99=float(np.percentile(ttfts, 99)),
        ttft_mean=float(np.mean(ttfts)),
        tbt_p50=float(np.percentile(tbts, 50)) if len(tbts) else 0.0,
        tbt_p90=float(np.percentile(tbts, 90)) if len(tbts) else 0.0,
        tbt_mean=float(np.mean(tbts)) if len(tbts) else 0.0,
        e2e_p50=e2e_p50,
        e2e_p90=float(np.percentile(e2es, 90)),
        e2e_p99=float(np.percentile(e2es, 99)),
        e2e_mean=float(np.mean(e2es)),
        decode_tps=total_decode / eff_s,
        prefill_tps=total_prefill / eff_s,
        ttft_tail_ratio=float(np.percentile(ttfts, 99)) / max(1, ttft_p50),
        e2e_tail_ratio=float(np.percentile(e2es, 99)) / max(1, e2e_p50),
    )


def print_table(results: list[ExperimentResult], title: str = "") -> None:
    if not results:
        return
    if title:
        print(f"\n  {title}")
    w = 10
    header = f"  {'指标':<22}" + "".join(f"  {r.scheduler_name:>{w}}" for r in results)
    print(f"\n{header}")
    print("  " + "─" * (22 + (w + 2) * len(results)))

    def row(label, fmt, values):
        print(f"  {label:<22}" + "".join(f"  {fmt.format(v):>{w}}" for v in values))

    row("TTFT mean (ms)", "{:.0f}", [r.ttft_mean for r in results])
    row("TTFT P50 (ms)", "{:.0f}", [r.ttft_p50 for r in results])
    row("TTFT P90 (ms)", "{:.0f}", [r.ttft_p90 for r in results])
    row("TTFT P99 (ms)", "{:.0f}", [r.ttft_p99 for r in results])
    row("TTFT P99/P50", "{:.2f}x", [r.ttft_tail_ratio for r in results])
    print()
    row("TBT mean (ms)", "{:.2f}", [r.tbt_mean for r in results])
    row("TBT P50 (ms)", "{:.2f}", [r.tbt_p50 for r in results])
    row("TBT P90 (ms)", "{:.2f}", [r.tbt_p90 for r in results])
    print()
    row("E2E mean (ms)", "{:.0f}", [r.e2e_mean for r in results])
    row("E2E P50 (ms)", "{:.0f}", [r.e2e_p50 for r in results])
    row("E2E P90 (ms)", "{:.0f}", [r.e2e_p90 for r in results])
    row("E2E P99 (ms)", "{:.0f}", [r.e2e_p99 for r in results])
    row("E2E P99/P50", "{:.2f}x", [r.e2e_tail_ratio for r in results])
    print()
    row("Decode tok/s", "{:.0f}", [r.decode_tps for r in results])
    row("Prefill tok/s", "{:.0f}", [r.prefill_tps for r in results])
    row("完成请求数", "{:d}", [r.num_requests for r in results])

    # 对比分析
    baseline = results[0]
    print(f"\n  对比 FCFS 基线:")
    for r in results[1:]:
        ttft_d = (r.ttft_mean - baseline.ttft_mean) / max(1, baseline.ttft_mean) * 100
        e2e_d = (r.e2e_mean - baseline.e2e_mean) / max(1, baseline.e2e_mean) * 100
        tail_d = (r.ttft_tail_ratio - baseline.ttft_tail_ratio) / max(0.01, baseline.ttft_tail_ratio) * 100
        sign = lambda x: "+" if x >= 0 else ""
        print(f"    {r.scheduler_name:>8}: TTFT mean {sign(ttft_d)}{ttft_d:.1f}%  "
              f"E2E mean {sign(e2e_d)}{e2e_d:.1f}%  "
              f"TTFT尾延迟比 {sign(tail_d)}{tail_d:.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="DistLMSim 调度器对比实验")
    parser.add_argument("--qps", type=float, default=20.0,
                        help="请求到达率 (req/s)")
    parser.add_argument("--time_limit", type=float, default=30.0,
                        help="模拟时长 (秒)")
    parser.add_argument("--prefill_length", type=int, default=512,
                        help="平均 prefill token 数")
    parser.add_argument("--decode_length", type=int, default=128,
                        help="平均 decode token 数")
    parser.add_argument("--prefill_batch_size", type=int, default=2,
                        help="Prefill 批大小 (越小排队越深, 调度差异越大)")
    parser.add_argument("--decode_batch_size", type=int, default=32,
                        help="Decode 批大小")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--seeds", type=str, default="",
                        help="多种子 (逗号分隔, 取平均)")
    parser.add_argument("--length_cv", type=float, default=0.5,
                        help="请求长度变异系数 (越大调度差异越大)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else [args.seed]

    print()
    print("  DistLMSim 调度器对比实验")
    print(f"  模型:   Qwen3-30B-A3B (48 层, MoE)")
    print(f"  集群:   1 Prefill (4×A800 TP=4) + 1 Decode (4×A800 TP=4)")
    print(f"  互联:   NVLink 600GB/s + RDMA RoCEv2 200Gb/s")
    print(f"  请求:   prefill={args.prefill_length}±{int(args.prefill_length*args.length_cv)}, "
          f"decode={args.decode_length}±{int(args.decode_length*args.length_cv)} (normal, CV={args.length_cv})")
    print(f"  QPS={args.qps}  prefill_bs={args.prefill_batch_size}  "
          f"decode_bs={args.decode_batch_size}  时长={args.time_limit}s")
    print(f"  种子:   {seeds}")

    # 估算排队深度: 请求到达率 * 单请求 prefill 时间
    # prefill_time 约 = prefill_length * num_layers / (fp16_tflops * ...)
    # 粗略: 单请求 prefill ≈ 200ms (512 tokens, 48 layers, 25 TFLOPS)
    est_prefill_ms = args.prefill_length * 48 * 0.4  # 非常粗略
    est_queue_depth = args.qps * est_prefill_ms / 1000.0 / args.prefill_batch_size
    print(f"  估计队列深度: ~{est_queue_depth:.1f} (越高调度差异越明显)")
    if est_queue_depth < 1.5:
        print(f"  ⚠️  队列深度不足, 建议增大 --qps 或减小 --prefill_batch_size")
    print()

    # 多种子平均
    all_seed_results: dict[str, list[ExperimentResult]] = {k: [] for k, _, _ in SCHEDULERS}

    for seed in seeds:
        args.seed = seed
        if len(seeds) > 1:
            print(f"\n  ── Seed {seed} ──")

        results = []
        for key, name, desc in SCHEDULERS:
            r = run_one(key, name, desc, args)
            results.append(r)
            all_seed_results[key].append(r)

        if len(seeds) == 1:
            print_table(results)

    # 多种子汇总
    if len(seeds) > 1:
        print(f"\n  ── 多种子平均 ({len(seeds)} seeds) ──")
        averaged = []
        for key, name, desc in SCHEDULERS:
            rs = all_seed_results[key]
            avg = ExperimentResult(key, name, desc)
            for fld in [f.name for f in ExperimentResult.__dataclass_fields__.values()
                        if f.type in (float, int) and f.name != "num_requests"]:
                vals = [getattr(r, fld) for r in rs]
                setattr(avg, fld, float(np.mean(vals)))
            avg.num_requests = int(np.mean([r.num_requests for r in rs]))
            averaged.append(avg)
        print_table(averaged, title=f"多种子平均 ({len(seeds)} seeds)")


if __name__ == "__main__":
    main()
