"""Level 2 DFlash vs DSpark 对比实验

验证新功能:
1. Domain-aware acceptance profiles (math/code/chat)
2. DFlash (纯并行) vs DSpark (半自回归) 接受率差异
3. Confidence scheduling 在不同负载下的自适应效果

用法:
  python3 examples/demo_level2_dspark.py
"""

import sys
import logging
import argparse

sys.path.insert(0, ".")

from main import create_disaggregated_simulator


def run_experiment(mode, domain, qps, enable_sched, time_limit=30, seed=42):
    """运行单组实验。"""
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

    dcfg = sim.config.disaggregated
    dcfg.enable_speculative_decoding = True
    dcfg.speculative_mode = mode
    dcfg.block_size = 7
    dcfg.draft_num_layers = 5
    dcfg.draft_embedding_dim = 512
    dcfg.enable_confidence_scheduling = enable_sched
    dcfg.default_domain = domain

    # 强制所有请求使用指定 domain
    original_create = sim._create_request
    def patched_create(arrival_time):
        req = original_create(arrival_time)
        req.domain = domain
        return req
    sim._create_request = patched_create

    ms = sim.run()

    completed = [m for m in ms._request_metrics.values() if m.decode_end_time > 0]
    if not completed:
        return None

    import numpy as np
    tbts = [m.tbt for m in completed if m.tbt > 0]
    e2es = [m.e2e_latency for m in completed]
    total_decode = sum(m.decode_tokens for m in completed)
    wall_time = max(m.decode_end_time for m in completed)
    first_arrival = min(m.arrival_time for m in completed)
    effective_s = (wall_time - first_arrival) / 1000.0

    return {
        "completed": len(completed),
        "tbt_p50": float(np.percentile(tbts, 50)) if tbts else 0.0,
        "e2e_p50": float(np.percentile(e2es, 50)),
        "decode_tps": total_decode / effective_s if effective_s > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Level 2: DFlash vs DSpark")
    parser.add_argument("--time_limit", type=float, default=30.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    print("=" * 76)
    print("  Level 2: DFlash vs DSpark — Domain-Aware Acceptance Profiles")
    print("=" * 76)
    print(f"  模型: Qwen3-30B-A3B (48层 MoE)  |  集群: PD Disagg 2×4 A800 TP=4")
    print(f"  Draft: 5层 dim=512, block_size=7  |  时长: {args.time_limit}s")
    print("=" * 76)

    # ─── Experiment 1: Domain sensitivity ─────────────────────────────
    print(f"\n{'─'*76}")
    print("  实验 1: Domain 敏感性 (QPS=10, 无 confidence scheduling)")
    print(f"{'─'*76}")
    print(f"  {'Mode':<10} {'Domain':<8} {'TBT P50':>10} {'E2E P50':>10} {'Decode tps':>12} {'Reqs':>6}")
    print(f"  {'─'*60}")

    for mode in ["dflash", "dspark"]:
        for domain in ["math", "code", "chat"]:
            r = run_experiment(mode, domain, qps=10, enable_sched=False,
                               time_limit=args.time_limit)
            if r:
                print(f"  {mode:<10} {domain:<8} {r['tbt_p50']:>8.3f}ms "
                      f"{r['e2e_p50']:>8.1f}ms {r['decode_tps']:>10.1f} {r['completed']:>6}")

    # ─── Experiment 2: Load adaptivity ─────────────────────────────────
    print(f"\n{'─'*76}")
    print("  实验 2: 负载自适应 (DSpark, domain=mixed)")
    print(f"{'─'*76}")
    print(f"  {'QPS':>6} {'Sched':>8} {'TBT P50':>10} {'E2E P50':>10} {'Decode tps':>12} {'Reqs':>6}")
    print(f"  {'─'*56}")

    for qps in [5, 10, 20, 50]:
        for enable_sched in [False, True]:
            r = run_experiment("dspark", "mixed", qps=qps,
                               enable_sched=enable_sched, time_limit=args.time_limit)
            if r:
                sched_str = "ON" if enable_sched else "OFF"
                print(f"  {qps:>6} {sched_str:>8} {r['tbt_p50']:>8.3f}ms "
                      f"{r['e2e_p50']:>8.1f}ms {r['decode_tps']:>10.1f} {r['completed']:>6}")

    # ─── Experiment 3: DFlash vs DSpark summary ────────────────────────
    print(f"\n{'─'*76}")
    print("  实验 3: DFlash vs DSpark 综合对比 (QPS=10, mixed domain)")
    print(f"{'─'*76}")

    baseline = run_experiment("dspark", "mixed", qps=10, enable_sched=False,
                              time_limit=args.time_limit)
    dflash = run_experiment("dflash", "mixed", qps=10, enable_sched=False,
                            time_limit=args.time_limit)
    dspark_sched = run_experiment("dspark", "mixed", qps=10, enable_sched=True,
                                  time_limit=args.time_limit)

    if baseline and dflash and dspark_sched:
        print(f"  {'配置':<25} {'TBT P50':>10} {'E2E P50':>10} {'Decode tps':>12}")
        print(f"  {'─'*60}")
        print(f"  {'DSpark (no sched)':<25} {baseline['tbt_p50']:>8.3f}ms "
              f"{baseline['e2e_p50']:>8.1f}ms {baseline['decode_tps']:>10.1f}")
        print(f"  {'DFlash (no sched)':<25} {dflash['tbt_p50']:>8.3f}ms "
              f"{dflash['e2e_p50']:>8.1f}ms {dflash['decode_tps']:>10.1f}")
        print(f"  {'DSpark + sched':<25} {dspark_sched['tbt_p50']:>8.3f}ms "
              f"{dspark_sched['e2e_p50']:>8.1f}ms {dspark_sched['decode_tps']:>10.1f}")

        # Speedup comparison
        if dflash['tbt_p50'] > 0:
            dspark_vs_dflash = dflash['tbt_p50'] / baseline['tbt_p50']
            print(f"\n  DSpark vs DFlash TBT: {dspark_vs_dflash:.2f}x "
                  f"(DSpark {'faster' if dspark_vs_dflash < 1 else 'slower'})")

    print(f"\n{'='*76}")
    print("  实验完成!")
    print(f"{'='*76}")


if __name__ == "__main__":
    main()
