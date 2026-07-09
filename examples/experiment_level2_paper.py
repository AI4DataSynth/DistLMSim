"""Level 2 Speculative Decoding 论文级实验

支撑三个 claim:

Claim 1 (Sec 4.3.1): DSpark 每 cycle 接受长度 > DFlash
  → 半自回归架构 (Markov head) 缓解 suffix decay

Claim 2 (Sec 4.2): Domain-aware acceptance profiles
  → math > code > chat, DSpark domain 敏感度更低

Claim 3 (Sec 5.4): Confidence scheduling 负载自适应
  → 高并发下动态截断 verify budget, 维持吞吐

用法:
  python3 examples/experiment_level2_paper.py
  python3 examples/experiment_level2_paper.py --time_limit 60 --qps 10
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")

from main import create_disaggregated_simulator


def run_one(mode, domain, qps, enable_sched, time_limit, seed, block_size=7):
    """运行单组实验，返回详细指标。"""
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
    dcfg.block_size = block_size
    dcfg.draft_num_layers = 5
    dcfg.draft_embedding_dim = 512
    dcfg.enable_confidence_scheduling = enable_sched
    dcfg.default_domain = domain
    dcfg.bonus_token = True

    # 强制 domain
    original_create = sim._create_request
    def patched_create(arrival_time):
        req = original_create(arrival_time)
        req.domain = domain
        return req
    sim._create_request = patched_create

    ms = sim.run()

    completed = [r for r in sim.ctx.requests.values()
                 if r.status.value == "completed"]

    if not completed:
        return None

    # 基本延迟指标
    tbts = []
    e2es = []
    for r in completed:
        if r.decode_end_time and r.decode_start_time:
            decode_time = r.decode_end_time - r.decode_start_time
            n_tokens = r.decode_tokens
            if n_tokens > 1:
                tbts.append(decode_time / (n_tokens - 1))
        e2es.append(r.e2e_latency)

    # Speculative decoding cycle 指标
    cycles = [r.total_spec_cycles for r in completed if r.total_spec_cycles > 0]
    accepted = [r.total_spec_accepted for r in completed if r.total_spec_cycles > 0]

    avg_accepted_per_cycle = 0.0
    if cycles and accepted:
        total_acc = sum(accepted)
        total_cyc = sum(cycles)
        avg_accepted_per_cycle = total_acc / total_cyc if total_cyc > 0 else 0.0

    # 吞吐
    total_decode_tokens = sum(r.decode_tokens for r in completed)
    wall_time = max(r.decode_end_time for r in completed if r.decode_end_time)
    first_arrival = min(r.arrival_time for r in completed)
    effective_s = (wall_time - first_arrival) / 1000.0
    decode_tps = total_decode_tokens / effective_s if effective_s > 0 else 0.0

    return {
        "mode": mode,
        "domain": domain,
        "qps": qps,
        "sched": enable_sched,
        "completed": len(completed),
        "tbt_p50": float(np.percentile(tbts, 50)) if tbts else 0.0,
        "e2e_p50": float(np.percentile(e2es, 50)) if e2es else 0.0,
        "decode_tps": decode_tps,
        "avg_accepted_per_cycle": avg_accepted_per_cycle,
        "avg_cycles_per_req": np.mean(cycles) if cycles else 0.0,
    }


def print_table(headers, rows, col_widths=None):
    """打印对齐表格。"""
    if not col_widths:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) + 2
                      for i, h in enumerate(headers)]
    header_line = "".join(str(h).rjust(w) for h, w in zip(headers, col_widths))
    print(f"  {header_line}")
    print(f"  {'─' * sum(col_widths)}")
    for row in rows:
        line = "".join(str(v).rjust(w) for v, w in zip(row, col_widths))
        print(f"  {line}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_limit", type=float, default=60.0)
    parser.add_argument("--qps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/level2_paper.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    results = []

    # ════════════════════════════════════════════════════════════════════
    # Claim 1: DSpark accepted length > DFlash
    # ════════════════════════════════════════════════════════════════════
    print("=" * 78)
    print("  Claim 1: DSpark 每 cycle 接受长度 > DFlash (半自回归缓解 suffix decay)")
    print("=" * 78)
    print(f"  配置: QPS={args.qps}, block_size=7, domain=mixed, no scheduling")
    print(f"  时长: {args.time_limit}s\n")

    headers = ["Drafter", "Avg Accepted/Cycle", "Avg Cycles/Req",
               "TBT P50 (ms)", "E2E P50 (ms)", "Decode tps"]
    rows = []

    for mode in ["dflash", "dspark"]:
        r = run_one(mode, "mixed", args.qps, False, args.time_limit, args.seed)
        if r:
            results.append(r)
            rows.append([
                mode.upper(),
                f"{r['avg_accepted_per_cycle']:.2f}",
                f"{r['avg_cycles_per_req']:.1f}",
                f"{r['tbt_p50']:.3f}",
                f"{r['e2e_p50']:.1f}",
                f"{r['decode_tps']:.1f}",
            ])

    print_table(headers, rows)

    if len(rows) == 2:
        dflash_acc = float(rows[0][1])
        dspark_acc = float(rows[1][1])
        if dflash_acc > 0:
            improvement = (dspark_acc - dflash_acc) / dflash_acc * 100
            print(f"\n  → DSpark accepted/cycle 比 DFlash 高 {improvement:.1f}%")

    # ════════════════════════════════════════════════════════════════════
    # Claim 2: Domain-aware acceptance profiles
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 78}")
    print("  Claim 2: Domain-aware acceptance (math > code > chat)")
    print("=" * 78)
    print(f"  DSpark domain 敏感度应低于 DFlash\n")

    headers = ["Drafter", "Domain", "Accepted/Cycle", "TBT P50 (ms)"]
    rows = []

    for mode in ["dflash", "dspark"]:
        for domain in ["math", "code", "chat"]:
            r = run_one(mode, domain, args.qps, False, args.time_limit, args.seed)
            if r:
                results.append(r)
                rows.append([
                    mode.upper(),
                    domain,
                    f"{r['avg_accepted_per_cycle']:.2f}",
                    f"{r['tbt_p50']:.3f}",
                ])

    print_table(headers, rows)

    # 计算 domain sensitivity: (math - chat) / math
    for mode in ["dflash", "dspark"]:
        mode_rows = [r for r in rows if r[0] == mode.upper()]
        if len(mode_rows) >= 3:
            math_acc = float(mode_rows[0][2])
            chat_acc = float(mode_rows[2][2])
            if math_acc > 0:
                sensitivity = (math_acc - chat_acc) / math_acc * 100
                print(f"\n  → {mode.upper()} domain sensitivity "
                      f"(math→chat drop): {sensitivity:.1f}%")

    # ════════════════════════════════════════════════════════════════════
    # Claim 3: Confidence scheduling load adaptivity
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 78}")
    print("  Claim 3: Confidence scheduling 负载自适应")
    print("=" * 78)
    print(f"  DSpark, block_size=14 (更大块让 scheduler 有截断空间)")
    print(f"  混合 domain (math/code/chat 各 1/3, 不同请求有不同 confidence)\n")

    headers = ["QPS", "Sched", "Accepted/Cycle", "TBT P50 (ms)",
               "E2E P50 (ms)", "Decode tps", "Reqs"]
    rows = []

    for qps in [5, 10, 20, 50, 100]:
        for sched in [False, True]:
            # 使用 larger block_size 让 scheduler 有更多截断空间
            r = run_one("dspark", "mixed", qps, sched, args.time_limit,
                        args.seed, block_size=14)
            if r:
                results.append(r)
                rows.append([
                    str(qps),
                    "ON" if sched else "OFF",
                    f"{r['avg_accepted_per_cycle']:.2f}",
                    f"{r['tbt_p50']:.3f}",
                    f"{r['e2e_p50']:.1f}",
                    f"{r['decode_tps']:.1f}",
                    str(r['completed']),
                ])

    print_table(headers, rows)

    # 高负载下 sched ON vs OFF 的吞吐对比
    high_load_results = [r for r in results
                         if r.get("qps", 0) >= 50 and r["mode"] == "dspark"]
    if len(high_load_results) >= 2:
        off_list = [r for r in high_load_results if not r["sched"]]
        on_list = [r for r in high_load_results if r["sched"]]
        if off_list and on_list:
            off = off_list[-1]
            on = on_list[-1]
            if off["decode_tps"] > 0:
                tps_improvement = (on["decode_tps"] - off["decode_tps"]) / off["decode_tps"] * 100
                print(f"\n  → 高负载 (QPS≥50) scheduling 吞吐变化: "
                      f"{tps_improvement:+.1f}%")

    # ════════════════════════════════════════════════════════════════════
    # Save results
    # ════════════════════════════════════════════════════════════════════
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 78}")
    print(f"  全部结果已保存: {args.output}")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
