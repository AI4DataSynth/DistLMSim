"""生成论文级 LaTeX 表格

从 results/level2_paper.json 读取数据，输出三张表格:
  Table 1: DFlash vs DSpark (Claim 1)
  Table 2: Domain-aware acceptance (Claim 2)
  Table 3: Confidence scheduling (Claim 3)

用法:
  python3 examples/generate_paper_tables.py
  python3 examples/generate_paper_tables.py --output tables.tex
"""

import json
import argparse


def load_data(path="results/level2_paper.json"):
    with open(path) as f:
        return json.load(f)


def table1_dflash_vs_dspark(data):
    """Claim 1: DFlash vs DSpark head-to-head (block_size=7, mixed domain)."""
    rows = [r for r in data
            if r["domain"] == "mixed" and not r["sched"]
            and r.get("qps", 0) == 10
            and r["avg_accepted_per_cycle"] < 5]

    dflash = [r for r in rows if r["mode"] == "dflash"][0]
    dspark = [r for r in rows if r["mode"] == "dspark"][0]

    acc_imp = (dspark["avg_accepted_per_cycle"] - dflash["avg_accepted_per_cycle"]) / dflash["avg_accepted_per_cycle"] * 100
    tbt_imp = (dflash["tbt_p50"] - dspark["tbt_p50"]) / dflash["tbt_p50"] * 100
    cyc_imp = (dflash["avg_cycles_per_req"] - dspark["avg_cycles_per_req"]) / dflash["avg_cycles_per_req"] * 100
    e2e_imp = (dflash["e2e_p50"] - dspark["e2e_p50"]) / dflash["e2e_p50"] * 100

    lines = [
        r"\begin{table}[t]",
        r"\caption{DFlash vs.\ DSpark head-to-head comparison (block size $\gamma{=}7$, QPS{=}10, mixed domain)."
        r" DSpark's semi-autoregressive architecture achieves higher accepted length per cycle by mitigating suffix decay.}",
        r"\label{tab:dflash-vs-dspark}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Drafter} & \textbf{Accepted/Cycle} ($\tau$) & \textbf{Cycles/Req} & \textbf{TBT P50} (ms) & \textbf{E2E P50} (ms) \\",
        r"\midrule",
        f"DFlash (parallel)    & {dflash['avg_accepted_per_cycle']:.2f} & {dflash['avg_cycles_per_req']:.1f} & {dflash['tbt_p50']:.3f} & {dflash['e2e_p50']:.1f} \\\\",
        f"DSpark (semi-AR)     & \\textbf{{{dspark['avg_accepted_per_cycle']:.2f}}} & \\textbf{{{dspark['avg_cycles_per_req']:.1f}}} & \\textbf{{{dspark['tbt_p50']:.3f}}} & \\textbf{{{dspark['e2e_p50']:.1f}}} \\\\",
        r"\midrule",
        f"\\textit{{Improvement}} & \\textbf{{+{acc_imp:.1f}\\%}} & -{cyc_imp:.1f}\\% & \\textbf{{$-{tbt_imp:.1f}$\\%}} & -{e2e_imp:.1f}\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def table2_domain_aware(data):
    """Claim 2: Domain-aware acceptance profiles."""
    domains = ["math", "code", "chat"]
    dflash_rows = {d: [r for r in data if r["mode"] == "dflash" and r["domain"] == d][0] for d in domains}
    dspark_rows = {d: [r for r in data if r["mode"] == "dspark" and r["domain"] == d][0] for d in domains}

    body_lines = []
    for d in domains:
        df = dflash_rows[d]
        ds = dspark_rows[d]
        imp = (ds["avg_accepted_per_cycle"] - df["avg_accepted_per_cycle"]) / df["avg_accepted_per_cycle"] * 100
        body_lines.append(
            f"{d.capitalize():10s} & {df['avg_accepted_per_cycle']:.2f} & {df['tbt_p50']:.3f} & "
            f"\\textbf{{{ds['avg_accepted_per_cycle']:.2f}}} & \\textbf{{{ds['tbt_p50']:.3f}}} & "
            f"+{imp:.1f}\\% \\\\"
        )

    dflash_sens = (dflash_rows["math"]["avg_accepted_per_cycle"] - dflash_rows["chat"]["avg_accepted_per_cycle"]) / dflash_rows["math"]["avg_accepted_per_cycle"] * 100
    dspark_sens = (dspark_rows["math"]["avg_accepted_per_cycle"] - dspark_rows["chat"]["avg_accepted_per_cycle"]) / dspark_rows["math"]["avg_accepted_per_cycle"] * 100

    body = "\n".join(body_lines)

    lines = [
        r"\begin{table}[t]",
        r"\caption{Domain-aware acceptance profiles across workload types (QPS{=}10, $\gamma{=}7$, no scheduling)."
        r" DSpark consistently outperforms DFlash across all domains.}",
        r"\label{tab:domain-aware}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{DFlash}} & \multicolumn{2}{c}{\textbf{DSpark}} & \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"\textbf{Domain} & $\tau$ & TBT (ms) & $\tau$ & TBT (ms) & \textbf{$\Delta\tau$} \\",
        r"\midrule",
        body,
        r"\midrule",
        f"\\textit{{Sensitivity}} & \\multicolumn{{2}}{{c}}{{{dflash_sens:.1f}\\%}} & \\multicolumn{{2}}{{c}}{{{dspark_sens:.1f}\\%}} & \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\raggedright\scriptsize\textit{Sensitivity} = (math $-$ chat) / math.",
        r"\end{table}",
    ]
    return "\n".join(lines)


def table3_scheduling(data):
    """Claim 3: Confidence scheduling load adaptivity (block_size=14)."""
    # block_size=14 results: τ > 4.2 (block_size=7 max τ ≈ 4.12)
    sched_data = [r for r in data
                  if r["mode"] == "dspark" and r["domain"] == "mixed"
                  and r["avg_accepted_per_cycle"] > 4.2]

    def fmt_e2e(v):
        return f"{v/1000:.1f}s" if v > 5000 else f"{v:.0f}ms"

    body_lines = []
    for qps in [5, 10, 20, 50, 100]:
        off = [r for r in sched_data if r["qps"] == qps and not r["sched"]]
        on = [r for r in sched_data if r["qps"] == qps and r["sched"]]
        if not off or not on:
            continue
        off, on = off[0], on[0]

        off_speed = 128.0 / (off["avg_cycles_per_req"] * off["tbt_p50"] * off["avg_accepted_per_cycle"] / 1000.0)
        on_speed = 128.0 / (on["avg_cycles_per_req"] * on["tbt_p50"] * on["avg_accepted_per_cycle"] / 1000.0)
        speed_imp = (on_speed - off_speed) / off_speed * 100
        tbt_imp = (off["tbt_p50"] - on["tbt_p50"]) / off["tbt_p50"] * 100

        # OFF row
        body_lines.append(
            f"\\multirow{{2}}{{*}}{{{qps}}} & "
            f"{off['avg_accepted_per_cycle']:.2f} & {off['tbt_p50']:.3f} & "
            f"{off_speed:.0f} & {fmt_e2e(off['e2e_p50'])} & {off['decode_tps']:.0f} \\\\"
        )
        # ON row (bold improvement)
        body_lines.append(
            f" & \\textbf{{{on['avg_accepted_per_cycle']:.2f}}} & \\textbf{{{on['tbt_p50']:.3f}}} & "
            f"\\textbf{{{on_speed:.0f}}} & \\textbf{{{fmt_e2e(on['e2e_p50'])}}} & {on['decode_tps']:.0f} \\\\"
        )
        # Improvement summary row
        sign = "+" if speed_imp > 0 else ""
        body_lines.append(
            f" & \\multicolumn{{5}}{{c}}{{\\scriptsize\\textit{{"
            f"$\\Delta\\tau$={on['avg_accepted_per_cycle']-off['avg_accepted_per_cycle']:+.2f}, "
            f"TBT $-{tbt_imp:.0f}$\\%, "
            f"Gen Speed \\textbf{{{sign}{speed_imp:.0f}\\%}}}}}} \\\\"
        )
        if qps < 100:
            body_lines.append("\\addlinespace")

    body = "\n".join(body_lines)

    lines = [
        r"\begin{table}[t]",
        r"\caption{Confidence-scheduled verification under varying system load ($\gamma{=}14$, mixed domain)."
        r" The scheduler dynamically truncates low-confidence suffix tokens, reducing per-cycle verification cost."
        r" While accepted length ($\tau$) decreases, the faster cycle time yields \textbf{39--40\% higher per-user"
        r" generation speed} at high concurrency (QPS$\geq$20). This is directionally consistent with the 60--85\%"
        r" improvement reported in the DSpark paper~\citep{cheng2026dspark}, with the gap attributable to our"
        r" conservative Roofline-based SPS curve."
        r" System throughput (Decode tps) remains unchanged, confirming the scheduler improves \emph{per-user"
        r" experience} without sacrificing aggregate capacity.}",
        r"\label{tab:scheduling}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r" & $\tau$ & \textbf{TBT P50} & \textbf{Gen Speed} & E2E P50 & Decode \\",
        r"\textbf{QPS} & (acc/cyc) & (ms) & (tok/s/user) & & tps \\",
        r"\midrule",
        body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\raggedright\scriptsize Each QPS group: top = Sched OFF (baseline), bottom = Sched ON (\textbf{bold})."
        r" E2E P50 at QPS$\geq$20 is dominated by queueing delay (system saturation).",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/level2_paper.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data = load_data(args.input)

    t1 = table1_dflash_vs_dspark(data)
    t2 = table2_domain_aware(data)
    t3 = table3_scheduling(data)

    full = f"% Auto-generated LaTeX tables for DFlash/DSpark simulator paper\n% Source: {args.input}\n\n{t1}\n\n{t2}\n\n{t3}\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(full)
        print(f"Tables written to {args.output}")
    else:
        print(full)


if __name__ == "__main__":
    main()
