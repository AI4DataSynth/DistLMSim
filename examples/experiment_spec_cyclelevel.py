"""Speculative Decoding Cycle-Level Analysis

对比 4 种 decode 模式的 cycle-level 性能:
  - Standard (K=0): 传统 autoregressive decode
  - DSpark (block=7): 半自回归 draft + Markov head
  - DFlash (block=7): 纯并行 draft (无 sequential head)
  - DSpark+CS (block=7): DSpark + Confidence Scheduling

额外扫描 DSpark block_size ∈ [3, 5, 7, 10] 的 draft/verify breakdown。

方法: 直接调用 SpeculativeDecodingEngine.compute_cycle() 收集
      cycle-level 统计量 (无模拟器事件循环噪声)。

配置:
  - Model: Qwen3-30B-A3B (48 layers, 128 experts, top-8, dim=2048)
  - Device: A800
  - PD-disaggregated, TP=4
  - prefill_length=512, decode_length=128
  - acceptance_rate=0.85, block_size=7

用法:
  python3 examples/experiment_spec_cyclelevel.py
"""

import sys
import json
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from distlmsim.config import (
    DeviceSKUConfig, ModelConfig, NVLinkConfig, RDMAConfig,
    NetworkTopologyConfig, DisaggregatedConfig, MetricsConfig,
)
from distlmsim.context import SimContext
from distlmsim.entities import Request, RequestStatus
from distlmsim.metrics import MetricsStore
from distlmsim.types import RDMAProtocolType, NetworkModelMode
from distlmsim.execution.speculative_decoder import SpeculativeDecodingEngine, CycleResult


# ─── 参数 ────────────────────────────────────────────────────────────────────

NUM_CYCLES = 2000       # 每模式采样 cycle 数
BATCH_SIZE = 8           # decode batch size
AVG_KV_LENGTH = 580      # prefill(512) + 部分 decode 的平均 KV 长度
SEED = 42


def make_model() -> ModelConfig:
    return ModelConfig(
        model_name="Qwen3-30B-A3B",
        num_layers=48,
        num_q_heads=32,
        num_kv_heads=4,
        embedding_dim=2048,
        num_experts=128,
        top_k_experts=8,
        vocab_size=151936,
    )


def make_ctx(model: ModelConfig, tp_size: int = 4) -> SimContext:
    device = DeviceSKUConfig()
    nvlink = NVLinkConfig(bandwidth_gbps=600.0)
    rdma = RDMAConfig(protocol=RDMAProtocolType.ROCE_V2, bandwidth_gbps=200.0)
    network = NetworkTopologyConfig(nvlink=nvlink, rdma=rdma, model_mode=NetworkModelMode.HYBRID)
    ms = MetricsStore(MetricsConfig())
    return SimContext(
        model_config=model,
        device_config=device,
        network_config=network,
        num_gpus_per_node=tp_size,
        tp_size=tp_size,
        metrics_store=ms,
    )


def make_batch(batch_size: int, kv_length: int) -> List[Request]:
    """创建一组虚拟 decode 请求 (已处于 decode 阶段)。"""
    reqs = []
    for i in range(batch_size):
        r = Request(
            id=i,
            arrival_time=0.0,
            prefill_tokens=512,
            decode_tokens=128,
            status=RequestStatus.DECODING,
            num_generated_tokens=kv_length - 512,  # 使 avg_kv ≈ kv_length
            domain="mixed",
        )
        reqs.append(r)
    return reqs


def make_engine(
    ctx: SimContext,
    mode: str,
    block_size: int = 7,
    acceptance_rate: float = 0.85,
    confidence_scheduling: bool = False,
    seed: int = SEED,
) -> SpeculativeDecodingEngine:
    """创建指定模式的 SpeculativeDecodingEngine。"""
    cfg = DisaggregatedConfig(
        enabled=True,
        enable_speculative_decoding=(mode != "standard"),
        speculative_mode=mode if mode != "standard" else "dspark",
        block_size=block_size,
        acceptance_rate=acceptance_rate,
        draft_num_layers=4,
        draft_embedding_dim=512,
        markov_rank=256,
        markov_head_type="vanilla",
        num_target_layer_ids=5,
        enable_confidence_scheduling=confidence_scheduling,
        confidence_threshold=0.6 if confidence_scheduling else 0.0,
        bonus_token=True,
    )
    # Standard mode: disable speculative decoding
    if mode == "standard":
        cfg.enable_speculative_decoding = False
    rng = random.Random(seed)
    return SpeculativeDecodingEngine(ctx, cfg, rng)


def run_cycles(
    engine: SpeculativeDecodingEngine,
    batch: List[Request],
    num_cycles: int,
) -> List[CycleResult]:
    """运行 num_cycles 轮 decode cycle，收集结果。"""
    results = []
    current_time = 0.0
    for _ in range(num_cycles):
        cr = engine.compute_cycle(batch, current_time)
        results.append(cr)
        current_time += cr.cycle_time_ms
    return results


@dataclass
class ModeStats:
    """一种模式的汇总统计。"""
    name: str
    mode: str
    block_size: int
    confidence_scheduling: bool
    tbt_p50_ms: float = 0.0
    tbt_p90_ms: float = 0.0
    avg_accepted_per_cycle: float = 0.0
    avg_draft_time_ms: float = 0.0
    avg_verify_time_ms: float = 0.0
    avg_cycle_time_ms: float = 0.0
    speedup_vs_standard: float = 0.0


def compute_stats(
    name: str, mode: str, block_size: int,
    confidence_scheduling: bool,
    results: List[CycleResult],
    baseline_tbt_p50: float = 0.0,
) -> ModeStats:
    cycle_times = [r.cycle_time_ms for r in results]
    accepted = [r.accepted_tokens for r in results]
    draft_times = [r.draft_time_ms for r in results]
    verify_times = [r.verify_time_ms for r in results]

    # TBT = cycle_time / accepted_tokens (effective per-token latency)
    tbts = [ct / max(1, acc) for ct, acc in zip(cycle_times, accepted)]

    tbt_p50 = float(np.percentile(tbts, 50))
    tbt_p90 = float(np.percentile(tbts, 90))

    stats = ModeStats(
        name=name,
        mode=mode,
        block_size=block_size,
        confidence_scheduling=confidence_scheduling,
        tbt_p50_ms=tbt_p50,
        tbt_p90_ms=tbt_p90,
        avg_accepted_per_cycle=float(np.mean(accepted)),
        avg_draft_time_ms=float(np.mean(draft_times)),
        avg_verify_time_ms=float(np.mean(verify_times)),
        avg_cycle_time_ms=float(np.mean(cycle_times)),
    )
    if baseline_tbt_p50 > 0:
        stats.speedup_vs_standard = baseline_tbt_p50 / tbt_p50 if tbt_p50 > 0 else 0.0
    return stats


# ─── 图表 ────────────────────────────────────────────────────────────────────

def plot_results(
    mode_stats: List[ModeStats],
    block_sweep_stats: List[ModeStats],
    output_dir: str,
):
    """生成论文级双面板图表。

    左图: 4 种模式 TBT P50 柱状图 + 加速比标注
    右图: DSpark 不同 block_size 的 draft/verify breakdown + accepted tokens
    """
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # ─── 左图: TBT P50 柱状图 ────────────────────────────────────────────
    colors = ["#4472C4", "#ED7D31", "#A5A5A5", "#70AD47"]
    names = [s.name for s in mode_stats]
    tbts = [s.tbt_p50_ms for s in mode_stats]
    speedups = [s.speedup_vs_standard for s in mode_stats]

    bars = ax1.bar(range(len(names)), tbts, color=colors, edgecolor="black",
                   linewidth=0.5, width=0.65)

    # 加速比标注
    for i, (bar, sp) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        label = f"{sp:.2f}x" if sp > 0 else "1.00x"
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                 label, ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_ylabel("TBT P50 (ms)")
    ax1.set_title("(a) TBT P50 by Decoding Mode")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    # ─── 右图: block_size sweep (draft/verify stacked bar + accepted line) ─
    block_sizes = [s.block_size for s in block_sweep_stats]
    draft_times = [s.avg_draft_time_ms for s in block_sweep_stats]
    verify_times = [s.avg_verify_time_ms for s in block_sweep_stats]
    accepted_tokens = [s.avg_accepted_per_cycle for s in block_sweep_stats]

    x = np.arange(len(block_sizes))
    width = 0.5

    bars_draft = ax2.bar(x, draft_times, width, label="Draft time",
                         color="#4472C4", edgecolor="black", linewidth=0.5)
    bars_verify = ax2.bar(x, verify_times, width, bottom=draft_times,
                          label="Verify time", color="#ED7D31",
                          edgecolor="black", linewidth=0.5)

    # Accepted tokens 在右轴
    ax2_r = ax2.twinx()
    ax2_r.plot(x, accepted_tokens, "o-", color="#70AD47", markersize=5,
               linewidth=1.5, label="Avg accepted tokens")
    ax2_r.set_ylabel("Accepted tokens", color="#70AD47")
    ax2_r.tick_params(axis="y", labelcolor="#70AD47", labelsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"B={b}" for b in block_sizes])
    ax2.set_xlabel("Block size")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("(b) DSpark Draft/Verify Breakdown")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)

    # 合并两个 legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               framealpha=0.9, edgecolor="gray")

    plt.tight_layout(pad=0.5)

    pdf_path = f"{output_dir}/spec_cyclelevel.pdf"
    png_path = f"{output_dir}/spec_cyclelevel.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")


# ─── 主流程 ──────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.WARNING)

    model = make_model()
    ctx = make_ctx(model, tp_size=4)
    batch = make_batch(BATCH_SIZE, AVG_KV_LENGTH)

    print("=" * 64)
    print("  Speculative Decoding Cycle-Level Analysis")
    print("=" * 64)
    print(f"  Model: {model.model_name} ({model.num_layers}L, {model.num_experts}E top-{model.top_k_experts})")
    print(f"  Device: A800, TP=4")
    print(f"  Batch size: {BATCH_SIZE}, avg KV length: {AVG_KV_LENGTH}")
    print(f"  Cycles per mode: {NUM_CYCLES}")
    print()

    # ─── Phase 1: 4 种模式对比 ────────────────────────────────────────────

    modes = [
        ("Standard",      "standard", 7, False),
        ("DSpark",        "dspark",   7, False),
        ("DFlash",        "dflash",   7, False),
        ("DSpark+CS",     "dspark",   7, True),
    ]

    # 先跑 Standard 拿 baseline
    print("--- Phase 1: Mode Comparison ---\n")

    mode_results = {}
    for name, mode, bs, cs in modes:
        engine = make_engine(ctx, mode, block_size=bs,
                             acceptance_rate=0.85,
                             confidence_scheduling=cs,
                             seed=SEED)
        results = run_cycles(engine, batch, NUM_CYCLES)
        mode_results[name] = results
        print(f"  [{name:12s}] collected {len(results)} cycles")

    # 计算统计 (baseline = Standard)
    baseline_stats = compute_stats("Standard", "standard", 0, False,
                                   mode_results["Standard"])
    baseline_stats.speedup_vs_standard = 1.0  # baseline 自身
    baseline_tbt = baseline_stats.tbt_p50_ms

    all_mode_stats = [baseline_stats]
    for name, mode, bs, cs in modes[1:]:
        stats = compute_stats(name, mode, bs, cs, mode_results[name],
                              baseline_tbt_p50=baseline_tbt)
        all_mode_stats.append(stats)

    # 打印表格
    print()
    print(f"{'Mode':<14s} {'TBT P50':>8s} {'TBT P90':>8s} {'Accepted':>9s} "
          f"{'Draft':>8s} {'Verify':>8s} {'Speedup':>8s}")
    print("-" * 70)
    for s in all_mode_stats:
        print(f"{s.name:<14s} {s.tbt_p50_ms:>7.3f}ms {s.tbt_p90_ms:>7.3f}ms "
              f"{s.avg_accepted_per_cycle:>8.2f} "
              f"{s.avg_draft_time_ms:>7.3f}ms {s.avg_verify_time_ms:>7.3f}ms "
              f"{s.speedup_vs_standard:>7.2f}x")

    # ─── Phase 2: DSpark block_size sweep ─────────────────────────────────

    print("\n--- Phase 2: DSpark Block Size Sweep ---\n")

    block_sizes = [3, 5, 7, 10]
    sweep_stats = []

    for bs in block_sizes:
        engine = make_engine(ctx, "dspark", block_size=bs,
                             acceptance_rate=0.85,
                             confidence_scheduling=False,
                             seed=SEED)
        results = run_cycles(engine, batch, NUM_CYCLES)
        stats = compute_stats(f"DSpark-B{bs}", "dspark", bs, False, results,
                              baseline_tbt_p50=baseline_tbt)
        sweep_stats.append(stats)
        print(f"  [B={bs:2d}] TBT P50={stats.tbt_p50_ms:.3f}ms  "
              f"draft={stats.avg_draft_time_ms:.3f}ms  "
              f"verify={stats.avg_verify_time_ms:.3f}ms  "
              f"accepted={stats.avg_accepted_per_cycle:.2f}  "
              f"speedup={stats.speedup_vs_standard:.2f}x")

    # ─── 保存结果 ─────────────────────────────────────────────────────────

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    result_dict = {
        "config": {
            "model": model.model_name,
            "num_layers": model.num_layers,
            "num_experts": model.num_experts,
            "top_k_experts": model.top_k_experts,
            "embedding_dim": model.embedding_dim,
            "device": "A800",
            "tp_size": 4,
            "batch_size": BATCH_SIZE,
            "avg_kv_length": AVG_KV_LENGTH,
            "acceptance_rate": 0.85,
            "num_cycles": NUM_CYCLES,
            "seed": SEED,
        },
        "mode_comparison": [
            {
                "name": s.name,
                "mode": s.mode,
                "block_size": s.block_size,
                "confidence_scheduling": s.confidence_scheduling,
                "tbt_p50_ms": round(s.tbt_p50_ms, 4),
                "tbt_p90_ms": round(s.tbt_p90_ms, 4),
                "avg_accepted_per_cycle": round(s.avg_accepted_per_cycle, 3),
                "avg_draft_time_ms": round(s.avg_draft_time_ms, 4),
                "avg_verify_time_ms": round(s.avg_verify_time_ms, 4),
                "avg_cycle_time_ms": round(s.avg_cycle_time_ms, 4),
                "speedup_vs_standard": round(s.speedup_vs_standard, 3),
            }
            for s in all_mode_stats
        ],
        "block_size_sweep": [
            {
                "block_size": s.block_size,
                "tbt_p50_ms": round(s.tbt_p50_ms, 4),
                "tbt_p90_ms": round(s.tbt_p90_ms, 4),
                "avg_accepted_per_cycle": round(s.avg_accepted_per_cycle, 3),
                "avg_draft_time_ms": round(s.avg_draft_time_ms, 4),
                "avg_verify_time_ms": round(s.avg_verify_time_ms, 4),
                "avg_cycle_time_ms": round(s.avg_cycle_time_ms, 4),
                "speedup_vs_standard": round(s.speedup_vs_standard, 3),
            }
            for s in sweep_stats
        ],
    }

    json_path = output_dir / "spec_cyclelevel.json"
    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\n  JSON: {json_path}")

    # ─── 绘图 ─────────────────────────────────────────────────────────────

    print()
    plot_results(all_mode_stats, sweep_stats, str(output_dir))

    # ─── 复制到论文目录 ───────────────────────────────────────────────────

    paper_fig_dir = Path("/Users/yunting/Documents/重要/codebase/大模型模拟/DistLMPaper/src/figures")
    if paper_fig_dir.exists():
        import shutil
        dst = paper_fig_dir / "spec_cyclelevel.pdf"
        shutil.copy2(output_dir / "spec_cyclelevel.pdf", dst)
        print(f"  Copied PDF to: {dst}")
    else:
        print(f"  Paper figures dir not found: {paper_fig_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
