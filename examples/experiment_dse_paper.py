"""Design Space Exploration (DSE) 实验 — 论文级图表

在多个维度上扫描设计空间，收集 TTFT/TBT/E2E 指标，
找出 Pareto 前沿 (TTFT P50 vs TBT P50 trade-off)，
生成论文级散点图和配置汇总表。

搜索维度:
  - 调度策略:   FCFS, SJF, PO, SRTF
  - KV 传输:    DIRECT, PIPELINED
  - Chunked Prefill: disabled (chunk_size=4096), enabled (chunk_size=512)
  - QPS:        5, 10, 20

用法:
  python3 examples/experiment_dse_paper.py
"""

import sys
import json
import logging
import itertools
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np

sys.path.insert(0, ".")

from distlmsim.config import (
    SimulationConfig,
    DeviceSKUConfig,
    ModelConfig,
    NVLinkConfig,
    RDMAConfig,
    NetworkTopologyConfig,
    DisaggregatedConfig,
    RequestGeneratorConfig,
    MetricsConfig,
)
from distlmsim.context import SimContext
from distlmsim.metrics.metrics_store import MetricsStore
from distlmsim.types import RDMAProtocolType, KVCacheTransferStrategy
from main import DisaggregatedSimulator


# ─── 设计空间定义 ──────────────────────────────────────────────────────────────

SCHEDULERS = ["fcfs", "sjf", "po", "srtf"]
KV_TRANSFER_STRATEGIES = ["DIRECT", "PIPELINED"]
CHUNKED_PREFILL_CONFIGS = [
    {"enabled": False, "chunk_size": 4096, "label": "NoChunk"},
    {"enabled": True,  "chunk_size": 512,  "label": "Chunk512"},
]
QPS_LIST = [5, 10, 20]


@dataclass
class DSEPoint:
    """一个设计空间配置点"""
    scheduler: str
    kv_transfer: str
    chunked_prefill: str   # "NoChunk" or "Chunk512"
    chunk_size: int
    qps: float

    def config_key(self) -> str:
        return (
            f"sched={self.scheduler}_kv={self.kv_transfer}_"
            f"chunk={self.chunked_prefill}_qps={int(self.qps)}"
        )


@dataclass
class DSEResult:
    """单个设计点的仿真结果"""
    config: DSEPoint
    ttft_p50: float = 0.0
    ttft_p99: float = 0.0
    tbt_p50: float = 0.0
    tbt_p99: float = 0.0
    e2e_p50: float = 0.0
    e2e_p99: float = 0.0
    completed: int = 0
    is_pareto: bool = False


# ─── 仿真器构建 ────────────────────────────────────────────────────────────────

def create_simulator(
    qps: float,
    scheduler: str,
    kv_transfer: str,
    chunked_prefill_enabled: bool,
    chunk_size: int,
    seed: int = 42,
    time_limit_s: float = 30.0,
) -> DisaggregatedSimulator:
    """创建带自定义配置的 DisaggregatedSimulator。"""

    device = DeviceSKUConfig()  # A800 defaults
    model = ModelConfig(
        model_name="Qwen3-30B-A3B",
        num_layers=48,
        num_q_heads=32,
        num_kv_heads=4,
        embedding_dim=2048,
        num_experts=128,
        top_k_experts=8,
    )
    nvlink = NVLinkConfig(bandwidth_gbps=600.0)
    rdma = RDMAConfig(protocol=RDMAProtocolType.ROCE_V2, bandwidth_gbps=200.0)
    network = NetworkTopologyConfig(nvlink=nvlink, rdma=rdma)

    metrics_config = MetricsConfig()
    ms = MetricsStore(metrics_config)

    ctx = SimContext(
        model_config=model,
        device_config=device,
        network_config=network,
        num_gpus_per_node=4,
        tp_size=4,
        metrics_store=ms,
    )

    kv_strategy = KVCacheTransferStrategy[kv_transfer]

    config = SimulationConfig(
        seed=seed,
        time_limit_s=time_limit_s,
        disaggregated=DisaggregatedConfig(
            enabled=True,
            num_prefill_nodes=1,
            num_decode_nodes=1,
            prefill_batch_size=8,
            decode_batch_size=32,
            kv_cache_transfer_strategy=kv_strategy,
            enable_chunked_prefill=chunked_prefill_enabled,
            prefill_chunk_size=chunk_size,
        ),
        request=RequestGeneratorConfig(
            qps=qps,
            prefill_length=512,
            decode_length=128,
            length_distribution="lognormal",
            length_cv=0.5,
        ),
        metrics=metrics_config,
    )

    return DisaggregatedSimulator(
        ctx, config,
        prefill_schedule_policy=scheduler,
        decode_schedule_policy=scheduler,
    )


# ─── 指标提取 ──────────────────────────────────────────────────────────────────

def extract_metrics(ms: MetricsStore) -> Dict:
    """从 MetricsStore 提取关键指标。"""
    completed = [m for m in ms._request_metrics.values() if m.decode_end_time > 0]
    if not completed:
        return {"completed": 0}

    ttfts = [m.ttft for m in completed]
    tbts = [m.tbt for m in completed if m.tbt > 0]
    e2es = [m.e2e_latency for m in completed]

    return {
        "completed": len(completed),
        "ttft_p50": float(np.percentile(ttfts, 50)),
        "ttft_p99": float(np.percentile(ttfts, 99)),
        "tbt_p50": float(np.percentile(tbts, 50)) if tbts else 0.0,
        "tbt_p99": float(np.percentile(tbts, 99)) if tbts else 0.0,
        "e2e_p50": float(np.percentile(e2es, 50)),
        "e2e_p99": float(np.percentile(e2es, 99)),
    }


# ─── Pareto 前沿 ───────────────────────────────────────────────────────────────

def find_pareto_frontier(
    results: List[DSEResult],
    x_key: str = "ttft_p50",
    y_key: str = "tbt_p50",
    minimize_x: bool = True,
    minimize_y: bool = True,
) -> List[DSEResult]:
    """找出 Pareto 最优前沿 (最小化两个目标)。"""
    valid = [r for r in results if r.completed > 0]
    if not valid:
        return []

    # 按 X 排序 (从小到大)
    sorted_pts = sorted(valid, key=lambda r: getattr(r, x_key))

    pareto = []
    best_y = float("inf") if minimize_y else float("-inf")

    for r in sorted_pts:
        y = getattr(r, y_key)
        if minimize_y and y < best_y:
            pareto.append(r)
            best_y = y
        elif not minimize_y and y > best_y:
            pareto.append(r)
            best_y = y

    return pareto


# ─── 主实验 ────────────────────────────────────────────────────────────────────

def run_experiment() -> List[DSEResult]:
    """运行完整的 DSE 扫描。"""
    all_results: List[DSEResult] = []
    total = (
        len(SCHEDULERS) * len(KV_TRANSFER_STRATEGIES)
        * len(CHUNKED_PREFILL_CONFIGS) * len(QPS_LIST)
    )
    idx = 0

    for scheduler, kv_transfer, chunk_cfg, qps in itertools.product(
        SCHEDULERS, KV_TRANSFER_STRATEGIES, CHUNKED_PREFILL_CONFIGS, QPS_LIST
    ):
        idx += 1
        point = DSEPoint(
            scheduler=scheduler,
            kv_transfer=kv_transfer,
            chunked_prefill=chunk_cfg["label"],
            chunk_size=chunk_cfg["chunk_size"],
            qps=qps,
        )

        print(f"  [{idx:3d}/{total}] {point.config_key()} ...", end=" ", flush=True)

        try:
            sim = create_simulator(
                qps=qps,
                scheduler=scheduler,
                kv_transfer=kv_transfer,
                chunked_prefill_enabled=chunk_cfg["enabled"],
                chunk_size=chunk_cfg["chunk_size"],
            )
            ms = sim.run()
            metrics = extract_metrics(ms)

            result = DSEResult(
                config=point,
                ttft_p50=metrics.get("ttft_p50", 0),
                ttft_p99=metrics.get("ttft_p99", 0),
                tbt_p50=metrics.get("tbt_p50", 0),
                tbt_p99=metrics.get("tbt_p99", 0),
                e2e_p50=metrics.get("e2e_p50", 0),
                e2e_p99=metrics.get("e2e_p99", 0),
                completed=metrics.get("completed", 0),
            )
            print(f"TTFT_P50={result.ttft_p50:.1f}ms  TBT_P50={result.tbt_p50:.2f}ms  "
                  f"E2E_P50={result.e2e_p50:.1f}ms  done={result.completed}")
        except Exception as e:
            result = DSEResult(config=point)
            print(f"FAILED: {e}")

        all_results.append(result)

    return all_results


# ─── 图表生成 ──────────────────────────────────────────────────────────────────

def plot_pareto_frontier(
    results: List[DSEResult],
    pareto: List[DSEResult],
    output_dir: Path,
):
    """生成 Pareto 前沿散点图 (论文级, 单列宽度 ~3.5 inches)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # 论文级样式
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    # 颜色映射: 调度策略
    scheduler_colors = {
        "fcfs": "#E74C3C",   # 红
        "sjf":  "#3498DB",   # 蓝
        "po":   "#2ECC71",   # 绿
        "srtf": "#F39C12",   # 橙
    }

    # 形状映射: KV 传输方式
    kv_markers = {
        "DIRECT":    "o",
        "PIPELINED": "s",
    }

    # Chunked prefill 用填充区分
    chunk_fills = {
        "NoChunk":    "full",
        "Chunk512":   "none",
    }

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # 画所有点 (用 QPS 区分大小, alpha 表示重叠度)
    for r in results:
        if r.completed == 0:
            continue
        color = scheduler_colors.get(r.config.scheduler, "#999999")
        marker = kv_markers.get(r.config.kv_transfer, "o")
        fill = chunk_fills.get(r.config.chunked_prefill, "full")

        # 大小随 QPS 增大，方便区分负载级别
        size_map = {5: 25, 10: 40, 20: 55}
        size = size_map.get(int(r.config.qps), 30)

        edgecolors = color if fill == "none" else "none"

        ax.scatter(
            r.ttft_p50, r.tbt_p50,
            c=color if fill == "full" else "none",
            marker=marker,
            s=size,
            alpha=0.45,
            edgecolors=edgecolors,
            linewidths=1.0 if fill == "none" else 0,
            zorder=2,
        )

    # Pareto 点高亮 (带描边)
    pareto_sorted = sorted(pareto, key=lambda r: r.ttft_p50)
    for r in pareto_sorted:
        color = scheduler_colors.get(r.config.scheduler, "#999999")
        marker = kv_markers.get(r.config.kv_transfer, "o")
        fill = chunk_fills.get(r.config.chunked_prefill, "full")
        edgecolors = color if fill == "none" else "#000000"

        ax.scatter(
            r.ttft_p50, r.tbt_p50,
            c=color if fill == "full" else "none",
            marker=marker,
            s=80,
            alpha=1.0,
            edgecolors=edgecolors,
            linewidths=1.5,
            zorder=4,
        )

    # 标注 QPS 级别 (选代表点)
    # 每个 (chunk, qps) 组合取一个代表点标注
    annotated = set()
    for r in results:
        if r.completed == 0:
            continue
        key = (r.config.chunked_prefill, int(r.config.qps))
        if key not in annotated:
            annotated.add(key)
            label = f"QPS={int(r.config.qps)}"
            if r.config.chunked_prefill == "Chunk512":
                label += "\n(chunk)"
            # 只在最左和最右标注，避免拥挤
            if int(r.config.qps) in (5, 20):
                ax.annotate(
                    label,
                    (r.ttft_p50, r.tbt_p50),
                    fontsize=6,
                    ha="left" if r.ttft_p50 < 1000 else "right",
                    va="bottom",
                    xytext=(4, 4),
                    textcoords="offset points",
                    color="#333333",
                )

    # X 轴 log scale (TTFT 跨越 3 个数量级)
    ax.set_xscale("log")
    ax.set_xlabel("TTFT P50 (ms)")
    ax.set_ylabel("TBT P50 (ms)")
    ax.set_title("Design Space: TTFT vs TBT")

    # 图例: 调度策略 (颜色)
    sched_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=5, label=s.upper())
        for s, c in scheduler_colors.items()
    ]
    # 图例: KV 传输 (形状)
    kv_handles = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor="#666666", markersize=5, label=k)
        for k, m in kv_markers.items()
    ]
    # 图例: Chunked prefill (填充)
    chunk_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#666666" if f == "full" else "none",
               markeredgecolor="#666666",
               markersize=5, label=k)
        for k, f in chunk_fills.items()
    ]

    legend1 = ax.legend(
        handles=sched_handles, title="Scheduler",
        loc="upper right", framealpha=0.8,
    )
    legend2 = ax.legend(
        handles=kv_handles + chunk_handles, title="KV / Chunk",
        loc="lower right", framealpha=0.8,
    )
    ax.add_artist(legend1)

    plt.tight_layout(pad=0.5)
    plt.savefig(output_dir / "dse_pareto.pdf")
    plt.savefig(output_dir / "dse_pareto.png")
    plt.close()
    print(f"\n  图表已保存: {output_dir / 'dse_pareto.pdf'}")


def print_pareto_table(pareto: List[DSEResult]):
    """打印 Pareto 前沿配置汇总表。"""
    if not pareto:
        print("\n  未找到 Pareto 最优配置")
        return

    pareto_sorted = sorted(pareto, key=lambda r: r.ttft_p50)

    print("\n" + "=" * 95)
    print("  Pareto 前沿配置 (TTFT P50 vs TBT P50, 最小化两个目标)")
    print("=" * 95)
    print(f"  {'#':<3} {'Scheduler':<10} {'KV Transfer':<12} {'Chunk':<10} "
          f"{'QPS':<5} {'TTFT P50':>10} {'TBT P50':>10} {'E2E P50':>10} {'Completed':>10}")
    print("  " + "-" * 88)

    for i, r in enumerate(pareto_sorted, 1):
        print(f"  {i:<3} {r.config.scheduler.upper():<10} {r.config.kv_transfer:<12} "
              f"{r.config.chunked_prefill:<10} {int(r.config.qps):<5} "
              f"{r.ttft_p50:>10.1f} {r.tbt_p50:>10.2f} {r.e2e_p50:>10.1f} "
              f"{r.completed:>10}")

    print("=" * 95)


def print_all_results_table(results: List[DSEResult]):
    """打印所有配置的结果汇总。"""
    valid = [r for r in results if r.completed > 0]
    if not valid:
        print("\n  无有效结果")
        return

    print("\n" + "=" * 105)
    print("  全量设计空间扫描结果")
    print("=" * 105)
    print(f"  {'#':<3} {'Scheduler':<10} {'KV':<11} {'Chunk':<10} "
          f"{'QPS':<5} {'TTFT_P50':>9} {'TTFT_P99':>9} {'TBT_P50':>9} "
          f"{'E2E_P50':>9} {'Done':>5} {'Pareto':>7}")
    print("  " + "-" * 98)

    for i, r in enumerate(valid, 1):
        pareto_mark = "*" if r.is_pareto else ""
        print(f"  {i:<3} {r.config.scheduler.upper():<10} {r.config.kv_transfer:<11} "
              f"{r.config.chunked_prefill:<10} {int(r.config.qps):<5} "
              f"{r.ttft_p50:>9.1f} {r.ttft_p99:>9.1f} {r.tbt_p50:>9.2f} "
              f"{r.e2e_p50:>9.1f} {r.completed:>5} {pareto_mark:>7}")

    print("=" * 105)
    print(f"  共 {len(valid)} 个有效配置, "
          f"其中 {sum(1 for r in valid if r.is_pareto)} 个 Pareto 最优点")


def save_results_json(results: List[DSEResult], pareto: List[DSEResult], output_dir: Path):
    """保存 JSON 结果。"""
    data = {
        "experiment": "DSE Pareto Frontier",
        "model": "Qwen3-30B-A3B",
        "device": "A800",
        "cluster": "1P(4xA800) + 1D(4xA800)",
        "search_dimensions": {
            "schedulers": SCHEDULERS,
            "kv_transfer": KV_TRANSFER_STRATEGIES,
            "chunked_prefill": [c["label"] for c in CHUNKED_PREFILL_CONFIGS],
            "qps": QPS_LIST,
        },
        "total_configs": len(results),
        "valid_configs": sum(1 for r in results if r.completed > 0),
        "pareto_count": len(pareto),
        "all_results": [],
        "pareto_frontier": [],
    }

    for r in results:
        entry = {
            "config": {
                "scheduler": r.config.scheduler,
                "kv_transfer": r.config.kv_transfer,
                "chunked_prefill": r.config.chunked_prefill,
                "chunk_size": r.config.chunk_size,
                "qps": r.config.qps,
            },
            "ttft_p50_ms": round(r.ttft_p50, 3),
            "ttft_p99_ms": round(r.ttft_p99, 3),
            "tbt_p50_ms": round(r.tbt_p50, 3),
            "tbt_p99_ms": round(r.tbt_p99, 3),
            "e2e_p50_ms": round(r.e2e_p50, 3),
            "e2e_p99_ms": round(r.e2e_p99, 3),
            "completed": r.completed,
            "is_pareto": r.is_pareto,
        }
        data["all_results"].append(entry)

    pareto_sorted = sorted(pareto, key=lambda r: r.ttft_p50)
    for r in pareto_sorted:
        data["pareto_frontier"].append({
            "scheduler": r.config.scheduler,
            "kv_transfer": r.config.kv_transfer,
            "chunked_prefill": r.config.chunked_prefill,
            "qps": r.config.qps,
            "ttft_p50_ms": round(r.ttft_p50, 3),
            "tbt_p50_ms": round(r.tbt_p50, 3),
            "e2e_p50_ms": round(r.e2e_p50, 3),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dse_pareto.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON 已保存: {json_path}")


# ─── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    print()
    print("  DistLMSim Design Space Exploration (DSE)")
    print(f"  模型:   Qwen3-30B-A3B (48 layers, 128 experts, top-8)")
    print(f"  集群:   1P(4xA800 TP=4) + 1D(4xA800 TP=4)")
    print(f"  请求:   prefill=512, decode=128 (lognormal, cv=0.5)")
    print()
    print(f"  搜索维度:")
    print(f"    调度策略:      {SCHEDULERS}")
    print(f"    KV 传输:       {KV_TRANSFER_STRATEGIES}")
    print(f"    Chunked Prefill: {[c['label'] for c in CHUNKED_PREFILL_CONFIGS]}")
    print(f"    QPS:           {QPS_LIST}")
    total = len(SCHEDULERS) * len(KV_TRANSFER_STRATEGIES) * len(CHUNKED_PREFILL_CONFIGS) * len(QPS_LIST)
    print(f"    总配置数:      {total}")
    print()

    # 运行实验
    results = run_experiment()

    # Pareto 分析
    pareto = find_pareto_frontier(results, x_key="ttft_p50", y_key="tbt_p50")

    # 标记 Pareto 点
    pareto_keys = {r.config.config_key() for r in pareto}
    for r in results:
        r.is_pareto = r.config.config_key() in pareto_keys

    # 输出
    print_all_results_table(results)
    print_pareto_table(pareto)

    # 保存
    output_dir = Path("results")
    save_results_json(results, pareto, output_dir)
    plot_pareto_frontier(results, pareto, output_dir)

    print("\n  实验完成!")


if __name__ == "__main__":
    main()
