"""模拟结果可视化脚本

运行 disaggregated 模拟并将结果保存为图表和 JSON 数据。
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")

import numpy as np

from main import create_disaggregated_simulator

# matplotlib 需要 non-GUI backend 以避免 macOS 弹窗问题
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def run_and_collect():
    """运行模拟并返回 MetricsStore。"""
    sim = create_disaggregated_simulator(
        num_gpus_per_node=4,
        qps=10.0,
        prefill_length=512,
        decode_length=128,
        prefill_batch_size=8,
        decode_batch_size=32,
        tp_size=4,
        rdma_bandwidth_gbps=200.0,
        time_limit_s=60.0,
    )
    return sim.run()


def extract_metrics(ms):
    """从 MetricsStore 提取可序列化数据。"""
    completed = [
        m for m in ms._request_metrics.values()
        if m.decode_end_time > 0
    ]
    ttfts = [m.ttft for m in completed]
    e2e = [m.e2e_latency for m in completed]
    tbts = [m.tbt for m in completed if m.tbt > 0]
    kv_times = [m.kv_cache_transfer_time for m in completed if m.kv_cache_transfer_time > 0]
    prefill_times = [m.prefill_time for m in completed if m.prefill_time > 0]
    decode_times = [m.decode_time for m in completed if m.decode_time > 0]
    scheduling_delays = [m.scheduling_delay for m in completed if m.scheduling_delay > 0]

    total_decode_tokens = sum(m.decode_tokens for m in completed if m.decode_tokens > 0)
    total_prefill_tokens = sum(m.prefill_tokens for m in completed if m.prefill_tokens > 0)
    wall_time = max(m.decode_end_time for m in completed)
    first_arrival = min(m.arrival_time for m in completed)
    effective_time_s = (wall_time - first_arrival) / 1000.0

    def pct(arr, p):
        return float(np.percentile(arr, p)) if arr else 0.0

    return {
        "num_completed": len(completed),
        "num_total": len(ms._request_metrics),
        "wall_time_ms": wall_time,
        "wall_time_s": wall_time / 1000.0,
        "prefill": {
            "total_tokens": total_prefill_tokens,
            "mean": float(np.mean(prefill_times)) if prefill_times else 0.0,
            "P50": pct(prefill_times, 50),
            "P99": pct(prefill_times, 99),
        },
        "kv_transfer": {
            "mean": float(np.mean(kv_times)) if kv_times else 0.0,
            "P50": pct(kv_times, 50),
            "P99": pct(kv_times, 99),
        },
        "ttft": {"P50": pct(ttfts, 50), "P90": pct(ttfts, 90), "P95": pct(ttfts, 95), "P99": pct(ttfts, 99)},
        "tbt": {"P50": pct(tbts, 50), "P90": pct(tbts, 90), "P95": pct(tbts, 95), "P99": pct(tbts, 99)},
        "decode": {
            "mean": float(np.mean(decode_times)) if decode_times else 0.0,
            "P50": pct(decode_times, 50),
        },
        "e2e": {"P50": pct(e2e, 50), "P90": pct(e2e, 90), "P95": pct(e2e, 95), "P99": pct(e2e, 99)},
        "throughput": {
            "decode_tokens_s": total_decode_tokens / effective_time_s if effective_time_s > 0 else 0.0,
            "prefill_tokens_s": total_prefill_tokens / effective_time_s if effective_time_s > 0 else 0.0,
        },
        "scheduling_delay": {
            "mean": float(np.mean(scheduling_delays)) if scheduling_delays else 0.0,
            "P50": pct(scheduling_delays, 50),
            "P99": pct(scheduling_delays, 99),
        },
        # 原始数据用于绘图
        "raw": {
            "ttfts": ttfts,
            "e2e": e2e,
            "tbts": tbts,
            "kv_times": kv_times,
            "prefill_times": prefill_times,
            "decode_times": decode_times,
            "scheduling_delays": scheduling_delays,
        },
    }


def save_json(data: dict, path: Path) -> None:
    """保存 JSON，去除 raw 数据。"""
    export = {k: v for k, v in data.items() if k != "raw"}
    export["timestamp"] = datetime.now().isoformat()
    path.write_text(json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8")


# ─── 绘图函数 ────────────────────────────────────────────────────────────────

COLORS = {
    "blue": "#3B82F6",
    "green": "#10B981",
    "orange": "#F59E0B",
    "red": "#EF4444",
    "purple": "#8B5CF6",
    "teal": "#14B8A6",
}

BAR_WIDTH = 0.6
FIG_DPI = 150
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 10


def _setup_ax(ax, title, ylabel, ylim=None):
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    if ylim:
        ax.set_ylim(ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E5E7EB")
    ax.spines["bottom"].set_color("#E5E7EB")
    ax.yaxis.grid(True, alpha=0.3)


def plot_latency_bar(data: dict, path: Path) -> None:
    """柱状图: TTFT / E2E / Prefill / Decode / KV Transfer 的 P50/P90/P99 对比。"""
    raw = data["raw"]
    ttfts, e2e = raw["ttfts"], raw["e2e"]
    prefill = raw["prefill_times"]
    decode = raw["decode_times"]
    kv = raw["kv_times"]

    metrics = ["TTFT", "E2E", "Prefill", "Decode", "KV Transfer"]
    datasets = [ttfts, e2e, prefill, decode, kv]
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["teal"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    percentiles = [50, 90, 99]
    pct_labels = ["P50", "P90", "P99"]

    for idx, (p, label) in enumerate(zip(percentiles, pct_labels)):
        ax = axes[idx]
        values = [np.percentile(ds, p) for ds in datasets]
        bars = ax.bar(metrics, values, color=colors, edgecolor="white", width=BAR_WIDTH)
        # 在柱子上显示数值
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold"
            )
        _setup_ax(ax, f"{label} Latency", "Latency (ms)")
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("DistLMSim — Latency Breakdown (ms)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cdf(data: dict, path: Path) -> None:
    """CDF 累积分布图: TTFT 和 E2E 延迟。"""
    raw = data["raw"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, label, color in [
        (axes[0], "TTFT (ms)", COLORS["blue"]),
        (axes[1], "E2E Latency (ms)", COLORS["green"]),
    ]:
        vals = sorted(raw["ttfts"] if "TTFT" in label else raw["e2e"])
        n = len(vals)
        cdf = np.arange(1, n + 1) / n
        ax.plot(vals, cdf, color=color, linewidth=2)
        ax.fill_between(vals, cdf, alpha=0.15, color=color)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(0.9, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(0.99, color="gray", linestyle="--", alpha=0.5)
        _setup_ax(ax, f"{label} CDF", "CDF")
        ax.set_xlabel(label, fontsize=FONT_SIZE_LABEL)
        ax.text(0.02, 0.55, "P50", fontsize=8, color="gray", transform=ax.transAxes)
        ax.text(0.02, 0.91, "P90", fontsize=8, color="gray", transform=ax.transAxes)
        ax.text(0.02, 0.995, "P99", fontsize=8, color="gray", transform=ax.transAxes)

    plt.suptitle("DistLMSim — Cumulative Distribution Function", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_throughput_and_load(data: dict, path: Path) -> None:
    """柱状图: 吞吐量和调度延迟。"""
    tp = data["throughput"]
    sd = data["scheduling_delay"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 吞吐量
    labels = ["Prefill\n(tokens/s)", "Decode\n(tokens/s)"]
    values = [tp["prefill_tokens_s"], tp["decode_tokens_s"]]
    colors = [COLORS["orange"], COLORS["blue"]]
    bars = axes[0].bar(labels, values, color=colors, edgecolor="white", width=BAR_WIDTH)
    for bar, val in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
            f"{val:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
    _setup_ax(axes[0], "Throughput", "Tokens / second")

    # 调度延迟
    sd_labels = ["Mean", "P50", "P99"]
    sd_values = [sd["mean"], sd["P50"], sd["P99"]]
    bars = axes[1].bar(sd_labels, sd_values, color=COLORS["red"], edgecolor="white", width=BAR_WIDTH)
    for bar, val in zip(bars, sd_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sd_values) * 0.01,
            f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold"
        )
    _setup_ax(axes[1], "Scheduling Delay", "Delay (ms)")

    plt.suptitle("DistLMSim — Throughput & Scheduling Delay", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_timeline(data: dict, path: Path) -> None:
    """甘特图风格: 前 30 个请求的生命周期时间线。"""
    ms = _dummy_ms_from_raw(data["raw"])
    completed = sorted(
        [m for m in ms._request_metrics.values() if m.decode_end_time > 0],
        key=lambda m: m.arrival_time
    )[:30]

    if not completed:
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    colors_map = {
        "prefill": COLORS["blue"],
        "kv_transfer": COLORS["orange"],
        "decode": COLORS["green"],
    }

    for i, m in enumerate(completed):
        base = i
        # Prefill
        if m.prefill_end_time > 0:
            start = m.prefill_start_time if m.prefill_start_time > 0 else m.arrival_time
            width = m.prefill_end_time - start
            ax.broken_barh(
                [(start, width)], (base - 0.3, 0.6),
                facecolors=colors_map["prefill"], alpha=0.8, edgecolor="white"
            )
        # KV Transfer
        if m.kv_cache_transfer_end > 0:
            start = m.kv_cache_transfer_start
            width = m.kv_cache_transfer_end - start
            ax.broken_barh(
                [(start, width)], (base - 0.3, 0.6),
                facecolors=colors_map["kv_transfer"], alpha=0.8, edgecolor="white"
            )
        # Decode
        if m.decode_end_time > 0:
            start = m.decode_start_time if m.decode_start_time > 0 else m.kv_cache_transfer_end
            width = m.decode_end_time - start
            ax.broken_barh(
                [(start, width)], (base - 0.3, 0.6),
                facecolors=colors_map["decode"], alpha=0.8, edgecolor="white"
            )

    ax.set_xlabel("Time (ms)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Request ID", fontsize=FONT_SIZE_LABEL)
    ax.set_yticks(range(len(completed)))
    ax.set_yticklabels([f"Req {m.request_id}" for m in completed], fontsize=8)
    ax.set_title("Request Timeline (First 30 Requests)", fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_map["prefill"], label="Prefill"),
        Patch(facecolor=colors_map["kv_transfer"], label="KV Transfer"),
        Patch(facecolor=colors_map["decode"], label="Decode"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _dummy_ms_from_raw(raw: dict):
    """从原始数据重建一个假的 MetricsStore 用于时间线绘图。"""
    from distlmsim.config import MetricsConfig
    from distlmsim.metrics.metrics_store import MetricsStore, RequestMetrics

    config = MetricsConfig()
    ms = MetricsStore(config)
    # 用原始数据重建 request metrics
    for i in range(len(raw["ttfts"])):
        rm = RequestMetrics(request_id=i)
        rm.arrival_time = 0  # 简化处理
        rm.prefill_start_time = 0
        rm.prefill_end_time = raw["ttfts"][i]  # 近似
        rm.kv_cache_transfer_start = rm.prefill_end_time
        kv_val = raw["kv_times"][i] if i < len(raw["kv_times"]) else 0
        rm.kv_cache_transfer_end = rm.kv_cache_transfer_start + kv_val
        rm.decode_start_time = rm.kv_cache_transfer_end
        decode_val = raw["decode_times"][i] if i < len(raw["decode_times"]) else (raw["decode_times"][-1] if raw["decode_times"] else 10)
        rm.decode_end_time = rm.decode_start_time + decode_val
        ms._request_metrics[i] = rm
    return ms


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 64)
    print("  DistLMSim — 模拟结果可视化")
    print("=" * 64)
    print()

    # 1. 运行模拟
    print("▶ 运行模拟...")
    ms = run_and_collect()
    print(f"  ✓ 完成 {ms.get_completed_count()} 个请求")
    print()

    # 2. 提取数据
    data = extract_metrics(ms)

    # 3. 保存 JSON
    json_path = results_dir / "simulation_results.json"
    save_json(data, json_path)
    print(f"✓ JSON 结果: {json_path}")

    # 4. 生成图表
    print()
    print("▶ 生成图表...")

    latency_path = results_dir / "latency_breakdown.png"
    plot_latency_bar(data, latency_path)
    print(f"  ✓ 延迟柱状图: {latency_path}")

    cdf_path = results_dir / "latency_cdf.png"
    plot_cdf(data, cdf_path)
    print(f"  ✓ CDF 分布图: {cdf_path}")

    throughput_path = results_dir / "throughput_scheduling.png"
    plot_throughput_and_load(data, throughput_path)
    print(f"  ✓ 吞吐量 & 调度延迟: {throughput_path}")

    timeline_path = results_dir / "request_timeline.png"
    plot_timeline(data, timeline_path)
    print(f"  ✓ 请求时间线: {timeline_path}")

    print()
    print("=" * 64)
    print("  完成！所有结果已保存至 results/")
    print("=" * 64)


if __name__ == "__main__":
    main()
