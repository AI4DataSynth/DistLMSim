"""MoE 专家负载均衡实验

测试不同 Zipf 分布参数下，4 种负载均衡策略的效果对比。

实验设计:
  - 变化 Zipf α 参数 (0.5, 1.0, 1.5, 2.0)，模拟不同程度的 token 分布偏斜
  - 对比 4 种策略: DefaultRouting, EPLB, RealisticEPLB, OmniPlacement
  - 测量: max_load, avg_load, deviation, num_migrations
  - 模型: Qwen3-30B-A3B (128 experts, top-8, 48 layers)
  - EP=8 (8 GPU 节点)

预期结果:
  - α=0.5 (均匀分布): 所有策略效果相近
  - α=1.0 (轻度偏斜): EPLB/RealisticEPLB/OmniPlacement 开始显示优势
  - α=1.5 (中度偏斜): 负载均衡策略显著降低 max_load
  - α=2.0 (重度偏斜): DefaultRouting max_load 急剧上升，其他策略保持相对平稳

用法:
  python3 examples/experiment_moe_load.py
  python3 examples/experiment_moe_load.py --alphas 0.5,1.0,1.5,2.0 --num_tokens 1000
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from distlmsim.parallelism.expert_parallel import (
    DefaultRoutingScheduler,
    EPLBScheduler,
    RealisticEPLBScheduler,
    OmniPlacementScheduler,
)


def generate_expert_demand(
    num_layers: int,
    num_experts: int,
    top_k: int,
    num_tokens: int,
    zipf_alpha: float,
    seed: int = 42,
) -> np.ndarray:
    """生成 Zipf 分布的专家需求矩阵。

    Args:
        num_layers: 模型层数
        num_experts: 专家总数
        top_k: 每 token 选择的专家数
        num_tokens: batch 中的 token 数
        zipf_alpha: Zipf 分布参数 (越大越偏斜)
        seed: 随机种子

    Returns:
        shape [num_layers, num_experts] 的专家调用次数矩阵
    """
    rng = np.random.default_rng(seed)
    
    # Zipf 分布概率
    ranks = np.arange(1, num_experts + 1)
    probs = 1.0 / (ranks ** zipf_alpha)
    probs = probs / probs.sum()
    
    layer_expert_demand = np.zeros((num_layers, num_experts), dtype=int)
    
    for layer in range(num_layers):
        # 每层每 token 选择 top_k 个专家
        for _ in range(num_tokens):
            chosen = rng.choice(num_experts, size=top_k, replace=False, p=probs)
            for exp_id in chosen:
                layer_expert_demand[layer, exp_id] += 1
    
    return layer_expert_demand


def run_experiment(
    alphas: List[float],
    num_layers: int = 48,
    num_experts: int = 128,
    top_k: int = 8,
    num_tokens: int = 1000,
    ep_size: int = 8,
    seed: int = 42,
) -> Dict:
    """运行 MoE 负载均衡实验。

    Returns:
        实验结果字典
    """
    results = {
        "config": {
            "num_layers": num_layers,
            "num_experts": num_experts,
            "top_k": top_k,
            "num_tokens": num_tokens,
            "ep_size": ep_size,
            "alphas": alphas,
        },
        "strategies": {},
    }

    strategies = [
        ("DefaultRouting", DefaultRoutingScheduler(num_experts, ep_size, top_k)),
        ("EPLB", EPLBScheduler(num_experts, ep_size, top_k)),
        ("RealisticEPLB", RealisticEPLBScheduler(num_experts, ep_size, top_k)),
        ("OmniPlacement", OmniPlacementScheduler(num_experts, ep_size, top_k)),
    ]

    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"Zipf α = {alpha}")
        print(f"{'='*60}")

        layer_expert_demand = generate_expert_demand(
            num_layers, num_experts, top_k, num_tokens, alpha, seed
        )

        # 统计原始需求分布
        avg_demand_per_layer = np.mean(layer_expert_demand, axis=0)
        print(f"\n原始专家需求分布 (平均每层):")
        print(f"  Max: {avg_demand_per_layer.max():.1f}")
        print(f"  Min: {avg_demand_per_layer.min():.1f}")
        print(f"  Mean: {avg_demand_per_layer.mean():.1f}")
        print(f"  Std: {avg_demand_per_layer.std():.1f}")

        alpha_results = {}

        for name, scheduler in strategies:
            print(f"\n策略: {name}")

            # 重置 scheduler 状态 (RealisticEPLB/OmniPlacement 有状态)
            if hasattr(scheduler, 'batch_count'):
                scheduler.batch_count = 0
                scheduler.historical_load = None
                scheduler.placement = []
            if hasattr(scheduler, 'placement_P'):
                scheduler.placement_P = None

            result = scheduler.compute_load_distribution(layer_expert_demand)

            print(f"  Max load:  {result.max_load}")
            print(f"  Avg load:  {result.avg_load}")
            print(f"  Deviation: {result.deviation:.1f}")
            if result.num_migrations > 0:
                print(f"  Migrations: {result.num_migrations}")

            alpha_results[name] = {
                "max_load": result.max_load,
                "avg_load": result.avg_load,
                "deviation": result.deviation,
                "num_migrations": result.num_migrations,
            }

        results["strategies"][alpha] = alpha_results

    return results


def plot_results(results: Dict, output_dir: Path):
    """绘制实验结果图表。"""
    alphas = results["config"]["alphas"]
    strategies = list(results["strategies"][alphas[0]].keys())

    # 提取数据
    max_loads = {s: [] for s in strategies}
    avg_loads = {s: [] for s in strategies}
    deviations = {s: [] for s in strategies}
    migrations = {s: [] for s in strategies}

    for alpha in alphas:
        for strategy in strategies:
            r = results["strategies"][alpha][strategy]
            max_loads[strategy].append(r["max_load"])
            avg_loads[strategy].append(r["avg_load"])
            deviations[strategy].append(r["deviation"])
            migrations[strategy].append(r["num_migrations"])

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MoE Expert Load Balancing: Zipf α vs Strategy Performance", fontsize=16)

    # 1. Max Load
    ax = axes[0, 0]
    for strategy in strategies:
        ax.plot(alphas, max_loads[strategy], marker='o', label=strategy, linewidth=2)
    ax.set_xlabel("Zipf α (token distribution skew)")
    ax.set_ylabel("Max GPU Load")
    ax.set_title("Max GPU Load (lower is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Load Deviation
    ax = axes[0, 1]
    for strategy in strategies:
        ax.plot(alphas, deviations[strategy], marker='s', label=strategy, linewidth=2)
    ax.set_xlabel("Zipf α")
    ax.set_ylabel("Load Deviation (max - avg)")
    ax.set_title("Load Imbalance (lower is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Max Load / Avg Load Ratio
    ax = axes[1, 0]
    for strategy in strategies:
        ratios = [m / a if a > 0 else 0 for m, a in zip(max_loads[strategy], avg_loads[strategy])]
        ax.plot(alphas, ratios, marker='^', label=strategy, linewidth=2)
    ax.set_xlabel("Zipf α")
    ax.set_ylabel("Max Load / Avg Load Ratio")
    ax.set_title("Load Imbalance Ratio (closer to 1.0 is better)")
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Migrations
    ax = axes[1, 1]
    for strategy in strategies:
        if any(m > 0 for m in migrations[strategy]):
            ax.plot(alphas, migrations[strategy], marker='D', label=strategy, linewidth=2)
    ax.set_xlabel("Zipf α")
    ax.set_ylabel("Number of Migrations")
    ax.set_title("Migration Overhead (lower is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "moe_load_balancing.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "moe_load_balancing.pdf", bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {output_dir / 'moe_load_balancing.png'}")


def main():
    parser = argparse.ArgumentParser(description="MoE 专家负载均衡实验")
    parser.add_argument("--alphas", type=str, default="0.5,1.0,1.5,2.0",
                        help="Zipf α 参数列表 (逗号分隔)")
    parser.add_argument("--num_tokens", type=int, default=1000,
                        help="每 batch 的 token 数")
    parser.add_argument("--num_layers", type=int, default=48,
                        help="模型层数")
    parser.add_argument("--num_experts", type=int, default=128,
                        help="专家总数")
    parser.add_argument("--top_k", type=int, default=8,
                        help="每 token 选择的专家数")
    parser.add_argument("--ep_size", type=int, default=8,
                        help="Expert Parallel 大小 (GPU 数)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="输出目录")

    args = parser.parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 70)
    print("  MoE Expert Load Balancing Experiment")
    print("=" * 70)
    print(f"\n实验配置:")
    print(f"  Zipf α 参数:  {alphas}")
    print(f"  模型:         Qwen3-30B-A3B ({args.num_layers} 层, {args.num_experts} 专家, Top-{args.top_k})")
    print(f"  每 batch token 数: {args.num_tokens}")
    print(f"  EP 大小:      {args.ep_size} GPUs")
    print(f"  随机种子:     {args.seed}")

    results = run_experiment(
        alphas=alphas,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        top_k=args.top_k,
        num_tokens=args.num_tokens,
        ep_size=args.ep_size,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 保存 JSON 结果
    json_path = output_dir / "moe_load_balancing.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {json_path}")

    # 绘制图表
    plot_results(results, output_dir)

    print("\n实验完成!")


if __name__ == "__main__":
    main()
