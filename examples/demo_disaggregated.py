"""存算分离 (Disaggregated Prefill/Decode) 推理模拟演示

集群拓扑:
  ┌─────────────────┐      RDMA 200 Gb/s      ┌─────────────────┐
  │  Prefill 节点    │ ◄═══════════════════════► │  Decode 节点     │
  │  Node 0          │     (RoCEv2)             │  Node 1          │
  │  ┌─────────────┐ │                          │  ┌─────────────┐ │
  │  │ GPU 0 (A800)│ │                          │  │ GPU 4 (A800)│ │
  │  │ GPU 1 (A800)│ │    NVLink/NVSwitch       │  │ GPU 5 (A800)│ │
  │  │ GPU 2 (A800)│ │    600 GB/s              │  │ GPU 6 (A800)│ │
  │  │ GPU 3 (A800)│ │    (节点内 TP=4)         │  │ GPU 7 (A800)│ │
  │  └─────────────┘ │                          │  └─────────────┘ │
  └─────────────────┘                          └─────────────────┘

推理流程:
  1. 请求到达 → Prefill 节点 (Node 0) 执行 prefill (批量)
  2. Prefill 完成 → KV Cache 通过 RDMA 传输到 Decode 节点
  3. Decode 节点 (Node 1) 执行逐 token decode
  4. 所有 decode token 完成 → 请求结束

用法:
  python3 examples/demo_disaggregated.py
  python3 examples/demo_disaggregated.py --qps 20 --prefill_length 1024
"""

import sys
import logging
import argparse

sys.path.insert(0, ".")

from main import create_disaggregated_simulator


def parse_args():
    parser = argparse.ArgumentParser(description="DistLMSim 存算分离演示")
    parser.add_argument("--qps", type=float, default=10.0, help="请求到达率 (requests/sec)")
    parser.add_argument("--prefill_length", type=int, default=512, help="Prefill token 数")
    parser.add_argument("--decode_length", type=int, default=128, help="Decode token 数")
    parser.add_argument("--prefill_batch_size", type=int, default=8, help="Prefill 批大小")
    parser.add_argument("--decode_batch_size", type=int, default=32, help="Decode 批大小")
    parser.add_argument("--tp_size", type=int, default=4, help="张量并行度 (每节点 GPU 数)")
    parser.add_argument("--rdma_bandwidth", type=float, default=200.0, help="RDMA 带宽 (Gbps)")
    parser.add_argument("--time_limit", type=float, default=60.0, help="模拟时长 (秒)")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    print("=" * 64)
    print("  DistLMSim — 存算分离推理模拟")
    print("=" * 64)
    print()
    print("集群配置:")
    print(f"  Prefill 节点: Node 0, {args.tp_size}x A800 (TP={args.tp_size})")
    print(f"  Decode  节点: Node 1, {args.tp_size}x A800 (TP={args.tp_size})")
    print(f"  节点内互联:   NVLink/NVSwitch 600 GB/s")
    print(f"  节点间互联:   RDMA (RoCEv2) {args.rdma_bandwidth} Gbps")
    print()
    print("推理配置:")
    print(f"  模型:          Qwen3-30B-A3B (48层, MoE 128专家 Top-8)")
    print(f"  QPS:           {args.qps}")
    print(f"  Prefill 长度:  {args.prefill_length} tokens")
    print(f"  Decode 长度:   {args.decode_length} tokens")
    print(f"  Prefill BS:    {args.prefill_batch_size}")
    print(f"  Decode BS:     {args.decode_batch_size}")
    print(f"  模拟时长:      {args.time_limit}s")
    print()

    sim = create_disaggregated_simulator(
        num_gpus_per_node=args.tp_size,
        qps=args.qps,
        prefill_length=args.prefill_length,
        decode_length=args.decode_length,
        prefill_batch_size=args.prefill_batch_size,
        decode_batch_size=args.decode_batch_size,
        tp_size=args.tp_size,
        rdma_bandwidth_gbps=args.rdma_bandwidth,
        time_limit_s=args.time_limit,
    )

    metrics = sim.run()
    metrics.print_summary()


if __name__ == "__main__":
    main()
