"""示例: 2节点集群，TP=4 PP=2 部署 Qwen3-30B-A3B

集群配置:
- 2 台 A800 DGX 节点 (各 8 GPU)
- 节点间 200 Gb/s RoCEv2 RDMA
- 节点内 NVSwitch 全互联

模型部署:
- TP=4 (同节点 4 GPU 张量并行)
- PP=2 (2 个流水线 stage，每 stage 24 层)
- DP=1 (单副本)

用法:
    python examples/demo_2node_tp4_pp2.py
"""

import sys
sys.path.insert(0, ".")

from distlmsim.config import (
    SimulationConfig,
    ClusterConfig,
    NodeSKUConfig,
    DeviceSKUConfig,
    NetworkTopologyConfig,
    NVLinkConfig,
    RDMAConfig,
    ReplicaConfig,
    ModelConfig,
    SchedulingConfig,
    RequestGeneratorConfig,
    DisaggregatedConfig,
)
from distlmsim.types import (
    DeviceSKUType,
    RDMAProtocolType,
    NetworkModelMode,
    GlobalSchedulerType,
)
from distlmsim.parallelism.parallelism_planner import ParallelismPlanner


def main():
    # 设备配置
    device = DeviceSKUConfig(
        device_type=DeviceSKUType.A800,
        fp16_tflops=25.0,
        memory_gb=80.0,
    )
    node_sku = NodeSKUConfig(num_gpus=8, device_sku=device)

    # 网络配置
    nvlink = NVLinkConfig(bandwidth_gbps=600.0)
    rdma = RDMAConfig(protocol=RDMAProtocolType.ROCE_V2, bandwidth_gbps=200.0)
    network = NetworkTopologyConfig(
        nvlink=nvlink,
        rdma=rdma,
        model_mode=NetworkModelMode.ANALYTICAL,
    )

    # 模型配置
    model = ModelConfig(
        model_name="Qwen3-30B-A3B",
        num_layers=48,
        num_q_heads=32,
        num_kv_heads=4,
        embedding_dim=2048,
        num_experts=128,
        top_k_experts=8,
    )

    # 副本配置
    replica = ReplicaConfig(
        model=model,
        device_sku=device,
        tensor_parallel_size=4,
        num_pipeline_stages=2,
    )

    # 集群配置
    cluster = ClusterConfig(
        num_nodes=2,
        node_sku=node_sku,
        network=network,
        num_replicas=1,
        replica=replica,
    )

    # 使用规划器验证并行策略
    planner = ParallelismPlanner(model, cluster)
    plan = planner.recommend_plan()

    print("推荐的并行策略:")
    print(f"  TP={plan.tp_size}, PP={plan.pp_size}, DP={plan.dp_size}, EP={plan.ep_size}")
    print(f"  每副本 GPU 数: {plan.gpus_per_replica}")
    print(f"  副本数: {plan.num_replicas}")
    print(f"  Stage->Node 映射: {plan.stage_node_mapping}")

    # 枚举所有可行方案
    all_plans = planner.enumerate_plans()
    print(f"\n所有可行方案 ({len(all_plans)} 种):")
    for p in all_plans[:10]:
        print(f"  TP={p.tp_size}, PP={p.pp_size}, DP={p.dp_size}")


if __name__ == "__main__":
    main()
