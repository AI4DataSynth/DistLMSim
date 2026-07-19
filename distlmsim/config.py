"""DistLMSim 配置系统

采用嵌套 dataclass 层级，支持 CLI argparse 自动生成。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

from distlmsim.types import (
    DeviceSKUType,
    NodeSKUType,
    InterconnectType,
    RDMAProtocolType,
    NetworkModelMode,
    DeploymentMode,
    GlobalSchedulerType,
    ReplicaSchedulerType,
    KVCacheTransferStrategy,
)


# ─── 设备 SKU 配置 ────────────────────────────────────────────────────────────

@dataclass
class DeviceSKUConfig:
    """GPU 设备参数"""
    device_type: DeviceSKUType = DeviceSKUType.A800
    fp16_tflops: float = 25.0          # FP16 算力 (TFLOPS)
    memory_gb: float = 80.0            # 显存 (GB)
    memory_bandwidth_gbps: float = 2039.0  # 显存带宽 (GB/s)


@dataclass
class NodeSKUConfig:
    """节点参数"""
    node_type: NodeSKUType = NodeSKUType.A800_DGX
    num_gpus: int = 8
    device_sku: DeviceSKUConfig = field(default_factory=DeviceSKUConfig)


# ─── 网络配置 ─────────────────────────────────────────────────────────────────

@dataclass
class NVLinkConfig:
    """NVLink/NVSwitch 互联配置 (节点内)"""
    interconnect_type: InterconnectType = InterconnectType.NVLINK_SWITCH
    bandwidth_gbps: float = 600.0      # 单链路双向带宽 (GB/s)，A800 NVLink3
    num_links_per_gpu: int = 12        # 每 GPU NVLink 链路数
    latency_us: float = 1.5            # 基础延迟 (微秒)
    # NVSwitch 参数
    nvswitch_bandwidth_gbps: float = 900.0  # NVSwitch 端口带宽


@dataclass
class RDMAConfig:
    """RDMA 网络配置 (节点间)"""
    protocol: RDMAProtocolType = RDMAProtocolType.ROCE_V2
    bandwidth_gbps: float = 200.0      # 链路带宽 (Gbps)，如 200Gb/s RoCEv2
    latency_us: float = 2.0            # 基础延迟 (微秒)
    # RoCEv2 特有参数
    congestion_control: str = "DCQCN"  # 拥塞控制算法
    congestion_alpha: float = 0.05     # 拥塞敏感度 (每增加一个并发流，带宽降低比例)
    ecn_enabled: bool = True           # ECN 显式拥塞通知
    pfc_enabled: bool = True           # PFC 优先级流控
    # InfiniBand 特有参数
    ib_subnet_manager: bool = True     # IB 子网管理器
    ib_service_level: int = 0          # IB 服务等级


@dataclass
class NetworkTopologyConfig:
    """网络拓扑配置"""
    nvlink: NVLinkConfig = field(default_factory=NVLinkConfig)
    rdma: RDMAConfig = field(default_factory=RDMAConfig)
    model_mode: NetworkModelMode = NetworkModelMode.HYBRID
    # 拓扑结构
    topology_type: str = "fat_tree"    # fat_tree, leaf_spine, torus
    num_switch_layers: int = 2         # 交换机层数
    oversubscription_ratio: float = 1.0  # 收敛比 (1.0 = 无收敛)
    # NCCL 开销参数
    nccl_cpu_launch_overhead_ms: float = 0.02
    nccl_cpu_skew_overhead_per_device_ms: float = 0.0


# ─── 集群配置 ─────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """模型参数"""
    model_name: str = "Qwen3-30B-A3B"
    num_layers: int = 48
    num_q_heads: int = 32
    num_kv_heads: int = 4
    embedding_dim: int = 2048
    mlp_hidden_dim: int = 0            # 0 表示从模型自动推导
    num_experts: int = 128             # MoE 专家数，0 表示 Dense
    top_k_experts: int = 8             # Top-K 路由
    expert_intermediate_dim: int = 0   # MoE 专家中间维度，0 表示从模型自动推导
    vocab_size: int = 151936


@dataclass
class ReplicaConfig:
    """模型副本配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    device_sku: DeviceSKUConfig = field(default_factory=DeviceSKUConfig)
    # 并行策略
    tensor_parallel_size: int = 1
    num_pipeline_stages: int = 1
    expert_parallel_size: int = 1
    enable_expert_parallel: bool = False
    # 调度器
    scheduler_type: ReplicaSchedulerType = ReplicaSchedulerType.SARATHI
    max_batch_size: int = 256
    max_num_tokens: int = 16384


@dataclass
class ClusterConfig:
    """集群配置"""
    num_nodes: int = 2
    node_sku: NodeSKUConfig = field(default_factory=NodeSKUConfig)
    network: NetworkTopologyConfig = field(default_factory=NetworkTopologyConfig)
    # 副本配置
    num_replicas: int = 1
    replica: ReplicaConfig = field(default_factory=ReplicaConfig)
    # 部署模式
    deployment_mode: DeploymentMode = DeploymentMode.COLOCATED


# ─── 存算分离配置 ──────────────────────────────────────────────────────────────

@dataclass
class DisaggregatedConfig:
    """Prefill/Decode 存算分离配置"""
    enabled: bool = False
    num_prefill_nodes: int = 1          # Prefill 专用节点数
    num_decode_nodes: int = 1           # Decode 专用节点数
    # KV Cache 传输
    kv_cache_transfer_strategy: KVCacheTransferStrategy = KVCacheTransferStrategy.DIRECT
    kv_cache_compression: bool = False  # KV Cache 压缩 (如 FP8)
    kv_cache_compression_ratio: float = 2.0
    # Store-and-Forward 参数
    store_forward_write_bw_gbps: float = 12.0   # 写入中间存储带宽 (如 NVMe: ~12 GB/s = 96 Gbps)
    store_forward_read_bw_gbps: float = 12.0    # 从中间存储读取带宽
    store_forward_latency_us: float = 100.0     # 存储 I/O 基础延迟 (μs)
    # 调度参数
    prefill_batch_size: int = 32        # Prefill 批大小 (上限，实际受 GPU 显存约束)
    decode_batch_size: int = 256        # Decode 批大小 (上限，实际受 GPU 显存约束)
    gpu_memory_utilization: float = 0.90  # GPU 显存利用率 (0-1, 预留 10% 给框架开销)
    # Chunked Prefill
    enable_chunked_prefill: bool = True
    prefill_chunk_size: int = 4096
    # Speculative Decoding
    enable_speculative_decoding: bool = False
    speculation_length: int = 4         # Draft model 生成的 candidate token 数 K
    acceptance_rate: float = 0.8        # 平均接受率 (0-1)
    # Draft model: 用较小的模型参数近似 (层数少 = 计算快)
    draft_num_layers: int = 4           # Draft model 层数 (target 48)
    draft_embedding_dim: int = 512      # Draft model 隐藏维度 (target 2048)
    # DSpark / DFlash 配置 (DeepSeek 投机解码方案)
    speculative_mode: str = "standard"  # "standard" | "dspark" | "dflash"
    block_size: int = 7                 # DSpark: 每轮起草的 token 块大小
    markov_rank: int = 256              # DSpark: Markov head 低秩维度
    markov_head_type: str = "vanilla"   # "vanilla" | "gated" | "rnn"
    num_target_layer_ids: int = 5       # DSpark: 从目标模型抽取的中间层数
    confidence_threshold: float = 0.0   # DSpark: 置信度早停阈值 (0=不启用)
    enable_confidence_scheduling: bool = False  # DSpark: 负载感知置信度调度
    # Draft model & SPS profiling
    draft_model_name: str = "dspark_5l_512d"  # Draft model 标识 (用于查找 profiling)
    sps_profile_path: str = ""            # SPS 曲线文件路径 (空=自动查找)
    bonus_token: bool = True              # 是否添加 bonus token (标准投机解码保证)
    # Acceptance Profile (DFlash/DSpark 位置级接受率)
    acceptance_profile_path: str = ""     # 外部 profile JSON 路径 (空=使用内置)
    default_domain: str = "mixed"         # 默认 workload domain (math/code/chat/mixed)
    dflash_pos1_alpha: float = 0.88       # DFlash 位置 1 基础接受率
    dflash_decay_rate: float = 0.03       # DFlash 位置衰减率 (快速, suffix decay)
    dspark_pos1_alpha: float = 0.88       # DSpark 位置 1 基础接受率 (继承 DFlash)
    dspark_decay_rate: float = 0.008      # DSpark 位置衰减率 (平稳, Markov head 缓解)
    # MoE Expert Load Imbalance
    moe_expert_load_zipf_alpha: float = 1.0  # Zipf α 控制 token→expert 分布偏斜度
                                              # 1.0 = 均匀, >1.0 = 偏斜


# ─── 调度配置 ─────────────────────────────────────────────────────────────────

@dataclass
class SchedulingConfig:
    """调度配置"""
    global_scheduler_type: GlobalSchedulerType = GlobalSchedulerType.ROUND_ROBIN
    replica_scheduler_type: ReplicaSchedulerType = ReplicaSchedulerType.SARATHI
    # 请求迁移
    enable_request_migration: bool = False
    migration_interval_ms: float = 1000.0
    # 负载均衡
    load_balancing_interval_ms: float = 500.0


# ─── 请求生成配置 ──────────────────────────────────────────────────────────────

@dataclass
class RequestGeneratorConfig:
    """请求生成配置"""
    generator_type: str = "synthetic"   # synthetic, trace_replay
    # 合成请求参数
    qps: float = 10.0                   # 请求到达率 (requests/sec)
    prefill_length: int = 2048          # 平均 prefill token 数
    decode_length: int = 512            # 平均 decode token 数
    # Trace 回放
    trace_file: Optional[str] = None
    # 请求到达间隔分布
    interval_distribution: str = "poisson"  # poisson, gamma
    gamma_shape: float = 2.0            # Gamma 分布 shape 参数 k (gamma 模式)
    # 请求长度分布
    length_distribution: str = "normal"  # fixed, normal, lognormal
    length_generator_type: str = "normal"  # fixed, normal, zipf
    length_cv: float = 0.3              # 变异系数 (normal/lognormal)
    prefill_length_cv: float = -1.0     # prefill 专用 CV (-1 = 使用 length_cv)
    decode_length_cv: float = -1.0      # decode 专用 CV (-1 = 使用 length_cv)
    zipf_alpha: float = 1.5             # Zipf 分布 alpha 参数 (zipf 模式)


# ─── 指标配置 ─────────────────────────────────────────────────────────────────

@dataclass
class MetricsConfig:
    """指标收集配置"""
    enable_detailed_logging: bool = True
    output_dir: str = "results"
    # 分位数
    percentiles: list = field(default_factory=lambda: [50, 90, 95, 99])
    # 可视化
    enable_plots: bool = False


# ─── 顶层配置 ─────────────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """模拟器顶层配置"""
    seed: int = 42
    log_level: str = "INFO"
    time_limit_s: float = 60.0          # 模拟时间上限 (秒)

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    disaggregated: DisaggregatedConfig = field(default_factory=DisaggregatedConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    request: RequestGeneratorConfig = field(default_factory=RequestGeneratorConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    @classmethod
    def from_cli(cls) -> "SimulationConfig":
        """从命令行参数构建配置。"""
        parser = argparse.ArgumentParser(description="DistLMSim 分布式推理模拟器")

        # 顶层参数
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--log_level", type=str, default="INFO")
        parser.add_argument("--time_limit_s", type=float, default=60.0)

        # 集群参数
        parser.add_argument("--num_nodes", type=int, default=2)
        parser.add_argument("--num_gpus_per_node", type=int, default=8)
        parser.add_argument("--num_replicas", type=int, default=1)

        # 网络参数
        parser.add_argument("--rdma_protocol", type=str, default="ROCE_V2",
                            choices=["ROCE_V2", "INFINIBAND", "TCP_IP"])
        parser.add_argument("--rdma_bandwidth_gbps", type=float, default=200.0)
        parser.add_argument("--nvlink_bandwidth_gbps", type=float, default=600.0)
        parser.add_argument("--network_model_mode", type=str, default="HYBRID",
                            choices=["ANALYTICAL", "PROFILING", "HYBRID"])

        # 并行策略
        parser.add_argument("--tensor_parallel_size", type=int, default=1)
        parser.add_argument("--num_pipeline_stages", type=int, default=1)
        parser.add_argument("--expert_parallel_size", type=int, default=1)

        # 模型参数
        parser.add_argument("--model_name", type=str, default="Qwen3-30B-A3B")
        parser.add_argument("--num_layers", type=int, default=48)

        # 调度参数
        parser.add_argument("--global_scheduler", type=str, default="ROUND_ROBIN")
        parser.add_argument("--replica_scheduler", type=str, default="SARATHI")

        # 存算分离
        parser.add_argument("--disaggregated", action="store_true")
        parser.add_argument("--num_prefill_nodes", type=int, default=1)
        parser.add_argument("--num_decode_nodes", type=int, default=1)

        # 请求参数
        parser.add_argument("--qps", type=float, default=10.0)
        parser.add_argument("--prefill_length", type=int, default=2048)
        parser.add_argument("--decode_length", type=int, default=512)

        args = parser.parse_args()

        # 构建嵌套配置
        device_sku = DeviceSKUConfig()
        node_sku = NodeSKUConfig(device_sku=device_sku)
        nvlink = NVLinkConfig(bandwidth_gbps=args.nvlink_bandwidth_gbps)
        rdma = RDMAConfig(
            protocol=RDMAProtocolType[args.rdma_protocol],
            bandwidth_gbps=args.rdma_bandwidth_gbps,
        )
        network = NetworkTopologyConfig(
            nvlink=nvlink,
            rdma=rdma,
            model_mode=NetworkModelMode[args.network_model_mode],
        )
        model = ModelConfig(model_name=args.model_name, num_layers=args.num_layers)
        replica = ReplicaConfig(
            model=model,
            device_sku=device_sku,
            tensor_parallel_size=args.tensor_parallel_size,
            num_pipeline_stages=args.num_pipeline_stages,
            expert_parallel_size=args.expert_parallel_size,
        )
        cluster = ClusterConfig(
            num_nodes=args.num_nodes,
            node_sku=node_sku,
            network=network,
            num_replicas=args.num_replicas,
            replica=replica,
        )
        disaggregated = DisaggregatedConfig(
            enabled=args.disaggregated,
            num_prefill_nodes=args.num_prefill_nodes,
            num_decode_nodes=args.num_decode_nodes,
        )
        scheduling = SchedulingConfig(
            global_scheduler_type=GlobalSchedulerType[args.global_scheduler],
            replica_scheduler_type=ReplicaSchedulerType[args.replica_scheduler],
        )
        request = RequestGeneratorConfig(
            qps=args.qps,
            prefill_length=args.prefill_length,
            decode_length=args.decode_length,
        )

        return cls(
            seed=args.seed,
            log_level=args.log_level,
            time_limit_s=args.time_limit_s,
            cluster=cluster,
            disaggregated=disaggregated,
            scheduling=scheduling,
            request=request,
        )
