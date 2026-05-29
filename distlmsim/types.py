"""DistLMSim 枚举类型定义"""

from enum import Enum, auto


class DeviceSKUType(Enum):
    """GPU 设备型号"""
    A800 = auto()
    A100 = auto()
    H100 = auto()
    H200 = auto()


class NodeSKUType(Enum):
    """节点型号"""
    A800_DGX = auto()  # 8x A800 + NVSwitch
    H100_DGX = auto()  # 8x H100 + NVSwitch


class InterconnectType(Enum):
    """节点内互联类型"""
    NVLINK_SWITCH = auto()  # NVSwitch 全互联 (A800 DGX)
    NVLINK_MESH = auto()    # NVLink 部分互联


class RDMAProtocolType(Enum):
    """节点间 RDMA 协议类型"""
    ROCE_V2 = auto()      # RoCEv2 over Converged Ethernet
    INFINIBAND = auto()   # InfiniBand
    TCP_IP = auto()       # TCP/IP (降级方案，用于对比)


class NetworkModelMode(Enum):
    """网络建模模式"""
    ANALYTICAL = auto()   # 解析模型 (带宽/延迟公式)
    PROFILING = auto()    # Profiling 数据驱动
    HYBRID = auto()       # 混合：有 profiling 用 profiling，否则回退解析


class ParallelismType(Enum):
    """并行策略类型"""
    TENSOR = auto()       # 张量并行 (TP)
    PIPELINE = auto()     # 流水线并行 (PP)
    DATA = auto()         # 数据并行 (DP)
    EXPERT = auto()       # 专家并行 (EP)


class DeploymentMode(Enum):
    """部署模式"""
    COLOCATED = auto()        # Prefill/Decode 混合部署
    DISAGGREGATED = auto()    # Prefill/Decode 分离部署


class NodeRole(Enum):
    """节点角色 (存算分离模式下)"""
    MIXED = auto()       # 混合节点 (同时处理 prefill 和 decode)
    PREFILL = auto()     # Prefill 专用节点
    DECODE = auto()      # Decode 专用节点


class GlobalSchedulerType(Enum):
    """全局调度策略"""
    ROUND_ROBIN = auto()
    RANDOM = auto()
    LEAST_OUTSTANDING = auto()
    TOPOLOGY_AWARE = auto()  # 拓扑感知调度


class ReplicaSchedulerType(Enum):
    """副本级调度策略"""
    SARATHI = auto()     # 块级 prefill 分块
    VLLM = auto()        # PagedAttention
    ORCA = auto()        # 迭代级调度
    FCFS = auto()        # 先到先服务


class KVCacheTransferStrategy(Enum):
    """KV Cache 传输策略 (存算分离模式)"""
    DIRECT = auto()      # Prefill 完成后直接传输到 Decode 节点
    PIPELINED = auto()   # 流水线传输 (边算边传)
    STORE_FORWARD = auto()  # 先存到共享存储再取


class EventType(Enum):
    """事件类型"""
    REQUEST_ARRIVAL = auto()
    GLOBAL_SCHEDULE = auto()
    REPLICA_SCHEDULE = auto()
    BATCH_STAGE_ARRIVAL = auto()
    BATCH_STAGE_END = auto()
    BATCH_END = auto()
    # 存算分离事件
    PREFILL_COMPLETE = auto()
    KV_CACHE_TRANSFER_START = auto()
    KV_CACHE_TRANSFER_END = auto()
    DECODE_START = auto()
    # 专家并行事件
    EXPERT_ASSIGNMENT = auto()
    EXPERT_COMM_START = auto()
    EXPERT_COMM_END = auto()
