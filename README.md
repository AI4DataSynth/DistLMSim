# DistLMSim

分布式大模型推理服务离散事件模拟器。支持 A800 NVLink 节点内互联和 RDMA (RoCEv2/InfiniBand) 节点间互联，面向在线推理服务场景。

## 快速开始

```bash
# 运行存算分离演示 (1 Prefill 节点 + 1 Decode 节点，各 4 张 A800)
python3 examples/demo_disaggregated.py

# 或者使用 main.py
python3 main.py --demo
```

## 集群架构

本演示模拟 **存算分离 (Disaggregated Prefill/Decode)** 部署模式：

```
请求到达 ──► ┌─────────────────┐   RDMA 200Gb/s   ┌─────────────────┐
             │  Prefill 节点    │ ════════════════► │  Decode 节点     │
             │  Node 0          │   KV Cache 传输   │  Node 1          │
             │  ┌─────────────┐ │                   │  ┌─────────────┐ │
             │  │ GPU 0 (A800)│ │                   │  │ GPU 4 (A800)│ │
             │  │ GPU 1 (A800)│ │  NVLink 600GB/s   │  │ GPU 5 (A800)│ │
             │  │ GPU 2 (A800)│ │  (节点内 TP=4)     │  │ GPU 6 (A800)│ │
             │  │ GPU 3 (A800)│ │                   │  │ GPU 7 (A800)│ │
             │  └─────────────┘ │                   │  └─────────────┘ │
             └─────────────────┘                   └─────────────────┘
```

**推理流程**：

1. 请求到达 → Prefill 节点 (Node 0) 批量执行 prefill
2. Prefill 完成 → KV Cache 通过 **RDMA** 传输到 Decode 节点
3. Decode 节点 (Node 1) 逐 token 迭代生成
4. 所有 decode token 完成 → 请求结束

## 教程

### 1. 运行默认演示

```bash
python3 examples/demo_disaggregated.py
```

输出示例：

```
================================================================
  DistLMSim 模拟结果汇总
================================================================
  完成请求数:    596
  模拟时长:      61165.4 ms (61.17 s)

  --- Prefill ---
  总 prefill tokens: 305152
  Prefill 延迟 (ms): mean=782.32, P50=785.20

  --- KV Cache 传输 (RDMA) ---
  传输延迟 (ms):  mean=1.06, P50=1.06, P99=1.06

  --- TTFT (Time to First Token, ms) ---
    P50: 1949.04    P90: 2569.16    P99: 3096.55

  --- TBT (Time Between Tokens, ms) ---
    P50: 5.34

  --- 吞吐量 ---
    Decode tokens/s: 1247.2
    Prefill tokens/s: 4989.0

  --- 节点负载 ---
    Prefill 节点: {0: 596}
    Decode  节点: {1: 596}
================================================================
```

### 2. 调整参数

```bash
# 提高 QPS (请求到达率)
python3 examples/demo_disaggregated.py --qps 20

# 增大 prefill 长度
python3 examples/demo_disaggregated.py --prefill_length 2048

# 使用 InfiniBand (400 Gbps) 替代 RoCEv2
python3 examples/demo_disaggregated.py --rdma_bandwidth 400

# 调整批大小
python3 examples/demo_disaggregated.py --prefill_batch_size 16 --decode_batch_size 64

# 查看详细日志 (含每个 batch 的处理信息)
python3 examples/demo_disaggregated.py --verbose
```

所有可用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--qps` | 10.0 | 请求到达率 (requests/sec) |
| `--prefill_length` | 512 | Prefill token 数 |
| `--decode_length` | 128 | Decode token 数 |
| `--prefill_batch_size` | 8 | Prefill 批大小 |
| `--decode_batch_size` | 32 | Decode 批大小 |
| `--tp_size` | 4 | 张量并行度 (每节点 GPU 数) |
| `--rdma_bandwidth` | 200.0 | RDMA 带宽 (Gbps) |
| `--time_limit` | 60.0 | 模拟时长 (秒) |
| `--verbose` | false | 详细日志 |

### 3. 指标说明

| 指标 | 含义 |
|------|------|
| **TTFT** (Time to First Token) | 首 token 延迟 = prefill 完成时间 - 请求到达时间，含排队延迟 |
| **TBT** (Time Between Tokens) | Decode 阶段每 token 间隔 |
| **E2E Latency** | 端到端延迟 = decode 完成 - 请求到达 |
| **KV Cache 传输延迟** | KV Cache 通过 RDMA 从 Prefill 节点传输到 Decode 节点的耗时 |
| **Prefill tokens/s** | Prefill 阶段吞吐 |
| **Decode tokens/s** | Decode 阶段吞吐 |

### 4. 通信建模原理

**节点内 — NVLink/NVSwitch**：

A800 DGX 节点内 8 GPU 通过 NVSwitch 全互联，带宽 600 GB/s (双向)。张量并行 (TP) 的 All-Reduce 通信走 NVLink：

```
All-Reduce 时间 = 2 × (N-1)/N × 数据量 / NVSwitch带宽 + 延迟
```

**节点间 — RDMA**：

节点间通过 RDMA NIC 连接，支持 RoCEv2 (200 Gb/s) 和 InfiniBand。KV Cache 传输走 RDMA：

```
传输时间 = 数据量 / 有效带宽 + 基础延迟
有效带宽 = 链路带宽 × (1 - 协议开销) × 拥塞因子
```

协议开销：RoCEv2 ≈ 4.7%，InfiniBand ≈ 1.8%

## 项目结构

```
DistLMSim/
├── main.py                          # 入口 + DisaggregatedSimulator
├── distlmsim/
│   ├── config.py                    # 配置系统 (dataclass 层级)
│   ├── types.py                     # 枚举定义
│   ├── entities.py                  # 核心实体 (Request, Batch, ExecutionTime...)
│   ├── events.py                    # 事件流定义
│   ├── topology/                    # 网络拓扑建模
│   │   ├── network_topology.py      #   拓扑图
│   │   ├── nvlink_model.py          #   NVLink/NVSwitch 模型
│   │   ├── rdma_model.py            #   RDMA 模型 (RoCEv2/IB)
│   │   └── communication_cost.py    #   通信开销计算器
│   ├── cluster/                     # 集群管理
│   │   ├── node.py                  #   物理节点
│   │   ├── cluster.py               #   集群
│   │   └── resource_manager.py      #   GPU 资源分配
│   ├── scheduling/                  # 分布式调度
│   │   ├── global_scheduler.py      #   全局调度 (RR/Random/LOR/拓扑感知)
│   │   ├── replica_scheduler.py     #   副本级调度 (Sarathi/vLLM/Orca)
│   │   ├── disaggregated_scheduler.py #  存算分离调度
│   │   └── migration.py             #   请求迁移
│   ├── parallelism/                 # 并行策略
│   │   ├── tensor_parallel.py       #   TP (节点内 NVLink)
│   │   ├── pipeline_parallel.py     #   PP (可跨节点 RDMA)
│   │   ├── expert_parallel.py       #   EP (跨节点 All-to-All)
│   │   └── parallelism_planner.py   #   并行策略规划器
│   ├── execution/                   # 执行时间预测
│   │   ├── execution_time_predictor.py # 计算时间 (解析/RF)
│   │   └── network_time_predictor.py   # 网络时间
│   ├── request/                     # 请求生成
│   │   └── request_generator.py     #   合成/Trace回放
│   └── metrics/                     # 指标收集
│       └── metrics_store.py         #   TTFT/TBT/E2E/吞吐量
├── data/profiling/                  # Profiling 数据 (后续填充)
└── examples/
    ├── demo_disaggregated.py        # 存算分离演示
    └── demo_2node_tp4_pp2.py        # 并行策略规划演示
```

## 依赖

- Python 3.10+
- numpy

```bash
pip install numpy
```

## 基于

衍生自 [TRADIOS](../TRADIOS) 单机多卡 MoE 推理模拟器，复用其 A800 profiling 数据格式、执行时间预测方法和调度器设计。
