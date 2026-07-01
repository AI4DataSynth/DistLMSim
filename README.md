# DistLMSim

Distributed Large Language Model inference service discrete event simulator. Supports A800 NVLink intra-node interconnect and RDMA (RoCEv2/InfiniBand) inter-node interconnect for PD-disaggregated online inference service scenarios.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run disaggregated prefill/decode demo (1 Prefill node + 1 Decode node, 4×A800 each)
python3 examples/demo_disaggregated.py

# Or use main.py
python3 main.py --demo
```

## Cluster Architecture

This demo simulates **disaggregated Prefill/Decode** deployment mode:

```
Request Arrival ──► ┌─────────────────┐   RDMA 200Gb/s   ┌─────────────────┐
                    │  Prefill Node   │ ════════════════► │  Decode Node    │
                    │  Node 0         │   KV Cache Trans. │  Node 1         │
                    │  ┌─────────────┐ │                   │  ┌─────────────┐ │
                    │  │ GPU 0 (A800)│ │                   │  │ GPU 4 (A800)│ │
                    │  │ GPU 1 (A800)│ │  NVLink 600GB/s   │  │ GPU 5 (A800)│ │
                    │  │ GPU 2 (A800)│ │  (Intra-node TP=4)│  │ GPU 6 (A800)│ │
                    │  │ GPU 3 (A800)│ │                   │  │ GPU 7 (A800)│ │
                    │  └─────────────┘ │                   │  └─────────────┘ │
                    └─────────────────┘                   └─────────────────┘
```

**Inference Flow**:

1. Request arrives → Prefill node (Node 0) executes prefill in batches
2. Prefill complete → KV Cache transferred to Decode node via **RDMA**
3. Decode node (Node 1) generates tokens iteratively
4. All decode tokens complete → Request ends

## Tutorial

### 1. Run Default Demo

```bash
python3 examples/demo_disaggregated.py
```

Sample output:

```
================================================================
  DistLMSim — 存算分离推理模拟
================================================================

集群配置:
  Prefill 节点: Node 0, 4x A800 (TP=4)
  Decode  节点: Node 1, 4x A800 (TP=4)
  节点内互联:   NVLink/NVSwitch 600 GB/s
  节点间互联:   RDMA (RoCEv2) 200.0 Gbps

推理配置:
  模型:          Qwen3-30B-A3B (48层, MoE 128专家 Top-8)
  QPS:           10.0
  Prefill 长度:  512 tokens
  Decode 长度:   128 tokens
  Prefill BS:    8
  Decode BS:     32
  模拟时长:      60.0s

================================================================
  DistLMSim 模拟结果汇总
================================================================
  完成请求数:    596
  模拟时长:      73628.8 ms (73.63 s)

  --- Prefill ---
  总 prefill tokens: 305152
  Prefill 延迟 (ms): mean=949.23, P50=985.28

  --- TTFT (ms) ---
    P50: 9129.40    P90: 13534.11    P99: 14521.62

  --- TBT (ms) ---
    P50: 2.51

  --- 吞吐量 ---
    Decode tokens/s: 1036.1
    Prefill tokens/s: 4144.5
================================================================
```

### 2. Adjust Parameters

```bash
# Increase QPS (request arrival rate)
python3 examples/demo_disaggregated.py --qps 20

# Increase prefill length
python3 examples/demo_disaggregated.py --prefill_length 2048

# Use InfiniBand (400 Gbps) instead of RoCEv2
python3 examples/demo_disaggregated.py --rdma_bandwidth 400

# Adjust batch sizes
python3 examples/demo_disaggregated.py --prefill_batch_size 16 --decode_batch_size 64

# Enable verbose logging (includes per-batch processing info)
python3 examples/demo_disaggregated.py --verbose
```

All available parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--qps` | 10.0 | Request arrival rate (requests/sec) |
| `--prefill_length` | 512 | Number of prefill tokens |
| `--decode_length` | 128 | Number of decode tokens |
| `--prefill_batch_size` | 8 | Prefill batch size |
| `--decode_batch_size` | 32 | Decode batch size |
| `--tp_size` | 4 | Tensor parallelism degree (GPUs per node) |
| `--rdma_bandwidth` | 200.0 | RDMA bandwidth (Gbps) |
| `--time_limit` | 60.0 | Simulation duration (seconds) |
| `--verbose` | false | Enable verbose logging |

### 3. Metrics Explained

| Metric | Meaning |
|--------|---------|
| **TTFT** (Time to First Token) | First token latency = prefill completion time - request arrival time, includes queueing delay |
| **TBT** (Time Between Tokens) | Per-token interval during decode phase |
| **E2E Latency** | End-to-end latency = decode completion - request arrival |
| **KV Cache Transfer Latency** | Time to transfer KV Cache from Prefill node to Decode node via RDMA |
| **Prefill tokens/s** | Throughput during prefill phase |
| **Decode tokens/s** | Throughput during decode phase |

### 4. Communication Modeling

**Intra-node — NVLink/NVSwitch**:

Within an A800 DGX node, 8 GPUs are fully interconnected via NVSwitch with 600 GB/s bidirectional bandwidth. Tensor Parallel (TP) All-Reduce communication uses NVLink:

```
All-Reduce Time = 2 × (N-1)/N × data_size / NVSwitch_bandwidth + latency
```

**Inter-node — RDMA**:

Nodes are connected via RDMA NICs, supporting RoCEv2 (200 Gb/s) and InfiniBand. KV Cache transfer uses RDMA:

```
Transfer Time = data_size / effective_bandwidth + base_latency
effective_bandwidth = link_bandwidth × (1 - protocol_overhead) × congestion_factor
```

Protocol overhead: RoCEv2 ≈ 4.7%, InfiniBand ≈ 1.8%

### 5. Scheduling Policies

DistLMSim supports 9 request scheduling policies:

| Policy | Description | Use Case |
|--------|-------------|----------|
| **FCFS** | First-Come-First-Served | Baseline, fair scheduling |
| **SJF** | Shortest Job First (by prefill tokens) | Optimize average TTFT |
| **LJF** | Longest Job First (by prefill tokens) | Worst-case reference |
| **SRTF** | Shortest Remaining Time First (by decode tokens) | Optimize E2E latency |
| **Random** | Random selection | No-policy baseline |
| **MLFQ** | Multi-Level Feedback Queue | Adaptive priority scheduling |
| **PO** | Priority Ordering (short=FCFS, long=SJF) | Balanced approach |
| **OPT** | Score-based with noise factor | Near-optimal scheduling |
| **LightLLM** | Separate prefill/decode batches | LightLLM-style scheduling |

Run scheduler comparison experiment:

```bash
python3 examples/experiment_schedulers.py --qps 20 --time_limit 30
```

### 6. DSpark / DFlash Speculative Decoding (DeepSeek)

DistLMSim supports DeepSeek's DSpark and DFlash speculative decoding schemes
([DeepSpec](https://github.com/deepseek-ai/DeepSpec)):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `speculative_mode` | `"standard"` | `"standard"` / `"dspark"` / `"dflash"` |
| `block_size` | 7 | Tokens drafted per speculation round |
| `markov_rank` | 256 | Low-rank dimension for Markov head |
| `markov_head_type` | `"vanilla"` | `"vanilla"` / `"gated"` / `"rnn"` |
| `num_target_layer_ids` | 5 | Target model layers tapped for features |
| `confidence_threshold` | 0.0 | Early stopping threshold (0=disabled) |
| `enable_confidence_scheduling` | `false` | Load-aware confidence scheduling |
| `draft_num_layers` | 5 | Draft model transformer layers |
| `draft_embedding_dim` | 512 | Draft model hidden dimension |
| `acceptance_rate` | 0.8 | Average token acceptance rate (0-1) |

**DSpark** uses semi-autoregressive drafting with Markov heads:
- Block-based: generates `block_size` tokens per round
- Markov head models token-to-token dependency via low-rank embedding (vocab→rank→vocab)
- Taps into target model's intermediate layers for feature extraction
- Confidence scheduling enables early stopping for low-confidence blocks

**DFlash** extends DSpark with parallel block drafting and draft-verify pipelining.

Run speculative decoding experiment:

```bash
python3 examples/experiment_speculative_decoding.py
```

### 7. MoE Expert Load Balancing

For Mixture-of-Experts (MoE) models, DistLMSim supports 4 expert load balancing strategies:

| Strategy | Description | Max Load Reduction |
|----------|-------------|-------------------|
| **DefaultRouting** | Standard Top-K routing, no balancing | Baseline |
| **EPLB** | Capacity factor (1.1) truncation | ~84% reduction |
| **RealisticEPLB** | Waterfill routing + redundant experts + periodic rebalance | Best load balance |
| **OmniPlacement** | Greedy swap optimization with budget control | Near-optimal placement |

Run MoE load balancing experiment:

```bash
python3 examples/experiment_moe_load.py
```

## Project Structure

```
DistLMSim/
├── main.py                          # Entry point + DisaggregatedSimulator
├── requirements.txt                 # Python dependencies
├── distlmsim/
│   ├── types.py                     # Enum definitions (Layer 0)
│   ├── interfaces.py                # Protocol interfaces for DAG decoupling (Layer 1)
│   ├── config.py                    # Configuration system (dataclass hierarchy)
│   ├── entities.py                  # Core entities (Request, Batch, ExecutionTime...)
│   ├── context.py                   # SimContext: shared runtime state (Layer 5)
│   ├── events.py                    # Event flow definitions
│   ├── topology/                    # Network topology modeling
│   │   ├── network_topology.py      #   Topology graph (path-based BW/latency)
│   │   ├── nvlink_model.py          #   NVLink/NVSwitch model (+ profiling lookup)
│   │   ├── rdma_model.py            #   RDMA model (RoCEv2/IB + profiling + congestion)
│   │   ├── communication_cost.py    #   Communication cost calculator
│   │   └── overlap_processor.py     #   Communication-computation overlap model
│   ├── cluster/                     # Cluster management
│   │   ├── node.py                  #   Physical nodes and GPU devices
│   │   ├── cluster.py               #   Cluster abstraction
│   │   └── resource_manager.py      #   GPU resource allocation
│   ├── scheduling/                  # Distributed scheduling
│   │   ├── global_scheduler.py      #   Global scheduling (RR/Random/LOR/TopologyAware)
│   │   ├── replica_scheduler.py     #   Replica-level scheduling (Sarathi/vLLM/Orca)
│   │   ├── advanced_schedulers.py   #   MLFQ/PO/OPT/LightLLM schedulers
│   │   ├── disaggregated_scheduler.py # Disaggregated prefill/decode scheduling
│   │   └── migration.py             #   Request migration
│   ├── parallelism/                 # Parallelism strategies
│   │   ├── tensor_parallel.py       #   TP (intra-node NVLink)
│   │   ├── pipeline_parallel.py     #   PP (stage partitioning + bubble ratio)
│   │   ├── expert_parallel.py       #   EP (all-to-all) + MoE load balancing
│   │   └── parallelism_planner.py   #   Parallelism strategy planner
│   ├── execution/                   # Execution time prediction
│   │   ├── execution_time_predictor.py # Analytical (Roofline) / Profiling / RandomForest
│   │   ├── network_time_predictor.py   # Network time prediction
│   │   └── speculative_decoder.py   #   Speculative decoding modeling
│   ├── request/                     # Request generation
│   │   └── request_generator.py     #   Synthetic + Trace replay + MoE distributions
│   ├── metrics/                     # Metrics collection
│   │   └── metrics_store.py         #   TTFT/TBT/E2E/throughput metrics
│   ├── analysis/                    # Analysis modules
│   │   ├── memory_analysis.py       #   Per-GPU inference memory + OOM detection
│   │   ├── mfu_analysis.py          #   Model FLOPs Utilization
│   │   └── timeline_analysis.py     #   Chrome Trace JSON timeline
│   └── design/                      # Design space exploration
│       └── design_space_explorer.py #   TP/EP/scheduler enumeration + Pareto
├── tests/                           # Unit tests (313 tests)
│   ├── run_tests.py                 #   Test runner
│   ├── test_e2e.py                  #   End-to-end integration tests
│   └── test_*.py                    #   Per-module unit tests
├── data/profiling/                  # Profiling data (operator latencies)
├── scripts/                         # GPU profiling & benchmark scripts
└── examples/
    ├── demo_disaggregated.py        # Disaggregated prefill/decode demo
    ├── demo_2node_tp4_pp2.py        # Parallelism planning demo
    ├── experiment_accuracy.py       # Roofline accuracy experiment
    ├── experiment_hybrid_accuracy.py # Hybrid backend comparison
    ├── experiment_schedulers.py     # 9 scheduling strategies comparison
    ├── experiment_moe_load.py       # MoE expert load balancing
    ├── experiment_speculative_decoding.py # Speculative decoding tuning
    ├── experiment_pd_vs_colocated.py # PD disaggregated vs colocated
    ├── experiment_chunked_prefill.py # Chunked prefill analysis
    ├── experiment_kv_transfer.py    # KV cache transfer strategies
    └── visualize_results.py         # Result visualization
```

## Dependencies

- Python 3.10+
- numpy, scipy, scikit-learn

```bash
pip install -r requirements.txt
```

## Running Tests

```bash
# Run all 313 unit tests
python3 tests/run_tests.py

# Run specific test file
python3 -m unittest tests/test_scheduling.py

# Run with verbose output
python3 -m unittest tests/test_e2e.py -v
```

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| 9 scheduling policies (FCFS–LightLLM) | ✅ Complete | All implemented and tested |
| 4 MoE load balancing strategies | ✅ Complete | All implemented and tested |
| NVLink/RDMA communication models | ✅ Complete | Analytical + profiling modes |
| Communication-computation overlap | ✅ Complete | Ratio-based + bandwidth-aware |
| Roofline execution time predictor | ✅ Complete | Compute/memory-bound modeling |
| Hybrid backend (Profiled + RF) | ✅ Complete | Linear regression + RandomForest |
| Speculative decoding modeling | ✅ Complete | Standard + DSpark/DFlash (DeepSeek) |
| Sarathi replica scheduler | ✅ Complete | Chunked prefill + decode mixing |
| vLLM replica scheduler | ✅ Complete | PagedAttention block mgmt + preemption |
| Orca replica scheduler | ✅ Complete | Iteration-level, full prefill batching |
| Pipeline parallel stage mapping | ✅ Complete | Stage-to-node assignment |
| TopologyAware global scheduler | ✅ Complete | Hash-affinity routing |
| Event-driven simulator | ✅ Complete | heapq-based discrete event loop |
| Memory analysis | ✅ Complete | KV cache, OOM detection |
| MFU analysis | ✅ Complete | Prefill/decode FLOPs utilization |
| Chrome Trace timeline | ✅ Complete | Chrome Trace Viewer compatible |
| Design space exploration | ✅ Complete | Pareto frontier + SLO constraints |

## Based On

Derived from [TRADIOS](../TRADIOS), a single-node multi-GPU MoE inference simulator. Reuses its A800 profiling data format, execution time prediction methods, and scheduler designs.
