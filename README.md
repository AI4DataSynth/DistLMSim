# DistLMSim

Distributed Large Language Model inference service discrete event simulator. Supports A800 NVLink intra-node interconnect and RDMA (RoCEv2/InfiniBand) inter-node interconnect for online inference service scenarios.

## Quick Start

```bash
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
  DistLMSim Simulation Results Summary
================================================================
  Completed Requests: 596
  Simulation Duration: 61165.4 ms (61.17 s)

  --- Prefill ---
  Total Prefill Tokens: 305152
  Prefill Latency (ms): mean=782.32, P50=785.20

  --- KV Cache Transfer (RDMA) ---
  Transfer Latency (ms): mean=1.06, P50=1.06, P99=1.06

  --- TTFT (Time to First Token, ms) ---
    P50: 1949.04    P90: 2569.16    P99: 3096.55

  --- TBT (Time Between Tokens, ms) ---
    P50: 5.34

  --- Throughput ---
    Decode tokens/s: 1247.2
    Prefill tokens/s: 4989.0

  --- Node Load ---
    Prefill Node: {0: 596}
    Decode Node:  {1: 596}
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
python3 examples/experiment_schedulers.py --qps 20 --prefill_batch_size 2 --time_limit 30
```

### 6. MoE Expert Load Balancing

For Mixture-of-Experts (MoE) models, DistLMSim supports 4 expert load balancing strategies:

| Strategy | Description | Max Load Reduction |
|----------|-------------|-------------------|
| **DefaultRouting** | Standard Top-K routing, no balancing | Baseline |
| **EPLB** | Capacity factor (1.1) truncation | ~84% reduction |
| **RealisticEPLB** | Waterfill routing + redundant experts + periodic rebalance | Best load balance |
| **OmniPlacement** | Greedy swap optimization with budget control | Near-optimal placement |

Example comparison (Zipf α=1.5, 128 experts, EP=8):

```
Default     : max=868  avg=125  deviation=743.0
EPLB        : max=138  avg=125  (capacity-truncated; effective deviation=13.0)
Realistic   : max=314  avg=125  migrations=0  (single batch, no rebalance triggered)
OmniPlace   : max=858  avg=125  migrations=4  (budget_N=4, limited swaps)
```

> **Note:** EPLB reports raw (pre-truncation) deviation. RealisticEPLB and OmniPlacement
> effectiveness depends on configuration: rebalance frequency, redundant expert count,
> and swap budget. With adequate budget they outperform Default; the example above uses
> conservative defaults to show baseline behavior.

## Project Structure

```
DistLMSim/
├── main.py                          # Entry point + DisaggregatedSimulator
├── distlmsim/
│   ├── config.py                    # Configuration system (dataclass hierarchy)
│   ├── types.py                     # Enum definitions
│   ├── entities.py                  # Core entities (Request, Batch, ExecutionTime...)
│   ├── events.py                    # Event flow definitions
│   ├── topology/                    # Network topology modeling
│   │   ├── network_topology.py      #   Topology graph
│   │   ├── nvlink_model.py          #   NVLink/NVSwitch model
│   │   ├── rdma_model.py            #   RDMA model (RoCEv2/IB)
│   │   ├── communication_cost.py    #   Communication cost calculator
│   │   └── overlap_processor.py     #   Communication-computation overlap model
│   ├── cluster/                     # Cluster management
│   │   ├── node.py                  #   Physical nodes
│   │   ├── cluster.py               #   Cluster abstraction
│   │   └── resource_manager.py      #   GPU resource allocation
│   ├── scheduling/                  # Distributed scheduling
│   │   ├── global_scheduler.py      #   Global scheduling (RR/Random/LOR; topology-aware*)
│   │   ├── replica_scheduler.py     #   Replica-level scheduling (Sarathi*/vLLM*/Orca*)
│   │   ├── advanced_schedulers.py   #   MLFQ/PO/OPT/LightLLM schedulers
│   │   ├── disaggregated_scheduler.py # Disaggregated prefill/decode scheduling
│   │   └── migration.py             #   Request migration
│   ├── parallelism/                 # Parallelism strategies
│   │   ├── tensor_parallel.py       #   TP (intra-node NVLink)
│   │   ├── pipeline_parallel.py     #   PP (cross-node RDMA)
│   │   ├── expert_parallel.py       #   EP (cross-node All-to-All) + MoE load balancing
│   │   └── parallelism_planner.py   #   Parallelism strategy planner
│   ├── execution/                   # Execution time prediction
│   │   ├── execution_time_predictor.py # Compute time (Roofline analytical/profiling/RF)
│   │   └── network_time_predictor.py   # Network time prediction
│   ├── request/                     # Request generation
│   │   └── request_generator.py     #   Synthetic/Trace replay + MoE expert distributions
│   ├── metrics/                     # Metrics collection
│   │   └── metrics_store.py         #   TTFT/TBT/E2E/throughput metrics
│   ├── analysis/                    # Analysis modules (ported from Charon)
│   │   ├── memory_analysis.py       #   Per-GPU inference memory estimation + OOM detection
│   │   ├── mfu_analysis.py          #   Model FLOPs Utilization for prefill/decode
│   │   └── timeline_analysis.py     #   Chrome Trace JSON timeline generation
│   └── design/                      # Design space exploration (ported from Charon)
│       └── design_space_explorer.py #   TP/EP/scheduler enumeration + Pareto analysis
├── tests/                           # Unit tests (313 tests)
│   ├── run_tests.py                 #   Test runner
│   ├── test_config.py               #   Configuration tests
│   ├── test_types.py                #   Enum type tests
│   ├── test_entities.py             #   Entity tests
│   ├── test_topology.py             #   Topology tests
│   ├── test_cluster.py              #   Cluster tests
│   ├── test_scheduling.py           #   Scheduling tests
│   ├── test_parallelism.py          #   Parallelism tests
│   ├── test_execution.py            #   Execution predictor tests
│   ├── test_metrics.py              #   Metrics tests
│   ├── test_e2e.py                  #   End-to-end tests
│   ├── test_memory_analysis.py      #   Memory analysis tests
│   ├── test_mfu_analysis.py         #   MFU analysis tests
│   ├── test_timeline_analysis.py    #   Timeline analysis tests
│   ├── test_overlap_processor.py    #   Overlap processor tests
│   ├── test_design_space_explorer.py #  Design space exploration tests
│   └── test_roofline_model.py       #   Roofline model tests
├── data/profiling/                  # Profiling data (to be populated)
└── examples/
    ├── demo_disaggregated.py        # Disaggregated prefill/decode demo
    ├── experiment_schedulers.py     # Scheduler comparison experiment
    ├── demo_2node_tp4_pp2.py        # Parallelism planning demo
    └── visualize_results.py         # Result visualization script
```

> `*` Starred items indicate stub/partial implementations. See [Implementation Status](#implementation-status) below.

## Dependencies

- Python 3.10+
- numpy

```bash
pip install numpy
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
| Memory analysis | ✅ Complete | KV cache, OOM detection |
| MFU analysis | ✅ Complete | Prefill/decode FLOPs utilization |
| Chrome Trace timeline | ✅ Complete | Chrome Trace Viewer compatible |
| Design space exploration | ✅ Complete | Pareto frontier + SLO constraints |
| Replica schedulers (Sarathi/vLLM/Orca) | ⚠️ Stub | `form_batch()` raises `NotImplementedError`; scheduling is handled in `main.py` |
| TopologyAware global scheduler | ⚠️ Stub | Falls back to RoundRobin |
| Event-driven simulator (`DistributedInferenceSimulator`) | ⚠️ Partial | Queue-based `DisaggregatedSimulator` in `main.py` is the primary simulator |

## Based On

Derived from [TRADIOS](../TRADIOS), a single-node multi-GPU MoE inference simulator. Reuses its A800 profiling data format, execution time prediction methods, and scheduler designs.
