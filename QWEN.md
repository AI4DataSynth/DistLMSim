# QWEN.md — DistLMSim Project Guide

## Project Overview

DistLMSim is a **discrete event simulator** for distributed Large Language Model (LLM) inference services. It models a cluster of A800 GPU nodes with NVLink intra-node and RDMA inter-node interconnects, focusing on online inference workloads.

- **Language:** Python 3.10+
- **Primary dependency:** numpy (only external dependency)
- **Version:** 0.1.0
- **Comments and docstrings:** Written in Chinese (中文)
- **Derived from:** [TRADIOS](../TRADIOS), a single-node multi-GPU MoE inference simulator

## Architecture

The simulator models **disaggregated Prefill/Decode** deployment: 1 Prefill node + 1 Decode node, each with multiple A800 GPUs connected via NVLink/NVSwitch (intra-node) and RDMA (inter-node).

### Core Module Structure (`distlmsim/`)

| Module | Purpose |
|---|---|
| `config.py` | Nested dataclass config hierarchy (DeviceSKU → NVLink → RDMA → Cluster → Model → Disaggregated → Request → Metrics → SimulationConfig). Supports `argparse` CLI auto-generation via `SimulationConfig.from_cli()`. |
| `types.py` | All enum definitions: DeviceSKUType, RDMAProtocolType, EventType, ParallelismType, DeploymentMode, scheduler types, etc. |
| `entities.py` | Core data structures: `Request` (with lifecycle status), `Batch`, `BatchStage`, `ExecutionTime` (per-layer time breakdown), `Node`, `Replica`. |
| `events.py` | Discrete event system: `BaseEvent` ABC with `handle_event()` returning `List[BaseEvent]` — processed via `heapq` priority queue. Includes `RequestArrivalEvent`, `BatchStageArrivalEvent`, `BatchStageEndEvent`, etc. |
| `topology/` | Network topology modeling: `NVLinkModel` (NVSwitch all-reduce), `RDMAModel` (RoCEv2/InfiniBand transfer), `communication_cost.py`, `network_topology.py`, `overlap_processor.py` (communication-computation overlap with ratio-based and bandwidth-aware slowdown models, ported from Charon). |
| `cluster/` | Cluster management: `node.py` (physical nodes), `cluster.py`, `resource_manager.py` (GPU allocation). |
| `scheduling/` | Scheduling: `global_scheduler.py` (RR/Random/LOR/topology-aware), `replica_scheduler.py` (Sarathi/vLLM/Orca), `advanced_schedulers.py` (MLFQ/PO/OPT/LightLLM), `disaggregated_scheduler.py`, `migration.py`. |
| `parallelism/` | Parallelism strategies: `tensor_parallel.py` (TP via NVLink), `pipeline_parallel.py` (PP via RDMA), `expert_parallel.py` (EP + MoE load balancing), `parallelism_planner.py`. |
| `execution/` | Execution time prediction: `execution_time_predictor.py` (Analytical/Profiling/RandomForest predictors), `network_time_predictor.py`. |
| `request/` | Request generation: `request_generator.py` (synthetic Poisson/Gamma arrival + Fixed/Normal/Lognormal/Zipf length distributions, trace replay). |
| `metrics/` | Metrics collection: `metrics_store.py` (TTFT, TBT, E2E latency, throughput, KV cache transfer latency, percentiles P50/P90/P95/P99). |
| `analysis/` | Analysis modules (ported from Charon): `memory_analysis.py` (per-GPU inference memory estimation with KV cache, OOM detection), `mfu_analysis.py` (Model FLOPs Utilization for prefill/decode), `timeline_analysis.py` (Chrome Trace JSON generation from MetricsStore). |
| `design/` | Design space exploration (ported from Charon): `design_space_explorer.py` (Cartesian product enumeration of TP/EP/scheduler combos, rule-based pruning, Pareto frontier analysis with SLO constraints). |

### Entry Points

- **`main.py`** — Contains `DisaggregatedSimulator` (queue-based scheduling sim) and `DistributedInferenceSimulator` (event-driven, not fully implemented). Also has `create_disaggregated_simulator()` factory function for quick setup.
- **`examples/demo_disaggregated.py`** — Interactive demo with CLI args.
- **`examples/experiment_schedulers.py`** — Scheduler comparison experiment.
- **`examples/demo_2node_tp4_pp2.py`** — Parallelism planning demo.

### Key Design Patterns

1. **Configuration:** Deeply nested `@dataclass` hierarchy. Top-level `SimulationConfig` contains `ClusterConfig`, `DisaggregatedConfig`, `SchedulingConfig`, `RequestGeneratorConfig`, `MetricsConfig`.
2. **Event system:** Abstract `BaseEvent` with `handle_event()` returning `List[BaseEvent]` — processed via `heapq` priority queue.
3. **SimContext:** `SimContext` dataclass holds shared runtime state (models, network configs, predictors, metrics store, request pool).
4. **Predictors:** Abstract `ExecutionTimePredictor` base class with `AnalyticalPredictor` (Roofline model: `time = max(FLOPS/peak_FLOPS, bytes/mem_BW)` with compute/memory efficiency factors), profiling-based (linear regression from CSV), and RandomForest implementations.
5. **Schedulers:** 9 scheduling policies (FCFS, SJF, LJF, SRTF, Random, MLFQ, PO, OPT, LightLLM) selectable per prefill/decode phase.
6. **MoE Load Balancing:** 4 strategies (DefaultRouting, EPLB, RealisticEPLB with waterfill, OmniPlacement with greedy swap).

## Building and Running

```bash
# Install dependency
pip install numpy

# Quick demo (disaggregated prefill/decode, 1P+1D node, 4xA800 each)
python3 main.py --demo
python3 examples/demo_disaggregated.py

# With custom parameters
python3 examples/demo_disaggregated.py --qps 20 --prefill_length 2048 --rdma_bandwidth 400

# Scheduler comparison experiment
python3 examples/experiment_schedulers.py --qps 20 --prefill_batch_size 2 --time_limit 30

# Parallelism planning demo
python3 examples/demo_2node_tp4_pp2.py
```

### Running Tests

Tests are in `tests/` (git-ignored, run locally). 313 unit tests covering all modules.

```bash
# All tests
python3 tests/run_tests.py

# Specific test file
python3 -m unittest tests/test_scheduling.py

# Verbose
python3 -m unittest tests/test_e2e.py -v
```

### CLI Parameters (via `SimulationConfig.from_cli()`)

Key parameters: `--seed`, `--time_limit_s`, `--num_nodes`, `--num_gpus_per_node`, `--rdma_protocol`, `--rdma_bandwidth_gbps`, `--nvlink_bandwidth_gbps`, `--tensor_parallel_size`, `--num_pipeline_stages`, `--expert_parallel_size`, `--model_name`, `--num_layers`, `--global_scheduler`, `--replica_scheduler`, `--disaggregated`, `--qps`, `--prefill_length`, `--decode_length`.

## Development Conventions

- **Language:** Code and docstrings are in Chinese (中文). Follow this convention for comments and documentation.
- **Typing:** Extensive use of `dataclasses`, `Enum`, type hints (`from __future__ import annotations`). Follow the existing pattern.
- **Imports:** Absolute imports from `distlmsim.*` (e.g., `from distlmsim.config import ModelConfig`).
- **Naming:** snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE_CASE for enum values and constants.
- **Configuration:** All configurable parameters go in `config.py` as dataclass fields with sensible defaults.
- **No external frameworks:** Only numpy is allowed. No pytest (uses unittest), no third-party libs beyond numpy.
- **Tests:** Use `unittest` framework. Test files are in `tests/` (git-ignored). Tests follow `test_<module>.py` naming.
- **Logging:** Use Python `logging` module. Debug-level for per-batch details, Info for summary.
- **Git:** `.gitignore` excludes `tests/`, `results/`, `data/profiling/**/*.csv`, `.qwen/`, `__pycache__/`, `*.pyc`.

## Communication Modeling Details

- **Intra-node (NVLink/NVSwitch):** All-reduce formula: `2 × (N-1)/N × data_size / NVSwitch_bandwidth + latency`
- **Inter-node (RDMA):** Transfer formula: `data_size / effective_bandwidth + base_latency`, where `effective_bandwidth = link_bandwidth × (1 - protocol_overhead) × congestion_factor`. Protocol overhead: RoCEv2 ≈ 4.7%, InfiniBand ≈ 1.8%.

## Data

- `data/profiling/` — Profiling data directory (CSV files git-ignored due to size). Used by profiling-based execution time predictors.
