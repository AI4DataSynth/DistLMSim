# DistLMSim Module Specification

> **Version:** 0.1.0  
> **Module Count:** 35 modules (29 package modules + 6 `__init__.py` re-export files)  
> **Dependency Graph:** Strict DAG (Directed Acyclic Graph), zero circular dependencies  
> **Verification:** 313/313 tests passed, E2E output identical to original codebase

---

## Dependency DAG Overview

Modules are organized into 8 layers. Each layer depends **only** on lower layers.

```
Layer 0  types
Layer 1  interfaces, entities, config
Layer 2  events
Layer 3  topology/*, request/*
Layer 4  execution/*, parallelism/*
Layer 5  context, cluster/*, metrics/*
Layer 6  scheduling/*, analysis/*, design/*
Layer 7  main.py (simulator entry point)
```

```
                          ┌─────────┐
                          │ main.py │  Layer 7
                          └────┬────┘
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
     ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐
     │ scheduling/* │  │  analysis/*  │  │   design/*      │  Layer 6
     └──────┬───────┘  └──────┬───────┘  └────────┬────────┘
            │                 │                    │
     ┌──────┴──────┐  ┌──────┴───────┐   ┌───────┴────────┐
     │  cluster/*  │  │  metrics/*   │   │                │  Layer 5
     └──────┬──────┘  └──────┬───────┘   │                │
            │                │            │                │
     ┌──────┴────────────────┴────────────┴──────┐
     │               context.py                  │  Layer 5
     └──────┬────────────────┬───────────────────┘
            │                │
     ┌──────┴──────┐  ┌─────┴──────────┐
     │ execution/* │  │ parallelism/*  │  Layer 4
     └──────┬──────┘  └─────┬──────────┘
            │               │
     ┌──────┴───────┐  ┌────┴─────────┐
     │  topology/*  │  │  request/*   │  Layer 3
     └──────┬───────┘  └────┬─────────┘
            │               │
     ┌──────┴───────────────┴──────────┐
     │            events.py            │  Layer 2
     └──────────────┬──────────────────┘
                    │
     ┌──────────────┼──────────────────┐
     ▼              ▼                  ▼
  types.py    interfaces.py       config.py     Layer 0-1
                  │
              entities.py
```

---

## Module Specifications

### Layer 0: Foundation Types

#### `distlmsim/types.py`
- **Layer:** 0
- **Dependencies:** None (stdlib only)
- **Input:** —
- **Output:** Enum types consumed by all other modules
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `DeviceSKUType` | Enum | GPU device models (A800, A100, H100, H200) |
  | `NodeSKUType` | Enum | Node models (A800_DGX, H100_DGX) |
  | `InterconnectType` | Enum | Intra-node interconnect (NVLINK_SWITCH, NVLINK_MESH) |
  | `RDMAProtocolType` | Enum | Inter-node protocol (ROCE_V2, INFINIBAND, TCP_IP) |
  | `NetworkModelMode` | Enum | Network modeling mode (ANALYTICAL, PROFILING, HYBRID) |
  | `ParallelismType` | Enum | Parallelism type (TENSOR, PIPELINE, DATA, EXPERT) |
  | `DeploymentMode` | Enum | Deployment mode (COLOCATED, DISAGGREGATED) |
  | `NodeRole` | Enum | Node role (MIXED, PREFILL, DECODE) |
  | `GlobalSchedulerType` | Enum | Global scheduling strategy |
  | `ReplicaSchedulerType` | Enum | Replica-level scheduling strategy |
  | `KVCacheTransferStrategy` | Enum | KV cache transfer strategy (DIRECT, PIPELINED, STORE_FORWARD) |
  | `EventType` | Enum | Discrete event types for simulation |

---

### Layer 1: Interfaces, Entities, Configuration

#### `distlmsim/interfaces.py`
- **Layer:** 1
- **Dependencies:** `types` (for type hints only)
- **Input:** —
- **Output:** Protocol interfaces for cross-module abstraction
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `ReplicaSelector` | Protocol | `select_replica(request_id: int) -> int` — used by events to route requests |
  | `MetricsRecorder` | Protocol | `record_request_arrival()`, `record_request_scheduled()`, `record_kv_cache_transfer_start/end()`, `record_decode_start()` — used by events to record metrics |
  | `ClusterView` | Protocol | `replicas -> Dict[int, object]`, `nodes -> Dict[int, object]` — used by scheduling to access cluster state |

#### `distlmsim/entities.py`
- **Layer:** 1
- **Dependencies:** `types` (via `RequestStatus` enum)
- **Input:** —
- **Output:** Core data structures for the simulation domain
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `RequestStatus` | Enum | Request lifecycle states (WAITING, PREFILLING, DECODING, etc.) |
  | `ExecutionTime` | dataclass | Per-layer execution time breakdown (attention, MLP, comm, etc.) |
  | `Request` | dataclass | Inference request (id, arrival_time, prefill/decode_tokens, status tracking) |
  | `Batch` | dataclass | Batch of requests for execution |
  | `BatchStage` | dataclass | Pipeline stage within a batch |
  | `Replica` | dataclass | Model replica with associated nodes |
  | `Node` | dataclass | Physical compute node |

#### `distlmsim/config.py`
- **Layer:** 1
- **Dependencies:** `types`
- **Input:** —
- **Output:** Nested dataclass configuration system with CLI auto-generation
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `DeviceSKUConfig` | dataclass | GPU device specs (peak_flops, memory_bw, memory_gb) |
  | `NodeSKUConfig` | dataclass | Node specs (num_gpus, interconnect_type) |
  | `NVLinkConfig` | dataclass | NVLink parameters (bandwidth_gbps) |
  | `RDMAConfig` | dataclass | RDMA parameters (protocol, bandwidth, congestion_alpha) |
  | `NetworkTopologyConfig` | dataclass | Network topology (nvlink + rdma configs) |
  | `ModelConfig` | dataclass | LLM model parameters (layers, heads, embedding_dim, experts) |
  | `ReplicaConfig` | dataclass | Replica configuration |
  | `ClusterConfig` | dataclass | Cluster topology configuration |
  | `DisaggregatedConfig` | dataclass | PD-disaggregated serving parameters |
  | `SchedulingConfig` | dataclass | Scheduling strategy configuration |
  | `RequestGeneratorConfig` | dataclass | Workload generation parameters |
  | `MetricsConfig` | dataclass | Metrics collection configuration |
  | `SimulationConfig` | dataclass | Top-level config aggregating all sub-configs; `from_cli()` factory |

---

### Layer 2: Event System

#### `distlmsim/events.py`
- **Layer:** 2
- **Dependencies:** `types` (EventType), `interfaces` (ReplicaSelector, MetricsRecorder)
- **Input:** Protocol-typed scheduler and metrics references
- **Output:** Event classes for discrete-event simulation
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `BaseEvent` | ABC | Base event with `handle_event(scheduler, metrics_store) -> List[BaseEvent]` |
  | `RequestArrivalEvent` | class | Request arrives → triggers global scheduling |
  | `GlobalScheduleEvent` | class | Global scheduler assigns request to replica |
  | `ReplicaScheduleEvent` | class | Replica-level scheduling |
  | `BatchStageArrivalEvent` | class | Batch arrives at pipeline stage (with PP comm time, cross-node detection) |
  | `BatchStageEndEvent` | class | Pipeline stage completes → next stage or batch end |
  | `BatchEndEvent` | class | All pipeline stages complete |
  | `PrefillCompleteEvent` | class | Prefill done → triggers KV cache transfer |
  | `KVCacheTransferStartEvent` | class | KV cache RDMA transfer begins |
  | `KVCacheTransferEndEvent` | class | KV cache transfer done → decode starts |
  | `DecodeStartEvent` | class | Decode phase begins |
  | `ExpertAssignmentEvent` | class | MoE expert routing |
  | `ExpertCommStartEvent` | class | MoE all-to-all communication begins |
  | `ExpertCommEndEvent` | class | MoE communication completes |

---

### Layer 3: Network Topology, Communication, Request Generation

#### `distlmsim/topology/network_topology.py`
- **Layer:** 3
- **Dependencies:** `config` (NetworkTopologyConfig), `types` (RDMAProtocolType)
- **Input:** Network topology configuration
- **Output:** Path-based bandwidth/latency queries
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `NetworkLink` | dataclass | Link with bandwidth, latency, oversubscription |
  | `SwitchNode` | dataclass | Network switch node |
  | `NetworkTopology` | class | `from_config()`, `get_path()`, `get_effective_bandwidth()`, `get_latency()` |

#### `distlmsim/topology/nvlink_model.py`
- **Layer:** 3
- **Dependencies:** `config` (NVLinkConfig), `types` (InterconnectType)
- **Input:** NVLink configuration, optional profiling CSV
- **Output:** Intra-node communication time estimates
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `NVLinkModel` | class | `get_allreduce_time(tp, data_size)`, `get_send_recv_time()`, `get_alltoall_time()` |

#### `distlmsim/topology/rdma_model.py`
- **Layer:** 3
- **Dependencies:** `config` (RDMAConfig), `types` (RDMAProtocolType)
- **Input:** RDMA configuration, optional profiling data
- **Output:** Inter-node communication time estimates
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `RDMAModel` | class | `get_transfer_time(bytes, concurrent)`, `get_allreduce_time()`, `get_alltoall_time()`, `get_send_recv_time()` |

#### `distlmsim/topology/communication_cost.py`
- **Layer:** 3
- **Dependencies:** `config`, `nvlink_model`, `rdma_model`, `network_topology`, `types`
- **Input:** Communication parameters (TP size, data size, node pair)
- **Output:** Unified communication cost across parallelism strategies
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `CommunicationBreakdown` | dataclass | Breakdown of communication time by phase |
  | `CommunicationCostCalculator` | class | `tensor_parallel_allreduce()`, `pipeline_parallel_send_recv()`, `expert_parallel_alltoall()`, `kv_cache_transfer()` |

#### `distlmsim/topology/overlap_processor.py`
- **Layer:** 3
- **Dependencies:** `entities` (ExecutionTime)
- **Input:** Compute and communication time pairs
- **Output:** Adjusted wall-clock time with overlap modeling
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `OverlapConfig` | dataclass | Overlap parameters (compute_slowdown, comm_slowdown) |
  | `OverlapPair` | dataclass | Compute-communication pair |
  | `OverlapResult` | dataclass | Adjusted times after overlap |
  | `OverlapProcessor` | class | `make_compute_comm_pair()`, `apply_ratio_slowdown()`, `apply_bandwidth_aware_slowdown()`, `process_pairs()` |

#### `distlmsim/request/request_generator.py`
- **Layer:** 3
- **Dependencies:** `config` (RequestGeneratorConfig), `entities` (Request), `events` (BaseEvent, RequestArrivalEvent)
- **Input:** Workload generation configuration
- **Output:** Synthetic or trace-replayed request streams
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `PoissonIntervalGenerator` | class | Exponential inter-arrival times |
  | `GammaIntervalGenerator` | class | Gamma-distributed inter-arrival times |
  | `FixedLengthGenerator` | class | Fixed request lengths |
  | `NormalLengthGenerator` | class | Normal-distributed lengths |
  | `LognormalLengthGenerator` | class | Log-normal distributed lengths |
  | `ZipfLengthGenerator` | class | Zipf-distributed (power-law) lengths |
  | `SyntheticRequestGenerator` | class | Full synthetic workload generator |
  | `TraceReplayRequestGenerator` | class | CSV trace replay generator |

---

### Layer 4: Execution & Parallelism

#### `distlmsim/execution/execution_time_predictor.py`
- **Layer:** 4
- **Dependencies:** `config` (ModelConfig, DeviceSKUConfig), `entities` (ExecutionTime)
- **Input:** Model config, device config, profiling data path
- **Output:** Per-layer execution time predictions
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `ExecutionTimePredictor` | ABC | `get_execution_time(num_tokens, batch_size, kv_cache_size, is_prefill) -> ExecutionTime` |
  | `AnalyticalPredictor` | class | Roofline model with efficiency factors (η_c=0.85, η_m=0.90) |
  | `ProfilingBasedPredictor` | class | Linear regression on profiled operator latencies |
  | `RandomForestPredictor` | class | Random forest regression for unseen shapes |
  | `create_predictor()` | factory | Creates predictor by type ("auto"/"analytical"/"profiled"/"random_forest") |

#### `distlmsim/execution/network_time_predictor.py`
- **Layer:** 4
- **Dependencies:** `config`, `topology/nvlink_model`, `topology/rdma_model`, `types`
- **Input:** Network topology config, NVLink/RDMA models
- **Output:** Unified network time predictions for all parallelism strategies
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `NetworkTimePredictor` | class | `get_tp_allreduce_time()`, `get_pp_send_recv_time()`, `get_ep_alltoall_time()`, `get_kv_cache_transfer_time()` |

#### `distlmsim/execution/speculative_decoder.py`
- **Layer:** 4
- **Dependencies:** `config` (ModelConfig, DisaggregatedConfig), `entities` (Request), `execution_time_predictor` (AnalyticalPredictor), `context` (SimContext)
- **Input:** SimContext, speculative decoding config, RNG
- **Output:** Speculative decoding cycle time and acceptance counts
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `SpeculativeDecoder` | class | `compute_draft_step_time()`, `compute_verify_time()`, `sample_acceptance()`, `compute_cycle_time() -> (time_ms, accepted_tokens)` |

#### `distlmsim/parallelism/tensor_parallel.py`
- **Layer:** 4
- **Dependencies:** `config` (ModelConfig)
- **Input:** Model configuration, TP size
- **Output:** TP execution parameters and communication data sizes
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `TPExecutionParams` | dataclass | Per-GPU execution parameters under TP |
  | `TensorParallelModel` | class | `get_execution_params()`, `get_allreduce_data_size()`, `get_compute_flops_per_layer()` |

#### `distlmsim/parallelism/pipeline_parallel.py`
- **Layer:** 4
- **Dependencies:** `config` (ModelConfig)
- **Input:** Model configuration, PP size, GPUs per node
- **Output:** Pipeline stage partitioning and scheduling
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `PipelineStage` | dataclass | Stage with layer range and GPU assignment |
  | `PipelineSchedule` | dataclass | Schedule with stage ordering and timing |
  | `PipelineParallelModel` | class | `create_stages()`, `get_stage_comm_data_size()`, `get_pipeline_bubble_ratio()`, `create_schedule()` |

#### `distlmsim/parallelism/expert_parallel.py`
- **Layer:** 4
- **Dependencies:** `config` (ModelConfig, DeviceSKUConfig)
- **Input:** Model config, EP size, expert routing distribution
- **Output:** Expert placement, routing, all-to-all communication, load balancing
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `ExpertPlacement` | dataclass | Expert-to-GPU mapping |
  | `ExpertRoutingResult` | dataclass | Token-to-expert routing with load info |
  | `ExpertParallelModel` | class | `create_expert_placement()`, `route_tokens()`, `get_alltoall_data_size()` |
  | `MoELoadResult` | dataclass | Per-GPU load breakdown |
  | `DefaultRoutingScheduler` | class | No load balancing |
  | `EPLBScheduler` | class | Capacity truncation |
  | `RealisticEPLBScheduler` | class | Waterfill routing + redundant experts + periodic rebalance |
  | `OmniPlacementScheduler` | class | Greedy swap optimization |

#### `distlmsim/parallelism/parallelism_planner.py`
- **Layer:** 4
- **Dependencies:** `config` (ModelConfig, ClusterConfig), `topology/communication_cost`
- **Input:** Model and cluster configuration
- **Output:** Recommended parallelism strategy
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `ParallelismPlan` | dataclass | TP/PP/DP/EP sizes and node mapping |
  | `ParallelismPlanner` | class | `recommend_plan()`, `enumerate_plans()` |

---

### Layer 5: Context, Cluster, Metrics

#### `distlmsim/context.py`
- **Layer:** 5
- **Dependencies:** `config`, `entities`, `topology/*` (nvlink_model, rdma_model, overlap_processor), `execution/execution_time_predictor`, `metrics/metrics_store`
- **Input:** Model, device, and network configs
- **Output:** Fully initialized SimContext with all shared state
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `SimContext` | dataclass | Holds model_config, device_config, network_config, nvlink_model, rdma_model, overlap_processor, time_predictor, requests dict, metrics_store, cluster params, profiling config. Auto-initializes sub-models in `__post_init__()` |

#### `distlmsim/cluster/node.py`
- **Layer:** 5
- **Dependencies:** `config` (NodeSKUConfig), `types` (NodeRole)
- **Input:** Node SKU configuration
- **Output:** Physical node with GPU management
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `GPUDevice` | dataclass | Individual GPU with memory tracking |
  | `PhysicalNode` | class | GPU list management, memory allocation/release, replica association |

#### `distlmsim/cluster/cluster.py`
- **Layer:** 5
- **Dependencies:** `config` (ClusterConfig), `cluster/node`, `cluster/resource_manager`, `entities` (Replica), `topology/network_topology`, `types` (NodeRole)
- **Input:** Cluster configuration
- **Output:** Fully initialized cluster with nodes, replicas, and network topology
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `Cluster` | class | `from_config()` factory, `replicas`/`nodes` properties, GPU-to-node mapping |

#### `distlmsim/cluster/resource_manager.py`
- **Layer:** 5
- **Dependencies:** `cluster/node` (PhysicalNode), `topology/network_topology`
- **Input:** Physical nodes and network topology
- **Output:** Resource allocation decisions
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `ResourceManager` | class | `allocate_replica()`, `release_replica()`, `assign_node_roles()` |

#### `distlmsim/metrics/metrics_store.py`
- **Layer:** 5
- **Dependencies:** `config` (MetricsConfig)
- **Input:** Metrics configuration
- **Output:** Per-request latency metrics and aggregate statistics
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `RequestMetrics` | dataclass | Per-request metrics (ttft, tbt, e2e_latency, prefill_time, decode_time) |
  | `MetricsStore` | class | `record_request_arrival/scheduled/prefill_start/end/decode_start/end/kv_cache_transfer_start/end()`, `set_request_tokens()`, `finalize()`, `print_summary()`, `get_completed_count()` |

---

### Layer 6: Scheduling, Analysis, Design Space

#### `distlmsim/scheduling/global_scheduler.py`
- **Layer:** 6
- **Dependencies:** `config` (SchedulingConfig), `types` (GlobalSchedulerType), `interfaces` (ClusterView)
- **Input:** Scheduling config, cluster view
- **Output:** Request-to-replica routing decisions
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `BaseGlobalScheduler` | ABC | `select_replica(request_id) -> int`, `from_config()` factory |
  | `RoundRobinGlobalScheduler` | class | Round-robin assignment |
  | `RandomGlobalScheduler` | class | Random replica selection |
  | `LeastOutstandingGlobalScheduler` | class | Least-queue-depth selection |
  | `TopologyAwareGlobalScheduler` | class | Hash-affinity routing |

#### `distlmsim/scheduling/replica_scheduler.py`
- **Layer:** 6
- **Dependencies:** `entities` (Batch, Request, RequestStatus), `events` (BaseEvent)
- **Input:** Request stream, batch completion notifications
- **Output:** Batch formation decisions
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `BaseReplicaScheduler` | ABC | `on_request_arrival()`, `on_batch_end()`, `form_batch()` |
  | `SarathiReplicaScheduler` | class | Chunked prefill + decode mixing |
  | `VllmReplicaScheduler` | class | PagedAttention (stub) |
  | `OrcaReplicaScheduler` | class | Iteration-level (stub) |

#### `distlmsim/scheduling/advanced_schedulers.py`
- **Layer:** 6
- **Dependencies:** `entities` (Request)
- **Input:** Waiting queue of requests, batch size
- **Output:** Selected subset of requests
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `AdvancedSchedulers` | class | `select_mlfq()`, `select_po()`, `select_opt()`, `select_lightllm_prefill()`, `select_lightllm_decode()` |

#### `distlmsim/scheduling/disaggregated_scheduler.py`
- **Layer:** 6
- **Dependencies:** `config` (DisaggregatedConfig, ModelConfig), `entities` (Request), `events` (BaseEvent), `interfaces` (ClusterView), `types` (NodeRole)
- **Input:** Disaggregated config, cluster view
- **Output:** Prefill/decode node assignment and coordination events
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `DisaggregatedScheduler` | class | `initialize()`, `schedule_prefill()`, `schedule_decode()`, `on_prefill_complete() -> List[BaseEvent]` |

#### `distlmsim/scheduling/migration.py`
- **Layer:** 6
- **Dependencies:** `entities` (Request), `events` (BaseEvent), `interfaces` (ClusterView)
- **Input:** Cluster state, migration parameters
- **Output:** Migration plans and execution events
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `MigrationPlan` | dataclass | Request migration with cost/benefit estimates |
  | `RequestMigrationManager` | class | `evaluate_migrations()`, `execute_migration() -> List[BaseEvent]` |

#### `distlmsim/analysis/memory_analysis.py`
- **Layer:** 6
- **Dependencies:** `config` (ModelConfig, DeviceSKUConfig, ReplicaConfig)
- **Input:** Model and device configuration
- **Output:** Memory usage breakdown per GPU
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `MemoryBreakdown` | dataclass | Parameter/KV cache/activation/communication buffer sizes |
  | `MemoryAnalyzer` | class | Peak GPU memory estimation |

#### `distlmsim/analysis/mfu_analysis.py`
- **Layer:** 6
- **Dependencies:** `config` (ModelConfig, DeviceSKUConfig, ReplicaConfig)
- **Input:** Model, device, and replica configuration
- **Output:** Model FLOPs Utilization (MFU) metrics
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `MFUResult` | dataclass | Prefill/decode/overall MFU |
  | `MFUAnalyzer` | class | MFU computation |

#### `distlmsim/analysis/timeline_analysis.py`
- **Layer:** 6
- **Dependencies:** `metrics/metrics_store` (MetricsStore, RequestMetrics)
- **Input:** Completed MetricsStore
- **Output:** Chrome Trace JSON for visualization
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `TraceEvent` | dataclass | Chrome Trace event format |
  | `TimelineAnalyzer` | class | Generates Chrome Trace JSON from MetricsStore |

#### `distlmsim/design/design_space_explorer.py`
- **Layer:** 6
- **Dependencies:** `types` (GlobalSchedulerType, ReplicaSchedulerType)
- **Input:** Design space configuration
- **Output:** Pareto-optimal design points
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `DesignSpaceConfig` | dataclass | Search space definition |
  | `DesignPoint` | dataclass | Single configuration in the design space |
  | `DesignResult` | dataclass | Evaluation result for a design point |
  | `PruningRule` | dataclass | Rule for pruning infeasible configurations |
  | `DesignSpaceExplorer` | class | Enumerate, prune, and evaluate design points |

---

### Layer 7: Simulator Entry Point

#### `main.py` (project root, not inside distlmsim/)
- **Layer:** 7
- **Dependencies:** All lower layers
- **Input:** SimulationConfig, SimContext
- **Output:** DisaggregatedSimulator.run() → MetricsStore
- **Public API:**
  | Name | Type | Description |
  |------|------|-------------|
  | `SimContext` | re-export | Imported from `distlmsim.context` |
  | `DisaggregatedSimulator` | class | Two-phase queue-based PD-disaggregated simulator. Supports 9 scheduling policies (fcfs/sjf/ljf/srtf/random/mlfq/po/opt/lightllm). `run() -> MetricsStore` |
  | `DistributedInferenceSimulator` | class | Event-driven simulator for TP+PP scenarios |
  | `create_disaggregated_simulator()` | factory | Quick-creation helper with sensible defaults |

---

## Dependency Matrix (Adjacency List)

Each entry: `module → [direct dependencies]`

```
types               → []
interfaces          → []
entities            → [types]
config              → [types]
events              → [types, interfaces]
topology/network_topology → [config, types]
topology/nvlink_model     → [config, types]
topology/rdma_model       → [config, types]
topology/communication_cost → [config, topology/nvlink_model, topology/rdma_model, topology/network_topology, types]
topology/overlap_processor → [entities]
request/request_generator → [config, entities, events]
execution/execution_time_predictor → [config, entities]
execution/network_time_predictor   → [config, topology/nvlink_model, topology/rdma_model, types]
execution/speculative_decoder      → [config, entities, execution/execution_time_predictor, context]
parallelism/tensor_parallel    → [config]
parallelism/pipeline_parallel  → [config]
parallelism/expert_parallel    → [config]
parallelism/parallelism_planner → [config, topology/communication_cost]
context              → [config, entities, topology/nvlink_model, topology/rdma_model, topology/overlap_processor, execution/execution_time_predictor, metrics/metrics_store]
cluster/node         → [config, types]
cluster/cluster      → [config, cluster/node, cluster/resource_manager, entities, topology/network_topology, types]
cluster/resource_manager → [cluster/node, topology/network_topology]
metrics/metrics_store → [config]
scheduling/global_scheduler      → [config, types, interfaces]
scheduling/replica_scheduler     → [entities, events]
scheduling/advanced_schedulers   → [entities]
scheduling/disaggregated_scheduler → [config, entities, events, interfaces, types]
scheduling/migration → [entities, events, interfaces]
analysis/memory_analysis  → [config]
analysis/mfu_analysis     → [config]
analysis/timeline_analysis → [metrics/metrics_store]
design/design_space_explorer → [types]
main.py                 → [context, config, entities, execution/*, topology/*, parallelism/*, metrics/*, scheduling/*]
```

---

## DAG Verification

The dependency graph has been verified to contain **zero cycles**:
- All `TYPE_CHECKING` imports have been eliminated
- Cross-layer references use `Protocol` interfaces (defined at Layer 1)
- `SimContext` extracted to dedicated module to break execution→simulator dependency
- `events.py` depends only on Layer 0-1 modules (`types`, `interfaces`)

---

## __init__.py Re-export Modules

Each subpackage has an `__init__.py` that re-exports public symbols. These are not counted as separate modules — they are thin re-export layers:

| Package | Re-exports |
|---------|-----------|
| `distlmsim/__init__.py` | `__version__` |
| `distlmsim/topology/__init__.py` | NetworkTopology, NVLinkModel, RDMAModel, CommunicationCostCalculator, OverlapProcessor, OverlapConfig |
| `distlmsim/execution/__init__.py` | ExecutionTimePredictor, NetworkTimePredictor |
| `distlmsim/parallelism/__init__.py` | TensorParallelModel, PipelineParallelModel, ExpertParallelModel, ParallelismPlanner |
| `distlmsim/scheduling/__init__.py` | BaseGlobalScheduler, BaseReplicaScheduler, DisaggregatedScheduler, RequestMigrationManager |
| `distlmsim/cluster/__init__.py` | PhysicalNode, Cluster, ResourceManager |
| `distlmsim/metrics/__init__.py` | MetricsStore |
| `distlmsim/request/__init__.py` | BaseRequestGenerator |
| `distlmsim/analysis/__init__.py` | MemoryAnalyzer, MFUAnalyzer, TimelineAnalyzer |
| `distlmsim/design/__init__.py` | DesignSpaceExplorer |

---

## Non-Package Files

| Path | Description |
|------|-------------|
| `main.py` | Simulator entry point (Layer 7) |
| `examples/*.py` | 11 experiment/demo scripts |
| `tests/*.py` | 18 test files (313 tests) |
| `scripts/*.py` | 14 GPU profiling/benchmark scripts |
| `data/profiling/` | Profiled operator latency CSV files |
| `setup.py` | Package installation |
