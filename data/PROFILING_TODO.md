# Profiling Data Collection TODO

本文档列出 DistLMSim 需要采集的所有 profiling 数据，按优先级和 GPU 设备组织。

**状态总览 (2026-06-18):** 8/10 项完成 ✅ | 1 项部分完成 ⚠️ | 3 项被阻塞 ❌ (GPU6 不可达)

## 当前已有数据

| 设备 | 模型 | 文件 | 行数 | 状态 |
|------|------|------|------|------|
| A800 | Qwen3-30B-A3B | `compute/.../attention.csv` | **166** | ✅ TP=1+2+4, decode 非零 |
| A800 | Qwen3-30B-A3B | `compute/.../mlp.csv` | **15** | ✅ 1-16384 tokens |
| A800 | Qwen3-30B-A3B | `compute/.../expert.csv` | **12** | ✅ 重新采集 |
| A800 | Qwen3-30B-A3B | `compute/.../eplb.csv` | **12** | ✅ 重新采集 (EPLB 开销) |
| A800 | Llama-2-13B | `compute/.../attention.csv` | **37** | ✅ TP=1, prefill+decode |
| A800 | Llama-2-13B | `compute/.../mlp.csv` | **14** | ✅ 1-8192 tokens |
| A800 DGX | - | `network/.../all_reduce.csv` | **26** | ✅ TP=2 + TP=4 |
| A800 DGX | - | `network/.../expert_comm.csv` | **42** | ✅ EP=2 + EP=4 |
| A800↔A800 | - | `network/.../tcp_transfer.csv` | **11** | ✅ TCP/IP 1KB-1GB |

## 交叉验证 (GPU4 vs GPU5, A800)

| num_tokens | GPU4 mlp_up_proj (ms) | GPU5 mlp_up_proj (ms) | 差异 |
|-----------|----------------------|----------------------|------|
| 1 | 0.0410 | 0.0522 | 27.5% (kernel launch 噪声) |
| 64 | 0.0655 | 0.0553 | 15.6% (kernel launch 噪声) |
| 1024 | 0.2345 | 0.2355 | **0.4%** |
| 4096 | 0.9810 | 0.9810 | **0.0%** |
| 16384 | 6.0498 | 6.1307 | **1.3%** |

**结论:** 大 token (≥1024) 计算密集型操作跨 GPU 差异 <1.5%，数据高度一致。
小 token 操作受 kernel launch 开销影响较大，属于正常现象。

## 可用 GPU 资源

| 机器 | IP | GPU | 数量 | RDMA | 状态 |
|------|-----|-----|------|------|------|
| GPU4 | 100.64.0.6 | A800 80GB | 4 | ✅ ConnectX-5 (25Gbps) | 常被占用 |
| GPU5 | 10.21.16.123 | A800 80GB | 4 | ❌ | - |
| GPU6 | 10.21.16.124 | H100 80GB PCIe | 2 | ✅ ConnectX-5 (25Gbps) | - |
| GPU2 | 100.64.0.4 | Quadro RTX 6000 | 1 | ❌ | 算力小 |
| GPU3 | 100.64.0.5 | Quadro RTX 6000 | 1 | ❌ | 通常有30-40GB空闲 |

---

## TODO 清单

### 🔴 P0: 关键缺失 — 直接影响模拟器准确性

- [x] **T1: A800 compute profiling — 补全 num_tokens 范围** ✅ 2026-06-18
  - 文件: `compute/a800/Qwen/Qwen3-30B-A3B/mlp.csv` (15 rows, 1-16384)
  - 采集环境: GPU4 (A800), PyTorch 2.10, CUDA 12.8

- [x] **T2: A800 attention profiling — 修复 decode attention = 0.0** ✅ 2026-06-18
  - 文件: `compute/a800/Qwen/Qwen3-30B-A3B/attention.csv` (106 rows: 43 prefill + 63 decode)
  - decode attention 范围: 0.15ms (batch=1,kv=64) → 24.65ms (batch=256,kv=4096)
  - 采集环境: GPU4 (A800), PyTorch 2.10, CUDA 12.8

- [ ] **T3: H100 compute profiling — 全新采集** ❌ 被阻塞
  - 目录: `compute/h100/Qwen/Qwen3-30B-A3B/`
  - 文件: `mlp.csv`, `attention.csv`, `expert.csv`, `eplb.csv`
  - 参数同 A800 但 H100 硬件规格不同 (FP16=49.5 TFLOPS, HBM3=3350 GB/s)
  - 机器: GPU6 (H100) — **当前 SSH 不可达 (Network unreachable)**

- [x] **T4: A800 network profiling — 补全 TP=4** ✅ 2026-06-18
  - 文件: `network/a800_dgx/all_reduce.csv` (26 rows: 13 TP=2 + 13 TP=4)
  - TP=2: 0.013ms (2KB) → 0.68ms (8MB)
  - TP=4: 0.023ms (2KB) → 1.04ms (8MB)
  - 采集环境: GPU4 (4x A800, NVLink), torchrun + NCCL

### 🟡 P1: 重要补充 — 提升模拟器通用性

- [ ] **T5: H100 network profiling — NVLink all-reduce** ❌ 被阻塞
  - 目录: `network/h100_pcie/`
  - 文件: `all_reduce.csv`
  - H100 PCIe 无 NVSwitch，NVLink P2P 带宽与 A800 不同
  - 机器: GPU6 (2x H100) — **当前 SSH 不可达**

- [⚠️] **T6: 跨节点传输 profiling** — TCP/IP 部分 ✅, RDMA 部分 ❌
  - ✅ `tcp_transfer.csv` (11 rows): GPU4↔GPU5 TCP/IP, 1KB-1GB
    - 带宽: 1.2 Gbps (1KB) → 7.3 Gbps (1GB), 稳定在 ~7 Gbps
    - 采集环境: GPU4 (10.21.16.122) → GPU5 (10.21.16.123), 10Gbps 管理网络
  - ❌ RDMA 部分需要 GPU4↔GPU6 (ConnectX-5 25Gbps RDMA 直连)，但 GPU6 不可达

- [x] **T7: expert_comm.csv 补全 — 多 EP 配置** ✅ 2026-06-18
  - 文件: `network/a800_dgx/expert_comm.csv` (42 rows: EP=2 + EP=4)
  - EP=2: 0.038ms (1KB) → 144ms (1GB)
  - EP=4: 0.017ms (1KB) → 159ms (1GB)
  - 采集环境: GPU4 (4x A800, NVLink), torchrun + NCCL all-to-all

- [x] **T8: A800 multi-TP attention profiling** ✅ 2026-06-18
  - 文件: `compute/a800/Qwen/Qwen3-30B-A3B/attention.csv` (追加 TP=2: 30行 + TP=4: 30行)
  - attention.csv 现在包含 166 行: TP=1 (106) + TP=2 (30) + TP=4 (30)
  - 采集环境: GPU4 (A800)

### 🟢 P2: 锦上添花 — 支持更多模型

- [x] **T9: Llama-2-13B profiling (A800)** ✅ 2026-06-18
  - 目录: `compute/a800/Meta/Llama-2-13b-chat-hf/`
  - `mlp.csv`: 14 rows (1-8192 tokens)
  - `attention.csv`: 37 rows (13 prefill + 24 decode)
  - 采集环境: GPU4 (A800), PyTorch 2.10, CUDA 12.8

- [ ] **T10: H100 Llama-2-13B profiling** ❌ 被阻塞
  - 目录: `compute/h100/Meta/Llama-2-13b-chat-hf/`
  - 文件: `mlp.csv`, `attention.csv`
  - 机器: GPU6 — **当前 SSH 不可达**

---

## CSV 格式规范

### compute/mlp.csv
```
n_head,n_kv_head,n_embd,n_expanded_embd,vocab_size,use_gated_mlp,num_tokens,num_tensor_parallel_workers,
time_stats.emb.min,time_stats.emb.max,time_stats.emb.mean,time_stats.emb.median,time_stats.emb.std,
time_stats.input_layernorm.min,...(共 10 组 x 5 列 = 50 列时间)
```

### compute/attention.csv
```
n_embd,n_q_head,n_kv_head,block_size,num_tensor_parallel_workers,max_model_len,batch_size,prefill_chunk_size,kv_cache_size,is_prefill,attention_backend,
time_stats.attn_input_reshape.min,...(共 5 组 x 5 列 = 25 列时间)
```

### compute/expert.csv
```
expert_id,num_experts,top_k_experts,n_embd,n_expanded_embd,num_tokens,batch_size,seq_len,
time_stats.expert_mlp.min,...(5 列)
```

### compute/eplb.csv
```
num_experts,batch_size_range,batch_size,num_tokens,max_expert_load,avg_expert_load,expert_load_deviation,
time_stats.eplb_overhead.min,...(5 列)
```

### network/all_reduce.csv
```
rank,num_workers,size,collective,devices_per_node,max_devices_per_node,
time_stats.all_reduce.min,...(5 列)
```

### network/expert_comm.csv
```
expert_parallel_size,network_device,data_size_bytes,tokens_per_device,num_tokens,
time_stats.expert_comm.median
```

---

## 采集工具

所有 profiling 数据通过 `DistLMTest/profiling/` 目录下的 benchmark 脚本采集：

| 脚本 | 输出 | 说明 |
|------|------|------|
| `profile_compute_mlp.py` | `mlp.csv` | MLP 路径 10 子操作 |
| `profile_compute_attention.py` | `attention.csv` | Attention 5 子操作 (含 decode) |
| `profile_compute_attention_multitp.py` | (追加 attention.csv) | TP=2,4 attention |
| `profile_compute_expert.py` | `expert.csv` | MoE Expert MLP |
| `profile_compute_eplb.py` | `eplb.csv` | EPLB 调度开销 |
| `profile_llama13b.py` | mlp.csv + attention.csv | Llama-2-13B 一站式 |
| `profile_network_allreduce.py` | `all_reduce.csv` | NVLink All-Reduce (torchrun) |
| `profile_network_expert_comm.py` | `expert_comm.csv` | EP all-to-all (torchrun) |
| `profile_network_cross_node.py` | `tcp_transfer.csv` | 跨节点 TCP (server/client) |

一键运行: `bash DistLMTest/profiling/run_all_profiling.sh --gpu 0 --compute_out <dir> --network_out <dir>`
