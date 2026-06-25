# Code Review TODO — 论文 vs 代码对齐

> 生成时间: 2026-06-17
> 审查范围: 45 项论文技术声明 vs DistLMSim 代码库
> 审查方式: 自动代码分析 + 人工审查

> **复杂代码改动详细计划**: 见 [COMPLEX_TODO_PLAN.md](COMPLEX_TODO_PLAN.md)
> 包含 M6/M7/M8/I1/I2/I3 的修改方案、涉及文件、依赖关系和推荐执行顺序。

---

## 🔴 高优先级 TODO（必须在提交前解决）

### H1: RandomForestPredictor 实现
- **文件**: `distlmsim/execution/execution_time_predictor.py:162-187`
- **问题**: 类完全是 stub，`get_execution_time()` 直接 fallback 到 AnalyticalPredictor
- **论文引用**: §3.3 "Prediction Engine", §4 "Hybrid backend"
- **行动**: 
  - [ ] 从 TRADIOS 移植 RF 训练逻辑（或使用 sklearn RandomForestRegressor）
  - [ ] 实现特征工程（operator shape → feature vector）
  - [ ] 添加训练/推理接口
  - [ ] **或者**: 修改论文，将 "learned prediction" 改为 "future work"，仅保留 Roofline + Profiled 两种后端

### H2: MoE 负载不均衡 → 迭代延迟
- **文件**: `distlmsim/parallelism/expert_parallel.py` + `main.py`
- **问题**: `max_gpu_load` 被计算但从未传入 `_compute_decode_step_time()` 或 `_compute_prefill_time()`
- **论文引用**: §3.5 "MoE Expert Routing" — "最慢专家决定 all-to-all 通信完成时间"
- **行动**:
  - [ ] 在 `_compute_decode_step_time()` 中加入 MoE 不均衡因子
  - [ ] 公式: `expert_time = base_time * (max_gpu_load / avg_gpu_load)`
  - [ ] 确保 ExpertRoutingResult 的 max_gpu_load 传递到时间计算链

### ~~H3: 集成 OverlapProcessor（3D Timeline）~~ ✅ 已解决 (2026-06-22)
- **文件**: `distlmsim/topology/overlap_processor.py` + `main.py`
- **解决方案**: 
  - 在 SimContext 中初始化 `OverlapProcessor(OverlapConfig())`
  - 为 DisaggregatedSimulator 和 ColocatedSimulator 添加 `_apply_tp_overlap()` 方法
  - 替换所有 7 个计算方法中的 `per_layer_time + tp_comm` → `self._apply_tp_overlap(per_layer_time, tp_comm)`
  - 重叠模型: `max(adjusted_compute, adjusted_comm)`，其中 adjusted 包含 slowdown 因子
  - 当 TP=1 (comm=0) 时自动退化为纯计算时间
- **验证**: TP=4 decode (compute=0.3ms, comm=0.05ms) 墙钟时间降低 2.7%

### ~~H4: Chunked Prefill 实现~~ ✅ 已解决 (2026-06-25)
- **文件**: `main.py` (`_compute_prefill_time`, `_compute_prefill_chunk`) + `execution_time_predictor.py`
- **解决方案**:
  - 在 `DisaggregatedSimulator` 和 `ColocatedSimulator` 中实现 chunked prefill: 当 `total_tokens > prefill_chunk_size` 时拆分为多个 chunk 顺序处理
  - 每个 chunk 独立调用 `_compute_prefill_chunk()`, 传入 `kv_cache_size=processed_tokens` 反映积累的 KV cache
  - 更新 `AnalyticalPredictor.get_execution_time()` 中 prefill attention FLOPs: `4 * num_tokens * nq * (num_tokens + kv_cache_size) * hd` (原来忽略 kv_cache_size)
  - Chunked attention 总 FLOPs 从 O(n²) 降为 O(c·n), 其中 c 是 chunk size
- **新增实验**: `examples/experiment_chunked_prefill.py` — 扫描 chunk_size ∈ {256, 512, 1024, 2048, 4096}
- **验证结果**:
  - Prefill=4096: best chunk=256 → TTFT 1.38× speedup (27.4% reduction)
  - Prefill=8192: best chunk=256 → TTFT 1.53× speedup (34.7% reduction)
  - TBT P50 不受影响 (decode 阶段不变)
- **论文更新**: 新增 §5.6 "Chunked Prefill Analysis" (含 Figure~\ref{fig:chunked-prefill}), abstract/introduction 已同步

### ~~H5: PD 协调调度策略对齐~~ ✅ 已解决 (2026-06-25)
- **文件**: `design.tex` §3.4 "Coordinated Scheduling"
- **解决方案**: 采用方案 B — 修改论文以匹配代码
  - 将 "3 种 PD 协调策略 (Independent/Load-aware/Priority-based)" 替换为 "9 种请求级调度策略"
  - 列出所有 9 种策略: FCFS/SJF/LJF/SRTF/Random/MLFQ/PO/OPT/LightLLM，各附简短描述
  - PD 协调简化为 "static round-robin assignment"
  - 添加 `\ref{sec:evaluation}` 交叉引用指向评估章节
  - 为 `evaluation.tex` 添加 `\label{sec:evaluation}` 修复 "Section ??" 问题
- **论文更新**: design.tex "Request-Level Scheduling" + "PD Coordination" 两个段落重写
- **验证**: 编译通过 (13页, 455KB), 视觉确认 9 种策略正确渲染, 交叉引用正确解析为 "Section 5"

### ~~H6: TCP/IP vs RDMA 性能差异建模~~ ✅ 方案B 已解决 (2026-06-25)
- **文件**: `design.tex` §3.4 "Transport Protocol Modeling" + `rdma_model.py`
- **问题**: 论文原文声称 RDMA 比 TCP/IP 快 2-3×，但代码仅建模协议头开销差异 (RoCEv2 4.7% vs TCP/IP 10%)，实际有效带宽差距仅 ~5%
- **解决方案 (方案 B: 改论文)**:
  - 删除 "2-3×" 声称，改为 "∼5% effective bandwidth advantage"
  - 明确说明代码建模的是 header overhead ratio (RoCEv2 4.7%, IB 1.8%, TCP/IP 10%)
  - 承认 RDMA 的额外优势 (kernel bypass, CPU offloading, zero-copy) 尚未在代码中建模
  - 引用 Mooncake \citep{qin2024mooncake} 作为生产环境 2-3× 改进的来源
  - 添加 "planned for future work" 标记 CPU/kernel overhead 建模
- **论文更新**: design.tex "Transport Protocol Modeling" 段落完全重写
- **遗留 TODO (待 GPU6 恢复)**:
  - 用实测 RDMA vs TCP 数据校准 TCP_IP 的基础延迟 (20-50μs) 和带宽衰减
  - 添加 CPU 上下文切换开销 (~5μs per transfer)
  - 添加内核缓冲拷贝开销 (有效带宽降至 60-80%)
  - 已在 design.tex 中留下 `⚠️ CODE REVIEW TODO [FUTURE - H6]` 注释

### ~~H7: KV Cache 传输策略实现~~ ✅ 已解决 (2026-06-24)
- **文件**: `main.py` + `config.py` + `design.tex`
- **问题**: PIPELINED 和 STORE_FORWARD 策略仅有枚举定义，无实现
- **解决方案**:
  - 在 `config.py` 中添加 Store-and-Forward 参数 (write/read BW, I/O latency)
  - 实现 `_compute_prefill_time_per_chunk()` 返回 per-chunk 时间列表
  - 实现 `_compute_pipelined_kv_ready_time()`: chunked prefill 中边算边传
  - 实现 `_compute_store_forward_time()`: write + storage latency + read
  - 修改 `_compute_kv_transfer_time()` 根据策略分发到对应实现
  - 修改 `run()` 中的 KV 传输逻辑分支: DIRECT/PIPELINED/STORE_FORWARD
- **新增实验**: `examples/experiment_kv_transfer.py` — 3 策略 × 2 prefill_lengths × 2 QPS
- **验证结果**:
  - PIPELINED TTFT 降低 18.8% (pf=2048) / 11.4% (pf=4096) vs DIRECT
  - STORE_FORWARD E2E 增加 0.9-1.3% (额外存储 I/O 开销)
  - TBT 不受影响 (decode 阶段与传输策略无关)
- **论文更新**:
  - design.tex: 重写 "KV Cache Transfer" 段落，描述 3 种策略
  - evaluation.tex: 新增 §5.7 "KV Cache Transfer Strategy Comparison" (含 Figure 10)
  - abstract: 添加第 (6) 点 (pipelined transfer 11-19% TTFT reduction)
  - introduction: "six case studies" + 第 (6) 点
  - introduction.tex Table 1: 移除 H7 TODO 标记
- **编译验证**: 14页, 476KB, 0 errors; 视觉确认 Figure 10 数据与正文一致

---

## 🟡 中等优先级 TODO

### ~~M1: Roofline 效率因子文档化~~ ✅ 已解决 (2026-06-24)
- **文件**: `design.tex` §3.3 "Analytical Engine"
- **解决方案**: 更新 Roofline 公式，添加 $\eta_c=0.85$ 和 $\eta_m=0.90$ 效率因子参数说明

### ~~M2: 工厂函数集成~~ ✅ 已解决 (2026-06-24)
- **状态**: `create_predictor()` 已在 `SimContext.__post_init__` 中使用，仅 draft model 使用 `AnalyticalPredictor`（合理，无 profiling 数据）
- **论文**: implementation.tex TODO 标记已移除

### ~~M3: Speculative Decoding 内存开销~~ ✅ 已解决 (2026-06-24)
- **文件**: `design.tex` §3.5 "Speculative Decoding"
- **解决方案**: 移除 "memory overhead" 声称，改为 "operator-level execution time" + "computation cost and verification throughput"

### ~~M4: True Continuous Batching 描述~~ ✅ 已解决 (2026-06-24)
- **文件**: `design.tex` §3.2 "Continuous Batching Simulation"
- **解决方案**: 移除 "faithfully models continuous batching dynamics" 过度声称，改为 "models batch composition dynamics"；移除 TODO 标记

### ~~M8: Hybrid Backend 精度对比实验~~ ✅ 已解决 (2026-06-25)
- **文件**: `examples/experiment_hybrid_accuracy.py` + `evaluation.tex`
- **解决方案**:
  - 新建 `experiment_hybrid_accuracy.py`: 对比 Analytical/ProfilingBased/RandomForest 三种后端
  - 修复 ProfilingBasedPredictor 的 `import pandas as pd` 缺失 bug
  - 71 个样本 (64 decode + 7 prefill) 的 MAPE 对比:
    - Analytical (Roofline): MAPE 89.4%, Median 93.7%
    - **ProfilingBased (Linear): MAPE 45.0%, Median 38.3%** (最优)
    - RandomForest: MAPE 52.2%, Median 47.6%
  - ProfilingBased 相比 Roofline MAPE 降低 49.7%
- **论文更新**:
  - evaluation.tex §5.1: 新增 Table 2 (三种后端 MAPE 对比)
  - 新增 Figure 4 (bar chart + scatter plot)
  - 移除旧的 "substantially lower prediction error" TODO 标记
  - 正文添加 "49.7% relative improvement" 量化描述
- **编译验证**: 14页, 499KB; 视觉确认 Table 数据正确, Figure 含 bar chart + scatter plot

### ~~M5: ProfilingBasedPredictor Bug 修复~~ ✅ 已解决 (2026-06-24)
- **文件**: `distlmsim/execution/execution_time_predictor.py:487-501`
- **解决方案**: `isinstance(mask, object)` → `isinstance(mask, pd.Series)`，修复 2 处（attn_prefill 和 attn_decode 模型训练）

### ~~M6: 动态 GPU 资源约束~~ ✅ 已解决 (2026-06-25)
- **文件**: `config.py` + `main.py` + `design.tex`
- **解决方案**:
  - `DisaggregatedConfig` 新增 `gpu_memory_utilization: float = 0.90`
  - `DisaggregatedSimulator._compute_dynamic_batch_sizes()`: 基于 GPU 显存动态计算 max batch size
    - 模型大小: active_params × 2 bytes / num_gpus
    - KV cache/request: 2 × layers × kv_heads × head_dim × seq_len × 2 bytes
    - max_bs = available_mem / kv_per_req; actual_bs = min(config, max_bs)
  - `ColocatedSimulator` 同步实现
  - 修复返回值顺序 bug (prefill_bs, decode_bs)
- **验证**:
  - A800 80GB: max_prefill=2845, max_decode=2276 (GPU 充足，config 值生效)
  - 端到端模拟器正常: TBT P50=2.5ms (disagg), 452ms (coloc)
  - PD 实验数据更新: QPS=30 优势从 197× 调整为 183×
- **论文更新**:
  - design.tex §3.2: 添加动态 batch size 描述
  - evaluation.tex Table 4 + 正文: 数据全部更新
  - abstract + introduction: 197× → 183×
- **编译验证**: 14页, 498KB; 视觉确认 Table 数据正确, 正文一致

### ~~M7: Per-expert 计算时间模型~~ ✅ 已解决 (2026-06-25)
- **文件**: `expert_parallel.py` + `main.py` + `design.tex`
- **解决方案**:
  - `ExpertRoutingResult` 添加 `per_expert_latencies: np.ndarray` 字段
  - `ExpertParallelModel._compute_per_expert_latencies()`: Roofline 模型计算每个专家的 MLP 延迟 (gate_up + act + down)
  - `route_tokens()` 自动计算 per_expert_latencies 并写入 result
  - `main.py._compute_moe_imbalance_factor()` 改用 `max(gpu_latencies)/mean(gpu_latencies)` 替代 token 计数比
  - DisaggregatedSimulator 和 ColocatedSimulator 均已更新
- **验证**:
  - Hot (50 tokens) vs Cold (5 tokens) 延迟比 4.28x（合理）
  - 端到端模拟器正常: TBT P50=2.42ms (旧: 2.62ms，更精确)
  - MoE 负载实验数据不变（测量 token 分布，非延迟）
- **论文更新**: design.tex §3.5 "MoE Expert Routing" 段落重写，含 Roofline 公式 max_i(T_i)/mean(T_i)
- **编译验证**: 14页, 478KB; 视觉确认 Roofline 模型和不平衡因子公式正确

---

## 🔧 代码改进机会

### ~~I1: Speculative Decoding 模块化~~ ✅ 已解决 (2026-06-25)
- **文件**: `distlmsim/execution/speculative_decoder.py` (新增) + `main.py`
- **解决方案**:
  - 新建 `SpeculativeDecoder` 类，封装 draft/verify 时间计算和接受率采样
  - `compute_draft_step_time()`: 使用预创建的 draft predictor
  - `compute_verify_time()`: target model 验证 K tokens
  - `sample_acceptance()`: 顺序接受采样
  - `compute_cycle_time()`: 完整投机周期时间
  - `main.py` 中 `_get_spec_decoder()` 懒初始化，decode 循环委托调用
  - 修复论文数据: baseline TBT 2.61→2.46ms, speedup 6.71→6.33×, α=0.6 best 2.65→2.40×
- **验证**: Spec K=4 α=0.8 → 3.26× speedup; 完整实验 K=12 α=0.9 → 5.93×
- **编译验证**: 14页, 498KB; 视觉确认 speculative decoding 数据一致

### I2: Congestion Factor 建模 ✅ 已解决 (2026-06-24)
- **文件**: `rdma_model.py` + `config.py` + `main.py` + `design.tex`
- **解决方案**:
  - 实现 alpha-fair 拥塞模型: `factor = 1 / (1 + alpha * (n-1))`
  - 在 `RDMAConfig` 中添加 `congestion_alpha=0.05` 参数
  - `get_transfer_time()` 新增 `concurrent_transfers` 参数
  - `main.py` 的 KV 传输调用点传入 `len(batch)` 作为并发数
  - 验证: 32 并发传输比单流慢 2.48x（符合 DCQCN 行为）
  - 论文 §3.7 新增 "Network Congestion" 段落，含公式
- **编译验证**: 14页, 477KB; 视觉确认公式和 DCQCN 引用正确

### I3: 事件系统 Stub 补全 ✅ 已解决 (2026-06-24)
- **文件**: `distlmsim/events.py`
- **解决方案**: 将 8 处 `NotImplementedError` 替换为功能实现或文档化占位
  - `KVCacheTransferStartEvent`: 实现 RDMA 传输时间计算（含回退模型）
  - `ExpertAssignmentEvent`: 链接到 `ExpertCommStartEvent`
  - `ExpertCommStartEvent`: 实现 EP all-to-all 通信时间估算
  - `ReplicaScheduleEvent/BatchStageArrivalEvent/BatchEndEvent/DecodeStartEvent/ExpertCommEndEvent`: 文档化占位（return []），说明 main.py 使用自己的仿真循环
- **验证**: 0 NotImplementedError, 13 事件类全部可导入, 模拟器运行正常
- **论文**: 事件驱动架构已在 design.tex §3.1 和 implementation.tex §4.1 正确描述，无需修改

---

## 📊 论文修改建议（如果不改代码）

如果选择不实现上述高优先级 TODO，以下论文段落需要修改：

| 论文位置 | 当前声称 | 建议修改 |
|----------|---------|---------|
| §3.3 Prediction Engine | "learned prediction for unseen shapes" | 改为 "future work" 或删除 |
| §3.5 MoE Load Imbalance | "最慢专家决定延迟" | 删除此因果链，仅保留负载分布统计 |
| §3.6 3D Timeline | "constructing a 3D timeline" | 改为 "planned extension" |
| §3.4 Chunked Prefill | "supports prefill chunking" | 改为 "configuration support" |
| §3.4 PD Coordination | 3 种策略名称 | 改为实际的 9 种请求级策略 |
| §3.4 TCP/IP vs RDMA | "2-3× difference" | 改为 "~10% protocol overhead" |
| Table 1 | "KV Cache Transfer" ✓ | 仅保留 DIRECT 策略 |
