# 复杂代码改动修改计划 ✅ 全部完成 (2026-06-25)

本文档列出 DistLMSim 的复杂代码改动项，按依赖顺序排列。
所有 6 项均已完成并验证。

---

## 依赖关系图

```
M7 (per-expert latency) ──→ M8 (hybrid backend experiment)
                                    ↑
M6 (dynamic batch size) ────────────┘ (独立，但 M8 实验应包含此效果)

I2 (congestion factor)  ──── 独立
I1 (spec decode modular) ─── 独立
I3 (event stubs)        ──── 独立
```

---

## TODO #1: M7 — Per-Expert 计算时间模型

### 问题
`ExpertParallelModel.route_tokens()` 仅计算 `expert_loads`（每个专家的 token 数），
但不基于 token 数量推导每个专家的实际计算延迟。论文 §3.5 声称 "the slowest expert
determines the all-to-all communication completion time"，需要 per-expert latency 才能
真正建模这一行为。

### 修改方案
1. **`ExpertRoutingResult` 添加字段**:
   ```python
   per_expert_latencies: np.ndarray  # shape: [num_experts] 每个专家的计算延迟 (ms)
   ```

2. **`route_tokens()` 添加 per-expert latency 计算**:
   - 输入: `expert_loads[i]` (每个专家的 token 数), `model_config` (expert_hidden, expert_ffn_dim)
   - 公式: `latency_i = roofline(gate_up_flops + act_flops + down_flops, mem_bytes)`
   - 其中 FLOPs = `2 * tokens_i * (n_embd * expert_hidden * 2 + expert_hidden * expert_ffn_dim)`

3. **`main.py._compute_moe_imbalance_factor()` 改用 per_expert_latencies**:
   - 旧: `max_gpu_load / avg_gpu_load`（基于 token 数的粗粒度比）
   - 新: `max(per_expert_latencies) / mean(per_expert_latencies)`（基于时间的精确比）

4. **新增实验验证**: MoE 负载不均衡实验中，对比基于 token 数 vs 基于 latencies 的 imbalance factor

### 涉及文件
- `distlmsim/parallelism/expert_parallel.py` (ExpertRoutingResult + route_tokens)
- `main.py` (_compute_moe_imbalance_factor)
- `examples/experiment_moe_load.py` (验证)

### 依赖: 无
### 工作量: 中等 (~100 行代码)

---

## TODO #2: M6 — 动态 GPU 资源约束

### 问题
当前 batch size 由配置固定 (`prefill_batch_size=2, decode_batch_size=32`)，
不考虑 GPU 显存限制。实际上 batch size 受限于:
`KV_cache_per_request * batch_size + model_params ≤ GPU_memory`

### 修改方案
1. **添加 `GPUMemoryModel` 类**:
   ```python
   class GPUMemoryModel:
       def __init__(self, total_gb, model_config, device_config):
           self.model_params_gb = self._compute_model_params(model_config)
           self.available_gb = total_gb - self.model_params_gb

       def max_batch_size(self, kv_cache_per_request_bytes: int) -> int:
           return int(self.available_gb * 1e9 / kv_cache_per_request_bytes)

       def kv_cache_per_request(self, seq_len: int) -> int:
           return 2 * num_layers * num_kv_heads * head_dim * seq_len * 2  # bytes
   ```

2. **在 `DisaggregatedSimulator.__init__` 中初始化 `GPUMemoryModel`**:
   - Prefill GPU: 计算 max_prefill_batch_size
   - Decode GPU: 计算 max_decode_batch_size (基于 avg kv_cache_size)

3. **在 `run()` 的 batch composition 中使用动态上限**:
   - `actual_prefill_bs = min(config.prefill_batch_size, gpu_mem.max_batch_size(kv_per_req))`
   - `actual_decode_bs = min(config.decode_batch_size, gpu_mem.max_batch_size(avg_kv))`

4. **添加 OOM 模拟**: 如果请求的 KV cache 超出剩余显存，加入等待队列

### 涉及文件
- `distlmsim/scheduling/gpu_memory.py` (新增)
- `main.py` (两个 Simulator 类)
- `config.py` (添加 `gpu_memory_utilization` 参数)

### 依赖: 无
### 工作量: 中等 (~120 行代码)

---

## TODO #3: M8 — Hybrid Backend 精度对比实验

### 问题
论文 §5.1 声称 "substantially lower prediction error when using the hybrid backend"，
但 `experiment_accuracy.py` 仅对比 Roofline-only vs profiling 实测，未展示
ProfilingBasedPredictor 和 RandomForestPredictor 的精度提升。

### 修改方案
1. **扩展 `experiment_accuracy.py`**:
   - 添加 `--predictor_type` 参数: `["analytical", "profiled", "random_forest"]`
   - 对每种 predictor type，运行相同的样本集，计算 MAPE

2. **新增对比维度**:
   - 按操作类型分组: attention vs MLP vs MoE expert
   - 按 batch size 分组: small (1-16) vs large (32-128)
   - 按 phase 分组: prefill vs decode

3. **生成对比图表**:
   - 3 柱对比: Roofline MAPE vs Profiled MAPE vs RF MAPE
   - 分操作/分 batch size 的热力图

4. **更新论文 §5.1**:
   - 添加 Table: 3 种后端的 MAPE 对比
   - 添加 Figure: 分操作类型的精度对比柱状图

### 涉及文件
- `examples/experiment_accuracy.py` (主要修改)
- `evaluation.tex` (更新 §5.1)
- `results/simulation_accuracy.json` (扩展)

### 依赖: M7 (per-expert latency 改善 RF predictor 的 expert 预测精度)
### 工作量: 中等 (~80 行代码 + 论文更新)

---

## TODO #4: I2 — RDMA Congestion Factor 建模

### 问题
`rdma_model.py` 中 `_get_congestion_factor()` 始终返回 1.0（无拥塞），
论文 §3.7 提到 "congestion_factor" 但未建模。

### 修改方案
1. **简单模型**: 基于同时传输数量:
   ```python
   def _get_congestion_factor(self, concurrent_transfers: int = 1) -> float:
       # Alpha-fair model: factor = 1 + alpha * (concurrent - 1)
       alpha = 0.05  # 每增加一个并发传输，带宽降低 5%
       return 1.0 + alpha * max(0, concurrent_transfers - 1)
   ```

2. **在 `run()` 中追踪并发传输数**:
   - 维护 `active_transfers` 计数器
   - KV transfer start → +1, end → -1

### 涉及文件
- `distlmsim/topology/rdma_model.py`
- `main.py` (run 循环)

### 依赖: 无
### 工作量: 小 (~30 行代码)

---

## TODO #5: I1 — Speculative Decoding 模块化

### 问题
投机解码逻辑散布在 `main.py:230-312`（DisaggregatedSimulator）和
`main.py:760-820`（ColocatedSimulator）中，代码重复且难以维护。

### 修改方案
1. **新建 `distlmsim/execution/speculative_decoder.py`**:
   ```python
   class SpeculativeDecoder:
       def __init__(self, config, ctx):
           self.K = config.speculation_length
           self.alpha = config.acceptance_rate
           self.draft_predictor = ...

       def compute_speculation_cycle_time(self, batch_requests) -> tuple[float, int]:
           """返回 (cycle_time_ms, accepted_tokens)"""
           draft_time = self._compute_draft_time(batch_requests)
           verify_time = self._compute_verify_time(batch_requests)
           accepted = int(self.K * self.alpha)
           return draft_time + verify_time, accepted
   ```

2. **两个 Simulator 类改为委托调用**:
   ```python
   self._spec_decoder = SpeculativeDecoder(config, ctx)
   # 在 decode 循环中:
   if self._spec_decoder:
       cycle_time, accepted = self._spec_decoder.compute_speculation_cycle_time(batch)
   ```

### 涉及文件
- `distlmsim/execution/speculative_decoder.py` (新增)
- `main.py` (两个 Simulator 类)

### 依赖: 无
### 工作量: 中等 (~150 行代码)

---

## TODO #6: I3 — 事件系统 Stub 补全

### 问题
`events.py` 中多个事件的 `handle_event()` 是空实现:
- `ExpertCommStartEvent`
- `ExpertCommEndEvent`
- `PipelineStageStartEvent`
- `PipelineStageEndEvent`

### 修改方案
1. 实现 ExpertComm 事件: 更新 MetricsStore 中的 expert_comm 时间
2. 实现 PipelineStage 事件: 更新 pipeline bubble 时间
3. 如果暂不使用，添加 `raise NotImplementedError("EP/PP events not yet integrated")` 明确标记

### 涉及文件
- `distlmsim/events.py`

### 依赖: 无
### 工作量: 小 (~40 行代码)

---

## 执行结果

| 顺序 | TODO | 状态 | 完成日期 |
|------|------|------|----------|
| 1 | I2 (congestion) | ✅ | 2026-06-25 |
| 2 | I3 (event stubs) | ✅ | 2026-06-25 |
| 3 | M7 (per-expert) | ✅ | 2026-06-25 |
| 4 | M6 (dynamic batch) | ✅ | 2026-06-25 |
| 5 | I1 (spec decode) | ✅ | 2026-06-25 |
| 6 | M8 (hybrid experiment) | ✅ | 2026-06-25 |
