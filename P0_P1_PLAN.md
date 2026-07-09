# DistLMSim DSpark 模拟改进计划

> 基于论文-代码交叉验证，修复不一致之处并添加 MTP-1 baseline。
> 优先级: P0 = 必须修复 (影响数据正确性), P1 = 强烈建议 (实验完整性)

---

## P0-1: DSpark pos1 acceptance 0.88 → 0.93 [P0]

**问题**: 论文 Figure 2 显示 DSpark 在 Math domain 的 pos1 conditional acceptance = 0.93，
但代码内置 profile 为 0.88（与 DFlash 相同）。DSpark 继承 DFlash 的 parallel backbone，
pos1 应该相同或更高（论文说 DSpark inherits the high initial acceptance, starting at 0.93 on Math）。

**文件**: `distlmsim/execution/acceptance_profile.py`
**位置**: `_BUILTIN_DSPARK` 字典
**修改**: 将所有 domain 的 pos1 值调高以匹配论文 Figure 2

```
当前值 → 目标值:
  math:  [0.88, 0.88, 0.87, 0.87, 0.86, 0.86, 0.85]
      → [0.93, 0.91, 0.90, 0.89, 0.88, 0.88, 0.87]
  code:  [0.82, 0.82, 0.81, 0.81, 0.80, 0.80, 0.79]
      → [0.87, 0.85, 0.84, 0.83, 0.83, 0.82, 0.82]
  chat:  [0.72, 0.72, 0.71, 0.70, 0.70, 0.69, 0.69]
      → [0.76, 0.74, 0.73, 0.72, 0.71, 0.71, 0.70]
  mixed: [0.80, 0.80, 0.79, 0.79, 0.78, 0.78, 0.77]
      → [0.85, 0.83, 0.82, 0.81, 0.81, 0.80, 0.80]
```

**验证**: 重新跑实验，DSpark τ 应提升 ~5%，DSpark vs DFlash 改进幅度应更接近论文 16-18%。

---

## P0-2: Per-request accepted tokens [P0]

**问题**: `main.py` decode 循环中所有 request 统一增加 `result.accepted_tokens`，
但实际上每个 request 的 scheduled_length 和 acceptance 可能不同。

**文件**: `main.py`
**位置**: `DisaggregatedSimulator.run()` 的 decode 循环 (~line 760)

当前代码:
```python
for req in active:
    req.num_generated_tokens += result.accepted_tokens  # ← batch 级统一
```

修改为:
```python
for i, req in enumerate(active):
    # Per-request: 取 scheduled_length 和 acceptance 的最小值
    req_accepted = min(result.accepted_tokens, result.per_request_accepted[i])
    req.num_generated_tokens += req_accepted
    req.accepted_tokens_last_cycle = req_accepted
```

**前置**: `CycleResult` 需要新增 `per_request_accepted: List[int]` 字段。

**文件**: `distlmsim/execution/speculative_decoder.py`
**位置**: `CycleResult` dataclass 和 `_compute_speculative_cycle()`

修改 CycleResult:
```python
@dataclass
class CycleResult:
    cycle_time_ms: float = 0.0
    accepted_tokens: int = 1          # batch 最小值 (向后兼容)
    per_request_accepted: list = None  # per-request accepted tokens [P0]
    scheduled_length: int = 1
    ...
```

修改 _compute_speculative_cycle:
```python
# 当前: effective_accepted = min(accepted, min_scheduled) — 统一
# 修改: 每个 request 独立计算
per_request_accepted = []
for i, req in enumerate(batch_requests):
    req_accepted = min(accepted, scheduled_lengths[i])
    if self._config.bonus_token:
        req_accepted = min(req_accepted + 1, scheduled_lengths[i] + 1)
    per_request_accepted.append(max(1, req_accepted))
```

---

## P0-3: Per-request verify time (变长 batch) [P0]

**问题**: Verify 时间用 `total_verify_tokens = sum(1 + l_r)` 计算，这是正确的，
但 `num_tokens` 参数传给 `get_execution_time` 时，attention 计算假设所有 token 均匀分布。
变长 verify 下，不同 request 的 token 数不同，attention 时间应有差异。

**文件**: `distlmsim/execution/speculative_decoder.py`
**位置**: `_compute_speculative_cycle()` 的 verify 阶段

**当前状态**: 已经用 `total_verify_tokens` 计算 verify 时间，这是合理的近似。
不需要额外修改，因为 Roofline 模型只关心总 token 数和 batch size。

**结论**: P0-3 不需要修改，当前实现已足够准确。

---

## P1-1: MTP-1 baseline [P1]

**问题**: 论文的核心对比是 DSpark vs MTP-1 (Multi-Token Prediction, 1 token)。
MTP-1 是 DeepSeek 之前的生产方案：每轮只预测 1 个额外 token，无 draft model。

**文件**: `distlmsim/execution/speculative_decoder.py`
**位置**: `SpeculativeDecodingEngine.compute_cycle()`

添加 MTP-1 模式:
```python
if self._config.speculative_mode == "mtp1":
    # MTP-1: target model 生成 1 个 token + 验证 1 个 MTP token
    # 等同于 block_size=1 的标准投机解码，无 draft model
    draft_time = 0.0  # MTP head 是 target model 的一部分，无额外 draft 时间
    accepted = 1 if self._rng.random() < self._config.acceptance_rate else 0
    accepted = max(1, accepted)  # 至少接受 bonus token
    # Verify: 只验证 1+1=2 tokens
    ...
```

**配置**: `speculative_mode = "mtp1"`, `block_size = 1`, `acceptance_rate = 0.8`

**文件**: `distlmsim/config.py`
**位置**: `DisaggregatedConfig`

无需修改，`speculative_mode` 已支持字符串扩展。

---

## P1-2: MTP-3/5 static baseline [P1]

**问题**: 论文提到 MTP-3/5 在高并发下会严格降低吞吐（因为静态验证过多）。

**实现**: 通过 `speculative_mode = "dspark"` + `enable_confidence_scheduling = False`
+ `block_size = 3/5` 即可模拟。无需额外代码修改。

---

## 修改顺序

1. **P0-1** acceptance_profile.py (5 min)
2. **P0-2** speculative_decoder.py CycleResult + main.py (30 min)
3. **P1-1** speculative_decoder.py MTP-1 mode (15 min)
4. 运行测试验证 (10 min)
5. 重跑实验验证数值 (30 min)
6. 提交 (5 min)

总计: ~1.5h
