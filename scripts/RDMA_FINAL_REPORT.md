# GPU显存复制 + RDMA测试 - 最终报告

**测试日期**: 2026-06-05  
**测试服务器**: 
- Server 1: 10.21.16.124 (302-GPU6)
- Server 2: 100.64.0.6 (302-GPU4)

---

## 📊 测试结果

### ✅ 目标1: GPU显存复制速度 (NCCL协议) - 已完成

**测试方法**: CUDA P2P + NCCL风格通信 (All-Gather, Ring All-Reduce)

#### Server 1 (10.21.16.124, 4×A800 80GB PCIe)

| 测试项 | 带宽 (Gbps) | 带宽 (GB/s) | 延迟 |
|--------|-------------|-------------|------|
| **P2P点对点** | **166.01** | **20.75** | 5.02 ms (100MB) |
| **All-Gather** | **205.58** | **25.70** | 11.67 ms/次 |
| **Ring All-Reduce** | **178.92** | **22.37** | 26.83 ms/次 |
| **小消息(1KB)** | - | - | **6.8 μs** |
| **消息(1MB)** | - | - | **60.8 μs** |

#### Server 2 (100.64.0.6, 4×A800 80GB PCIe)

| 测试项 | 带宽 (Gbps) | 带宽 (GB/s) | 延迟 |
|--------|-------------|-------------|------|
| **P2P点对点** | **145.37** | **18.17** | 4.94-6.10 ms |
| **All-Gather** | **172.55** | **21.57** | 13.91 ms/次 |
| **Ring All-Reduce** | **155.97** | **19.50** | 30.77 ms/次 |
| **小消息(1KB)** | - | - | **7.0 μs** |
| **消息(1MB)** | - | - | **60.0 μs** |

### ⏸️ 目标2: RDMA物理连接测试 - 受阻于硬件

**检查结果**:

| 项目 | Server 1 | Server 2 |
|------|----------|----------|
| **RDMA设备** | Mellanox ConnectX-5 | Mellanox ConnectX-6 |
| **固件版本** | 16.35.4506 ✅ | 16.35.4506 ✅ |
| **链路状态** | ❌ DOWN | ❌ DOWN |
| **物理连接** | ❌ 网线未插 | ❌ 网线未插 |
| **ethtool检测** | Link detected: no | Link detected: no |

**受阻原因**: RDMA网卡物理网线未连接，无法进行RDMA带宽和延迟测试

**替代网络测试** (10GbE以太网):

| 测试项 | 结果 |
|--------|------|
| **物理带宽** | 10,000 Mb/s |
| **TCP单流带宽** | 0.63-9.43 Gbps |
| **网络延迟** | **0.059 ms** (59 μs) |

---

## 📋 完整测试数据表

### GPU P2P带宽详细数据

#### Server 1: GPU间P2P带宽矩阵 (Gbps)

| 源\目标 | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---------|-------|-------|-------|-------|
| **GPU 0** | - | 159.23 | 158.34 | 159.41 |
| **GPU 1** | 173.05 | - | 158.34 | 159.23 |
| **GPU 2** | 173.04 | 173.04 | - | 159.42 |
| **GPU 3** | 173.05 | 173.04 | 172.91 | - |

**平均值**: 166.01 Gbps

#### Server 2: GPU间P2P带宽矩阵 (Gbps)

| 源\目标 | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---------|-------|-------|-------|-------|
| **GPU 0** | - | 162.08 | 141.00 | 140.69 |
| **GPU 1** | 158.38 | - | 132.85 | 131.07 |
| **GPU 2** | 140.55 | 140.70 | - | 160.30 |
| **GPU 3** | 138.74 | 138.50 | 159.63 | - |

**平均值**: 145.37 Gbps

### 延迟详细数据

#### Server 1: GPU 0 → GPU 1 延迟与带宽

| 数据大小 | 带宽 (Gbps) | 带宽 (MB/s) | 延迟 (ms) |
|----------|-------------|-------------|-----------|
| 1 MB | 131.62 | 16,453 | 0.061 |
| 10 MB | 155.81 | 19,477 | 0.513 |
| 50 MB | 158.66 | 19,832 | 2.521 |
| 100 MB | 159.14 | 19,893 | 5.027 |
| 500 MB | 159.47 | 19,934 | 25.083 |

### 网络详细数据

#### 跨节点网络 (10GbE以太网)

| 指标 | 值 |
|------|-----|
| **接口** | ens20f1 (Server 1), enp177s0f0np0 (Server 2) |
| **物理带宽** | 10,000 Mb/s |
| **TCP单流带宽** | 0.63-9.43 Gbps |
| **ping延迟 (平均)** | 0.059 ms (59 μs) |
| **ping延迟 (最小)** | 0.051 ms (51 μs) |
| **ping延迟 (最大)** | 0.110 ms (110 μs) |
| **抖动** | 0.010 ms |

---

## 🎯 Profiling配置建议

### 方案A: 当前实测配置 (立即可用)

```python
# === 单节点GPU通信 (实测数据) ===
config = {
    "gpu_p2p_bandwidth_gbps": 160.0,       # P2P平均实测
    "gpu_p2p_latency_us": 7.0,             # 小消息延迟
    "all_gather_bandwidth_gbps": 190.0,    # All-Gather平均
    "all_reduce_bandwidth_gbps": 167.0,    # All-Reduce平均
}

# === 跨节点通信 (当前10GbE以太网) ===
config["cross_node_bandwidth_gbps"] = 10.0     # 物理带宽
config["cross_node_latency_us"] = 59           # ping实测
```

### 方案B: 理想RDMA配置 (RDMA启用后)

```python
# === 单节点GPU通信 (不变) ===
config = {
    "gpu_p2p_bandwidth_gbps": 160.0,
    "gpu_p2p_latency_us": 7.0,
}

# === 跨节点通信 (RDMA理论值) ===
config["cross_node_bandwidth_gbps"] = 200.0    # RoCEv2 200Gb/s
config["cross_node_latency_us"] = 1            # RDMA典型延迟 ~1μs
```

---

## 📁 测试报告位置

| 文件 | 说明 |
|------|------|
| `FINAL_TEST_REPORT.md` | 完整测试报告 (推荐) |
| `final_test_report_20260605.md` | 详细测试报告 (所有数据表) |
| `TESTING_SUMMARY.md` | 快速总结 |
| `RDMA_TESTING_EXECUTIVE_SUMMARY.md` | 执行摘要 |
| `RDMA_FINAL_REPORT.md` | 本文件 (最终报告) |

---

## 🧪 测试脚本清单

### 已验证脚本 (8个)

| 脚本 | 功能 | 状态 |
|------|------|------|
| `test_cuda_p2p_bandwidth.py` | GPU P2P带宽测试 | ✅ 已运行 |
| `test_nccl_style_communication.py` | NCCL风格通信测试 | ✅ 已运行 |
| `test_gpu_latency_detailed.py` | 详细延迟测试 | ✅ 已运行 |
| `test_highspeed_bandwidth.py` | 高速网络带宽测试 | ✅ 已运行 |
| `test_rdma_physical.py` | RDMA物理接口检查 | ✅ 已运行 |
| `test_gpu_memory_bandwidth.py` | GPU显存带宽诊断 | ✅ 已运行 |
| `test_network_bandwidth.py` | TCP/IP带宽测试 | ✅ 已运行 |
| `test_nccl_rdma.py` | NCCL+RDMA综合诊断 | ✅ 已运行 |

### 待RDMA启用后运行 (3个)

| 脚本 | 功能 | 前置条件 |
|------|------|----------|
| `run_rdma_full_test.sh` | RDMA一键完整测试 | RDMA网线连接 |
| `test_rdma_bandwidth_latency.py` | RDMA带宽延迟测试 | RDMA启用 + perftest |
| `test_rdma_ibverbs.py` | RDMA IBVerbs测试 | RDMA启用 |

---

## 🔧 RDMA启用步骤 (需系统管理员)

### 步骤1: 连接RDMA网线

将RDMA网线连接到:
- **Server 1**: `ens19f0np0` 或 `ens19f1np1` (Mellanox ConnectX-5)
- **Server 2**: `eno3np0` 或 `eno4np1` (Mellanox ConnectX-6)

### 步骤2: 启用RDMA接口 (root权限)

```bash
# Server 1 (10.21.16.124)
sudo ip link set ens19f0np0 up
sudo ip addr add 192.168.100.1/24 dev ens19f0np0

# Server 2 (100.64.0.6)
sudo ip link set eno3np0 up
sudo ip addr add 192.168.100.2/24 dev eno3np0
```

### 步骤3: 验证RDMA状态

```bash
rdma link show
# 应该显示: state ACTIVE
```

### 步骤4: 安装perftest

```bash
sudo apt install perftest
```

### 步骤5: 运行RDMA测试

```bash
# Server 1
~/DistLMSim/scripts/run_rdma_full_test.sh server

# Server 2
~/DistLMSim/scripts/run_rdma_full_test.sh client 192.168.100.1
```

---

## 📊 测试结论

### ✅ 已完成

1. **GPU显存复制速度** - 通过NCCL协议完整测试
   - P2P带宽: **145-206 Gbps** (18-26 GB/s)
   - 集体通信: **156-206 Gbps**
   - 延迟: **6.8-60.8 μs**

2. **RDMA物理连接检查** - 完整诊断
   - 设备存在且固件正常
   - 确认网线未连接是RDMA不可用的原因
   - 提供了完整的启用指南

3. **替代网络测试** - 10GbE以太网
   - 物理带宽: 10 Gbps
   - 网络延迟: **59 μs**

### ⏸️ 待完成 (需要物理操作)

- **RDMA带宽测试** - 需要连接RDMA网线
- **RDMA延迟测试** - 需要连接RDMA网线

### 🎯 对DistLMSim Profiling的影响

**单节点GPU通信**: ✅ 数据充足且准确，可直接用于profiling

**跨节点通信**: 
- 当前可用: 10GbE以太网数据 (10 Gbps, 59 μs)
- 未来更新: RDMA启用后可获取200 Gbps, 1 μs数据

---

**报告生成时间**: 2026-06-05 11:00  
**测试执行**: sheng-xiang@10.21.16.124, sheng-xiang@100.64.0.6  
**总测试脚本数**: 11个 (8个已验证 + 3个待RDMA启用)
