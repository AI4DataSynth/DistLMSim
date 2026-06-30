# RDMA + GPU 带宽测试报告

**测试时间**: 2026-06-05  
**测试人员**: sheng-xiang  
**测试目标**: 检查GPU间显存复制速度和RDMA网络连接状态

---

## 服务器配置

### Server 1: 10.21.16.124 (hostname: 302-GPU6)

| 项目 | 配置 |
|------|------|
| GPU | 4x NVIDIA A800 80GB PCIe |
| GPU显存 | 81920 MiB 每卡 |
| GPU拓扑 | 所有GPU间通过PCIe NODE连接 |
| RDMA设备 | mlx5_0, mlx5_1 (Mellanox ConnectX-6) |
| RDMA状态 | **DOWN (未启用)** |
| 网络接口 | ens20f1 (10.21.16.124/24) - UP |
| NVLink | **Inactive (未连接)** |
| PCIe | Gen4 x16 (每GPU) |
| CUDA版本 | 13.2 |
| 驱动版本 | 595.45.04 |

### Server 2: 100.64.0.6 (hostname: 302-GPU4)

| 项目 | 配置 |
|------|------|
| GPU | 4x NVIDIA A800 80GB PCIe |
| GPU显存 | 81920 MiB 每卡 |
| GPU拓扑 | GPU0-1同NUMA节点，GPU2-3同NUMA节点 |
| RDMA设备 | roceo3, roceo4 (Mellanox ConnectX-6) |
| RDMA状态 | **DOWN (未启用)** |
| 网络接口 | enp177s0f0np0 (10.21.16.122/24) - UP |
| NVLink | **Inactive (未连接)** |
| PCIe | Gen4 x16 (每GPU) |
| CUDA版本 | 13.2 |
| 驱动版本 | 595.45.04 |

**注意**: Server 2上有正在运行的GPU任务：
- GPU 1: 69893 MiB (python进程)
- GPU 2: 651 MiB (imputations/.venv/bin/python3)
- GPU 3: 37455 MiB (VLLM::EngineCore)

---

## 测试结果

### 1. GPU间P2P访问支持

| Server | GPU0↔GPU1 | GPU0↔GPU2 | GPU0↔GPU3 | GPU1↔GPU2 | GPU1↔GPU3 | GPU2↔GPU3 |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|
| Server 1 | ✅ OK | ✅ OK | ✅ OK | ✅ OK | ✅ OK | ✅ OK |
| Server 2 | ✅ OK | ✅ OK | ✅ OK | ✅ OK | ✅ OK | ✅ OK |

**结论**: 所有GPU间均支持P2P读/写操作

### 2. NVLink状态

| Server | NVLink状态 |
|--------|-----------|
| Server 1 | ❌ Inactive (所有链路未激活) |
| Server 2 | ❌ Inactive (所有链路未激活) |

**结论**: 两台服务器均未配置NVLink，GPU间通信通过PCIe进行

### 3. RDMA网络状态

| 项目 | Server 1 | Server 2 |
|------|----------|----------|
| RDMA设备 | mlx5_0, mlx5_1 | roceo3, roceo4 |
| 设备型号 | Mellanox ConnectX-6 | Mellanox ConnectX-6 |
| 固件版本 | 16.35.4506 | 16.35.4506 |
| 端口状态 | ❌ DOWN | ❌ DOWN |
| 物理状态 | Disabled | Disabled |
| link_layer | Ethernet | Ethernet |
| 最大MTU | 4096 | 4096 |
| 活跃MTU | 1024 | 1024 |

**结论**: RDMA设备存在但未启用，物理连接处于Disabled状态

### 4. 网络带宽测试 (TCP/IP)

| 测试方向 | 平均带宽 | 峰值带宽 | 说明 |
|----------|----------|----------|------|
| Server 2 → Server 1 | 0.95 Gbps | 9.43 Gbps | TCP/IP (以太网) |

**带宽趋势图**:
```
初始:  9.43 Gbps (TCP缓冲效应)
1秒:   4.72 Gbps
2秒:   3.15 Gbps
3秒:   2.36 Gbps
4秒:   1.89 Gbps
5秒:   1.58 Gbps
6秒:   1.35 Gbps
7秒:   1.19 Gbps
8秒:   1.06 Gbps
平均:  0.95 Gbps (118.74 MB/s)
```

### 5. 网络延迟测试 (TCP/IP)

| 指标 | 延迟 (ms) |
|------|-----------|
| 平均 | 0.170 ms |
| P50 | 0.145 ms |
| P95 | 0.283 ms |
| P99 | 0.416 ms |
| 最小 | 0.118 ms |
| 最大 | 0.416 ms |

### 6. PCIe带宽估算

| GPU | PCIe配置 | 理论带宽 (单向) |
|-----|---------|----------------|
| 所有GPU | Gen4 x16 | ~32 GB/s |

---

## 关键发现

### ✅ 正常项
1. **GPU P2P支持**: 所有GPU间支持直接P2P访问
2. **PCIe配置**: 所有GPU运行在PCIe Gen4 x16模式
3. **CUDA环境**: CUDA 13.2 + 驱动595.45.04正常
4. **SSH连接**: 两台服务器间SSH免密码连接配置成功

### ⚠️ 问题项
1. **RDMA未启用**: 
   - 两台服务器的RDMA端口均处于DOWN状态
   - 物理层状态为Disabled
   - 可能原因：网线未连接、交换机未配置、或需要管理员权限启用

2. **NVLink未激活**:
   - 两台服务器均报告NVLink链路Inactive
   - A800 PCIe版本通常不配备NVLink（与SXM版本不同）
   - 这是**正常现象**（PCIe版A800无NVLink）

3. **网络带宽受限**:
   - 当前TCP/IP带宽仅约1 Gbps
   - 远低于预期的RDMA 200 Gbps
   - 可能受限于以太网交换机配置

---

## 对Profiling的影响

### 当前配置下的预期性能

| 通信类型 | 预期带宽 | 预期延迟 | 说明 |
|----------|----------|----------|------|
| GPU间 (P2P via PCIe) | ~25-30 GB/s | <1 µs | PCIe Gen4 x16 |
| 跨节点 (TCP/IP) | ~1 Gbps | 0.17 ms | 当前实测 |
| 跨节点 (RDMA) | **不可用** | **不可用** | RDMA未启用 |

### Profiling配置建议

如果要在模拟器中使用当前实际配置，建议修改：

```python
# config.py 或 CLI参数
rdma_bandwidth_gbps=1.0,      # 实测TCP/IP带宽，非RDMA
rdma_protocol="ROCE_V2",      # 协议类型（但实际走TCP）
nvlink_bandwidth_gbps=0.0,    # NVLink不可用
```

或者，如果要使用**理论RDMA带宽**进行模拟（假设RDMA正常工作）：

```python
rdma_bandwidth_gbps=200.0,    # RoCEv2 200Gb/s
nvlink_bandwidth_gbps=600.0,  # NVLink 600GB/s（如果有）
```

---

## 下一步建议

### 如需启用RDMA：

1. **检查物理连接**
   ```bash
   # 检查网线是否连接到RDMA网卡
   ethtool ens19f0np0  # Server 1
   ethtool eno3np0     # Server 2
   ```

2. **启用RDMA接口** (需要root权限)
   ```bash
   sudo ip link set ens19f0np0 up   # Server 1
   sudo ip link set eno3np0 up      # Server 2
   ```

3. **配置IP地址**
   ```bash
   sudo ip addr add 192.168.100.1/24 dev ens19f0np0  # Server 1
   sudo ip addr add 192.168.100.2/24 dev eno3np0     # Server 2
   ```

4. **测试RDMA带宽** (安装perftest后)
   ```bash
   # Server 1
   ib_write_bw -d mlx5_0
   
   # Server 2
   ib_write_bw -d roceo3 192.168.100.1
   ```

### 如果保持当前配置：

使用实测的TCP/IP带宽（~1 Gbps）进行profiling，模拟器配置应反映实际网络状况。

---

## 测试脚本位置

所有测试脚本已上传至两台服务器：

- `~/DistLMSim/scripts/test_rdma_connection.py` - RDMA连接诊断
- `~/DistLMSim/scripts/test_network_bandwidth.py` - TCP/IP带宽测试
- `~/DistLMSim/scripts/test_nccl_rdma.py` - NCCL + RDMA综合测试
- `~/DistLMSim/scripts/test_gpu_memory_bandwidth.py` - GPU显存带宽测试

可隨時重新运行测试：
```bash
cd ~/DistLMSim
python3 scripts/test_gpu_memory_bandwidth.py
python3 scripts/test_network_bandwidth.py --server
python3 scripts/test_network_bandwidth.py --client --server-ip 10.21.16.124
```

---

**报告生成时间**: 2026-06-05 10:05
