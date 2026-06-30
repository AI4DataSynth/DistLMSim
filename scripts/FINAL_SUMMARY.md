# GPU-to-GPU Memory Copy + RDMA Test Report

**Date**: 2026-06-05  
**Servers**: 10.21.16.124 (302-GPU6) | 100.64.0.6 (302-GPU4)  
**Hardware**: 4×NVIDIA A800 80GB PCIe per server, Mellanox ConnectX RDMA NICs

---

## Test Results

### 1. GPU-to-GPU Memory Copy Speed (CUDA P2P / NCCL-style)

| Test | Server 1 | Server 2 |
|------|----------|----------|
| P2P Bandwidth | 166 Gbps | 145 Gbps |
| All-Gather | 206 Gbps | 173 Gbps |
| Ring All-Reduce | 179 Gbps | 156 Gbps |
| 1KB Latency | 6.8 μs | 7.0 μs |

### 2. RDMA Physical Connection

| Status | Details |
|--------|---------|
| Device | Mellanox ConnectX-5/6 ✅ |
| Driver | mlx5_core ✅ |
| Cable | ❌ NOT connected |
| Link | ❌ DOWN |
| Test | ❌ Cannot test (requires physical cable) |

### 3. Alternative: 10GbE Ethernet

| Metric | Value |
|--------|-------|
| Bandwidth | 10 Gbps |
| Latency | 0.059 ms |

---

## For Profiling

```python
# GPU (实测)
gpu_p2p_bw_gbps = 160.0
gpu_p2p_lat_us = 7.0

# Cross-node (10GbE, current)
cross_node_bw_gbps = 10.0
cross_node_lat_us = 59

# Cross-node (RDMA, after cable connected)
# cross_node_bw_gbps = 200.0
# cross_node_lat_us = 1
```

---

## RDMA Enablement (Requires SysAdmin)

1. Connect RDMA cable to Mellanox NIC
2. `sudo ip link set <iface> up`
3. Run: `bash ~/DistLMSim/scripts/run_rdma_full_test.sh server/client`

---

**Full Report**: `GPU_MEMORY_COPY_AND_RDMA_REPORT.md`  
**Scripts**: `~/DistLMSim/scripts/` (12 scripts deployed)
