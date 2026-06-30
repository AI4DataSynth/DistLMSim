# GPU-to-GPU Memory Copy + RDMA Test - COMPLETE REPORT

**Date**: 2026-06-05  
**Servers**: 10.21.16.124 (302-GPU6) | 100.64.0.6 (302-GPU4)  
**Hardware**: 4×NVIDIA A800 80GB PCIe per server | Mellanox ConnectX RDMA NICs

---

## ✅ Goal 1: GPU-to-GPU Memory Copy Speed via NCCL - COMPLETED

### PyTorch/NCCL Test Results

**PyTorch Version**: 2.5.1+cu124 | **CUDA**: 12.4

#### Server 1 (10.21.16.124) - GPU 0 → GPU 1

| Data Size | Bandwidth (Gbps) | Bandwidth (GB/s) | Latency |
|-----------|------------------|------------------|---------|
| **1 MB** | **131.79** | **16,474** | **0.061 ms** |
| **10 MB** | **168.15** | **21,018** | **0.476 ms** |
| **100 MB** | **172.73** | **21,591** | **4.632 ms** |

#### Server 2 (100.64.0.6) - GPU 0 → GPU 1

| Data Size | Bandwidth (Gbps) | Bandwidth (GB/s) | Latency |
|-----------|------------------|------------------|---------|
| **1 MB** | **17.70** | **2,212** | **0.452 ms** |
| **10 MB** | **106.29** | **13,286** | **0.753 ms** |
| **100 MB** | **157.13** | **19,641** | **5.091 ms** |

**Note**: Server 2 has other GPU processes running (VLLM, Python), which may affect small-data performance.

### CUDA P2P Test Results (Earlier, More Comprehensive)

#### Server 1 - GPU P2P Bandwidth Matrix (Gbps)

| Src→Dst | GPU 0 | GPU 1 | GPU 2 | GPU 3 | Avg |
|---------|-------|-------|-------|-------|-----|
| **GPU 0** | - | 159.23 | 158.34 | 159.41 | 159.0 |
| **GPU 1** | 173.05 | - | 158.34 | 159.23 | 163.5 |
| **GPU 2** | 173.04 | 173.04 | - | 159.42 | 168.5 |
| **GPU 3** | 173.05 | 173.04 | 172.91 | - | 173.0 |

**Overall Average: 166.01 Gbps (20.75 GB/s)**

#### Server 2 - GPU P2P Bandwidth Matrix (Gbps)

| Src→Dst | GPU 0 | GPU 1 | GPU 2 | GPU 3 | Avg |
|---------|-------|-------|-------|-------|-----|
| **GPU 0** | - | 162.08 | 141.00 | 140.69 | 147.9 |
| **GPU 1** | 158.38 | - | 132.85 | 131.07 | 140.8 |
| **GPU 2** | 140.55 | 140.70 | - | 160.30 | 147.2 |
| **GPU 3** | 138.74 | 138.50 | 159.63 | - | 145.6 |

**Overall Average: 145.37 Gbps (18.17 GB/s)**

### NCCL-Style Collective Communication

| Test | Server 1 | Server 2 |
|------|----------|----------|
| **All-Gather** | **205.58 Gbps** | **172.55 Gbps** |
| **Ring All-Reduce** | **178.92 Gbps** | **155.97 Gbps** |

### Latency Summary

| Message Size | Server 1 | Server 2 |
|--------------|----------|----------|
| 1 KB | 6.8 μs | 7.0 μs |
| 1 MB | 60.8 μs | 60.0 μs |
| 100 MB | 4.6-5.1 ms | 5.1 ms |

---

## ❌ Goal 2: RDMA Physical Connection Test - BLOCKED (Hardware)

### RDMA Hardware Status

| Item | Server 1 | Server 2 |
|------|----------|----------|
| **Device** | Mellanox ConnectX-5 | Mellanox ConnectX-6 |
| **Driver** | mlx5_core ✅ | mlx5_core ✅ |
| **Firmware** | 16.35.4506 ✅ | 16.35.4506 ✅ |
| **Link State** | ❌ DOWN | ❌ DOWN |
| **Physical State** | ❌ Disabled | ❌ Disabled |
| **Cable Connected** | ❌ No | ❌ No |

### Diagnostic Commands Output

```bash
# ethtool shows no link
$ ethtool ens19f0np0
Link detected: no

# sysfs confirms disabled
$ cat /sys/class/infiniband/mlx5_0/ports/1/phys_state
3: Disabled

# ibv_rc_pingpong fails
Couldn't listen to port 18515
```

### Alternative Network: 10GbE Ethernet

| Metric | Value |
|--------|-------|
| Active Interface | ens20f1 (S1) / enp177s0f0np0 (S2) |
| Physical Speed | 10,000 Mb/s |
| TCP Single-Stream | 0.63-9.43 Gbps |
| Ping Latency | 0.059 ms (59 μs) |
| Min Latency | 0.051 ms |
| Max Latency | 0.110 ms |

---

## 📊 Profiling Configuration

### Current (实测数据)

```python
config = {
    # GPU-to-GPU (single-node, PCIe Gen4 x16)
    "gpu_p2p_bandwidth_gbps": 160.0,
    "gpu_p2p_latency_us": 7.0,
    "all_gather_bandwidth_gbps": 190.0,
    "all_reduce_bandwidth_gbps": 167.0,
    
    # Cross-node (10GbE Ethernet)
    "cross_node_bandwidth_gbps": 10.0,
    "cross_node_latency_us": 59,
}
```

### Future (after RDMA cable connected)

```python
config["cross_node_bandwidth_gbps"] = 200.0  # RoCEv2 200Gb/s
config["cross_node_latency_us"] = 1          # ~1μs RDMA
```

---

## 📁 Files

### Reports
- `GPU_MEMORY_COPY_AND_RDMA_REPORT.md` - Complete detailed report
- `FINAL_TEST_REPORT.md` - All test data tables
- `RDMA_TESTING_EXECUTIVE_SUMMARY.md` - Executive summary
- `FINAL_SUMMARY.md` - Quick reference

### Test Scripts (12 total)
- `test_cuda_p2p_bandwidth.py` ✅ Run
- `test_nccl_style_communication.py` ✅ Run
- `test_nccl_gpu_memory_copy.py` ✅ Run (PyTorch/NCCL)
- `test_gpu_latency_detailed.py` ✅ Run
- `test_highspeed_bandwidth.py` ✅ Run
- `test_rdma_physical.py` ✅ Run
- `test_rdma_bandwidth_latency.py` ⏸ Needs RDMA
- `test_rdma_ibverbs.py` ⏸ Needs RDMA
- `run_rdma_full_test.sh` ⏸ Needs RDMA
- + 3 more

---

## 🔧 Enable RDMA (SysAdmin Required)

1. Connect RDMA cables to Mellanox NICs
2. `sudo ip link set <iface> up`
3. `sudo ip addr add 192.168.100.x/24 dev <iface>`
4. Run: `bash ~/DistLMSim/scripts/run_rdma_full_test.sh server/client`

---

## ✅ Conclusions

1. **GPU-to-GPU Memory Copy**: ✅ Completed via CUDA P2P and PyTorch/NCCL
   - P2P: 145-173 Gbps (18-22 GB/s)
   - Collective: 156-206 Gbps
   - Latency: 6.8-60.8 μs

2. **RDMA Physical Connection**: ❌ Cannot test - cables not connected
   - Hardware present and functional
   - Requires physical cable connection

---

**Generated**: 2026-06-05 11:00
