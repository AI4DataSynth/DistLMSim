# GPU-to-GPU NCCL + RDMA Test Report

**Date**: 2026-06-05

---

## ✅ Goal 1: GPU-to-GPU Memory Copy (Single-Node via NCCL) - COMPLETED

| Test | Server 1 | Server 2 |
|------|----------|----------|
| P2P Bandwidth | 166 Gbps (20.75 GB/s) | 145 Gbps (18.17 GB/s) |
| All-Gather | 206 Gbps (25.70 GB/s) | 173 Gbps (21.57 GB/s) |
| All-Reduce | 179 Gbps (22.37 GB/s) | 156 Gbps (19.50 GB/s) |
| Latency (1KB) | 6.8 μs | 7.0 μs |

**This is within a single server (GPU to GPU via PCIe).**

---

## ✅ Goal 2: RDMA Physical Connection Test - COMPLETED

### What We Tested

**1. RDMA CPU-to-CPU Bandwidth (ib_write_bw):**
```
Server 1 CPU RAM → RDMA NIC (mlx5_1) → 25GbE → RDMA NIC (roceo3) → Server 2 CPU RAM
Result: 2709.48 MB/s (~21.7 Gbps)
Message size: 64 KB
```

**2. RDMA CPU-to-CPU Latency (ib_write_lat):**
```
Result: 1.34 μs (2 bytes)
```

### What This IS NOT

**This is NOT GPU-to-GPU over RDMA.** The ib_write_bw test measures:
- CPU memory → RDMA NIC → Network → RDMA NIC → CPU memory

### What GPU-to-GPU over RDMA Requires

For GPU-to-GPU memory copy via RDMA, you need:
1. **GPUDirect RDMA (GDR)** - allows RDMA NIC to directly read/write GPU memory
2. **NCCL** - handles GPU-to-GPU communication over RDMA
3. **NCCL Cross-Node Test** - requires both servers to run simultaneously

Our NCCL cross-node test timed out due to:
- NCCL initialization complexity
- Requires proper IB Subnet Manager or RoCE configuration
- May need additional RDMA routing setup

---

## 📊 Correct Profiling Configuration

### Single-Node (GPU to GPU via PCIe) - ✅ Verified
```python
gpu_p2p_bandwidth_gbps = 160.0
gpu_p2p_latency_us = 7.0
```

### Cross-Node RDMA - CPU-to-CPU (Verified)
```python
rdma_cpu_bw_gbps = 21.7      # ib_write_bw实测
rdma_cpu_latency_us = 1.34   # ib_write_lat实测
```

### Cross-Node GPU-to-GPU over RDMA - Estimated
```python
# GPU-to-GPU via RDMA = CPU-to-CPU RDMA bandwidth minus PCIe overhead
# Typical efficiency: ~60-80% of raw RDma bandwidth
rdma_gpu_bw_gbps = 15.0 - 18.0    # Estimated
rdma_gpu_latency_us = 2.0 - 5.0   # Estimated (higher due to GPU sync)
```

**To get actual GPU-to-GPU over RDMA measurements:**
1. Enable GPUDirect RDMA: `sudo modprobe nvidia_peermem`
2. Run NCCL test with proper configuration
3. Use `nccl-tests/all_reduce_perf`

---

## Summary

| Measurement | Status | Value |
|-------------|--------|-------|
| GPU-to-GPU (PCIe, single-node) | ✅ Verified | 145-206 Gbps |
| RDMA CPU-to-CPU (25GbE) | ✅ Verified | 21.7 Gbps, 1.34 μs |
| GPU-to-GPU over RDMA | ⚠️ Estimated | ~15-18 Gbps |

---

**Report**: 2026-06-05
