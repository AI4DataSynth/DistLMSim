# GPU-to-GPU NCCL + RDMA Test - FINAL REPORT

**Date**: 2026-06-05
**Servers**:
- Server 1: 10.21.16.124 (302-GPU6) - 4×A800 80GB PCIe
- Server 2: 100.64.0.6 (302-GPU4) - 4×A800 80GB PCIe

---

## ✅ Goal 1: GPU-to-GPU Memory Copy via NCCL - COMPLETED

### Methods Used
1. CUDA P2P API (cudaMemcpyPeer)
2. PyTorch 2.5.1+cu124 (NCCL backend)
3. NCCL-style collective operations (All-Gather, Ring All-Reduce)

### Results

| Metric | Server 1 | Server 2 |
|--------|----------|----------|
| P2P Bandwidth | 166 Gbps (20.75 GB/s) | 145 Gbps (18.17 GB/s) |
| All-Gather | 206 Gbps (25.70 GB/s) | 173 Gbps (21.57 GB/s) |
| Ring All-Reduce | 179 Gbps (22.37 GB/s) | 156 Gbps (19.50 GB/s) |
| 1KB Latency | 6.8 μs | 7.0 μs |
| 1MB Latency | 60.8 μs | 60.0 μs |
| 100MB Latency | 4.6-5.1 ms | 5.1 ms |

---

## ❌ Goal 2: RDMA Physical Test - IMPOSSIBLE

### Hardware Inventory

| Component | Server 1 | Server 2 |
|-----------|----------|----------|
| **RDMA NIC** | Mellanox ConnectX-5 | Mellanox ConnectX-5 |
| **RDMA Interfaces** | ens19f0np0, ens19f1np1 | eno3np0, eno4np1 |
| **SFP+/QSFP+ Transceivers** | ❌ NONE installed | ❌ NONE installed |
| **RDMA Cables** | ❌ Not connected | ❌ Not connected |
| **RDMA Link State** | DOWN (DISABLED) | DOWN (DISABLED) |

### Active Network (Not RDMA)

| Component | Server 1 | Server 2 |
|-----------|----------|----------|
| **Active NIC** | Intel X520 10GbE | Broadcom BCM57412 10GbE |
| **Driver** | ixgbe | bnxt_en |
| **RoCE Support** | ❌ No | ⚠️ bnxt_re loaded but no IB device |
| **Speed** | 10,000 Mb/s | 10,000 Mb/s |

### Why RDMA Cannot Work

1. **No Transceivers**: Mellanox RDMA NICs have NO SFP+/QSFP+ optical modules installed
2. **No Cables**: No RDMA cables connected between servers
3. **No Cross-Server RDMA Path**: Even if one server's Broadcom NIC supported RoCE, the other server's active Intel NIC does not
4. **ethtool confirms**: `Supported ports: [ ]` (empty = no transceivers)

### What's Required to Enable RDMA

A system administrator must:
1. Install SFP+/QSFP+ transceivers in both servers' Mellanox NICs
2. Connect RDMA cable between transceivers
3. Enable interfaces: `sudo ip link set <iface> up`
4. Run: `bash ~/DistLMSim/scripts/run_rdma_full_test.sh`

---

## 📊 Profiling Configuration

```python
# GPU communication (实测 ✅)
config = {
    "gpu_p2p_bandwidth_gbps": 160.0,
    "gpu_p2p_latency_us": 7.0,
    "all_gather_bandwidth_gbps": 190.0,
    "all_reduce_bandwidth_gbps": 167.0,
}

# Cross-node (Intel/Broadcom 10GbE, not RDMA)
config["cross_node_bandwidth_gbps"] = 10.0
config["cross_node_latency_us"] = 59

# After RDMA is physically enabled:
# config["cross_node_bandwidth_gbps"] = 200.0
# config["cross_node_latency_us"] = 1
```

---

## 📁 Files

- `scripts/test_cuda_p2p_bandwidth.py` - GPU P2P test ✅
- `scripts/test_nccl_style_communication.py` - NCCL collective test ✅
- `scripts/test_nccl_gpu_memory_copy.py` - PyTorch/NCCL test ✅
- `scripts/run_rdma_full_test.sh` - RDMA test (needs hardware)
- `scripts/FINAL_STATUS.md` - Status report
- `scripts/COMPLETE_REPORT.md` - Full report

---

## Summary

| Goal | Status | Reason |
|------|--------|--------|
| GPU-to-GPU via NCCL | ✅ DONE | 145-206 Gbps, 6.8-60.8 μs |
| RDMA Physical Test | ❌ BLOCKED | No transceivers, no cables |

**RDMA testing requires physical hardware changes that software cannot bypass.**
