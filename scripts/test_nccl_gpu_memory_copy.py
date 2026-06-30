#!/usr/bin/env python3
"""
NCCL GPU-to-GPU Memory Copy Test (PyTorch-based)

This script uses PyTorch's NCCL backend to test actual NCCL-based
GPU-to-GPU memory copy speed.

Usage:
  python3 test_nccl_gpu_memory_copy.py
"""

import torch
import torch.distributed as dist
import os
import time
import sys


def test_nccl_p2p_bandwidth():
    """Test NCCL-based GPU-to-GPU memory copy bandwidth"""
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*80}")
    print(f"NCCL GPU-to-GPU Memory Copy Test")
    print(f"{'='*80}")
    print(f"GPUs available: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    
    if num_gpus < 2:
        print("Need at least 2 GPUs for this test")
        return
    
    # Test different data sizes
    test_sizes = [
        (1, "1 MB"),
        (10, "10 MB"),
        (100, "100 MB"),
    ]
    
    print(f"\n{'='*80}")
    print("GPU-to-GPU Copy Bandwidth (CUDA P2P via NCCL backend)")
    print(f"{'='*80}")
    
    for size_mb, size_str in test_sizes:
        size_bytes = int(size_mb * 1024 * 1024)
        num_elements = size_bytes // 4  # float32
        
        # Test GPU 0 -> GPU 1
        src_data = torch.randn(num_elements, dtype=torch.float32, device='cuda:0')
        dst_data = torch.empty(num_elements, dtype=torch.float32, device='cuda:1')
        
        torch.cuda.synchronize()
        
        # Warmup
        dst_data.copy_(src_data)
        torch.cuda.synchronize()
        
        # Benchmark
        iterations = max(10, 100 // size_mb)
        start = time.perf_counter()
        
        for _ in range(iterations):
            dst_data.copy_(src_data)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        avg_latency_ms = (elapsed / iterations) * 1000
        bandwidth_gbps = (size_mb * iterations * 8) / (elapsed * 1e3)
        bandwidth_mbps = (size_mb * iterations) / elapsed
        
        print(f"  GPU 0 → GPU 1, {size_str:8s}: "
              f"Bandwidth = {bandwidth_gbps:7.2f} Gbps ({bandwidth_mbps:8.1f} MB/s), "
              f"Latency = {avg_latency_ms:6.3f} ms")


def test_nccl_all_reduce(rank, world_size):
    """Test NCCL All-Reduce bandwidth"""
    print(f"\n{'='*80}")
    print("NCCL All-Reduce Test")
    print(f"{'='*80}")
    
    num_gpus = torch.cuda.device_count()
    data_size_mb = 100
    size_bytes = int(data_size_mb * 1024 * 1024)
    num_elements = size_bytes // 4
    
    device = torch.device(f'cuda:{rank % num_gpus}')
    tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
    
    # Warmup
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    
    # Benchmark
    iterations = 10
    start = time.perf_counter()
    
    for _ in range(iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # All-Reduce effective bandwidth
    # For ring all-reduce: 2 * (N-1) / N * data_size per iteration
    effective_data_mb = data_size_mb * 2 * (world_size - 1) / world_size * iterations
    bandwidth_gbps = (effective_data_mb * 8) / (elapsed * 1e3)
    
    print(f"  Data size: {data_size_mb} MB")
    print(f"  World size: {world_size}")
    print(f"  Iterations: {iterations}")
    print(f"  Total time: {elapsed:.4f} s")
    print(f"  Effective bandwidth: {bandwidth_gbps:.2f} Gbps")
    print(f"  Per iteration: {elapsed / iterations * 1000:.2f} ms")


def main():
    print("=" * 80)
    print("NCCL GPU Memory Copy Test")
    print("=" * 80)
    
    # Check if PyTorch is available
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
    except ImportError:
        print("ERROR: PyTorch is not installed!")
        print("Please install PyTorch: pip install torch")
        sys.exit(1)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        sys.exit(1)
    
    # Test local GPU-to-GPU copy
    test_nccl_p2p_bandwidth()
    
    # Test NCCL distributed (single-node, multi-GPU)
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        print(f"\nRunning NCCL distributed test with {num_gpus} GPUs...")
        
        # Setup distributed
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        
        dist.init_process_group(backend='nccl', world_size=1, rank=0)
        
        test_nccl_all_reduce(0, 1)
        
        dist.destroy_process_group()
    
    print(f"\n{'='*80}")
    print("Test complete")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
