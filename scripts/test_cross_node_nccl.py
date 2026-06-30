#!/usr/bin/env python3
"""
跨节点GPU-to-GPU通信测试 - 使用NCCL通过RDMA网络

测试从一台服务器的GPU到另一台服务器的GPU的显存数据传输

Usage:
  # 设置环境变量
  export NCCL_SOCKET_IFNAME=ens20f1  # 使用10GbE以太网作为控制平面
  export NCCL_IB_DISABLE=0           # 启用RDMA
  export NCCL_IB_HCA=mlx5            # 指定RDMA设备
  export NCCL_NET_GDR_LEVEL=2        # GPU Direct RDMA
  
  # Server 1 (rank 0)
  python3 test_cross_node_nccl.py --rank 0 --world-size 2 --master-ip 10.21.16.124
  
  # Server 2 (rank 1)
  python3 test_cross_node_nccl.py --rank 1 --world-size 2 --master-ip 10.21.16.124
"""

import os
import sys
import time
import torch
import torch.distributed as dist


def test_cross_node_gpu_communication(rank, world_size, master_ip):
    """测试跨节点GPU通信"""
    
    # 设置分布式环境
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_DISABLE'] = '0'
    os.environ['NCCL_IB_HCA'] = 'mlx5,roce'
    os.environ['NCCL_SOCKET_IFNAME'] = 'ens20f1,enp177s0f0np0'  # 控制平面
    os.environ['NCCL_NET_GDR_LEVEL'] = '2'  # GPUDirect RDMA
    
    print(f"Rank {rank}: 初始化NCCL...")
    print(f"  MASTER_ADDR={master_ip}")
    print(f"  NCCL_IB_DISABLE={os.environ['NCCL_IB_DISABLE']}")
    print(f"  NCCL_IB_HCA={os.environ['NCCL_IB_HCA']}")
    
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    num_gpus = torch.cuda.device_count()
    local_gpu = rank % num_gpus
    
    print(f"\nRank {rank}: 使用GPU {local_gpu}")
    print(f"  GPU: {torch.cuda.get_device_name(local_gpu)}")
    print(f"  显存: {torch.cuda.get_device_properties(local_gpu).total_mem / 1e9:.1f} GB")
    
    # 测试不同大小的数据传输
    test_sizes = [
        (1, "1 MB"),
        (10, "10 MB"),
        (100, "100 MB"),
    ]
    
    print(f"\n{'='*60}")
    print("跨节点GPU P2P通信测试")
    print(f"{'='*60}")
    
    for size_mb, size_str in test_sizes:
        size_bytes = int(size_mb * 1024 * 1024)
        num_elements = size_bytes // 4
        
        # 创建GPU张量
        if rank == 0:
            tensor = torch.randn(num_elements, dtype=torch.float32, device=f'cuda:{local_gpu}')
        else:
            tensor = torch.empty(num_elements, dtype=torch.float32, device=f'cuda:{local_gpu}')
        
        torch.cuda.synchronize()
        
        # 预热
        if rank == 0:
            dist.send(tensor, dst=1)
        else:
            dist.recv(tensor, src=0)
        torch.cuda.synchronize()
        
        # 测试
        iterations = max(5, 50 // size_mb)
        start = time.perf_counter()
        
        for _ in range(iterations):
            if rank == 0:
                dist.send(tensor, dst=1)
            else:
                dist.recv(tensor, src=0)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        if rank == 0:
            avg_latency_ms = (elapsed / iterations) * 1000
            bandwidth_gbps = (size_mb * iterations * 8) / (elapsed * 1e3)
            bandwidth_mbps = (size_mb * iterations) / elapsed
            
            print(f"  {size_str:8s}: "
                  f"Bandwidth = {bandwidth_gbps:7.2f} Gbps ({bandwidth_mbps:8.1f} MB/s), "
                  f"Latency = {avg_latency_ms:6.3f} ms")
        
        dist.barrier()
    
    print(f"\n{'='*60}")
    print("跨节点NCCL All-Reduce测试")
    print(f"{'='*60}")
    
    # All-Reduce测试
    data_size_mb = 100
    size_bytes = int(data_size_mb * 1024 * 1024)
    num_elements = size_bytes // 4
    
    tensor = torch.randn(num_elements, dtype=torch.float32, device=f'cuda:{local_gpu}')
    
    # 预热
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    
    # 测试
    iterations = 10
    start = time.perf_counter()
    
    for _ in range(iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    if rank == 0:
        effective_data_mb = data_size_mb * 2 * (world_size - 1) / world_size * iterations
        bandwidth_gbps = (effective_data_mb * 8) / (elapsed * 1e3)
        print(f"  All-Reduce (100 MB): {bandwidth_gbps:.2f} Gbps")
        print(f"  Per iteration: {elapsed / iterations * 1000:.2f} ms")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-ip", type=str, required=True)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("跨节点GPU-to-GPU NCCL通信测试")
    print("=" * 60)
    print(f"Host: {os.uname().nodename}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Rank: {args.rank}, World Size: {args.world_size}\n")
    
    test_cross_node_gpu_communication(args.rank, args.world_size, args.master_ip)
