#!/usr/bin/env python3
"""
完整的NCCL风格GPU通信测试

模拟NCCL all-reduce和all-gather操作，测试GPU间通信性能
"""

import ctypes
import sys
import time
import os
import json

# 加载CUDA库
try:
    cuda = ctypes.cdll.LoadLibrary("libcudart.so")
except Exception as e:
    print(f"CUDA加载失败: {e}")
    sys.exit(1)

cudaSuccess = 0

def check_cuda(err, msg="CUDA操作失败"):
    if err != 0:
        raise RuntimeError(f"{msg}: 错误代码 {err}")

def test_all_gather_bandwidth(num_gpus: int = 4, data_size_mb: int = 100, iterations: int = 20):
    """
    测试All-Gather操作的带宽
    All-Gather: 每个GPU发送数据给所有其他GPU
    """
    print(f"\n{'='*80}")
    print(f"All-Gather带宽测试 ({num_gpus} GPU, {data_size_mb} MB, {iterations} 迭代)")
    print(f"{'='*80}")
    
    size_per_gpu = int(data_size_mb * 1024 * 1024 / num_gpus)
    total_size = size_per_gpu * num_gpus
    
    # 为每个GPU分配内存
    pointers = []
    for gpu_id in range(num_gpus):
        cuda.cudaSetDevice(gpu_id)
        
        ptr = ctypes.c_void_p()
        cuda.cudaMalloc(ctypes.byref(ptr), total_size)
        cuda.cudaMemset(ptr, 0, total_size)
        pointers.append(ptr)
    
    # 启用P2P访问
    for src in range(num_gpus):
        for dst in range(num_gpus):
            if src != dst:
                cuda.cudaSetDevice(dst)
                cuda.cudaDeviceEnablePeerAccess(src, 0)
    
    # 预热
    cuda.cudaSetDevice(0)
    cuda.cudaMemcpyPeer(pointers[1], 1, pointers[0], 0, size_per_gpu)
    cuda.cudaDeviceSynchronize()
    
    # 测试All-Gather
    print(f"  开始测试...")
    start = time.perf_counter()
    
    for _ in range(iterations):
        for src_gpu in range(num_gpus):
            for dst_gpu in range(num_gpus):
                if src_gpu != dst_gpu:
                    cuda.cudaSetDevice(dst_gpu)
                    cuda.cudaMemcpyPeer(
                        ctypes.c_void_p(
                            pointers[dst_gpu].value + src_gpu * size_per_gpu
                        ),
                        dst_gpu,
                        pointers[src_gpu],
                        src_gpu,
                        size_per_gpu
                    )
    
    # 同步所有GPU
    for gpu_id in range(num_gpus):
        cuda.cudaSetDevice(gpu_id)
        cuda.cudaDeviceSynchronize()
    
    elapsed = time.perf_counter() - start
    
    # 计算带宽
    # All-Gather总数据量 = num_gpus * (num_gpus - 1) * size_per_gpu
    total_data_mb = num_gpus * (num_gpus - 1) * (data_size_mb / num_gpus)
    effective_bandwidth_gbps = (total_data_mb * iterations * 8) / (elapsed * 1e3)
    
    print(f"  总数据交换: {total_data_mb:.0f} MB × {iterations} = {total_data_mb * iterations:.0f} MB")
    print(f"  总时间: {elapsed:.4f} 秒")
    print(f"  有效带宽: {effective_bandwidth_gbps:.2f} Gbps")
    print(f"  每次迭代: {elapsed / iterations * 1000:.2f} ms")
    
    # 清理
    for gpu_id in range(num_gpus):
        cuda.cudaSetDevice(gpu_id)
        cuda.cudaFree(pointers[gpu_id])
    
    return effective_bandwidth_gbps

def test_ring_allreduce_bandwidth(num_gpus: int = 4, data_size_mb: int = 100, iterations: int = 20):
    """
    测试Ring All-Reduce操作的带宽
    Ring All-Reduce: GPU形成环，数据在环中传递
    """
    print(f"\n{'='*80}")
    print(f"Ring All-Reduce带宽测试 ({num_gpus} GPU, {data_size_mb} MB, {iterations} 迭代)")
    print(f"{'='*80}")
    
    chunk_size = int(data_size_mb * 1024 * 1024 / num_gpus)
    total_size = chunk_size * num_gpus
    
    # 分配内存
    pointers = []
    for gpu_id in range(num_gpus):
        cuda.cudaSetDevice(gpu_id)
        ptr = ctypes.c_void_p()
        cuda.cudaMalloc(ctypes.byref(ptr), total_size)
        cuda.cudaMemset(ptr, 0, total_size)
        pointers.append(ptr)
    
    # 启用P2P
    for src in range(num_gpus):
        for dst in range(num_gpus):
            if src != dst:
                cuda.cudaSetDevice(dst)
                cuda.cudaDeviceEnablePeerAccess(src, 0)
    
    # 预热
    cuda.cudaSetDevice(0)
    cuda.cudaMemcpyPeer(pointers[1], 1, pointers[0], 0, chunk_size)
    cuda.cudaDeviceSynchronize()
    
    # 测试Ring传递
    print(f"  开始测试...")
    start = time.perf_counter()
    
    for _ in range(iterations):
        # Reduce-Scatter阶段
        for step in range(num_gpus - 1):
            for gpu_id in range(num_gpus):
                next_gpu = (gpu_id + 1) % num_gpus
                cuda.cudaSetDevice(next_gpu)
                cuda.cudaMemcpyPeer(
                    ctypes.c_void_p(pointers[next_gpu].value + step * chunk_size),
                    next_gpu,
                    pointers[gpu_id],
                    gpu_id,
                    chunk_size
                )
        
        # All-Gather阶段
        for step in range(num_gpus - 1):
            for gpu_id in range(num_gpus):
                prev_gpu = (gpu_id - 1) % num_gpus
                cuda.cudaSetDevice(gpu_id)
                cuda.cudaMemcpyPeer(
                    pointers[gpu_id],
                    gpu_id,
                    pointers[prev_gpu],
                    prev_gpu,
                    chunk_size
                )
    
    # 同步
    for gpu_id in range(num_gpus):
        cuda.cudaSetDevice(gpu_id)
        cuda.cudaDeviceSynchronize()
    
    elapsed = time.perf_counter() - start
    
    # Ring All-Reduce带宽计算
    total_data_mb = data_size_mb * 2 * (num_gpus - 1)
    effective_bandwidth_gbps = (total_data_mb * iterations * 8) / (elapsed * 1e3)
    
    print(f"  总数据移动: {total_data_mb:.0f} MB × {iterations}")
    print(f"  总时间: {elapsed:.4f} 秒")
    print(f"  有效带宽: {effective_bandwidth_gbps:.2f} Gbps")
    print(f"  每次迭代: {elapsed / iterations * 1000:.2f} ms")
    
    # 清理
    for gpu_id in range(num_gpus):
        cuda.cudaSetDevice(gpu_id)
        cuda.cudaFree(pointers[gpu_id])
    
    return effective_bandwidth_gbps

def test_point_to_point_latency(num_gpus: int = 4):
    """测试点对点延迟"""
    print(f"\n{'='*80}")
    print(f"点对点延迟测试 ({num_gpus} GPU)")
    print(f"{'='*80}")
    
    # 测试不同大小的消息
    message_sizes = [
        (1024, "1 KB"),
        (10 * 1024, "10 KB"),
        (100 * 1024, "100 KB"),
        (1024 * 1024, "1 MB"),
    ]
    
    for size_bytes, size_str in message_sizes:
        iterations = max(100, 10000 // (size_bytes // 1024))
        
        cuda.cudaSetDevice(0)
        src_ptr = ctypes.c_void_p()
        cuda.cudaMalloc(ctypes.byref(src_ptr), size_bytes)
        cuda.cudaMemset(src_ptr, 0, size_bytes)
        
        cuda.cudaSetDevice(1)
        dst_ptr = ctypes.c_void_p()
        cuda.cudaMalloc(ctypes.byref(dst_ptr), size_bytes)
        
        cuda.cudaDeviceEnablePeerAccess(0, 0)
        
        # 预热
        cuda.cudaMemcpyPeer(dst_ptr, 1, src_ptr, 0, size_bytes)
        cuda.cudaDeviceSynchronize()
        
        # 测试延迟
        start = time.perf_counter()
        for _ in range(iterations):
            cuda.cudaMemcpyPeer(dst_ptr, 1, src_ptr, 0, size_bytes)
        cuda.cudaDeviceSynchronize()
        elapsed = time.perf_counter() - start
        
        avg_latency_us = (elapsed / iterations) * 1e6
        
        print(f"  {size_str:8s}: 平均延迟 = {avg_latency_us:.1f} μs ({elapsed / iterations * 1000:.3f} ms), "
              f"迭代={iterations}")
        
        cuda.cudaFree(dst_ptr)
        cuda.cudaSetDevice(0)
        cuda.cudaFree(src_ptr)

def main():
    print("=" * 80)
    print("NCCL风格GPU通信测试")
    print("=" * 80)
    print(f"主机名: {os.uname().nodename}")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    count = ctypes.c_int()
    cuda.cudaGetDeviceCount(ctypes.byref(count))
    num_gpus = count.value
    
    print(f"检测到 {num_gpus} 个GPU\n")
    
    if num_gpus < 2:
        print("需要至少2个GPU")
        return
    
    # 1. 点对点延迟测试
    test_point_to_point_latency(num_gpus)
    
    # 2. All-Gather带宽测试
    test_all_gather_bandwidth(num_gpus, data_size_mb=100, iterations=20)
    
    # 3. Ring All-Reduce带宽测试
    test_ring_allreduce_bandwidth(num_gpus, data_size_mb=100, iterations=10)
    
    print(f"\n{'='*80}")
    print("测试完成")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
