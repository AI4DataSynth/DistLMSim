#!/usr/bin/env python3
"""
GPU间显存复制延迟和带宽精确测试

使用CUDA API进行细粒度的延迟和带宽测量
"""

import ctypes
import sys
import time
import os

# 加载CUDA库
try:
    cuda = ctypes.cdll.LoadLibrary("libcudart.so")
except Exception as e:
    print(f"CUDA加载失败: {e}")
    sys.exit(1)

cudaSuccess = 0
cudaMemcpyDeviceToDevice = 3

def check_cuda_error(err, msg="CUDA操作失败"):
    if err != 0:
        raise RuntimeError(f"{msg}: 错误代码 {err}")

def test_detailed_bandwidth(src_gpu: int, dst_gpu: int):
    """详细的带宽和延迟测试"""
    print(f"\n{'='*80}")
    print(f"测试 GPU {src_gpu} → GPU {dst_gpu}")
    print(f"{'='*80}")
    
    old_device = ctypes.c_int()
    cuda.cudaGetDevice(ctypes.byref(old_device))
    
    cuda.cudaSetDevice(src_gpu)
    
    # 测试不同数据大小
    test_sizes = [
        (1, "1 MB"),
        (10, "10 MB"),
        (50, "50 MB"),
        (100, "100 MB"),
        (500, "500 MB"),
    ]
    
    for size_mb, size_str in test_sizes:
        size_bytes = int(size_mb * 1024 * 1024)
        
        # 分配内存
        src_ptr = ctypes.c_void_p()
        cuda.cudaMalloc(ctypes.byref(src_ptr), size_bytes)
        cuda.cudaMemset(src_ptr, 0, size_bytes)
        
        cuda.cudaSetDevice(dst_gpu)
        dst_ptr = ctypes.c_void_p()
        cuda.cudaMalloc(ctypes.byref(dst_ptr), size_bytes)
        
        # 启用P2P
        cuda.cudaDeviceEnablePeerAccess(src_gpu, 0)
        
        # 预热
        cuda.cudaMemcpyPeer(dst_ptr, dst_gpu, src_ptr, src_gpu, size_bytes)
        cuda.cudaDeviceSynchronize()
        
        # 测试延迟（单次复制）
        iterations = max(5, 100 // size_mb)
        
        start = time.perf_counter()
        for _ in range(iterations):
            cuda.cudaMemcpyPeer(dst_ptr, dst_gpu, src_ptr, src_gpu, size_bytes)
        cuda.cudaDeviceSynchronize()
        elapsed = time.perf_counter() - start
        
        avg_latency_ms = (elapsed / iterations) * 1000
        bandwidth_gbps = (size_mb * iterations * 8) / (elapsed * 1e3)
        bandwidth_mbps = (size_mb * iterations) / elapsed
        
        print(f"  {size_str:8s}: "
              f"带宽={bandwidth_gbps:7.2f} Gbps ({bandwidth_mbps:8.1f} MB/s), "
              f"延迟={avg_latency_ms:6.3f} ms, "
              f"迭代={iterations}")
        
        # 清理
        cuda.cudaFree(dst_ptr)
        cuda.cudaSetDevice(src_gpu)
        cuda.cudaFree(src_ptr)
    
    cuda.cudaSetDevice(old_device)

def main():
    print("=" * 80)
    print("GPU显存复制延迟和带宽精确测试")
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
    
    # 测试关键GPU对
    test_pairs = [
        (0, 1),  # 相邻GPU
        (0, 2),  # 跨NUMA节点
        (1, 3),  # 跨NUMA节点
    ]
    
    for src, dst in test_pairs:
        if src < num_gpus and dst < num_gpus:
            test_detailed_bandwidth(src, dst)
    
    print(f"\n{'='*80}")
    print("测试完成")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
