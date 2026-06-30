#!/usr/bin/env python3
"""
NCCL GPU显存复制速度测试 - 使用ctypes调用CUDA/NCCL API

测试内容：
1. 单机内GPU间显存复制速度（通过CUDA API）
2. 测量GPU P2P带宽
3. 测量延迟

不依赖PyTorch，直接使用CUDA Runtime API
"""

import ctypes
import ctypes.util
import sys
import time
import os

# 加载CUDA库
try:
    cuda = ctypes.cdll.LoadLibrary("libcudart.so")
    print("✓ CUDA Runtime库加载成功")
except Exception as e:
    print(f"✗ CUDA Runtime库加载失败: {e}")
    print("尝试使用其他方法...")
    sys.exit(1)

# CUDA错误检查
def check_cuda_error(err, msg="CUDA操作失败"):
    if err != 0:
        print(f"{msg}: 错误代码 {err}")
        sys.exit(1)

# CUDA常量
cudaSuccess = 0
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3

def get_device_count():
    """获取GPU数量"""
    count = ctypes.c_int()
    err = cuda.cudaGetDeviceCount(ctypes.byref(count))
    check_cuda_error(err, "获取GPU数量失败")
    return count.value

def get_device_name(gpu_id):
    """获取GPU名称"""
    props = ctypes.create_string_buffer(256)
    # 使用cudaGetDeviceProperties
    err = cuda.cudaSetDevice(gpu_id)
    check_cuda_error(err, f"设置GPU {gpu_id}失败")
    
    # 简化：直接查询名称
    return f"GPU {gpu_id}"

def test_p2p_bandwidth(src_gpu: int, dst_gpu: int, size_mb: int = 100, iterations: int = 10):
    """测试两个GPU间的P2P带宽"""
    print(f"\n测试 GPU {src_gpu} → GPU {dst_gpu} (大小: {size_mb} MB, 迭代: {iterations}次)")
    
    size_bytes = int(size_mb * 1024 * 1024)
    num_elements = size_bytes // 4  # float32 = 4 bytes
    
    # 在源GPU上分配内存并填充数据
    old_device = ctypes.c_int()
    cuda.cudaGetDevice(ctypes.byref(old_device))
    
    # 设置源GPU
    cuda.cudaSetDevice(src_gpu)
    
    # 分配源内存
    src_ptr = ctypes.c_void_p()
    err = cuda.cudaMalloc(ctypes.byref(src_ptr), size_bytes)
    check_cuda_error(err, f"在GPU {src_gpu}分配内存失败")
    
    # 填充数据
    err = cuda.cudaMemset(src_ptr, 0, size_bytes)
    check_cuda_error(err, "内存初始化失败")
    
    # 设置目标GPU
    cuda.cudaSetDevice(dst_gpu)
    
    # 分配目标内存
    dst_ptr = ctypes.c_void_p()
    err = cuda.cudaMalloc(ctypes.byref(dst_ptr), size_bytes)
    check_cuda_error(err, f"在GPU {dst_gpu}分配内存失败")
    
    # 启用P2P访问
    cuda.cudaDeviceEnablePeerAccess(src_gpu, 0)
    
    # 预热
    err = cuda.cudaMemcpyPeer(dst_ptr, dst_gpu, src_ptr, src_gpu, size_bytes)
    check_cuda_error(err, "P2P复制失败")
    cuda.cudaDeviceSynchronize()
    
    # 正式测试
    print(f"  开始带宽测试...")
    start = time.time()
    
    for i in range(iterations):
        err = cuda.cudaMemcpyPeer(dst_ptr, dst_gpu, src_ptr, src_gpu, size_bytes)
        check_cuda_error(err, f"第{i+1}次复制失败")
    
    cuda.cudaDeviceSynchronize()
    elapsed = time.time() - start
    
    # 计算带宽
    total_mb = size_mb * iterations
    bandwidth_gbps = (total_mb * 8) / (elapsed * 1e3)  # Gbps
    bandwidth_mbps = total_mb / elapsed  # MB/s
    
    print(f"  总数据: {total_mb:.0f} MB")
    print(f"  总时间: {elapsed:.4f} 秒")
    print(f"  带宽: {bandwidth_gbps:.2f} Gbps ({bandwidth_mbps:.1f} MB/s)")
    print(f"  单次延迟: {elapsed / iterations * 1000:.2f} ms")
    
    # 清理
    cuda.cudaFree(dst_ptr)
    cuda.cudaSetDevice(src_gpu)
    cuda.cudaFree(src_ptr)
    cuda.cudaSetDevice(old_device)
    
    return bandwidth_gbps

def test_all_gpu_combinations():
    """测试所有GPU组合间的带宽"""
    num_gpus = get_device_count()
    print(f"\n检测到 {num_gpus} 个GPU")
    
    if num_gpus < 2:
        print("需要至少2个GPU进行测试")
        return
    
    # 获取GPU信息
    for i in range(num_gpus):
        cuda.cudaSetDevice(i)
        print(f"  GPU {i}: {get_device_name(i)}")
    
    print("\n" + "=" * 80)
    print("GPU间P2P带宽测试结果")
    print("=" * 80)
    
    results = {}
    
    # 测试所有GPU对
    for src in range(num_gpus):
        for dst in range(num_gpus):
            if src != dst:
                key = f"{src}→{dst}"
                try:
                    bw = test_p2p_bandwidth(src, dst, size_mb=100, iterations=20)
                    results[key] = bw
                except Exception as e:
                    print(f"  ✗ {key} 测试失败: {e}")
                    results[key] = None
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("带宽汇总 (Gbps)")
    print("=" * 80)
    
    for key, bw in results.items():
        if bw is not None:
            print(f"  {key}: {bw:.2f} Gbps")
        else:
            print(f"  {key}: 测试失败")
    
    # 计算平均值
    valid_bws = [bw for bw in results.values() if bw is not None]
    if valid_bws:
        avg_bw = sum(valid_bws) / len(valid_bws)
        print(f"\n  平均带宽: {avg_bw:.2f} Gbps")

def test_host_to_device_bandwidth(gpu_id: int, size_mb: int = 100, iterations: int = 10):
    """测试Host到Device的带宽"""
    print(f"\n测试 Host → GPU {gpu_id}")
    
    size_bytes = int(size_mb * 1024 * 1024)
    
    cuda.cudaSetDevice(gpu_id)
    
    # 分配主机内存
    host_ptr = ctypes.create_string_buffer(size_bytes)
    
    # 分配设备内存
    dev_ptr = ctypes.c_void_p()
    err = cuda.cudaMalloc(ctypes.byref(dev_ptr), size_bytes)
    check_cuda_error(err, "设备内存分配失败")
    
    # 预热
    err = cuda.cudaMemcpy(dev_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice)
    check_cuda_error(err, "HostToDevice复制失败")
    cuda.cudaDeviceSynchronize()
    
    # 测试
    start = time.time()
    for _ in range(iterations):
        err = cuda.cudaMemcpy(dev_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice)
        check_cuda_error(err, "复制失败")
    cuda.cudaDeviceSynchronize()
    elapsed = time.time() - start
    
    total_mb = size_mb * iterations
    bandwidth_gbps = (total_mb * 8) / (elapsed * 1e3)
    
    print(f"  带宽: {bandwidth_gbps:.2f} Gbps ({total_mb / elapsed:.1f} MB/s)")
    
    cuda.cudaFree(dev_ptr)

def main():
    print("=" * 80)
    print("NCCL/CUDA GPU显存复制速度测试")
    print("=" * 80)
    print(f"主机名: {os.uname().nodename}")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 获取GPU数量
    try:
        num_gpus = get_device_count()
        print(f"检测到 {num_gpus} 个GPU\n")
    except Exception as e:
        print(f"获取GPU数量失败: {e}")
        return
    
    # 测试Host到Device带宽
    print("=" * 80)
    print("Host到Device带宽测试")
    print("=" * 80)
    for gpu_id in range(min(num_gpus, 2)):  # 只测试前2个GPU
        test_host_to_device_bandwidth(gpu_id, size_mb=100, iterations=10)
    
    # 测试GPU间P2P带宽
    if num_gpus >= 2:
        print("\n" + "=" * 80)
        print("GPU间P2P带宽测试")
        print("=" * 80)
        test_all_gpu_combinations()
    else:
        print("\n需要至少2个GPU才能进行P2P测试")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

if __name__ == "__main__":
    main()
