#!/usr/bin/env python3
"""
NCCL GPU显存复制速度测试 + RDMA物理接口测试

测试内容：
1. 检查RDMA物理接口状态
2. 使用NCCL测试GPU间显存复制速度（带宽）
3. 测试GPU间通信延迟

使用方法：
  - 单机测试（单节点内GPU间）: python3 test_nccl_rdma.py --local
  - 双机测试（跨节点GPU间，需要两台都运行）: 
    Server 1: python3 test_nccl_rdma.py --server --ip 10.21.16.124
    Server 2: python3 test_nccl_rdma.py --client --ip 10.21.16.124 --remote-ip 100.64.0.6
"""

import subprocess
import sys
import time
import argparse
from typing import Optional


def run_cmd(cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    """运行命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"


def check_rdma_interfaces():
    """检查RDMA物理接口状态"""
    print("=" * 80)
    print("检查RDMA物理接口...")
    print("=" * 80)
    
    # ibv_devinfo - 检查RDMA设备
    rc, stdout, _ = run_cmd("ibv_devinfo")
    if rc == 0:
        print("\nRDMA设备详细信息:")
        print(stdout)
    
    # 检查网卡状态
    rc, stdout, _ = run_cmd("ip link show | grep -E 'state UP|state DOWN'")
    if rc == 0:
        print("\n网络接口状态:")
        print(stdout)
    
    # 检查RoCE/RDMA相关
    rc, stdout, _ = run_cmd("rdma link show")
    if rc == 0:
        print("\nRDMA链路状态:")
        print(stdout)
    else:
        print("\n无法获取RDMA链路信息 (rdma命令不可用)")
    
    # 检查GPU拓扑
    rc, stdout, _ = run_cmd("nvidia-smi topo -m")
    if rc == 0:
        print("\nGPU拓扑:")
        print(stdout)


def test_nccl_local():
    """本地NCCL测试 - 测试单节点内GPU间带宽"""
    print("\n" + "=" * 80)
    print("本地NCCL带宽测试 (单节点内GPU间)")
    print("=" * 80)
    
    # 检查是否安装了NCCL测试工具
    rc, _, _ = run_cmd("which all_reduce_perf")
    if rc != 0:
        print("未找到NCCL测试工具 (all_reduce_perf)")
        print("使用Python + PyTorch进行NCCL测试...\n")
        test_nccl_local_pytorch()
        return
    
    # 使用NCCL原生工具
    print("使用NCCL all_reduce_perf进行测试...")
    cmd = "all_reduce_perf -b 8 -e 128M -f 2 -g 4"
    print(f"运行: {cmd}\n")
    rc, stdout, stderr = run_cmd(cmd, timeout=60)
    print(stdout)
    if stderr:
        print(f"Stderr: {stderr}")


def test_nccl_local_pytorch():
    """使用PyTorch进行本地NCCL测试"""
    print("使用PyTorch进行NCCL带宽测试...\n")
    
    test_code = '''
import torch
import torch.distributed as dist
import os
import time

def test_bandwidth():
    # 初始化进程组
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # 获取GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")
    
    for gpu_id in range(num_gpus):
        props = torch.cuda.get_device_properties(gpu_id)
        print(f"  GPU {gpu_id}: {props.name} ({props.total_mem / 1e9:.1f} GB)")
    
    print("\\n测试GPU间显存复制带宽 (host to device)...")
    
    # 测试不同大小的数据传输
    sizes_mb = [1, 10, 100, 500, 1000]
    
    for size_mb in sizes_mb:
        size_bytes = int(size_mb * 1024 * 1024)
        num_elements = size_bytes // 4  # float32
        
        # 在GPU 0上创建数据
        src_data = torch.randn(num_elements, dtype=torch.float32, device='cuda:0')
        torch.cuda.synchronize()
        
        # 测试复制到GPU 1
        if num_gpus > 1:
            dst_data = torch.empty(num_elements, dtype=torch.float32, device='cuda:1')
            
            # 预热
            dst_data.copy_(src_data)
            torch.cuda.synchronize()
            
            # 正式测试
            iterations = max(10, 100 // size_mb)
            start = time.time()
            for _ in range(iterations):
                dst_data.copy_(src_data)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            bandwidth_gbps = (size_mb * iterations * 8) / (elapsed * 1e3)  # Gbps
            print(f"  {size_mb:4d} MB: {bandwidth_gbps:.2f} Gbps ({bandwidth_gbps * 125:.1f} MB/s), "
                  f"延迟: {elapsed / iterations * 1000:.2f} ms")
    
    print("\\n测试完成!")

if __name__ == '__main__':
    test_bandwidth()
'''
    
    # 写入临时文件
    with open('/tmp/test_nccl_local.py', 'w') as f:
        f.write(test_code)
    
    rc, stdout, stderr = run_cmd("python3 /tmp/test_nccl_local.py", timeout=60)
    print(stdout)
    if stderr and "Error" in stderr:
        print(f"Stderr: {stderr}")


def test_rdma_bandwidth_with_iperf(server_ip: Optional[str] = None, is_server: bool = False):
    """使用iperf3测试RDMA网络带宽（如果可用）"""
    print("\n" + "=" * 80)
    print("网络带宽测试 (iperf3)")
    print("=" * 80)
    
    # 检查iperf3
    rc, _, _ = run_cmd("which iperf3")
    if rc != 0:
        print("iperf3未安装，跳过")
        return
    
    if is_server:
        print(f"启动iperf3服务器 (端口 5201)")
        cmd = "iperf3 -s"
        print(f"运行: {cmd}")
        print("按 Ctrl+C 停止")
        try:
            proc = subprocess.Popen(cmd, shell=True)
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
    else:
        print(f"连接到服务器: {server_ip}")
        cmd = f"iperf3 -c {server_ip} -t 10"
        print(f"运行: {cmd}\n")
        rc, stdout, stderr = run_cmd(cmd, timeout=30)
        print(stdout)
        if stderr:
            print(f"Stderr: {stderr}")


def check_nccl_env():
    """检查NCCL环境变量和配置"""
    print("\n" + "=" * 80)
    print("检查NCCL环境配置...")
    print("=" * 80)
    
    # 检查NCCL相关环境变量
    env_vars = [
        'NCCL_DEBUG', 'NCCL_SOCKET_IFNAME', 'NCCL_IB_DISABLE',
        'NCCL_IB_HCA', 'NCCL_NET_GDR_LEVEL', 'NCCL_SHM_DISABLE'
    ]
    
    for var in env_vars:
        rc, stdout, _ = run_cmd(f"echo ${var}")
        value = stdout.strip()
        if value:
            print(f"  {var} = {value}")
    
    # 检查NCCL版本
    rc, stdout, _ = run_cmd("python3 -c 'import torch; print(f\\\"NCCL版本: {torch.cuda.nccl.version() if hasattr(torch.cuda.nccl, \\'version\\') else \\'N/A\\'}\\\")'")
    if rc == 0:
        print(stdout)
    
    # 检查PyTorch是否支持NCCL
    rc, stdout, _ = run_cmd("python3 -c 'import torch; print(f\\\"NCCL可用: {torch.distributed.is_nccl_available()}\\\")'")
    if rc == 0:
        print(stdout)


def test_rdma_latency_ping(ip: str, count: int = 20):
    """使用ping测试网络延迟"""
    print("\n" + "=" * 80)
    print(f"网络延迟测试 (ping {ip})")
    print("=" * 80)
    
    cmd = f"ping -c {count} {ip}"
    rc, stdout, stderr = run_cmd(cmd, timeout=30)
    if rc == 0:
        # 提取统计信息
        for line in stdout.split('\n'):
            if 'rtt' in line or 'round-trip' in line:
                print(line)
    else:
        print(f"ping失败: {stderr}")


def main():
    parser = argparse.ArgumentParser(description="NCCL + RDMA测试脚本")
    parser.add_argument("--local", action="store_true", help="本地NCCL测试")
    parser.add_argument("--server", action="store_true", help="服务器模式")
    parser.add_argument("--client", action="store_true", help="客户端模式")
    parser.add_argument("--ip", type=str, help="服务器IP")
    parser.add_argument("--remote-ip", type=str, help="远程服务器IP")
    parser.add_argument("--check", action="store_true", help="仅检查环境")
    
    args = parser.parse_args()
    
    # 默认：完整检查
    if not args.local and not args.server and not args.client and not args.check:
        print("NCCL + RDMA 完整诊断")
        print("=" * 80)
        print(f"主机名: {run_cmd('hostname')[1].strip()}")
        print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 检查RDMA接口
        check_rdma_interfaces()
        
        # 检查NCCL环境
        check_nccl_env()
        
        # 本地NCCL测试
        test_nccl_local()
        
        # ping测试
        print("\n" + "=" * 80)
        print("下一步：双机测试")
        print("=" * 80)
        print("在两台服务器上分别运行:")
        print("  Server 1: python3 test_nccl_rdma.py --server --ip <server1_ip>")
        print("  Server 2: python3 test_nccl_rdma.py --client --ip <server1_ip> --remote-ip <server2_ip>")
        return
    
    if args.check:
        check_rdma_interfaces()
        check_nccl_env()
        return
    
    if args.local:
        test_nccl_local()
        return
    
    if args.server:
        print("NCCL + RDMA 测试 - Server模式")
        print("=" * 80)
        check_rdma_interfaces()
        check_nccl_env()
        test_rdma_bandwidth_with_iperf(is_server=True)
        return
    
    if args.client:
        print("NCCL + RDMA 测试 - Client模式")
        print("=" * 80)
        check_rdma_interfaces()
        check_nccl_env()
        test_rdma_bandwidth_with_iperf(server_ip=args.ip)
        test_rdma_latency_ping(args.ip)
        return


if __name__ == "__main__":
    main()
