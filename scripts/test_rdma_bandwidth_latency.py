#!/usr/bin/env python3
"""
RDMA带宽和延迟测试 (RDMA启用后运行)

此脚本在RDMA物理连接启用后运行，用于测试：
1. RDMA网络带宽
2. RDMA网络延迟
3. GPU间通过RDMA的显存复制速度

前置条件:
- RDMA网线已连接
- RDMA接口已启用 (ip link set <iface> up)
- 已安装perftest (ib_write_bw, ib_write_lat)

使用方法:
  Server 1: python3 test_rdma_bandwidth_latency.py --server
  Server 2: python3 test_rdma_bandwidth_latency.py --client --server-ip <server1_rdma_ip>
"""

import subprocess
import sys
import time
import argparse
import os


def run_cmd(cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    """运行命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"


def check_rdma_ready() -> tuple[bool, str]:
    """检查RDMA是否就绪"""
    # 检查RDMA设备
    rc, stdout, _ = run_cmd("ibv_devinfo")
    if rc != 0:
        return False, "未找到RDMA设备"
    
    # 检查链路状态
    rc, stdout, _ = run_cmd("rdma link show")
    if rc == 0:
        if "state ACTIVE" in stdout or "state UP" in stdout:
            # 提取设备名
            for line in stdout.split('\n'):
                if "state ACTIVE" in line or "state UP" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        device = parts[1]
                        return True, f"RDMA就绪: {device}"
        else:
            return False, "RDMA链路DOWN，请先启用接口"
    
    return False, "无法获取RDMA状态"


def test_rdma_bandwidth(server_ip: str = None, is_server: bool = False):
    """测试RDMA带宽"""
    print("=" * 80)
    print("RDMA带宽测试 (ib_write_bw)")
    print("=" * 80)
    
    # 检查perftest
    rc, _, _ = run_cmd("which ib_write_bw")
    if rc != 0:
        print("✗ ib_write_bw未安装")
        print("安装: sudo apt install perftest")
        return
    
    if is_server:
        print(f"启动RDMA带宽测试服务器...")
        print("等待客户端连接...")
        cmd = "ib_write_bw -d mlx5_0"
        print(f"运行: {cmd}")
        print("\n在客户端运行: ib_write_bw -d <device> <server_ip>")
        try:
            proc = subprocess.Popen(cmd, shell=True)
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
    else:
        print(f"连接到RDMA服务器: {server_ip}")
        cmd = f"ib_write_bw -d mlx5_0 {server_ip}"
        print(f"运行: {cmd}\n")
        rc, stdout, stderr = run_cmd(cmd, timeout=30)
        print(stdout)
        if stderr:
            print(f"Stderr: {stderr}")


def test_rdma_latency(server_ip: str = None, is_server: bool = False):
    """测试RDMA延迟"""
    print("=" * 80)
    print("RDMA延迟测试 (ib_write_lat)")
    print("=" * 80)
    
    # 检查perftest
    rc, _, _ = run_cmd("which ib_write_lat")
    if rc != 0:
        print("✗ ib_write_lat未安装")
        print("安装: sudo apt install perftest")
        return
    
    if is_server:
        print(f"启动RDMA延迟测试服务器...")
        cmd = "ib_write_lat -d mlx5_0"
        print(f"运行: {cmd}")
        try:
            proc = subprocess.Popen(cmd, shell=True)
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
    else:
        print(f"连接到RDMA服务器: {server_ip}")
        cmd = f"ib_write_lat -d mlx5_0 {server_ip}"
        print(f"运行: {cmd}\n")
        rc, stdout, stderr = run_cmd(cmd, timeout=30)
        print(stdout)
        if stderr:
            print(f"Stderr: {stderr}")


def test_rdma_alltoall(server_ip: str = None, is_server: bool = False):
    """测试RDMA All-to-All带宽"""
    print("=" * 80)
    print("RDMA All-to-All测试 (ib_send_bw)")
    print("=" * 80)
    
    rc, _, _ = run_cmd("which ib_send_bw")
    if rc != 0:
        print("✗ ib_send_bw未安装，跳过")
        return
    
    if is_server:
        cmd = "ib_send_bw -d mlx5_0 --all"
        print(f"运行: {cmd}")
        try:
            proc = subprocess.Popen(cmd, shell=True)
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
    else:
        cmd = f"ib_send_bw -d mlx5_0 --all {server_ip}"
        print(f"运行: {cmd}\n")
        rc, stdout, stderr = run_cmd(cmd, timeout=30)
        print(stdout)


def main():
    parser = argparse.ArgumentParser(description="RDMA带宽和延迟测试")
    parser.add_argument("--server", action="store_true", help="服务器模式")
    parser.add_argument("--client", action="store_true", help="客户端模式")
    parser.add_argument("--server-ip", type=str, help="服务器RDMA IP地址")
    parser.add_argument("--check", action="store_true", help="仅检查RDMA状态")
    parser.add_argument("--bandwidth", action="store_true", help="仅测试带宽")
    parser.add_argument("--latency", action="store_true", help="仅测试延迟")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RDMA带宽和延迟测试")
    print("=" * 80)
    print(f"主机名: {os.uname().nodename}")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 检查RDMA状态
    ready, msg = check_rdma_ready()
    print(f"RDMA状态: {msg}")
    
    if not ready and not args.check:
        print("\n⚠ RDMA未就绪，无法进行测试")
        print("\n启用RDMA的步骤:")
        print("1. 连接RDMA网线")
        print("2. sudo ip link set <iface> up")
        print("3. sudo ip addr add <ip>/24 dev <iface>")
        print("4. 重新运行此脚本")
        return
    
    if args.check:
        return
    
    # 运行测试
    if args.server:
        if args.bandwidth:
            test_rdma_bandwidth(is_server=True)
        elif args.latency:
            test_rdma_latency(is_server=True)
        else:
            test_rdma_bandwidth(is_server=True)
            test_rdma_latency(is_server=True)
    
    elif args.client:
        if not args.server_ip:
            print("错误: 客户端模式需要指定 --server-ip")
            sys.exit(1)
        
        if args.bandwidth:
            test_rdma_bandwidth(server_ip=args.server_ip)
        elif args.latency:
            test_rdma_latency(server_ip=args.server_ip)
        else:
            test_rdma_bandwidth(server_ip=args.server_ip)
            test_rdma_latency(server_ip=args.server_ip)
    else:
        print("使用方法:")
        print(f"  检查状态: python3 {sys.argv[0]} --check")
        print(f"  服务器:   python3 {sys.argv[0]} --server")
        print(f"  客户端:   python3 {sys.argv[0]} --client --server-ip <server_rdma_ip>")


if __name__ == "__main__":
    main()
