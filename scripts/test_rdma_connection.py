#!/usr/bin/env python3
"""
RDMA连接测试脚本

用于测试两台服务器之间的实际RDMA网络连接情况，包括：
1. 检查RDMA设备是否可用
2. 测试网络带宽（使用ib_write_bw或perftest工具）
3. 测试网络延迟
4. 收集RDMA配置信息

使用方法：
  在两台服务器上分别运行：
  - Server 1: python3 test_rdma_connection.py --server
  - Server 2: python3 test_rdma_connection.py --client --server-ip <server1_ip>
"""

import subprocess
import sys
import argparse
import time
from typing import Optional


def run_command(cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    """运行命令并返回 (returncode, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"


def check_rdma_devices():
    """检查RDMA设备是否可用"""
    print("=" * 60)
    print("检查RDMA设备...")
    print("=" * 60)
    
    # 检查ibv_devices
    rc, stdout, stderr = run_command("ibv_devices")
    if rc == 0:
        print("✓ 找到RDMA设备:")
        print(stdout)
    else:
        print("✗ 未找到RDMA设备或ibv_devices未安装")
        print(f"  Error: {stderr}")
        return False
    
    # 检查ibstatus
    rc, stdout, stderr = run_command("ibstatus")
    if rc == 0:
        print("\n✓ RDMA设备状态:")
        print(stdout)
    else:
        print("\n✗ 无法获取RDMA状态 (ibstatus未安装或无权限)")
    
    # 检查ibv_devinfo
    rc, stdout, stderr = run_command("ibv_devinfo")
    if rc == 0:
        print("\n✓ RDMA设备详细信息:")
        print(stdout)
    else:
        print("\n✗ 无法获取设备详细信息")
    
    return True


def check_perftest_tools() -> dict[str, bool]:
    """检查perftest工具是否安装"""
    print("\n" + "=" * 60)
    print("检查perftest工具...")
    print("=" * 60)
    
    tools = {
        "ib_write_bw": False,
        "ib_read_bw": False,
        "ib_send_bw": False,
        "ib_write_lat": False,
        "ib_read_lat": False,
        "ib_send_lat": False,
    }
    
    for tool in tools.keys():
        rc, _, _ = run_command(f"which {tool}")
        tools[tool] = (rc == 0)
        status = "✓" if tools[tool] else "✗"
        print(f"  {status} {tool}")
    
    return tools


def test_rdma_bandwidth_server(server_ip: Optional[str] = None):
    """作为server启动带宽测试"""
    print("\n" + "=" * 60)
    print("启动RDMA带宽测试 (Server模式)...")
    print("=" * 60)
    print(f"等待client连接到: {server_ip or '本地'}")
    print("按 Ctrl+C 停止测试\n")
    
    # 尝试ib_write_bw
    cmd = f"ib_write_bw -d mlx5_0"
    print(f"运行命令: {cmd}")
    try:
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()
    except KeyboardInterrupt:
        print("\n测试已停止")
        proc.terminate()


def test_rdma_bandwidth_client(server_ip: str):
    """作为client启动带宽测试"""
    print("\n" + "=" * 60)
    print(f"启动RDMA带宽测试 (Client模式)...")
    print("=" * 60)
    print(f"连接到server: {server_ip}\n")
    
    # 尝试ib_write_bw
    cmd = f"ib_write_bw -d mlx5_0 {server_ip}"
    print(f"运行命令: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("测试超时")


def test_rdma_latency(server_ip: Optional[str] = None):
    """测试RDMA延迟"""
    print("\n" + "=" * 60)
    print("RDMA延迟测试...")
    print("=" * 60)
    
    if server_ip:
        cmd = f"ib_write_lat -d mlx5_0 {server_ip}"
    else:
        cmd = f"ib_write_lat -d mlx5_0"
    
    print(f"运行命令: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        print(result.stdout)
    except subprocess.TimeoutExpired:
        print("测试超时")


def collect_network_info():
    """收集网络配置信息"""
    print("\n" + "=" * 60)
    print("网络配置信息...")
    print("=" * 60)
    
    # 网卡信息
    rc, stdout, _ = run_command("ip addr show")
    if rc == 0:
        print("\n网络接口:")
        print(stdout)
    
    # 路由信息
    rc, stdout, _ = run_command("ip route show")
    if rc == 0:
        print("\n路由表:")
        print(stdout)
    
    # 检查RoCE/RDMA相关模块
    rc, stdout, _ = run_command("lsmod | grep -E 'mlx|ib_|rdma'")
    if rc == 0:
        print("\n已加载的RDMA模块:")
        print(stdout)
    
    # GPU信息
    rc, stdout, _ = run_command("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader")
    if rc == 0:
        print("\nGPU设备:")
        print(stdout)


def main():
    parser = argparse.ArgumentParser(description="RDMA连接测试脚本")
    parser.add_argument("--server", action="store_true", help="以server模式运行")
    parser.add_argument("--client", action="store_true", help="以client模式运行")
    parser.add_argument("--server-ip", type=str, help="Server IP地址 (client模式必需)")
    parser.add_argument("--bandwidth", action="store_true", help="仅测试带宽")
    parser.add_argument("--latency", action="store_true", help="仅测试延迟")
    parser.add_argument("--info", action="store_true", help="仅收集信息")
    
    args = parser.parse_args()
    
    # 如果没有指定模式，运行完整的诊断
    if not args.server and not args.client and not args.info:
        print("RDMA连接诊断工具")
        print("=" * 60)
        print(f"主机名: {subprocess.run('hostname', shell=True, capture_output=True, text=True).stdout.strip()}")
        print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 收集信息
        collect_network_info()
        
        # 检查RDMA设备
        has_rdma = check_rdma_devices()
        
        # 检查工具
        tools = check_perftest_tools()
        
        if not has_rdma:
            print("\n⚠ 未检测到RDMA设备，无法进行带宽/延迟测试")
            print("  请确保:")
            print("  1. 已安装RDMA驱动 (mlnx-ofed 或类似)")
            print("  2. RDMA设备已正确配置")
            return
        
        if not any(tools.values()):
            print("\n⚠ 未安装perftest工具，无法进行带宽/延迟测试")
            print("  安装方法: sudo apt install perftest (Ubuntu) 或 sudo yum install perftest (CentOS)")
            return
        
        print("\n" + "=" * 60)
        print("下一步操作:")
        print("=" * 60)
        print("在两台服务器上分别运行:")
        print(f"  Server 1: python3 {sys.argv[0]} --server")
        print(f"  Server 2: python3 {sys.argv[0]} --client --server-ip <server1_ip>")
        return
    
    # Server模式
    if args.server:
        print("RDMA测试 - Server模式")
        print("=" * 60)
        collect_network_info()
        check_rdma_devices()
        check_perftest_tools()
        
        if args.bandwidth:
            test_rdma_bandwidth_server()
        elif args.latency:
            test_rdma_latency()
        else:
            print("\n完整测试...")
            test_rdma_bandwidth_server()
    
    # Client模式
    elif args.client:
        if not args.server_ip:
            print("错误: client模式需要指定 --server-ip")
            sys.exit(1)
        
        print("RDMA测试 - Client模式")
        print("=" * 60)
        
        if args.bandwidth:
            test_rdma_bandwidth_client(args.server_ip)
        elif args.latency:
            test_rdma_latency(args.server_ip)
        else:
            test_rdma_bandwidth_client(args.server_ip)
    
    # 仅收集信息
    elif args.info:
        collect_network_info()
        check_rdma_devices()
        check_perftest_tools()


if __name__ == "__main__":
    main()
