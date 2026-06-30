#!/usr/bin/env python3
"""
RDMA延迟和带宽测试 - 使用IBVerbs API

使用ibv_rc_pingpong测试RDMA延迟和带宽
即使没有perftest，也可以使用IBVerbs原生工具

前提条件:
- RDMA设备存在
- RDMA链路UP (需要先启用接口)

使用方法:
  需要先启用RDMA接口 (root权限):
  sudo ip link set ens19f0np0 up
  sudo ip addr add 192.168.100.1/24 dev ens19f0np0
  
  Server 1: python3 test_rdma_ibverbs.py --server
  Server 2: python3 test_rdma_ibverbs.py --client --server-ip <rdma_ip>
"""

import subprocess
import sys
import time
import argparse
import os
import signal


def run_cmd(cmd: str, timeout: int = 10) -> tuple[int, str, str]:
    """运行命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"


def check_rdma_status() -> tuple[bool, str]:
    """检查RDMA状态"""
    # 检查设备
    rc, stdout, _ = run_cmd("ibv_devinfo")
    if rc != 0:
        return False, "未找到RDMA设备"
    
    # 检查链路
    rc, stdout, _ = run_cmd("rdma link show")
    if rc == 0 and ("state ACTIVE" in stdout or "state UP" in stdout):
        # 提取设备名
        for line in stdout.split('\n'):
            if "state" in line and ("ACTIVE" in line or "UP" in line):
                parts = line.split()
                device = parts[0] if parts else "unknown"
                return True, f"RDMA就绪: {device}"
    
    return False, "RDMA链路DOWN"


def test_rdma_pingpong(server_ip: str = None, is_server: bool = False):
    """使用ibv_rc_pingpong测试RDMA延迟"""
    print("=" * 80)
    print("RDMA延迟测试 (ibv_rc_pingpong)")
    print("=" * 80)
    
    if is_server:
        print(f"启动RDMA pingpong服务器...")
        print("等待客户端连接...")
        print("")
        print("运行: ibv_rc_pingpong -d mlx5_0")
        print("")
        
        try:
            proc = subprocess.Popen(
                "ibv_rc_pingpong -d mlx5_0",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待Ctrl+C
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
    else:
        if not server_ip:
            print("错误: 客户端模式需要指定--server-ip")
            return
        
        print(f"连接到RDMA服务器: {server_ip}")
        print("运行: ibv_rc_pingpong -d roceo3 {server_ip}")
        print("")
        
        try:
            proc = subprocess.Popen(
                f"ibv_rc_pingpong -d roceo3 {server_ip}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待输出
            try:
                stdout, stderr = proc.communicate(timeout=30)
                print(stdout)
                if stderr:
                    print(f"Stderr: {stderr}")
            except subprocess.TimeoutExpired:
                proc.terminate()
                print("测试超时")
        except Exception as e:
            print(f"错误: {e}")


def test_rdma_with_perftest(server_ip: str = None, is_server: bool = False):
    """使用perftest测试RDMA带宽"""
    print("=" * 80)
    print("RDMA带宽测试 (perftest)")
    print("=" * 80)
    
    # 检查perftest
    rc, _, _ = run_cmd("which ib_write_bw")
    if rc != 0:
        print("✗ perftest未安装")
        print("安装: sudo apt install perftest")
        return
    
    if is_server:
        print(f"启动RDMA带宽测试服务器...")
        print("运行: ib_write_bw -d mlx5_0")
        print("")
        
        try:
            proc = subprocess.Popen("ib_write_bw -d mlx5_0", shell=True)
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
    else:
        if not server_ip:
            print("错误: 客户端模式需要指定--server-ip")
            return
        
        print(f"连接到RDMA服务器: {server_ip}")
        print(f"运行: ib_write_bw -d roceo3 {server_ip}")
        print("")
        
        try:
            proc = subprocess.Popen(f"ib_write_bw -d roceo3 {server_ip}", shell=True)
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()


def test_rdma_latency(server_ip: str = None, is_server: bool = False):
    """使用perftest测试RDMA延迟"""
    print("=" * 80)
    print("RDMA延迟测试 (perftest)")
    print("=" * 80)
    
    rc, _, _ = run_cmd("which ib_write_lat")
    if rc != 0:
        print("✗ ib_write_lat未安装")
        return
    
    if is_server:
        print(f"启动RDMA延迟测试服务器...")
        print("运行: ib_write_lat -d mlx5_0")
        print("")
        
        try:
            proc = subprocess.Popen("ib_write_lat -d mlx5_0", shell=True)
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
    else:
        if not server_ip:
            print("错误: 客户端模式需要指定--server-ip")
            return
        
        print(f"连接到RDMA服务器: {server_ip}")
        print(f"运行: ib_write_lat -d roceo3 {server_ip}")
        print("")
        
        try:
            proc = subprocess.Popen(f"ib_write_lat -d roceo3 {server_ip}", shell=True)
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()


def main():
    parser = argparse.ArgumentParser(description="RDMA延迟和带宽测试")
    parser.add_argument("--server", action="store_true", help="服务器模式")
    parser.add_argument("--client", action="store_true", help="客户端模式")
    parser.add_argument("--server-ip", type=str, help="服务器RDMA IP")
    parser.add_argument("--check", action="store_true", help="仅检查RDMA状态")
    parser.add_argument("--pingpong", action="store_true", help="使用pingpong测试")
    parser.add_argument("--bandwidth", action="store_true", help="仅测试带宽")
    parser.add_argument("--latency", action="store_true", help="仅测试延迟")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RDMA测试 (IBVerbs)")
    print("=" * 80)
    print(f"主机名: {os.uname().nodename}")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 检查RDMA
    ready, msg = check_rdma_status()
    print(f"RDMA状态: {msg}\n")
    
    if args.check:
        if not ready:
            print("\nRDMA未就绪，请先启用接口:")
            print("  sudo ip link set ens19f0np0 up   # Server 1")
            print("  sudo ip link set eno3np0 up      # Server 2")
        return
    
    if not ready:
        print("❌ RDMA未就绪，无法进行测试")
        print("\n请先连接RDMA网线并启用接口")
        return
    
    # 运行测试
    if args.server:
        if args.pingpong:
            test_rdma_pingpong(is_server=True)
        elif args.bandwidth:
            test_rdma_with_perftest(is_server=True)
        elif args.latency:
            test_rdma_latency(is_server=True)
        else:
            test_rdma_with_perftest(is_server=True)
            test_rdma_latency(is_server=True)
    
    elif args.client:
        if not args.server_ip:
            print("错误: 客户端模式需要指定--server-ip")
            sys.exit(1)
        
        if args.pingpong:
            test_rdma_pingpong(server_ip=args.server_ip)
        elif args.bandwidth:
            test_rdma_with_perftest(server_ip=args.server_ip)
        elif args.latency:
            test_rdma_latency(server_ip=args.server_ip)
        else:
            test_rdma_with_perftest(server_ip=args.server_ip)
            test_rdma_latency(server_ip=args.server_ip)
    
    else:
        print("使用方法:")
        print(f"  检查状态: python3 {sys.argv[0]} --check")
        print(f"  服务器:   python3 {sys.argv[0]} --server")
        print(f"  客户端:   python3 {sys.argv[0]} --client --server-ip <rdma_ip>")


if __name__ == "__main__":
    main()
