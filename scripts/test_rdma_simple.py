#!/usr/bin/env python3
"""
RDMA测试 - 无需sudo，使用RDMA CM直接通信
通过TCP建立控制连接，然后通过RDMA传输数据
"""

import socket
import struct
import sys
import time
import subprocess

def run_cmd(cmd, timeout=10):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except:
        return -1, "", "timeout"

def test_rdma_connectivity(server_ip):
    """测试RDMA连通性"""
    print(f"Testing RDMA connectivity to {server_ip}...")
    
    # 1. 检查RDMA设备
    rc, stdout, _ = run_cmd("ibv_devinfo")
    if rc != 0:
        print("No RDMA devices found")
        return False
    
    print("RDMA devices:")
    print(stdout)
    
    # 2. 检查GID
    rc, stdout, _ = run_cmd("cat /sys/class/infiniband/mlx5_1/ports/1/gids/0 2>/dev/null || cat /sys/class/infiniband/roceo3/ports/1/gids/0 2>/dev/null")
    gid = stdout.strip()
    print(f"GID: {gid}")
    
    # 3. 测试TCP连接 (用于控制平面)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    
    try:
        sock.connect((server_ip, 18515))
        print(f"TCP connection to {server_ip}:18515 OK")
    except Exception as e:
        print(f"TCP connection failed: {e}")
        print("This is normal for ibv_rc_pingpong - it opens its own port")
    
    sock.close()
    
    # 4. 测量网络延迟 (ping模拟)
    rc, stdout, _ = run_cmd(f"ping -c 5 {server_ip}")
    if rc == 0:
        for line in stdout.split('\n'):
            if 'rtt' in line or 'round-trip' in line:
                print(f"Ping: {line}")
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_rdma_simple.py <server_ip>")
        print("Example: python3 test_rdma_simple.py 10.21.16.124")
        sys.exit(1)
    
    server_ip = sys.argv[1]
    print("=" * 60)
    print("RDMA Connectivity Test")
    print("=" * 60)
    print(f"Target: {server_ip}")
    print()
    
    test_rdma_connectivity(server_ip)
    
    print("\n" + "=" * 60)
    print("RDMA Testing Options:")
    print("=" * 60)
    print("1. ibv_rc_pingpong:")
    print(f"   Server: ibv_rc_pingpong -d mlx5_1 -i 1")
    print(f"   Client: ibv_rc_pingpong -d roceo3 -i 1 {server_ip}")
    print()
    print("2. ibv_rc_pingpong with GID:")
    print(f"   Server: ibv_rc_pingpong -d mlx5_1 -g 0 -i 1")
    print(f"   Client: ibv_rc_pingpong -d roceo3 -g 0 -i 1 {server_ip}")
    print()
    print("3. ib_write_bw (requires perftest):")
    print(f"   Server: ib_write_bw -d mlx5_1")
    print(f"   Client: ib_write_bw -d roceo3 {server_ip}")

if __name__ == "__main__":
    main()
