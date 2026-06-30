#!/usr/bin/env python3
"""
网络带宽测试脚本 (TCP/IP方式，无需RDMA/perftest)

由于RDMA端口处于DOWN状态，此脚本使用TCP/IP方式测试两台服务器之间的实际网络带宽。
使用iperf3或者Python自带的socket进行带宽测试。

使用方法：
  - Server 1: python3 test_network_bandwidth.py --server
  - Server 2: python3 test_network_bandwidth.py --client --server-ip 10.21.16.124
"""

import socket
import time
import argparse
import sys
import threading
from typing import Optional


def run_server(listen_port: int = 5001):
    """启动带宽测试服务器"""
    print("=" * 60)
    print("网络带宽测试 - Server模式")
    print("=" * 60)
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', listen_port))
    server_socket.listen(1)
    
    print(f"等待客户端连接... (端口 {listen_port})")
    conn, addr = server_socket.accept()
    print(f"客户端已连接: {addr}")
    
    # 接收数据并计算带宽
    total_bytes = 0
    start_time = time.time()
    chunk_size = 1024 * 1024  # 1MB
    
    try:
        while True:
            data = conn.recv(chunk_size)
            if not data:
                break
            total_bytes += len(data)
            
            # 每5秒报告一次
            elapsed = time.time() - start_time
            if elapsed >= 5.0:
                bandwidth_gbps = (total_bytes * 8) / (elapsed * 1e9)
                print(f"  接收: {total_bytes / 1e6:.2f} MB, "
                      f"带宽: {bandwidth_gbps:.2f} Gbps ({bandwidth_gbps * 1000 / 8:.2f} MB/s)")
                total_bytes = 0
                start_time = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()
        server_socket.close()
    
    print("\n服务器已停止")


def run_client(server_ip: str, port: int = 5001, duration: int = 10):
    """启动带宽测试客户端"""
    print("=" * 60)
    print("网络带宽测试 - Client模式")
    print("=" * 60)
    print(f"连接到服务器: {server_ip}:{port}")
    print(f"测试时长: {duration} 秒\n")
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, port))
    print("已连接到服务器，开始发送数据...\n")
    
    # 发送数据
    total_bytes = 0
    start_time = time.time()
    chunk_size = 1024 * 1024  # 1MB
    data = b'x' * chunk_size
    
    try:
        while time.time() - start_time < duration:
            client_socket.sendall(data)
            total_bytes += chunk_size
            
            # 每秒报告一次
            elapsed = time.time() - start_time
            if int(elapsed) > 0 and int(elapsed) % 1 == 0:
                bandwidth_gbps = (total_bytes * 8) / (elapsed * 1e9)
                print(f"  已发送: {total_bytes / 1e6:.2f} MB, "
                      f"带宽: {bandwidth_gbps:.2f} Gbps ({bandwidth_gbps * 1000 / 8:.2f} MB/s)")
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        client_socket.close()
    
    elapsed = time.time() - start_time
    bandwidth_gbps = (total_bytes * 8) / (elapsed * 1e9)
    print(f"\n测试完成:")
    print(f"  总发送: {total_bytes / 1e9:.2f} GB")
    print(f"  总时间: {elapsed:.2f} 秒")
    print(f"  平均带宽: {bandwidth_gbps:.2f} Gbps ({bandwidth_gbps * 1000 / 8:.2f} MB/s)")


def run_latency_server(listen_port: int = 5002, count: int = 100):
    """启动延迟测试服务器"""
    print("=" * 60)
    print("网络延迟测试 - Server模式")
    print("=" * 60)
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', listen_port))
    server_socket.listen(1)
    
    print(f"等待客户端连接... (端口 {listen_port})")
    
    for i in range(count):
        conn, addr = server_socket.accept()
        # 立即回复
        conn.send(b'pong')
        conn.close()
        if i % 10 == 0:
            print(f"  处理连接 {i+1}/{count}")
    
    server_socket.close()
    print("\n延迟测试服务器已停止")


def test_latency(server_ip: str, port: int = 5002, count: int = 100):
    """测试网络延迟"""
    print("=" * 60)
    print("网络延迟测试")
    print("=" * 60)
    print(f"目标服务器: {server_ip}:{port}")
    print(f"测试次数: {count}\n")
    
    latencies = []
    
    for i in range(count):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        start = time.time()
        sock.connect((server_ip, port))
        # 接收回复
        sock.recv(1024)
        sock.close()
        elapsed = (time.time() - start) * 1000  # ms
        latencies.append(elapsed)
        
        if i % 10 == 0:
            print(f"  [{i+1}/{count}] 延迟: {elapsed:.3f} ms")
    
    # 统计
    latencies.sort()
    avg = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    
    print(f"\n延迟统计:")
    print(f"  平均: {avg:.3f} ms")
    print(f"  P50:  {p50:.3f} ms")
    print(f"  P95:  {p95:.3f} ms")
    print(f"  P99:  {p99:.3f} ms")
    print(f"  最小: {latencies[0]:.3f} ms")
    print(f"  最大: {latencies[-1]:.3f} ms")


def main():
    parser = argparse.ArgumentParser(description="网络带宽测试脚本")
    parser.add_argument("--server", action="store_true", help="以server模式运行")
    parser.add_argument("--client", action="store_true", help="以client模式运行")
    parser.add_argument("--server-ip", type=str, help="Server IP地址 (client模式必需)")
    parser.add_argument("--port", type=int, default=5001, help="端口号 (默认 5001)")
    parser.add_argument("--duration", type=int, default=10, help="测试时长 (秒)")
    parser.add_argument("--latency", action="store_true", help="测试延迟")
    parser.add_argument("--latency-server", action="store_true", help="启动延迟测试服务器")
    
    args = parser.parse_args()
    
    if args.latency_server:
        run_latency_server()
    elif args.server:
        run_server(args.port)
    elif args.client:
        if not args.server_ip:
            print("错误: client模式需要指定 --server-ip")
            sys.exit(1)
        
        if args.latency:
            test_latency(args.server_ip, args.port)
        else:
            run_client(args.server_ip, args.port, args.duration)
    else:
        print("使用方法:")
        print(f"  Server: python3 {sys.argv[0]} --server")
        print(f"  Client: python3 {sys.argv[0]} --client --server-ip <server_ip>")
        print(f"  延迟测试: python3 {sys.argv[0]} --client --server-ip <server_ip> --latency")
        print(f"  延迟服务器: python3 {sys.argv[0]} --latency-server")


if __name__ == "__main__":
    main()
