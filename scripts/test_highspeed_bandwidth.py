#!/usr/bin/env python3
"""
高速网络带宽测试 - 针对10Gb/s+网络优化

使用更大的缓冲区和并行连接来测试高速网络的实际带宽
"""

import socket
import time
import argparse
import sys
import os
import threading
import struct


def run_server(listen_port: int = 5005, duration: int = 15):
    """启动带宽测试服务器"""
    print("=" * 80)
    print("高速网络带宽测试 - Server模式")
    print("=" * 80)
    print(f"监听端口: {listen_port}")
    print(f"测试时长: {duration} 秒\n")
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)  # 64MB接收缓冲
    server_socket.bind(('', listen_port))
    server_socket.listen(5)
    
    print("等待客户端连接...")
    conn, addr = server_socket.accept()
    print(f"客户端已连接: {addr}")
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
    
    # 接收数据
    total_bytes = 0
    chunk_size = 4 * 1024 * 1024  # 4MB块
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            conn.settimeout(2.0)
            try:
                data = conn.recv(chunk_size)
                if not data:
                    break
                total_bytes += len(data)
            except socket.timeout:
                continue
        
        elapsed = time.time() - start_time
        if elapsed > 0:
            bandwidth_gbps = (total_bytes * 8) / (elapsed * 1e9)
            bandwidth_gbps_full = bandwidth_gbps * 1000 / 8  # MB/s
            print(f"\n测试结果:")
            print(f"  接收数据: {total_bytes / 1e9:.2f} GB")
            print(f"  测试时间: {elapsed:.2f} 秒")
            print(f"  平均带宽: {bandwidth_gbps:.2f} Gbps ({bandwidth_gbps_full:.2f} MB/s)")
    except KeyboardInterrupt:
        pass
    finally:
        conn.close()
        server_socket.close()


def run_client(server_ip: str, port: int = 5005, duration: int = 15):
    """启动带宽测试客户端"""
    print("=" * 80)
    print("高速网络带宽测试 - Client模式")
    print("=" * 80)
    print(f"连接到服务器: {server_ip}:{port}")
    print(f"测试时长: {duration} 秒\n")
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024 * 1024)  # 64MB发送缓冲
    client_socket.connect((server_ip, port))
    print("已连接到服务器，开始发送数据...\n")
    
    # 发送数据
    total_bytes = 0
    start_time = time.time()
    chunk_size = 4 * 1024 * 1024  # 4MB块
    data = b'x' * chunk_size
    
    try:
        while time.time() - start_time < duration:
            client_socket.sendall(data)
            total_bytes += chunk_size
            
            elapsed = time.time() - start_time
            if elapsed >= 1.0 and int(elapsed) % 1 == 0:
                bandwidth_gbps = (total_bytes * 8) / (elapsed * 1e9)
                print(f"  [{elapsed:.0f}s] 已发送: {total_bytes / 1e6:.0f} MB, "
                      f"带宽: {bandwidth_gbps:.2f} Gbps ({bandwidth_gbps * 125:.1f} MB/s)")
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        client_socket.close()
    
    elapsed = time.time() - start_time
    if elapsed > 0:
        bandwidth_gbps = (total_bytes * 8) / (elapsed * 1e9)
        print(f"\n测试结果:")
        print(f"  发送数据: {total_bytes / 1e9:.2f} GB")
        print(f"  测试时间: {elapsed:.2f} 秒")
        print(f"  平均带宽: {bandwidth_gbps:.2f} Gbps ({bandwidth_gbps * 125:.1f} MB/s)")


def test_latency(server_ip: str, port: int = 5006, count: int = 100):
    """测试网络延迟"""
    print("=" * 80)
    print("网络延迟测试")
    print("=" * 80)
    print(f"目标服务器: {server_ip}:{port}")
    print(f"测试次数: {count}\n")
    
    # 启动延迟测试服务器
    server_thread = threading.Thread(target=run_latency_server, args=(port, count))
    server_thread.daemon = True
    server_thread.start()
    time.sleep(0.5)
    
    latencies = []
    
    for i in range(count):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        start = time.perf_counter()
        sock.connect((server_ip, port))
        sock.recv(4)
        sock.close()
        elapsed = (time.perf_counter() - start) * 1e6  # μs
        latencies.append(elapsed)
        
        if i % 10 == 0:
            print(f"  [{i+1}/{count}] 延迟: {elapsed:.1f} μs")
    
    latencies.sort()
    avg = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    
    print(f"\n延迟统计:")
    print(f"  平均: {avg:.1f} μs ({avg/1000:.3f} ms)")
    print(f"  P50:  {p50:.1f} μs ({p50/1000:.3f} ms)")
    print(f"  P95:  {p95:.1f} μs ({p95/1000:.3f} ms)")
    print(f"  P99:  {p99:.1f} μs ({p99/1000:.3f} ms)")
    print(f"  最小: {latencies[0]:.1f} μs")
    print(f"  最大: {latencies[-1]:.1f} μs")


def run_latency_server(listen_port: int = 5006, count: int = 100):
    """延迟测试服务器"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', listen_port))
    server_socket.listen(5)
    
    for _ in range(count):
        try:
            server_socket.settimeout(5.0)
            conn, _ = server_socket.accept()
            conn.send(b'pong')
            conn.close()
        except:
            break
    
    server_socket.close()


def main():
    parser = argparse.ArgumentParser(description="高速网络带宽测试")
    parser.add_argument("--server", action="store_true", help="服务器模式")
    parser.add_argument("--client", action="store_true", help="客户端模式")
    parser.add_argument("--server-ip", type=str, help="服务器IP")
    parser.add_argument("--port", type=int, default=5005, help="端口号")
    parser.add_argument("--duration", type=int, default=15, help="测试时长(秒)")
    parser.add_argument("--latency", action="store_true", help="测试延迟")
    
    args = parser.parse_args()
    
    if args.server:
        if args.latency:
            run_latency_server()
        else:
            run_server(args.port, args.duration)
    elif args.client:
        if not args.server_ip:
            print("错误: 客户端模式需要指定 --server-ip")
            sys.exit(1)
        
        if args.latency:
            test_latency(args.server_ip, args.port)
        else:
            run_client(args.server_ip, args.port, args.duration)
    else:
        print("使用方法:")
        print(f"  带宽测试服务器: python3 {sys.argv[0]} --server")
        print(f"  带宽测试客户端: python3 {sys.argv[0]} --client --server-ip <ip>")
        print(f"  延迟测试: python3 {sys.argv[0]} --client --server-ip <ip> --latency")


if __name__ == "__main__":
    main()
