#!/bin/bash
# RDMA完整测试脚本 - 无需sudo，使用ibv_rc_pingpong
# 用法:
#   Server 1: bash run_rdma_test.sh server
#   Server 2: bash run_rdma_test.sh client 10.21.16.124

set -e

SERVER_IP="${2:-}"
PORT=18515

echo "============================================================"
echo "RDMA Bandwidth & Latency Test"
echo "============================================================"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname)"
echo ""

# 检查RDMA状态
echo "Step 1: Check RDMA Status"
echo "------------------------------------------------------------"
ibv_devinfo | grep -E "hca_id|state" | head -10
echo ""

if [ "$1" = "server" ]; then
    echo "Step 2: Start RDMA Server"
    echo "------------------------------------------------------------"
    echo "Starting ibv_rc_pingpong server on port $PORT..."
    echo ""
    echo "Run on client: ibv_rc_pingpong -d roceo3 -g 0 -p $PORT $SERVER_IP"
    echo ""
    
    # 测试不同消息大小
    for msg_size in 64 256 1024 4096 16384 65536 262144 1048576; do
        echo "Testing message size: $msg_size bytes"
        ibv_rc_pingpong -d mlx5_1 -g 0 -s $msg_size -p $PORT -i 100 &
        SERVER_PID=$!
        sleep 2
        
        # 等待客户端连接（超时10秒）
        TIMEOUT=10
        while [ $TIMEOUT -gt 0 ]; do
            if ! kill -0 $SERVER_PID 2>/dev/null; then
                break
            fi
            sleep 1
            TIMEOUT=$((TIMEOUT - 1))
        done
        
        wait $SERVER_PID 2>/dev/null || true
        sleep 1
    done
    
elif [ "$1" = "client" ] && [ -n "$SERVER_IP" ]; then
    echo "Step 2: Test RDMA to $SERVER_IP"
    echo "------------------------------------------------------------"
    echo ""
    
    # 测试不同消息大小
    for msg_size in 64 256 1024 4096 16384 65536 262144 1048576; do
        echo "=== Message Size: $msg_size bytes ==="
        timeout 15 ibv_rc_pingpong -d roceo3 -g 0 -s $msg_size -p $PORT -i 100 $SERVER_IP 2>&1 | tail -5
        echo ""
        sleep 1
    done
    
else
    echo "Usage:"
    echo "  Server: bash $0 server"
    echo "  Client: bash $0 client <server_ip>"
    exit 1
fi

echo "============================================================"
echo "Test Complete"
echo "============================================================"
