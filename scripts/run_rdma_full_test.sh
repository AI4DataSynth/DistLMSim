#!/bin/bash
# RDMA完整测试脚本 - RDMA启用后一键运行
# 使用方法: bash run_rdma_full_test.sh [server|client] [server_ip]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_IP="${2:-}"

echo "================================================================================"
echo "RDMA完整测试套件"
echo "================================================================================"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "主机名: $(hostname)"
echo ""

# 步骤1: 检查RDMA状态
echo "步骤 1/5: 检查RDMA状态..."
echo "--------------------------------------------------------------------------------"
ibv_devinfo | grep -E "hca_id|state|rate|link_layer" || echo "未找到RDMA设备"
echo ""

# 检查链路是否UP
if rdma link show 2>/dev/null | grep -q "state ACTIVE\|state UP"; then
    echo "✅ RDMA链路活跃"
    RDMA_READY=true
else
    echo "❌ RDMA链路DOWN"
    echo ""
    echo "请先启用RDMA接口:"
    echo "  sudo ip link set ens19f0np0 up   # Server 1"
    echo "  sudo ip link set eno3np0 up      # Server 2"
    echo ""
    exit 1
fi

# 步骤2: 检查perftest工具
echo "步骤 2/5: 检查测试工具..."
echo "--------------------------------------------------------------------------------"
for tool in ib_write_bw ib_write_lat ib_read_bw ib_send_bw; do
    if which $tool &>/dev/null; then
        echo "  ✅ $tool"
    else
        echo "  ❌ $tool (未安装)"
    fi
done
echo ""

# 步骤3: 测试RDMA带宽
if [ "$1" = "server" ]; then
    echo "步骤 3/5: 启动RDMA带宽测试服务器..."
    echo "--------------------------------------------------------------------------------"
    echo "等待客户端连接..."
    echo ""
    ib_write_bw -d mlx5_0 &
    BW_PID=$!
    sleep 2
    
    echo "步骤 4/5: 启动RDMA延迟测试服务器..."
    echo "--------------------------------------------------------------------------------"
    ib_write_lat -d mlx5_0 &
    LAT_PID=$!
    sleep 2
    
    echo "步骤 5/5: 启动RDMA All-to-All测试服务器..."
    echo "--------------------------------------------------------------------------------"
    ib_send_bw -d mlx5_0 --all &
    A2A_PID=$!
    
    echo ""
    echo "所有RDMA测试服务器已启动"
    echo ""
    echo "在客户端运行:"
    echo "  bash $0 client <server_rdma_ip>"
    echo ""
    
    # 等待客户端连接
    wait $BW_PID $LAT_PID $A2A_PID 2>/dev/null || true

elif [ "$1" = "client" ] && [ -n "$SERVER_IP" ]; then
    echo "步骤 3/5: 测试RDMA带宽..."
    echo "--------------------------------------------------------------------------------"
    ib_write_bw -d roceo3 $SERVER_IP || echo "ib_write_bw测试失败"
    echo ""
    
    echo "步骤 4/5: 测试RDMA延迟..."
    echo "--------------------------------------------------------------------------------"
    ib_write_lat -d roceo3 $SERVER_IP || echo "ib_write_lat测试失败"
    echo ""
    
    echo "步骤 5/5: 测试RDMA All-to-All..."
    echo "--------------------------------------------------------------------------------"
    ib_send_bw -d roceo3 --all $SERVER_IP || echo "ib_send_bw测试失败"
    
else
    echo "使用方法:"
    echo "  服务器模式: bash $0 server"
    echo "  客户端模式: bash $0 client <server_rdma_ip>"
    exit 1
fi

echo ""
echo "================================================================================"
echo "RDMA测试完成"
echo "================================================================================"
