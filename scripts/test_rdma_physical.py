#!/usr/bin/env python3
"""
RDMA物理接口测试 - 检查并测试RDMA网络

测试内容：
1. 检查RDMA接口状态
2. 如果RDMA可用，测试RDMA带宽和延迟
3. 如果RDMA不可用，提供配置建议
"""

import subprocess
import sys
import time


def run_cmd(cmd: str, timeout: int = 10) -> tuple[int, str, str]:
    """运行命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"


def check_rdma_devices():
    """检查RDMA设备"""
    print("=" * 80)
    print("RDMA设备检查")
    print("=" * 80)
    
    # ibv_devinfo
    rc, stdout, _ = run_cmd("ibv_devinfo")
    if rc == 0:
        print("\n✓ RDMA设备信息:")
        print(stdout)
        return True
    else:
        print("\n✗ 未找到RDMA设备")
        return False


def check_rdma_links():
    """检查RDMA链路状态"""
    print("\n" + "=" * 80)
    print("RDMA链路状态")
    print("=" * 80)
    
    rc, stdout, _ = run_cmd("rdma link show")
    if rc == 0:
        print(stdout)
        
        # 检查是否有UP的链路
        if "state ACTIVE" in stdout or "state UP" in stdout:
            print("\n✓ 检测到活跃的RDMA链路")
            return True
        else:
            print("\n⚠ RDMA链路处于DOWN状态")
            return False
    else:
        print("无法获取RDMA链路信息")
        return False


def test_rdma_with_perftest():
    """使用perftest测试RDMA（如果已安装）"""
    print("\n" + "=" * 80)
    print("RDMA带宽测试 (perftest)")
    print("=" * 80)
    
    rc, _, _ = run_cmd("which ib_write_bw")
    if rc != 0:
        print("✗ perftest未安装")
        print("\n安装方法:")
        print("  Ubuntu/Debian: sudo apt install perftest")
        print("  CentOS/RHEL: sudo yum install perftest")
        return False
    
    print("✓ perftest已安装")
    print("\n要测试RDMA带宽，请在两台服务器上分别运行:")
    print("  Server 1: ib_write_bw -d <rdma_device>")
    print("  Server 2: ib_write_bw -d <rdma_device> <server1_ip>")
    return True


def check_network_interfaces():
    """检查网络接口"""
    print("\n" + "=" * 80)
    print("网络接口状态")
    print("=" * 80)
    
    rc, stdout, _ = run_cmd("ip -br addr show")
    if rc == 0:
        print(stdout)
    
    # 检查可能的RDMA接口
    rdma_interfaces = []
    rc, stdout, _ = run_cmd("ip link show | grep -E 'mtu 9000|state UP' | awk '{print $2}' | tr -d ':'")
    if rc == 0:
        interfaces = stdout.strip().split('\n')
        rdma_interfaces = [iface for iface in interfaces if iface]
    
    if rdma_interfaces:
        print(f"\n活跃的接口: {', '.join(rdma_interfaces)}")
    else:
        print("\n未找到活跃的网络接口")


def check_gpu_to_rdma_mapping():
    """检查GPU和RDMA网卡的拓扑关系"""
    print("\n" + "=" * 80)
    print("GPU-RDMA拓扑关系")
    print("=" * 80)
    
    rc, stdout, _ = run_cmd("nvidia-smi topo -m")
    if rc == 0:
        print(stdout)


def provide_rdma_setup_guide():
    """提供RDMA配置指南"""
    print("\n" + "=" * 80)
    print("RDMA配置指南（如需启用RDMA）")
    print("=" * 80)
    
    print("""
如果RDMA链路处于DOWN状态，按以下步骤启用：

1. 检查物理连接
   - 确保RDMA网卡已连接网线
   - 检查网线是否连接到支持RDMA的交换机

2. 启用RDMA接口（需要root权限）
   Server 1 (10.21.16.124):
   ```bash
   sudo ip link set ens19f0np0 up
   sudo ip link set ens19f1np1 up
   sudo ip addr add 192.168.100.1/24 dev ens19f0np0
   ```
   
   Server 2 (100.64.0.6):
   ```bash
   sudo ip link set eno3np0 up
   sudo ip link set eno4np1 up
   sudo ip addr add 192.168.100.2/24 dev eno3np0
   ```

3. 测试RDMA连通性
   ```bash
   # 检查链路状态
   ibv_devinfo
   rdma link show
   
   # 测试带宽（需要安装perftest）
   # Server 1:
   ib_write_bw -d mlx5_0
   
   # Server 2:
   ib_write_bw -d roceo3 192.168.100.1
   ```

4. 配置NCCL使用RDMA
   ```bash
   export NCCL_IB_DISABLE=0
   export NCCL_SOCKET_IFNAME=ens20f1
   export NCCL_IB_HCA=mlx5
   export NCCL_NET_GDR_LEVEL=2
   ```

5. 测试NCCL带宽
   ```bash
   # 使用nccl-tests
   all_reduce_perf -b 1M -e 1G -f 2 -g 4
   ```
""")


def main():
    print("=" * 80)
    print("RDMA物理接口测试")
    print("=" * 80)
    print(f"主机名: {run_cmd('hostname')[1].strip()}")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 检查RDMA设备
    has_devices = check_rdma_devices()
    
    if not has_devices:
        print("\n⚠ 未检测到RDMA设备，无法进行RDMA测试")
        return
    
    # 检查RDMA链路
    has_active_links = check_rdma_links()
    
    # 检查网络接口
    check_network_interfaces()
    
    # 检查GPU-RDMA拓扑
    check_gpu_to_rdma_mapping()
    
    # 检查perftest
    has_perftest = test_rdma_with_perftest()
    
    if not has_active_links:
        print("\n" + "=" * 80)
        print("⚠ RDMA链路未激活")
        print("=" * 80)
        print("\n当前RDMA端口处于DOWN状态，无法进行RDMA带宽测试。")
        print("这可能是由于：")
        print("  1. 网线未连接")
        print("  2. 交换机未配置RDMA")
        print("  3. 接口未启用")
        
        provide_rdma_setup_guide()
    else:
        print("\n✓ RDMA链路活跃，可以进行带宽测试")


if __name__ == "__main__":
    main()
