#!/usr/bin/env python3
"""
GPU显存带宽测试 - 不依赖PyTorch

使用CUDA示例程序或nvidia-smi来测试GPU显存带宽
"""

import subprocess
import sys
import time


def run_cmd(cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    """运行命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"


def check_gpu_info():
    """检查GPU基本信息"""
    print("=" * 80)
    print("GPU信息检查")
    print("=" * 80)
    
    # nvidia-smi
    rc, stdout, _ = run_cmd("nvidia-smi")
    if rc == 0:
        print(stdout)
    
    # GPU时钟频率
    rc, stdout, _ = run_cmd("nvidia-smi --query-gpu=clocks.sm.clocks.gr,clocks.mem --format=csv")
    if rc == 0:
        print("\nGPU时钟频率:")
        print(stdout)


def test_gpu_memory_bandwidth_cuda():
    """使用CUDA bandwidthTest测试显存带宽"""
    print("\n" + "=" * 80)
    print("GPU显存带宽测试 (CUDA bandwidthTest)")
    print("=" * 80)
    
    # 检查是否有bandwidthTest
    rc, stdout, _ = run_cmd("find /usr/local/cuda -name bandwidthTest 2>/dev/null | head -1")
    if rc == 0 and stdout.strip():
        bw_test_path = stdout.strip()
        print(f"找到bandwidthTest: {bw_test_path}")
        print("\n运行测试...\n")
        rc, stdout, stderr = run_cmd(f"{bw_test_path} -htod", timeout=60)
        print(stdout)
        if stderr:
            print(f"Stderr: {stderr}")
        return
    
    # 尝试其他位置
    rc, stdout, _ = run_cmd("which bandwidthTest 2>/dev/null")
    if rc == 0 and stdout.strip():
        print(f"运行bandwidthTest...\n")
        rc, stdout, stderr = run_cmd("bandwidthTest -htod", timeout=60)
        print(stdout)
        if stderr:
            print(f"Stderr: {stderr}")
        return
    
    print("未找到CUDA bandwidthTest工具")
    print("尝试使用其他方法测试...\n")
    
    # 使用nvidia-smi进行简单测试
    test_with_nvidia_smi()


def test_with_nvidia_smi():
    """使用nvidia-smi进行显存测试"""
    print("=" * 80)
    print("使用nvidia-smi监控显存")
    print("=" * 80)
    
    # 获取GPU显存使用
    rc, stdout, _ = run_cmd(
        "nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free "
        "--format=csv,noheader"
    )
    if rc == 0:
        print("\n当前显存使用情况:")
        print("GPU   已用     总计     可用")
        print(stdout)
    
    # GPU功耗和温度
    rc, stdout, _ = run_cmd(
        "nvidia-smi --query-gpu=index,power.draw,temperature.gpu,utilization.gpu "
        "--format=csv,noheader"
    )
    if rc == 0:
        print("\nGPU功耗和利用率:")
        print("GPU   功耗(W)  温度(°C)  利用率(%)")
        print(stdout)


def check_p2p_access():
    """检查GPU间P2P访问支持"""
    print("\n" + "=" * 80)
    print("GPU P2P访问检查")
    print("=" * 80)
    
    # 检查CUDA样本
    rc, stdout, _ = run_cmd("find /usr/local/cuda -name deviceQuery 2>/dev/null | head -1")
    if rc == 0 and stdout.strip():
        print(f"运行deviceQuery...\n")
        rc, stdout, stderr = run_cmd(stdout.strip(), timeout=30)
        print(stdout)
        return
    
    print("未找到deviceQuery工具")
    print("使用nvidia-smi拓扑信息替代...\n")
    
    # nvidia-smi拓扑
    rc, stdout, _ = run_cmd("nvidia-smi topo -m")
    if rc == 0:
        print(stdout)
    
    # P2P支持检查
    rc, stdout, _ = run_cmd("nvidia-smi topo -p2p r")
    if rc == 0:
        print("\nP2P读支持:")
        print(stdout)
    
    rc, stdout, _ = run_cmd("nvidia-smi topo -p2p w")
    if rc == 0:
        print("\nP2P写支持:")
        print(stdout)


def check_nvlink():
    """检查NVLink状态"""
    print("\n" + "=" * 80)
    print("NVLink状态检查")
    print("=" * 80)
    
    rc, stdout, _ = run_cmd("nvidia-smi nvlink --status")
    if rc == 0:
        print(stdout)
    
    rc, stdout, _ = run_cmd("nvidia-smi nvlink --capabilities")
    if rc == 0:
        print("\nNVLink能力:")
        print(stdout)


def test_pcie_bandwidth():
    """估算PCIe带宽"""
    print("\n" + "=" * 80)
    print("PCIe带宽估算")
    print("=" * 80)
    
    # 获取PCIe信息
    rc, stdout, _ = run_cmd(
        "nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current "
        "--format=csv,noheader"
    )
    if rc == 0:
        print("\nPCIe链路配置:")
        print("GPU   代次   宽度")
        print(stdout)
        
        print("\nPCIe理论带宽:")
        print("  PCIe Gen3 x16: ~16 GB/s (单向)")
        print("  PCIe Gen4 x16: ~32 GB/s (单向)")
        print("  A800 PCIe版本: Gen4")


def main():
    print("GPU显存带宽诊断工具")
    print("=" * 80)
    print(f"主机名: {run_cmd('hostname')[1].strip()}")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # GPU信息
    check_gpu_info()
    
    # 显存带宽测试
    test_gpu_memory_bandwidth_cuda()
    
    # P2P访问
    check_p2p_access()
    
    # NVLink
    check_nvlink()
    
    # PCIe带宽
    test_pcie_bandwidth()
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
