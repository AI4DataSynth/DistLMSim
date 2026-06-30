"""DistLMSim 模拟运行时上下文

持有所有共享状态：配置、通信模型、预测器、指标收集器等。
从 main.py 提取为独立模块，消除 speculative_decoder→main 的循环依赖。

依赖层次: Layer 5
  输入: config, entities, topology (nvlink_model, rdma_model, overlap_processor),
        execution (execution_time_predictor), metrics (metrics_store)
  输出: SimContext (被 simulator, speculative_decoder 等高层模块消费)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from distlmsim.config import (
    DeviceSKUConfig,
    ModelConfig,
    NetworkTopologyConfig,
)
from distlmsim.entities import Request
from distlmsim.execution.execution_time_predictor import (
    ExecutionTimePredictor,
    create_predictor,
)
from distlmsim.metrics.metrics_store import MetricsStore
from distlmsim.topology.nvlink_model import NVLinkModel
from distlmsim.topology.overlap_processor import OverlapConfig, OverlapProcessor
from distlmsim.topology.rdma_model import RDMAModel


@dataclass
class SimContext:
    """模拟运行时上下文，持有所有共享状态。"""

    model_config: ModelConfig
    device_config: DeviceSKUConfig
    network_config: NetworkTopologyConfig
    # 通信模型
    nvlink_model: NVLinkModel = field(default=None)
    rdma_model: RDMAModel = field(default=None)
    # 通信-计算重叠模型
    overlap_processor: OverlapProcessor = field(default=None)
    # 执行时间预测
    time_predictor: ExecutionTimePredictor = field(default=None)
    # 请求池
    requests: Dict[int, Request] = field(default_factory=dict)
    # 指标
    metrics_store: MetricsStore = field(default=None)
    # 集群参数
    num_gpus_per_node: int = 4
    prefill_node_id: int = 0
    decode_node_id: int = 1
    tp_size: int = 4
    # Profiling 配置
    profiling_dir: Optional[str] = None
    predictor_type: str = "auto"

    def __post_init__(self):
        if self.nvlink_model is None:
            self.nvlink_model = NVLinkModel(
                self.network_config.nvlink, self.num_gpus_per_node
            )
        if self.rdma_model is None:
            self.rdma_model = RDMAModel(
                self.network_config.rdma,
                congestion_alpha=self.network_config.rdma.congestion_alpha,
            )
        if self.overlap_processor is None:
            self.overlap_processor = OverlapProcessor(OverlapConfig())
        if self.time_predictor is None:
            self.time_predictor = create_predictor(
                self.model_config,
                self.device_config,
                self.profiling_dir,
                self.predictor_type,
            )
