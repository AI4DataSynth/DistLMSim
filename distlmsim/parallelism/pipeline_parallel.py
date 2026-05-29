"""流水线并行 (Pipeline Parallelism) 模型

PP 将模型按层切分为多个 stage，stage 间通过 send/recv 通信。
同节点 stage 走 NVLink，跨节点 stage 走 RDMA。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from distlmsim.config import ModelConfig


@dataclass
class PipelineStage:
    """流水线 stage 定义。"""
    stage_id: int
    start_layer: int        # 起始层 (inclusive)
    end_layer: int          # 结束层 (exclusive)
    node_id: int            # 所在节点
    gpu_ids: List[int]      # 使用的 GPU (如果 TP>1)

    @property
    def num_layers(self) -> int:
        return self.end_layer - self.start_layer


@dataclass
class PipelineSchedule:
    """流水线调度方案。"""
    stages: List[PipelineStage]
    schedule_type: str       # "1f1b", "gpipe", "interleaved"
    num_micro_batches: int   # 微批数量


class PipelineParallelModel:
    """流水线并行模型。

    负责:
    1. 将模型层均匀切分为 PP stages
    2. 分配 stages 到节点 (考虑 TP 亲和性)
    3. 计算 stage 间通信数据量和时间
    4. 选择流水线调度策略 (1F1B, GPipe, Interleaved)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        pp_size: int,
        tp_size: int = 1,
    ):
        self._model = model_config
        self._pp_size = pp_size
        self._tp_size = tp_size

    def create_stages(
        self,
        node_gpu_mapping: dict[int, List[int]],
    ) -> List[PipelineStage]:
        """创建流水线 stages。

        将模型层均匀切分，并映射到物理节点/GPU。

        Args:
            node_gpu_mapping: stage_id -> GPU ID 列表

        Returns:
            PipelineStage 列表
        """
        layers_per_stage = self._model.num_layers // self._pp_size
        stages = []

        for stage_id in range(self._pp_size):
            start_layer = stage_id * layers_per_stage
            end_layer = (
                (stage_id + 1) * layers_per_stage
                if stage_id < self._pp_size - 1
                else self._model.num_layers
            )
            gpu_ids = node_gpu_mapping.get(stage_id, [])
            node_id = gpu_ids[0] // 8 if gpu_ids else 0  # TODO: 正确映射

            stages.append(PipelineStage(
                stage_id=stage_id,
                start_layer=start_layer,
                end_layer=end_layer,
                node_id=node_id,
                gpu_ids=gpu_ids,
            ))

        return stages

    def get_stage_comm_data_size(self, num_tokens: int) -> int:
        """计算 stage 间通信的数据量 (bytes)。

        通信数据 = 激活张量 = [num_tokens, embedding_dim] (float16)

        Args:
            num_tokens: batch 中的 token 数

        Returns:
            数据量 (bytes)
        """
        if self._pp_size <= 1:
            return 0

        return num_tokens * self._model.embedding_dim * 2  # float16

    def get_pipeline_bubble_ratio(self, num_micro_batches: int) -> float:
        """计算流水线气泡比例。

        1F1B 调度: bubble_ratio = (PP - 1) / num_micro_batches
        GPipe: bubble_ratio = (PP - 1) / num_micro_batches

        Returns:
            气泡比例 (0-1)
        """
        if self._pp_size <= 1 or num_micro_batches <= 0:
            return 0.0
        return (self._pp_size - 1) / num_micro_batches

    def create_schedule(
        self,
        num_micro_batches: int = 4,
        schedule_type: str = "1f1b",
    ) -> PipelineSchedule:
        """创建流水线调度方案。

        Args:
            num_micro_batches: 微批数量
            schedule_type: 调度策略 ("1f1b", "gpipe", "interleaved")

        Returns:
            PipelineSchedule
        """
        # TODO: 实现完整的 stage 到节点的映射
        stages = self.create_stages({i: [i * self._tp_size] for i in range(self._pp_size)})
        return PipelineSchedule(
            stages=stages,
            schedule_type=schedule_type,
            num_micro_batches=num_micro_batches,
        )
