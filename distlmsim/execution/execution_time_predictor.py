"""执行时间预测器

基于 RandomForest 或解析模型预测模型各子阶段的执行时间。
复用 TRADIOS 的预测器模式。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from distlmsim.config import ModelConfig, DeviceSKUConfig
from distlmsim.entities import ExecutionTime


class ExecutionTimePredictor(ABC):
    """执行时间预测器基类。

    预测模型前向传播中各子阶段的执行时间。
    """

    @abstractmethod
    def get_execution_time(
        self,
        num_tokens: int,
        batch_size: int,
        kv_cache_size: int,
        is_prefill: bool,
    ) -> ExecutionTime:
        """获取单层执行时间。

        Args:
            num_tokens: batch 中的总 token 数
            batch_size: batch 中的请求数
            kv_cache_size: KV Cache 当前大小
            is_prefill: 是否为 prefill 阶段

        Returns:
            ExecutionTime 对象
        """
        ...


class AnalyticalPredictor(ExecutionTimePredictor):
    """解析模型预测器。

    基于 GPU FLOPS 和内存带宽的理论计算。
    适用于快速原型验证，无需 profiling 数据。
    """

    def __init__(
        self,
        model_config: ModelConfig,
        device_config: DeviceSKUConfig,
    ):
        self._model = model_config
        self._device = device_config

    def get_execution_time(
        self,
        num_tokens: int,
        batch_size: int,
        kv_cache_size: int,
        is_prefill: bool,
    ) -> ExecutionTime:
        # 理论计算
        # 计算时间 = FLOPs / GPU_FLOPS
        # 内存时间 = data_bytes / memory_bandwidth

        head_dim = self._model.embedding_dim // self._model.num_q_heads

        # Attention 计算量 (per layer)
        # Q projection: 2 * tokens * embd * (q_heads * head_dim)
        qkv_flops = 2 * num_tokens * self._model.embedding_dim * (
            self._model.num_q_heads + 2 * self._model.num_kv_heads
        ) * head_dim
        # Attention scores: 2 * tokens * q_heads * kv_len * head_dim (prefill)
        #                  2 * batch_size * q_heads * kv_len * head_dim (decode)
        if is_prefill:
            attn_flops = 2 * num_tokens * self._model.num_q_heads * num_tokens * head_dim
        else:
            attn_flops = 2 * batch_size * self._model.num_q_heads * kv_cache_size * head_dim

        # MLP (gated): 3 * 2 * tokens * embd * mlp_hidden
        mlp_hidden = self._model.mlp_hidden_dim or int(self._model.embedding_dim * 8 / 3)
        mlp_flops = 3 * 2 * num_tokens * self._model.embedding_dim * mlp_hidden

        total_flops = qkv_flops + attn_flops + mlp_flops
        gpu_flops_per_s = self._device.fp16_tflops * 1e12
        compute_time_ms = total_flops / gpu_flops_per_s * 1e3

        # 简单分配各子阶段
        attn_ratio = (qkv_flops + attn_flops) / total_flops
        mlp_ratio = mlp_flops / total_flops

        return ExecutionTime(
            attn_pre_proj_time=compute_time_ms * attn_ratio * 0.3,
            attn_rope_time=compute_time_ms * 0.02,
            attn_kv_cache_save_time=compute_time_ms * 0.05,
            attn_prefill_time=compute_time_ms * attn_ratio * 0.5 if is_prefill else 0.0,
            attn_decode_time=compute_time_ms * attn_ratio * 0.5 if not is_prefill else 0.0,
            attn_post_proj_time=compute_time_ms * attn_ratio * 0.2,
            mlp_up_proj_time=compute_time_ms * mlp_ratio * 0.4,
            mlp_act_time=compute_time_ms * mlp_ratio * 0.1,
            mlp_down_proj_time=compute_time_ms * mlp_ratio * 0.4,
            input_layernorm_time=compute_time_ms * 0.01,
            post_attention_layernorm_time=compute_time_ms * 0.01,
            add_time=compute_time_ms * 0.01,
        )


class RandomForestPredictor(ExecutionTimePredictor):
    """RandomForest 预测器。

    从 profiling CSV 数据训练 RandomForest 模型。
    复用 TRADIOS 的训练和缓存机制。

    TODO: 实现从 TRADIOS 移植的 RF 训练逻辑。
    """

    def __init__(self, model_config: ModelConfig, device_config: DeviceSKUConfig):
        self._model = model_config
        self._device = device_config
        self._fallback = AnalyticalPredictor(model_config, device_config)

    def get_execution_time(
        self,
        num_tokens: int,
        batch_size: int,
        kv_cache_size: int,
        is_prefill: bool,
    ) -> ExecutionTime:
        # TODO: 实现 RF 预测，当前回退到解析模型
        return self._fallback.get_execution_time(
            num_tokens, batch_size, kv_cache_size, is_prefill
        )
