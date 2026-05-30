"""执行时间预测器

基于 RandomForest、Profiling 线性回归或解析模型预测模型各子阶段的执行时间。
复用 TRADIOS 的预测器模式。
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np

from distlmsim.config import ModelConfig, DeviceSKUConfig
from distlmsim.entities import ExecutionTime

logger = logging.getLogger(__name__)


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


class ProfilingBasedPredictor(ExecutionTimePredictor):
    """基于 Profiling CSV 数据的线性回归预测器。

    从 TRADIOS 格式的 CSV profiling 数据中训练简单线性回归模型（np.polyfit），
    并预计算所有可能输入组合的预测值，存入字典实现 O(1) 查询。

    CSV 文件路径格式:
        <profiling_dir>/compute/<device>/<org>/<model>/mlp.csv
        <profiling_dir>/compute/<device>/<org>/<model>/attention.csv
        <profiling_dir>/network/all_reduce.csv

    如果 CSV 文件不存在，回退到 AnalyticalPredictor。
    """

    # MLP 子模型名称 -> CSV 目标列
    _MLP_TARGETS: Dict[str, str] = {
        "attn_pre_proj": "time_stats.attn_pre_proj.median",
        "mlp_up_proj": "time_stats.mlp_up_proj.median",
        "mlp_act": "time_stats.mlp_act.median",
        "mlp_down_proj": "time_stats.mlp_down_proj.median",
        "attn_post_proj": "time_stats.attn_post_proj.median",
        "attn_rope": "time_stats.attn_rope.median",
        "input_layernorm": "time_stats.input_layernorm.median",
        "post_attention_layernorm": "time_stats.post_attention_layernorm.median",
        "add": "time_stats.add.median",
    }

    # Attention 子模型名称 -> CSV 目标列
    _ATTN_TARGETS: Dict[str, str] = {
        "attn_prefill": "time_stats.attn_prefill.median",
        "attn_decode": "time_stats.attn_decode.median",
        "attn_kv_cache_save": "time_stats.attn_kv_cache_save.median",
    }

    def __init__(
        self,
        model_config: ModelConfig,
        device_config: DeviceSKUConfig,
        profiling_dir: str,
    ):
        self._model = model_config
        self._device = device_config
        self._profiling_dir = profiling_dir
        self._fallback = AnalyticalPredictor(model_config, device_config)

        # 线性回归系数: {sub_model_name: (slope, intercept)}
        self._mlp_models: Dict[str, Tuple[float, float]] = {}
        self._attn_models: Dict[str, Tuple[float, float]] = {}
        self._network_models: Dict[str, Tuple[float, float]] = {}

        # 预测缓存: {(num_tokens, batch_size, kv_cache_size, is_prefill): ExecutionTime}
        self._cache: Dict[Tuple, ExecutionTime] = {}

        self._load_and_train()

    def _csv_path(self, *relative_parts: str) -> str:
        """拼接 CSV 路径。"""
        return os.path.join(self._profiling_dir, *relative_parts)

    @staticmethod
    def _fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """一元线性回归 y = slope * x + intercept，返回 (slope, intercept)。"""
        coeffs = np.polyfit(x, y, deg=1)
        return float(coeffs[0]), float(coeffs[1])

    @staticmethod
    def _predict_linear(slope: float, intercept: float, x: float) -> float:
        """线性预测，结果不低于 0。"""
        return max(0.0, slope * x + intercept)

    def _load_csv(self, path: str) -> Optional[object]:
        """安全加载 CSV 文件，返回 pandas DataFrame 或 None。"""
        if not os.path.isfile(path):
            logger.warning("Profiling CSV 不存在: %s", path)
            return None
        try:
            import pandas as pd
            return pd.read_csv(path)
        except Exception as e:
            logger.warning("加载 CSV 失败 %s: %s", path, e)
            return None

    def _train_mlp_models(self) -> None:
        """从 mlp.csv 训练各子模型的线性回归。"""
        path = self._csv_path("mlp.csv")
        df = self._load_csv(path)
        if df is None:
            return
        feature_col = "num_tokens"
        if feature_col not in df.columns:
            logger.warning("mlp.csv 缺少列 %s", feature_col)
            return
        x = df[feature_col].values.astype(float)
        for name, target_col in self._MLP_TARGETS.items():
            if target_col in df.columns:
                y = df[target_col].values.astype(float)
                self._mlp_models[name] = self._fit_linear(x, y)
                logger.debug("训练 MLP 子模型 %s: slope=%.6f, intercept=%.6f",
                             name, *self._mlp_models[name])

    def _train_attention_models(self) -> None:
        """从 attention.csv 训练 attention 子模型。

        对 prefill: 使用 num_tokens 作为特征
        对 decode: 使用 batch_size 作为特征
        对 kv_cache_save: 使用 kv_cache_size 作为特征
        """
        path = self._csv_path("attention.csv")
        df = self._load_csv(path)
        if df is None:
            return

        # attn_prefill: 特征 = num_tokens (仅 is_prefill==1 行)
        target = "time_stats.attn_prefill.median"
        if target in df.columns and "num_tokens" in df.columns:
            mask = df["is_prefill"] == 1 if "is_prefill" in df.columns else slice(None)
            subset = df.loc[mask] if isinstance(mask, object) else df
            if len(subset) > 0:
                x = subset["num_tokens"].values.astype(float)
                y = subset[target].values.astype(float)
                self._attn_models["attn_prefill"] = self._fit_linear(x, y)

        # attn_decode: 特征 = batch_size (仅 is_prefill==0 行)
        target = "time_stats.attn_decode.median"
        if target in df.columns and "batch_size" in df.columns:
            mask = df["is_prefill"] == 0 if "is_prefill" in df.columns else slice(None)
            subset = df.loc[mask] if isinstance(mask, object) else df
            if len(subset) > 0:
                x = subset["batch_size"].values.astype(float)
                y = subset[target].values.astype(float)
                self._attn_models["attn_decode"] = self._fit_linear(x, y)

        # attn_kv_cache_save: 特征 = kv_cache_size
        target = "time_stats.attn_kv_cache_save.median"
        if target in df.columns and "kv_cache_size" in df.columns:
            x = df["kv_cache_size"].values.astype(float)
            y = df[target].values.astype(float)
            self._attn_models["attn_kv_cache_save"] = self._fit_linear(x, y)

    def _train_network_models(self) -> None:
        """从 all_reduce.csv 训练网络模型。特征 = size。"""
        path = self._csv_path("all_reduce.csv")
        df = self._load_csv(path)
        if df is None:
            return
        target = "time_stats.all_reduce.median"
        if target in df.columns and "size" in df.columns:
            x = df["size"].values.astype(float)
            y = df[target].values.astype(float)
            self._network_models["all_reduce"] = self._fit_linear(x, y)

    def _load_and_train(self) -> None:
        """加载所有 CSV 并训练模型。"""
        self._train_mlp_models()
        self._train_attention_models()
        self._train_network_models()

        if not self._mlp_models and not self._attn_models:
            logger.warning("ProfilingBasedPredictor: 未训练任何子模型，将回退到 AnalyticalPredictor")

    def _predict_from_models(
        self,
        num_tokens: int,
        batch_size: int,
        kv_cache_size: int,
        is_prefill: bool,
    ) -> ExecutionTime:
        """从线性回归模型预测各子阶段时间。"""
        et = ExecutionTime()

        # MLP 子模型: 特征 = num_tokens
        for attr, name in [
            ("attn_pre_proj_time", "attn_pre_proj"),
            ("mlp_up_proj_time", "mlp_up_proj"),
            ("mlp_act_time", "mlp_act"),
            ("mlp_down_proj_time", "mlp_down_proj"),
            ("attn_post_proj_time", "attn_post_proj"),
            ("attn_rope_time", "attn_rope"),
            ("input_layernorm_time", "input_layernorm"),
            ("post_attention_layernorm_time", "post_attention_layernorm"),
            ("add_time", "add"),
        ]:
            if name in self._mlp_models:
                slope, intercept = self._mlp_models[name]
                setattr(et, attr, self._predict_linear(slope, intercept, num_tokens))

        # Attention prefill/decode
        if is_prefill and "attn_prefill" in self._attn_models:
            slope, intercept = self._attn_models["attn_prefill"]
            et.attn_prefill_time = self._predict_linear(slope, intercept, num_tokens)
        elif not is_prefill and "attn_decode" in self._attn_models:
            slope, intercept = self._attn_models["attn_decode"]
            et.attn_decode_time = self._predict_linear(slope, intercept, batch_size)

        # KV cache save
        if "attn_kv_cache_save" in self._attn_models:
            slope, intercept = self._attn_models["attn_kv_cache_save"]
            et.attn_kv_cache_save_time = self._predict_linear(slope, intercept, kv_cache_size)

        return et

    def get_execution_time(
        self,
        num_tokens: int,
        batch_size: int,
        kv_cache_size: int,
        is_prefill: bool,
    ) -> ExecutionTime:
        cache_key = (num_tokens, batch_size, kv_cache_size, is_prefill)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self._mlp_models and not self._attn_models:
            # 无训练模型 -> 回退
            result = self._fallback.get_execution_time(
                num_tokens, batch_size, kv_cache_size, is_prefill
            )
        else:
            result = self._predict_from_models(
                num_tokens, batch_size, kv_cache_size, is_prefill
            )

        self._cache[cache_key] = result
        return result

    def precompute_cache(
        self,
        num_tokens_range: list[int],
        batch_size_range: list[int],
        kv_cache_size_range: list[int],
    ) -> None:
        """预计算所有参数组合的预测值，填充缓存以实现 O(1) 查询。

        Args:
            num_tokens_range: num_tokens 取值列表
            batch_size_range: batch_size 取值列表
            kv_cache_size_range: kv_cache_size 取值列表
        """
        for nt in num_tokens_range:
            for bs in batch_size_range:
                for kvs in kv_cache_size_range:
                    for is_pf in (True, False):
                        self.get_execution_time(nt, bs, kvs, is_pf)
        logger.info("预计算缓存完成，共 %d 条", len(self._cache))


def create_predictor(
    model_config: ModelConfig,
    device_config: DeviceSKUConfig,
    profiling_dir: Optional[str] = None,
) -> ExecutionTimePredictor:
    """工厂函数：创建执行时间预测器。

    如果 profiling_dir 存在且包含 CSV 文件，返回 ProfilingBasedPredictor；
    否则返回 AnalyticalPredictor。

    Args:
        model_config: 模型配置
        device_config: 设备配置
        profiling_dir: profiling 数据目录路径（可选）

    Returns:
        ExecutionTimePredictor 实例
    """
    if profiling_dir is not None and os.path.isdir(profiling_dir):
        # 检查是否至少有一个 CSV 文件
        has_csv = False
        for root, _dirs, files in os.walk(profiling_dir):
            if any(f.endswith(".csv") for f in files):
                has_csv = True
                break
        if has_csv:
            logger.info("使用 ProfilingBasedPredictor，数据目录: %s", profiling_dir)
            return ProfilingBasedPredictor(model_config, device_config, profiling_dir)
        else:
            logger.info("profiling 目录无 CSV 文件，回退到 AnalyticalPredictor")
    else:
        logger.info("未提供 profiling 目录，使用 AnalyticalPredictor")

    return AnalyticalPredictor(model_config, device_config)
