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
    """解析模型预测器 (含 Roofline 模型)。

    基于 Roofline 模型: time = max(compute_time, memory_time)
    - compute_time = FLOPS / (peak_FLOPS * compute_efficiency)
    - memory_time  = bytes  / (memory_bandwidth * memory_efficiency)

    对 prefill (compute-bound) 和 decode (memory-bound) 均适用。
    """

    # Roofline 效率因子
    COMPUTE_EFFICIENCY = 0.85
    MEMORY_EFFICIENCY = 0.90
    # FP16 每元素字节数
    BPE = 2

    def __init__(
        self,
        model_config: ModelConfig,
        device_config: DeviceSKUConfig,
    ):
        self._model = model_config
        self._device = device_config

    def _roofline_time_ms(
        self, flops: int, memory_bytes: int,
    ) -> float:
        """Roofline 模型: time = max(compute, memory)"""
        peak_flops = self._device.fp16_tflops * 1e12
        mem_bw = self._device.memory_bandwidth_gbps * 1e9  # GB/s → bytes/s
        compute_ms = flops / (peak_flops * self.COMPUTE_EFFICIENCY) * 1e3
        memory_ms = memory_bytes / (mem_bw * self.MEMORY_EFFICIENCY) * 1e3
        return max(compute_ms, memory_ms)

    def get_execution_time(
        self,
        num_tokens: int,
        batch_size: int,
        kv_cache_size: int,
        is_prefill: bool,
    ) -> ExecutionTime:
        m = self._model
        h = m.embedding_dim
        hd = h // m.num_q_heads
        nq = m.num_q_heads
        nkv = m.num_kv_heads
        mlp_hidden = m.mlp_hidden_dim or int(h * 8 / 3)
        bpe = self.BPE

        # --- QKV 投影 (per layer) ---
        qkv_out_dim = (nq + 2 * nkv) * hd
        qkv_flops = 2 * num_tokens * h * qkv_out_dim
        qkv_mem = (h * qkv_out_dim + num_tokens * h + num_tokens * qkv_out_dim) * bpe
        qkv_time = self._roofline_time_ms(qkv_flops, qkv_mem)

        # --- Attention (per layer) ---
        if is_prefill:
            # Chunked prefill: 新 chunk 需要 attend 到自身 + 之前积累的 KV cache
            effective_kv_len = num_tokens + kv_cache_size
            attn_flops = 4 * num_tokens * nq * effective_kv_len * hd
            # 内存: 读取已有 KV cache + 新 Q/K/V + 输出
            kv_read = 2 * batch_size * nkv * kv_cache_size * hd * bpe
            qkv_new = (3 * num_tokens * nq * hd) * bpe
            out_write = num_tokens * nq * hd * bpe
            attn_mem = kv_read + qkv_new + out_write
        else:
            attn_flops = 4 * batch_size * nq * kv_cache_size * hd
            # KV cache 读取: 2 * batch * nkv * kv_len * hd
            kv_read = 2 * batch_size * nkv * kv_cache_size * hd * bpe
            q_read = batch_size * nq * hd * bpe
            out_write = batch_size * nq * hd * bpe
            attn_mem = kv_read + q_read + out_write
        attn_time = self._roofline_time_ms(attn_flops, attn_mem)

        # --- O 投影 (per layer) ---
        o_flops = 2 * num_tokens * (nq * hd) * h
        o_mem = (nq * hd * h + num_tokens * nq * hd + num_tokens * h) * bpe
        o_time = self._roofline_time_ms(o_flops, o_mem)

        # --- MLP (SwiGLU, per layer) ---
        # gate + up: 2 个线性层, down: 1 个线性层
        gate_up_flops = 2 * 2 * num_tokens * h * mlp_hidden
        gate_up_mem = (2 * h * mlp_hidden + num_tokens * h + 2 * num_tokens * mlp_hidden) * bpe
        gate_up_time = self._roofline_time_ms(gate_up_flops, gate_up_mem)

        down_flops = 2 * num_tokens * mlp_hidden * h
        down_mem = (mlp_hidden * h + num_tokens * mlp_hidden + num_tokens * h) * bpe
        down_time = self._roofline_time_ms(down_flops, down_mem)

        # --- 小操作 (norm, rope, add, activation) ---
        # 这些是 memory-bound, 按数据量估算
        small_bytes = num_tokens * h * bpe
        mem_bw = self._device.memory_bandwidth_gbps * 1e9 * self.MEMORY_EFFICIENCY
        norm_time = small_bytes / mem_bw * 1e3  # 一次 norm
        rope_time = small_bytes / mem_bw * 1e3 * 0.5
        add_time = small_bytes / mem_bw * 1e3 * 0.3
        act_time = num_tokens * mlp_hidden * bpe / mem_bw * 1e3  # SiLU

        # KV cache save
        kv_save_bytes = 2 * batch_size * nkv * hd * bpe  # K + V 写入
        kv_save_time = kv_save_bytes / mem_bw * 1e3

        # --- MoE Expert MLP (per layer, 均衡负载下最慢专家的时间) ---
        expert_mlp_time = 0.0
        if m.num_experts > 0:
            expert_hidden = m.expert_intermediate_dim or int(h * 0.375)
            tokens_per_expert_avg = num_tokens * m.top_k_experts / m.num_experts
            # gate_up: h → expert_hidden (SwiGLU 风格，gate + up 合并)
            exp_gate_up_flops = 2 * tokens_per_expert_avg * (h * expert_hidden * 2)
            exp_gate_up_mem = (h * expert_hidden * 2 + tokens_per_expert_avg * h
                               + tokens_per_expert_avg * expert_hidden * 2) * bpe
            exp_gate_up_time = self._roofline_time_ms(exp_gate_up_flops, exp_gate_up_mem)
            # down: expert_hidden → h
            exp_down_flops = 2 * tokens_per_expert_avg * expert_hidden * h
            exp_down_mem = (expert_hidden * h + tokens_per_expert_avg * expert_hidden
                            + tokens_per_expert_avg * h) * bpe
            exp_down_time = self._roofline_time_ms(exp_down_flops, exp_down_mem)
            # 专家激活
            exp_act_time = tokens_per_expert_avg * expert_hidden * bpe / mem_bw * 1e3
            expert_mlp_time = exp_gate_up_time + exp_act_time + exp_down_time

        return ExecutionTime(
            attn_pre_proj_time=qkv_time * 0.5,
            attn_rope_time=rope_time,
            attn_kv_cache_save_time=kv_save_time,
            attn_prefill_time=attn_time if is_prefill else 0.0,
            attn_decode_time=attn_time if not is_prefill else 0.0,
            attn_post_proj_time=o_time,
            mlp_up_proj_time=gate_up_time * 0.5,
            mlp_act_time=act_time,
            mlp_down_proj_time=down_time,
            input_layernorm_time=norm_time,
            post_attention_layernorm_time=norm_time,
            add_time=add_time,
            expert_mlp_time=expert_mlp_time,
        )


class RandomForestPredictor(ExecutionTimePredictor):
    """RandomForest 预测器。

    从 profiling CSV 数据训练 RandomForest 模型。
    使用 sklearn RandomForestRegressor 进行特征工程和训练。

    特征工程：
    - Attention: [num_tokens, batch_size, kv_cache_size, is_prefill]
    - MLP: [num_tokens]

    训练数据来自 TRADIOS 格式的 CSV 文件。
    """

    def __init__(self, model_config: ModelConfig, device_config: DeviceSKUConfig, profiling_dir: str):
        from sklearn.ensemble import RandomForestRegressor

        self._model = model_config
        self._device = device_config
        self._profiling_dir = profiling_dir
        self._fallback = AnalyticalPredictor(model_config, device_config)

        # 模型字典：name -> RandomForestRegressor
        self._attn_models: Dict[str, RandomForestRegressor] = {}
        self._mlp_models: Dict[str, RandomForestRegressor] = {}
        # Expert MLP 模型
        self._expert_model: Optional[RandomForestRegressor] = None

        # 训练模型
        self._train_attention_models()
        self._train_mlp_models()
        self._train_expert_models()

        logger.info("RandomForestPredictor 训练完成: %d attention 子模型, %d MLP 子模型, expert=%s",
                    len(self._attn_models), len(self._mlp_models),
                    "yes" if self._expert_model else "no")

    def _csv_path(self, *relative_parts: str) -> str:
        """拼接 CSV 路径。"""
        return os.path.join(self._profiling_dir, *relative_parts)

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

    def _train_attention_models(self) -> None:
        """从 attention.csv 训练 attention 子模型。只使用 TP=1 数据。"""
        from sklearn.ensemble import RandomForestRegressor

        path = self._csv_path("attention.csv")
        df = self._load_csv(path)
        if df is None:
            return
        
        # Filter for TP=1 only (simulator default)
        if "num_tensor_parallel_workers" in df.columns:
            df = df[df["num_tensor_parallel_workers"] == 1]
            logger.debug("Filtering attention CSV for TP=1: %d rows", len(df))

        # attn_prefill: 特征 = [num_tokens] (仅 is_prefill==1 行)
        target = "time_stats.attn_prefill.median"
        if target in df.columns and "num_tokens" in df.columns:
            mask = df["is_prefill"] == True if "is_prefill" in df.columns else slice(None)
            subset = df.loc[mask] if "is_prefill" in df.columns else df
            if len(subset) > 0:
                X = subset[["num_tokens"]].values.astype(float)
                y = subset[target].values.astype(float)
                model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
                model.fit(X, y)
                self._attn_models["attn_prefill"] = model
                logger.debug("训练 attn_prefill RF 模型: %d 样本", len(X))

        # attn_decode: 特征 = [batch_size, kv_cache_size] (仅 is_prefill==0 行)
        target = "time_stats.attn_decode.median"
        if target in df.columns and "batch_size" in df.columns and "kv_cache_size" in df.columns:
            mask = df["is_prefill"] == False if "is_prefill" in df.columns else slice(None)
            subset = df.loc[mask] if "is_prefill" in df.columns else df
            if len(subset) > 0:
                X = subset[["batch_size", "kv_cache_size"]].values.astype(float)
                y = subset[target].values.astype(float)
                model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
                model.fit(X, y)
                self._attn_models["attn_decode"] = model
                logger.debug("训练 attn_decode RF 模型: %d 样本", len(X))

        # attn_kv_cache_save: 特征 = [kv_cache_size]
        target = "time_stats.attn_kv_cache_save.median"
        if target in df.columns and "kv_cache_size" in df.columns:
            X = df[["kv_cache_size"]].values.astype(float)
            y = df[target].values.astype(float)
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            model.fit(X, y)
            self._attn_models["attn_kv_cache_save"] = model
            logger.debug("训练 attn_kv_cache_save RF 模型: %d 样本", len(X))

    def _train_mlp_models(self) -> None:
        """从 mlp.csv 训练各子模型的 RandomForest。"""
        from sklearn.ensemble import RandomForestRegressor
        
        path = self._csv_path("mlp.csv")
        df = self._load_csv(path)
        if df is None:
            return
        feature_col = "num_tokens"
        if feature_col not in df.columns:
            logger.warning("mlp.csv 缺少列 %s", feature_col)
            return
        
        mlp_targets = {
            "emb": "time_stats.emb.median",
            "input_layernorm": "time_stats.input_layernorm.median",
            "attn_pre_proj": "time_stats.attn_pre_proj.median",
            "attn_rope": "time_stats.attn_rope.median",
            "attn_post_proj": "time_stats.attn_post_proj.median",
            "post_attention_layernorm": "time_stats.post_attention_layernorm.median",
            "mlp_up_proj": "time_stats.mlp_up_proj.median",
            "mlp_act": "time_stats.mlp_act.median",
            "mlp_down_proj": "time_stats.mlp_down_proj.median",
            "add": "time_stats.add.median",
        }
        
        X = df[[feature_col]].values.astype(float)
        for name, target_col in mlp_targets.items():
            if target_col in df.columns:
                y = df[target_col].values.astype(float)
                model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
                model.fit(X, y)
                self._mlp_models[name] = model
                logger.debug("训练 MLP 子模型 %s RF: %d 样本", name, len(X))

    def _train_expert_models(self) -> None:
        """从 expert.csv 训练 expert MLP 模型。"""
        from sklearn.ensemble import RandomForestRegressor

        path = self._csv_path("expert.csv")
        df = self._load_csv(path)
        if df is None:
            return
        target = "time_stats.expert_mlp.median"
        feature_col = "num_tokens"
        if target in df.columns and feature_col in df.columns:
            X = df[[feature_col]].values.astype(float)
            y = df[target].values.astype(float)
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            model.fit(X, y)
            self._expert_model = model
            logger.debug("训练 expert_mlp RF 模型: %d 样本", len(X))

    def get_execution_time(
        self,
        num_tokens: int,
        batch_size: int,
        kv_cache_size: int,
        is_prefill: bool,
    ) -> ExecutionTime:
        """使用 RF 模型预测执行时间。"""
        result = ExecutionTime()

        # Attention 预测
        if is_prefill and "attn_prefill" in self._attn_models:
            X = np.array([[num_tokens]], dtype=float)
            result.attn_prefill_time = float(self._attn_models["attn_prefill"].predict(X)[0])

        if not is_prefill and "attn_decode" in self._attn_models:
            # 双特征: [batch_size, kv_cache_size]
            X = np.array([[batch_size, kv_cache_size]], dtype=float)
            result.attn_decode_time = float(self._attn_models["attn_decode"].predict(X)[0])

        if "attn_kv_cache_save" in self._attn_models:
            X = np.array([[kv_cache_size]], dtype=float)
            result.attn_kv_cache_save_time = float(self._attn_models["attn_kv_cache_save"].predict(X)[0])

        # MLP 预测
        X = np.array([[num_tokens]], dtype=float)
        for name, model in self._mlp_models.items():
            pred = float(model.predict(X)[0])
            if name == "emb":
                result.emb_time = pred
            elif name == "input_layernorm":
                result.input_layernorm_time = pred
            elif name == "attn_pre_proj":
                result.attn_pre_proj_time = pred
            elif name == "attn_rope":
                result.attn_rope_time = pred
            elif name == "attn_post_proj":
                result.attn_post_proj_time = pred
            elif name == "post_attention_layernorm":
                result.post_attention_layernorm_time = pred
            elif name == "mlp_up_proj":
                result.mlp_up_proj_time = pred
            elif name == "mlp_act":
                result.mlp_act_time = pred
            elif name == "mlp_down_proj":
                result.mlp_down_proj_time = pred
            elif name == "add":
                result.add_time = pred

        # Expert MLP 预测
        if self._expert_model is not None and num_tokens > 0:
            X = np.array([[num_tokens]], dtype=float)
            result.expert_mlp_time = float(self._expert_model.predict(X)[0])

        # 如果 RF 模型未覆盖所有子模型，使用 fallback
        if result.total_time == 0.0:
            return self._fallback.get_execution_time(num_tokens, batch_size, kv_cache_size, is_prefill)

        return result


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
        self._mlp_models: Dict[str, Tuple[float, float, float, float]] = {}  # (small_const, slope, intercept, threshold)
        self._attn_models: Dict[str, Tuple[float, float]] = {}
        self._network_models: Dict[str, Tuple[float, float]] = {}
        # 双特征模型: attn_decode 使用 (batch_size * kv_cache_size) 作为特征
        self._attn_decode_model: Optional[Tuple[float, float]] = None
        # Expert MLP 模型
        self._expert_model: Optional[Tuple[float, float]] = None

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
        """从 mlp.csv 训练各子模型的分段线性回归。
        
        MLP 操作在 num_tokens <= 256 时受 kernel launch overhead 主导，时间基本不变；
        只有 num_tokens > 256 后才开始线性增长。因此使用分段模型：
        - 小 token (<= 256): 常数（取平均值）
        - 大 token (> 256): 线性回归
        """
        path = self._csv_path("mlp.csv")
        df = self._load_csv(path)
        if df is None:
            return
        feature_col = "num_tokens"
        if feature_col not in df.columns:
            logger.warning("mlp.csv 缺少列 %s", feature_col)
            return
        
        # 分段：小 token (<= 256) 和大 token (> 256)
        small_mask = df[feature_col] <= 256
        large_mask = df[feature_col] > 256
        small_df = df[small_mask]
        large_df = df[large_mask]
        
        for name, target_col in self._MLP_TARGETS.items():
            if target_col not in df.columns:
                continue
            
            # 小 token: 常数（平均值）
            small_const = float(small_df[target_col].mean()) if len(small_df) > 0 else 0.0
            
            # 大 token: 线性回归
            if len(large_df) >= 2:
                x_large = large_df[feature_col].values.astype(float)
                y_large = large_df[target_col].values.astype(float)
                slope, intercept = self._fit_linear(x_large, y_large)
            else:
                slope, intercept = 0.0, small_const
            
            # 存储分段模型: (small_const, slope, intercept, threshold)
            self._mlp_models[name] = (small_const, slope, intercept, 256.0)
            logger.debug("训练 MLP 子模型 %s (分段): const=%.6f (nt<=256), slope=%.6f, intercept=%.6f (nt>256)",
                         name, small_const, slope, intercept)

    def _train_attention_models(self) -> None:
        """从 attention.csv 训练 attention 子模型。

        对 prefill: 使用 num_tokens 作为特征
        对 decode: 使用 batch_size * kv_cache_size 作为特征 (attention 时间与两者的乘积成正比)
        对 kv_cache_save: 使用 kv_cache_size 作为特征
        
        只使用 TP=1 的数据（与模拟器默认配置一致）。
        """
        import pandas as pd

        path = self._csv_path("attention.csv")
        df = self._load_csv(path)
        if df is None:
            return
        
        # Filter for TP=1 only (simulator default)
        if "num_tensor_parallel_workers" in df.columns:
            df = df[df["num_tensor_parallel_workers"] == 1]
            logger.debug("Filtering attention CSV for TP=1: %d rows", len(df))

        # attn_prefill: 特征 = num_tokens (仅 is_prefill==1 行)
        target = "time_stats.attn_prefill.median"
        if target in df.columns and "num_tokens" in df.columns:
            mask = df["is_prefill"] == 1 if "is_prefill" in df.columns else slice(None)
            subset = df.loc[mask] if isinstance(mask, pd.Series) else df
            if len(subset) > 0:
                x = subset["num_tokens"].values.astype(float)
                y = subset[target].values.astype(float)
                self._attn_models["attn_prefill"] = self._fit_linear(x, y)

        # attn_decode: 特征 = batch_size * kv_cache_size (仅 is_prefill==0 行)
        # Attention decode 时间 ∝ batch_size × kv_cache_size (每个 token 需要 attend 到所有 KV)
        target = "time_stats.attn_decode.median"
        if target in df.columns and "batch_size" in df.columns and "kv_cache_size" in df.columns:
            mask = df["is_prefill"] == 0 if "is_prefill" in df.columns else slice(None)
            subset = df.loc[mask] if isinstance(mask, pd.Series) else df
            if len(subset) > 0:
                x = (subset["batch_size"] * subset["kv_cache_size"]).values.astype(float)
                y = subset[target].values.astype(float)
                self._attn_decode_model = self._fit_linear(x, y)
                logger.debug("训练 attn_decode 双特征模型: slope=%.10f, intercept=%.6f",
                             *self._attn_decode_model)

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

    def _train_expert_models(self) -> None:
        """从 expert.csv 训练 expert MLP 模型。特征 = num_tokens。"""
        path = self._csv_path("expert.csv")
        df = self._load_csv(path)
        if df is None:
            return
        target = "time_stats.expert_mlp.median"
        feature_col = "num_tokens"
        if target in df.columns and feature_col in df.columns:
            x = df[feature_col].values.astype(float)
            y = df[target].values.astype(float)
            self._expert_model = self._fit_linear(x, y)
            logger.debug("训练 expert_mlp 模型: slope=%.6f, intercept=%.6f", *self._expert_model)

    def _load_and_train(self) -> None:
        """加载所有 CSV 并训练模型。"""
        self._train_mlp_models()
        self._train_attention_models()
        self._train_network_models()
        self._train_expert_models()

        if not self._mlp_models and not self._attn_models and self._attn_decode_model is None:
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

        # MLP 子模型: 分段预测 (小 token 用常数，大 token 用线性)
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
                small_const, slope, intercept, threshold = self._mlp_models[name]
                if num_tokens <= threshold:
                    setattr(et, attr, small_const)
                else:
                    setattr(et, attr, self._predict_linear(slope, intercept, num_tokens))

        # Attention prefill/decode
        if is_prefill and "attn_prefill" in self._attn_models:
            slope, intercept = self._attn_models["attn_prefill"]
            et.attn_prefill_time = self._predict_linear(slope, intercept, num_tokens)
        elif not is_prefill and self._attn_decode_model is not None:
            # 双特征: batch_size * kv_cache_size
            slope, intercept = self._attn_decode_model
            feature = float(batch_size * kv_cache_size)
            et.attn_decode_time = self._predict_linear(slope, intercept, feature)

        # KV cache save
        if "attn_kv_cache_save" in self._attn_models:
            slope, intercept = self._attn_models["attn_kv_cache_save"]
            et.attn_kv_cache_save_time = self._predict_linear(slope, intercept, kv_cache_size)

        # Expert MLP
        if self._expert_model is not None and num_tokens > 0:
            slope, intercept = self._expert_model
            et.expert_mlp_time = self._predict_linear(slope, intercept, num_tokens)

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


class HighFidelityPredictor(ExecutionTimePredictor):
    """高保真预测器：直接从 profiling CSV 查表，无回归、无插值。

    对于每个 (batch_size, kv_cache_size, num_tokens, is_prefill) 组合，
    直接在 profiling CSV 中找到最接近的匹配行，返回其测量值。
    仅当精确匹配不存在时，才回退到 ProfilingBasedPredictor。

    支持 kernel fusion 校正：profiling 分别测量每个子操作（各有 kernel launch
    overhead），但生产框架（如 vLLM）使用 kernel fusion 合并多个子操作。
    通过双 fusion factor 分别校正 prefill 和 decode 阶段：
    - prefill_fusion_factor (默认 0.90): prefill 阶段主要是大矩阵乘法，fusion 收益小
    - decode_fusion_factor (默认 0.55): decode 阶段主要是小矩阵乘法+内存访问，fusion 收益大
    两个因子可独立校准，基于 H100 实测数据优化。

    此模式适用于对精度要求极高的场景（如论文验证、生产部署规划）。
    """

    # MLP 子操作名称 -> CSV 目标列
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

    def __init__(
        self,
        model_config: ModelConfig,
        device_config: DeviceSKUConfig,
        profiling_dir: str,
        prefill_fusion_factor: float = 0.90,
        decode_fusion_factor: float = 0.55,
        fusion_factor: float = None,
    ):
        import pandas as pd

        self._model = model_config
        self._device = device_config
        self._profiling_dir = profiling_dir
        # 支持旧的单一 fusion_factor 参数（向后兼容）
        if fusion_factor is not None:
            self._prefill_fusion_factor = fusion_factor
            self._decode_fusion_factor = fusion_factor
        else:
            self._prefill_fusion_factor = prefill_fusion_factor
            self._decode_fusion_factor = decode_fusion_factor
        # Size-dependent fusion scaling: kernel fusion benefit increases with
        # larger batch/token sizes. Calibrated from H100 vLLM measurements:
        #   pf=512 → fusion=0.85, pf=1024 → fusion=0.51
        # Model: fusion(size) = base_fusion * (size / ref_size) ^ (-alpha)
        # where ref_size=512 for prefill, ref_size=1 for decode, alpha=0.75
        self._fusion_alpha = 0.75
        self._prefill_ref_size = 512
        self._decode_ref_size = 1
        self._fallback = ProfilingBasedPredictor(model_config, device_config, profiling_dir)

        # 加载 profiling CSV
        base = os.path.join(profiling_dir, "compute", "a800",
                            model_config.model_name.replace("-", "/") if "/" not in model_config.model_name
                            else model_config.model_name)
        # 尝试多种路径格式
        for candidate in [
            base,
            os.path.join(profiling_dir, "compute", "a800", "Qwen", model_config.model_name),
        ]:
            if os.path.isdir(candidate):
                base = candidate
                break

        # 加载 attention CSV (filter TP=1)
        attn_path = os.path.join(base, "attention.csv")
        self._attn_df = None
        if os.path.exists(attn_path):
            self._attn_df = pd.read_csv(attn_path)
            if "num_tensor_parallel_workers" in self._attn_df.columns:
                self._attn_df = self._attn_df[self._attn_df["num_tensor_parallel_workers"] == 1]

        # 加载 mlp CSV
        mlp_path = os.path.join(base, "mlp.csv")
        self._mlp_df = None
        if os.path.exists(mlp_path):
            self._mlp_df = pd.read_csv(mlp_path)

        # 加载 expert CSV
        expert_path = os.path.join(base, "expert.csv")
        self._expert_df = None
        if os.path.exists(expert_path):
            self._expert_df = pd.read_csv(expert_path)

        # 缓存
        self._cache: Dict[Tuple, ExecutionTime] = {}

        attn_rows = len(self._attn_df) if self._attn_df is not None else 0
        mlp_rows = len(self._mlp_df) if self._mlp_df is not None else 0
        expert_rows = len(self._expert_df) if self._expert_df is not None else 0
        logger.info("HighFidelityPredictor 初始化: %d attention, %d MLP, %d expert 行",
                     attn_rows, mlp_rows, expert_rows)

    def get_execution_time(
        self,
        num_tokens: int,
        batch_size: int,
        kv_cache_size: int,
        is_prefill: bool = False,
    ) -> ExecutionTime:
        cache_key = (num_tokens, batch_size, kv_cache_size, is_prefill)
        if cache_key in self._cache:
            return self._cache[cache_key]

        et = self._lookup(num_tokens, batch_size, kv_cache_size, is_prefill)
        self._cache[cache_key] = et
        return et

    def _lookup(
        self,
        num_tokens: int,
        batch_size: int,
        kv_cache_size: int,
        is_prefill: bool,
    ) -> ExecutionTime:
        """从 profiling CSV 精确查表。"""
        import pandas as pd

        # 根据阶段计算 fusion factor
        # Prefill: size-dependent fusion (larger prefill → more fusion benefit)
        # Decode: fixed fusion (batch size varies in continuous batching, hard to model)
        if is_prefill:
            base_fusion = self._prefill_fusion_factor
            ref_size = self._prefill_ref_size
            size = max(num_tokens, 1)
            fusion = base_fusion * (size / ref_size) ** (-self._fusion_alpha)
            fusion = max(0.1, min(fusion, 1.0))
        else:
            fusion = self._decode_fusion_factor

        et = ExecutionTime()
        found_any = False

        # ─── Attention 查表 ─────────────────────────────────────────────
        if self._attn_df is not None:
            attn_mask = (
                (self._attn_df["batch_size"] == batch_size)
                & (self._attn_df["is_prefill"] == is_prefill)
            )
            attn_rows = self._attn_df[attn_mask]

            if len(attn_rows) > 0:
                if is_prefill:
                    if "prefill_chunk_size" in self._attn_df.columns:
                        closest_idx = (attn_rows["prefill_chunk_size"] - num_tokens).abs().idxmin()
                    else:
                        closest_idx = attn_rows.index[0]
                else:
                    closest_idx = (attn_rows["kv_cache_size"] - kv_cache_size).abs().idxmin()

                row = attn_rows.loc[closest_idx]

                # 提取 attention 子操作
                attn_cols = {
                    "attn_input_reshape": "time_stats.attn_input_reshape.mean",
                    "attn_kv_cache_save": "time_stats.attn_kv_cache_save.mean",
                    "attn_decode": "time_stats.attn_decode.mean",
                    "attn_prefill": "time_stats.attn_prefill.mean",
                    "attn_output_reshape": "time_stats.attn_output_reshape.mean",
                }
                for name, col in attn_cols.items():
                    if col in row.index and not pd.isna(row[col]):
                        val = float(row[col])
                        if name == "attn_input_reshape":
                            et.attn_input_reshape_time = val
                        elif name == "attn_kv_cache_save":
                            et.attn_kv_cache_save_time = val * fusion
                        elif name == "attn_decode" and not is_prefill:
                            et.attn_decode_time = val * fusion
                        elif name == "attn_prefill" and is_prefill:
                            et.attn_prefill_time = val * fusion
                        elif name == "attn_output_reshape":
                            et.attn_output_reshape_time = val
                found_any = True

        # ─── MLP 查表 (支持 log-linear 插值) ─────────────────────────
        if self._mlp_df is not None:
            nt_values = self._mlp_df["num_tokens"].values
            closest_nt = nt_values[np.argmin(np.abs(nt_values - num_tokens))]
            
            # 如果精确匹配，直接查表
            if closest_nt == num_tokens:
                closest_idx = (self._mlp_df["num_tokens"] - num_tokens).abs().idxmin()
                mlp_row = self._mlp_df.loc[closest_idx]
            else:
                # Log-linear 插值: 找到上下界
                lower_nts = nt_values[nt_values < num_tokens]
                upper_nts = nt_values[nt_values > num_tokens]
                
                if len(lower_nts) > 0 and len(upper_nts) > 0:
                    lower_nt = lower_nts.max()
                    upper_nt = upper_nts.min()
                    lower_row = self._mlp_df[self._mlp_df["num_tokens"] == lower_nt].iloc[0]
                    upper_row = self._mlp_df[self._mlp_df["num_tokens"] == upper_nt].iloc[0]
                    
                    # Log-linear 插值权重
                    log_ratio = (np.log(num_tokens) - np.log(lower_nt)) / (np.log(upper_nt) - np.log(lower_nt))
                    
                    # 对每个子操作插值
                    mlp_row = lower_row.copy()
                    for col in self._mlp_df.columns:
                        if col.startswith("time_stats.") and col.endswith(".mean"):
                            if not pd.isna(lower_row[col]) and not pd.isna(upper_row[col]):
                                lower_val = float(lower_row[col])
                                upper_val = float(upper_row[col])
                                interp_val = lower_val * (1 - log_ratio) + upper_val * log_ratio
                                mlp_row[col] = interp_val
                else:
                    closest_idx = (self._mlp_df["num_tokens"] - num_tokens).abs().idxmin()
                    mlp_row = self._mlp_df.loc[closest_idx]

            mlp_cols = {
                "input_layernorm": "time_stats.input_layernorm.mean",
                "attn_pre_proj": "time_stats.attn_pre_proj.mean",
                "attn_rope": "time_stats.attn_rope.mean",
                "attn_post_proj": "time_stats.attn_post_proj.mean",
                "post_attention_layernorm": "time_stats.post_attention_layernorm.mean",
                "mlp_up_proj": "time_stats.mlp_up_proj.mean",
                "mlp_act": "time_stats.mlp_act.mean",
                "mlp_down_proj": "time_stats.mlp_down_proj.mean",
                "add": "time_stats.add.mean",
            }
            for name, col in mlp_cols.items():
                if col in mlp_row.index and not pd.isna(mlp_row[col]):
                    val = float(mlp_row[col]) * fusion
                    if name == "input_layernorm":
                        et.input_layernorm_time = val
                    elif name == "attn_pre_proj":
                        et.attn_pre_proj_time = val
                    elif name == "attn_rope":
                        et.attn_rope_time = val
                    elif name == "attn_post_proj":
                        et.attn_post_proj_time = val
                    elif name == "post_attention_layernorm":
                        et.post_attention_layernorm_time = val
                    elif name == "mlp_up_proj":
                        et.mlp_up_proj_time = val
                    elif name == "mlp_act":
                        et.mlp_act_time = val
                    elif name == "mlp_down_proj":
                        et.mlp_down_proj_time = val
                    elif name == "add":
                        et.add_time = val
            found_any = True

        # ─── Expert MLP 查表 ───────────────────────────────────────────
        if self._expert_df is not None:
            closest_idx = (self._expert_df["num_tokens"] - num_tokens).abs().idxmin()
            expert_row = self._expert_df.loc[closest_idx]
            if "time_stats.expert_mlp.mean" in expert_row.index and not pd.isna(expert_row["time_stats.expert_mlp.mean"]):
                et.expert_mlp_time = float(expert_row["time_stats.expert_mlp.mean"]) * fusion
                found_any = True

        # 如果没有找到任何数据，回退到 ProfilingBasedPredictor
        if not found_any:
            return self._fallback.get_execution_time(num_tokens, batch_size, kv_cache_size, is_prefill)

        return et


def create_predictor(
    model_config: ModelConfig,
    device_config: DeviceSKUConfig,
    profiling_dir: Optional[str] = None,
    predictor_type: str = "auto",
) -> ExecutionTimePredictor:
    """工厂函数：创建执行时间预测器。

    Args:
        model_config: 模型配置
        device_config: 设备配置
        profiling_dir: profiling 数据目录路径（可选）
        predictor_type: 预测器类型 ("auto", "analytical", "profiled", "random_forest", "high_fidelity")
            - "auto": 根据 profiling_dir 自动选择（默认）
            - "analytical": 强制使用 AnalyticalPredictor
            - "profiled": 强制使用 ProfilingBasedPredictor（需要 profiling_dir）
            - "random_forest": 强制使用 RandomForestPredictor（需要 profiling_dir）
            - "high_fidelity": 使用 HighFidelityPredictor（精确查表，需要 profiling_dir）

    Returns:
        ExecutionTimePredictor 实例
    """
    # 检查 profiling 目录
    has_csv = False
    if profiling_dir is not None and os.path.isdir(profiling_dir):
        for root, _dirs, files in os.walk(profiling_dir):
            if any(f.endswith(".csv") for f in files):
                has_csv = True
                break

    # 根据类型选择预测器
    if predictor_type == "analytical":
        logger.info("使用 AnalyticalPredictor（强制）")
        return AnalyticalPredictor(model_config, device_config)

    elif predictor_type == "profiled":
        if not has_csv:
            logger.warning("请求 ProfilingBasedPredictor 但 profiling 目录无效，回退到 AnalyticalPredictor")
            return AnalyticalPredictor(model_config, device_config)
        logger.info("使用 ProfilingBasedPredictor，数据目录: %s", profiling_dir)
        return ProfilingBasedPredictor(model_config, device_config, profiling_dir)

    elif predictor_type == "random_forest":
        if not has_csv:
            logger.warning("请求 RandomForestPredictor 但 profiling 目录无效，回退到 AnalyticalPredictor")
            return AnalyticalPredictor(model_config, device_config)
        logger.info("使用 RandomForestPredictor，数据目录: %s", profiling_dir)
        return RandomForestPredictor(model_config, device_config, profiling_dir)

    elif predictor_type == "high_fidelity":
        if not has_csv:
            logger.warning("请求 HighFidelityPredictor 但 profiling 目录无效，回退到 AnalyticalPredictor")
            return AnalyticalPredictor(model_config, device_config)
        logger.info("使用 HighFidelityPredictor（精确查表模式），数据目录: %s", profiling_dir)
        return HighFidelityPredictor(model_config, device_config, profiling_dir)

    else:  # auto
        if has_csv:
            logger.info("自动选择 ProfilingBasedPredictor，数据目录: %s", profiling_dir)
            return ProfilingBasedPredictor(model_config, device_config, profiling_dir)
        else:
            logger.info("未提供有效 profiling 目录，使用 AnalyticalPredictor")
            return AnalyticalPredictor(model_config, device_config)
