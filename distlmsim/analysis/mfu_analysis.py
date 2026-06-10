"""MFU (Model FLOPs Utilization) 分析模块

计算推理过程中的模型算力利用率，衡量 GPU 计算效率。
从 Charon 项目的 MFU 分析器移植，适配 DistLMSim 的推理场景。

MFU 定义:
    MFU = actual_flops_per_second / peak_flops_per_second

推理 FLOPs:
    - Prefill (per request): 2 * active_params * prefill_length
    - Decode (per token):    2 * active_params

    active_params:
        Dense = 全部参数
        MoE   = shared_params + top_k * expert_params_per_expert

每 GPU FLOPS (TP 分片):
    per_gpu_flops = total_flops / tp_size
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from distlmsim.config import ModelConfig, DeviceSKUConfig, ReplicaConfig


@dataclass
class MFUResult:
    """MFU 分析结果"""
    # FLOPs 分解
    total_params: int = 0
    active_params_per_token: int = 0
    prefill_flops_per_request: int = 0
    decode_flops_per_token: int = 0
    # 时间 (ms)
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    total_time_ms: float = 0.0
    # 峰值算力
    peak_flops_per_gpu: float = 0.0
    num_gpus: int = 1
    # MFU
    prefill_mfu: float = 0.0
    decode_mfu: float = 0.0
    overall_mfu: float = 0.0
    # 吞吐量
    prefill_flops_per_s: float = 0.0
    decode_flops_per_s: float = 0.0
    # 分解
    attention_flops: int = 0
    feedforward_flops: int = 0
    embedding_flops: int = 0
    other_flops: int = 0

    def summary(self) -> Dict:
        return {
            "total_params": self.total_params,
            "active_params_per_token": self.active_params_per_token,
            "prefill_mfu": round(self.prefill_mfu, 6),
            "decode_mfu": round(self.decode_mfu, 6),
            "overall_mfu": round(self.overall_mfu, 6),
            "prefill_flops_per_s": round(self.prefill_flops_per_s, 2),
            "decode_flops_per_s": round(self.decode_flops_per_s, 2),
            "peak_flops_per_gpu": self.peak_flops_per_gpu,
            "num_gpus": self.num_gpus,
            "prefill_time_ms": round(self.prefill_time_ms, 3),
            "decode_time_ms": round(self.decode_time_ms, 3),
        }


class MFUAnalyzer:
    """推理 MFU 分析器

    计算分布式推理的模型算力利用率。
    支持 Dense 和 MoE 模型，以及 TP 并行。

    使用方法:
        analyzer = MFUAnalyzer(model_config, device_config, replica_config)
        result = analyzer.analyze(
            prefill_length=2048, decode_length=512,
            prefill_time_ms=50.0, decode_time_ms=500.0,
        )
    """

    def __init__(
        self,
        model_config: ModelConfig,
        device_config: DeviceSKUConfig,
        replica_config: Optional[ReplicaConfig] = None,
    ):
        self._model = model_config
        self._device = device_config
        self._tp = replica_config.tensor_parallel_size if replica_config else 1
        self._ep = replica_config.expert_parallel_size if replica_config else 1

    @property
    def _is_moe(self) -> bool:
        return self._model.num_experts > 0

    @property
    def _head_dim(self) -> int:
        return self._model.embedding_dim // self._model.num_q_heads

    # --------------------------------------------------------
    # FLOPs 估算
    # --------------------------------------------------------

    def _count_attn_params_per_layer(self) -> int:
        h = self._model.embedding_dim
        hd = self._head_dim
        nq = self._model.num_q_heads
        nkv = self._model.num_kv_heads
        return h * (nq * hd) + h * (nkv * hd) * 2 + (nq * hd) * h

    def _count_ffn_params_per_layer(self) -> int:
        h = self._model.embedding_dim
        mlp = self._model.mlp_hidden_dim or int(h * 8 / 3)
        return 3 * h * mlp

    def _count_expert_params_per_expert(self) -> int:
        h = self._model.embedding_dim
        mlp = self._model.mlp_hidden_dim or int(h * 8 / 3)
        return 3 * h * mlp

    def _count_norm_params_per_layer(self) -> int:
        return 2 * self._model.embedding_dim

    def _count_shared_params(self) -> int:
        return self._model.vocab_size * self._model.embedding_dim

    def count_total_params(self) -> int:
        """计算模型总参数量"""
        m = self._model
        if self._is_moe:
            gate = m.embedding_dim * m.num_experts
            shared_per_layer = (
                self._count_attn_params_per_layer()
                + self._count_norm_params_per_layer()
                + gate
            )
            expert_per_layer = (
                m.num_experts * self._count_expert_params_per_expert()
            )
            per_layer = shared_per_layer + expert_per_layer
        else:
            per_layer = (
                self._count_attn_params_per_layer()
                + self._count_ffn_params_per_layer()
                + self._count_norm_params_per_layer()
            )
        return per_layer * m.num_layers + self._count_shared_params()

    def _count_active_params_per_token(self) -> int:
        """计算每 token 激活的参数量。

        Dense: 全部参数
        MoE: shared + top_k * expert_params_per_expert
        """
        m = self._model
        if self._is_moe:
            gate = m.embedding_dim * m.num_experts
            shared_per_layer = (
                self._count_attn_params_per_layer()
                + self._count_norm_params_per_layer()
                + gate
            )
            shared_total = shared_per_layer * m.num_layers + self._count_shared_params()
            expert_active = (
                m.top_k_experts
                * self._count_expert_params_per_expert()
                * m.num_layers
            )
            return shared_total + expert_active
        else:
            return self.count_total_params()

    def _estimate_flops_breakdown(self, seq_len: int) -> Dict[str, int]:
        """估算 FLOPs 按组件分解 (2 * params * seq_len)"""
        m = self._model
        num_layers = m.num_layers

        attn_params = self._count_attn_params_per_layer() * num_layers
        norm_params = self._count_norm_params_per_layer() * num_layers
        embed_params = self._count_shared_params()

        if self._is_moe:
            k = m.top_k_experts
            ffn_active = k * self._count_expert_params_per_expert() * num_layers
            gate_params = m.embedding_dim * m.num_experts * num_layers
        else:
            ffn_active = self._count_ffn_params_per_layer() * num_layers
            gate_params = 0

        return {
            "attention": 2 * attn_params * seq_len,
            "feedforward": 2 * ffn_active * seq_len,
            "embedding": 2 * embed_params * seq_len,
            "other": 2 * (norm_params + gate_params) * seq_len,
        }

    # --------------------------------------------------------
    # 主分析入口
    # --------------------------------------------------------

    def analyze(
        self,
        prefill_length: int = 2048,
        decode_length: int = 512,
        prefill_time_ms: float = 0.0,
        decode_time_ms: float = 0.0,
        batch_size: int = 1,
    ) -> MFUResult:
        """分析推理 MFU。

        Args:
            prefill_length: prefill 序列长度
            decode_length: decode 生成长度
            prefill_time_ms: prefill 阶段耗时 (ms, 单请求)
            decode_time_ms: decode 阶段总耗时 (ms, 单请求)
            batch_size: 批次大小

        Returns:
            MFUResult
        """
        result = MFUResult()
        tp = max(self._tp, 1)

        # 参数量
        result.total_params = self.count_total_params()
        result.active_params_per_token = self._count_active_params_per_token()

        # FLOPs
        # Prefill: 2 * active_params * prefill_length (矩阵乘 2MNK)
        result.prefill_flops_per_request = (
            2 * result.active_params_per_token * prefill_length
        )
        # Decode: 2 * active_params per token
        result.decode_flops_per_token = 2 * result.active_params_per_token

        # FLOPs 分解
        breakdown = self._estimate_flops_breakdown(prefill_length)
        result.attention_flops = breakdown["attention"]
        result.feedforward_flops = breakdown["feedforward"]
        result.embedding_flops = breakdown["embedding"]
        result.other_flops = breakdown["other"]

        # 峰值算力 (FP16 TFLOPS → FLOPS)
        result.peak_flops_per_gpu = self._device.fp16_tflops * 1e12
        result.num_gpus = tp

        # 时间
        result.prefill_time_ms = prefill_time_ms
        result.decode_time_ms = decode_time_ms
        result.total_time_ms = prefill_time_ms + decode_time_ms

        # 每 GPU FLOPS (TP 分片后每 GPU 做 1/TP 的工作)
        prefill_flops_per_gpu = result.prefill_flops_per_request // tp
        decode_flops_per_token_per_gpu = result.decode_flops_per_token // tp

        # Prefill MFU
        if prefill_time_ms > 0:
            prefill_s = prefill_time_ms / 1000.0
            result.prefill_flops_per_s = prefill_flops_per_gpu / prefill_s
            if result.peak_flops_per_gpu > 0:
                result.prefill_mfu = (
                    result.prefill_flops_per_s / result.peak_flops_per_gpu
                )

        # Decode MFU
        if decode_time_ms > 0 and decode_length > 0:
            decode_s = decode_time_ms / 1000.0
            total_decode_flops_per_gpu = decode_flops_per_token_per_gpu * decode_length
            result.decode_flops_per_s = total_decode_flops_per_gpu / decode_s
            if result.peak_flops_per_gpu > 0:
                result.decode_mfu = (
                    result.decode_flops_per_s / result.peak_flops_per_gpu
                )

        # Overall MFU (加权平均)
        total_flops_per_gpu = (
            prefill_flops_per_gpu
            + decode_flops_per_token_per_gpu * decode_length
        )
        total_time_s = result.total_time_ms / 1000.0
        if total_time_s > 0 and result.peak_flops_per_gpu > 0:
            result.overall_mfu = (
                (total_flops_per_gpu / total_time_s) / result.peak_flops_per_gpu
            )

        return result

    def analyze_from_metrics(
        self,
        prefill_length: int,
        decode_length: int,
        prefill_time_ms: float,
        decode_time_ms: float,
    ) -> MFUResult:
        """从 MetricsStore 的数据分析 MFU (便捷接口)。

        对一批完成请求的平均 prefill/decode 时间计算 MFU。

        Args:
            prefill_length: 平均 prefill 长度
            decode_length: 平均 decode 长度
            prefill_time_ms: 平均 prefill 时间 (ms)
            decode_time_ms: 平均 decode 时间 (ms)
        """
        return self.analyze(
            prefill_length=prefill_length,
            decode_length=decode_length,
            prefill_time_ms=prefill_time_ms,
            decode_time_ms=decode_time_ms,
        )

    @staticmethod
    def compute_mfu(
        total_flops: int,
        peak_flops: float,
        time_s: float,
    ) -> float:
        """计算 MFU 的静态方法。

        Args:
            total_flops: 总 FLOPs
            peak_flops: 峰值 FLOPS (单 GPU)
            time_s: 执行时间 (秒)

        Returns:
            MFU 值 (0-1)
        """
        if peak_flops <= 0 or time_s <= 0:
            return 0.0
        return total_flops / (peak_flops * time_s)
