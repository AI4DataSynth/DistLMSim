"""推理内存分析模块

估算分布式推理过程中单 GPU 的峰值显存占用。
从 Charon 项目的内存分析器移植，适配 DistLMSim 的配置和数据结构。

推理显存组成:
1. 模型参数 (Weights): 以 FP16/BF16 存储
2. KV Cache: 每层每请求的 K/V 张量缓存
3. 激活 (Activations): 当前步的中间计算结果
4. 通信缓冲区: TP/EP 通信所需的临时缓冲
5. 临时缓冲区: 计算过程中的临时分配

MoE 模型:
- 专家参数按 EP 分片
- 路由缓冲区: batch * seq * num_experts * sizeof(float32)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from distlmsim.config import ModelConfig, DeviceSKUConfig, ReplicaConfig


# 推理默认精度每元素字节数 (FP16/BF16)
_DEFAULT_BPE = 2


@dataclass
class MemoryBreakdown:
    """单 GPU 推理内存分解 (单位: bytes)"""
    # 模型参数
    params_bytes: int = 0
    # MoE 专家参数
    expert_params_bytes: int = 0
    # KV Cache
    kv_cache_bytes: int = 0
    # 激活 (当前步)
    activations_bytes: int = 0
    # 通信缓冲区
    comm_buffers_bytes: int = 0
    # MoE 路由缓冲区
    routing_buffers_bytes: int = 0
    # 临时缓冲区
    temp_buffers_bytes: int = 0
    # 汇总
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0  # 含碎片 (~10%)
    # GPU 容量
    gpu_capacity_bytes: int = 0
    memory_utilization: float = 0.0
    is_oom: bool = False
    # 分解比例
    breakdown_pct: Dict[str, float] = field(default_factory=dict)
    # 辅助信息
    num_params_total: int = 0
    num_params_per_gpu: int = 0
    kv_cache_per_request_bytes: int = 0
    max_batch_before_oom: int = 0

    @staticmethod
    def to_gb(value_bytes: int) -> float:
        """字节转 GB"""
        return value_bytes / (1024 ** 3)

    def summary(self) -> Dict:
        """返回摘要字典"""
        return {
            "params_gb": round(self.to_gb(self.params_bytes), 3),
            "expert_params_gb": round(self.to_gb(self.expert_params_bytes), 3),
            "kv_cache_gb": round(self.to_gb(self.kv_cache_bytes), 3),
            "activations_gb": round(self.to_gb(self.activations_bytes), 3),
            "comm_buffers_gb": round(self.to_gb(self.comm_buffers_bytes), 3),
            "routing_buffers_gb": round(self.to_gb(self.routing_buffers_bytes), 3),
            "temp_buffers_gb": round(self.to_gb(self.temp_buffers_bytes), 3),
            "peak_allocated_gb": round(self.to_gb(self.peak_allocated_bytes), 3),
            "peak_reserved_gb": round(self.to_gb(self.peak_reserved_bytes), 3),
            "gpu_capacity_gb": round(self.to_gb(self.gpu_capacity_bytes), 3),
            "memory_utilization": round(self.memory_utilization, 4),
            "is_oom": self.is_oom,
            "num_params_b": self.num_params_total,
            "max_batch_before_oom": self.max_batch_before_oom,
        }


class MemoryAnalyzer:
    """推理内存分析器

    估算分布式推理场景下单 GPU 的峰值显存占用。
    支持 Dense 和 MoE 模型，以及 TP/EP 并行策略的显存分片。

    使用方法:
        analyzer = MemoryAnalyzer(model_config, device_config, replica_config)
        breakdown = analyzer.analyze(batch_size=32, prefill_length=2048, decode_length=512)
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
        self._bpe = _DEFAULT_BPE

    @property
    def _is_moe(self) -> bool:
        return self._model.num_experts > 0

    @property
    def _head_dim(self) -> int:
        return self._model.embedding_dim // self._model.num_q_heads

    # --------------------------------------------------------
    # 参数量计算
    # --------------------------------------------------------

    def _count_attn_params_per_layer(self) -> int:
        """每层 Attention 参数量 (支持 GQA)"""
        h = self._model.embedding_dim
        hd = self._head_dim
        nq = self._model.num_q_heads
        nkv = self._model.num_kv_heads
        # Q, K, V, O 投影
        q = h * (nq * hd)
        k = h * (nkv * hd)
        v = h * (nkv * hd)
        o = (nq * hd) * h
        return q + k + v + o

    def _count_ffn_params_per_layer(self) -> int:
        """每层 Dense FFN 参数量 (SwiGLU: gate + up + down)"""
        h = self._model.embedding_dim
        mlp = self._model.mlp_hidden_dim or int(h * 8 / 3)
        return 3 * h * mlp

    def _count_expert_params_per_layer(self) -> int:
        """每层所有 MoE 专家的参数量"""
        if not self._is_moe:
            return 0
        h = self._model.embedding_dim
        mlp = self._model.mlp_hidden_dim or int(h * 8 / 3)
        # 每个专家: gate + up + down = 3 * h * mlp
        return self._model.num_experts * 3 * h * mlp

    def _count_norm_params_per_layer(self) -> int:
        """每层 LayerNorm/RMSNorm 参数量 (两个 norm)"""
        return 2 * self._model.embedding_dim

    def _count_shared_params(self) -> int:
        """非层参数 (Embedding + LM Head, 通常共享权重)"""
        return self._model.vocab_size * self._model.embedding_dim

    def count_total_params(self) -> int:
        """计算模型总参数量"""
        m = self._model
        if self._is_moe:
            # MoE: 共享 (attn + norm + gate) + 专家
            gate_params = m.embedding_dim * m.num_experts
            shared_per_layer = (
                self._count_attn_params_per_layer()
                + self._count_norm_params_per_layer()
                + gate_params
            )
            expert_per_layer = self._count_expert_params_per_layer()
            per_layer = shared_per_layer + expert_per_layer
        else:
            per_layer = (
                self._count_attn_params_per_layer()
                + self._count_ffn_params_per_layer()
                + self._count_norm_params_per_layer()
            )
        return per_layer * m.num_layers + self._count_shared_params()

    def _count_params_per_gpu(self) -> tuple[int, int]:
        """计算单 GPU 的参数量。

        返回:
            (shared_params_per_gpu, expert_params_per_gpu)
        """
        m = self._model
        tp = max(self._tp, 1)
        ep = max(self._ep, 1)

        if self._is_moe:
            gate_params = m.embedding_dim * m.num_experts
            shared_per_layer = (
                self._count_attn_params_per_layer()
                + self._count_norm_params_per_layer()
                + gate_params
            )
            shared_total = shared_per_layer * m.num_layers + self._count_shared_params()
            expert_total = self._count_expert_params_per_layer() * m.num_layers
            return shared_total // tp, expert_total // ep
        else:
            total = (
                (self._count_attn_params_per_layer()
                 + self._count_ffn_params_per_layer()
                 + self._count_norm_params_per_layer())
                * m.num_layers
                + self._count_shared_params()
            )
            return total // tp, 0

    # --------------------------------------------------------
    # KV Cache 估算
    # --------------------------------------------------------

    def _kv_cache_per_request(self, total_seq_len: int) -> int:
        """单请求的 KV Cache 大小 (bytes, 单 GPU)"""
        m = self._model
        hd = self._head_dim
        nkv = m.num_kv_heads
        num_layers = m.num_layers
        tp = max(self._tp, 1)
        # K + V: 2 * num_layers * num_kv_heads * head_dim * bpe / TP
        return 2 * num_layers * nkv * hd * self._bpe * total_seq_len // tp

    # --------------------------------------------------------
    # 主分析入口
    # --------------------------------------------------------

    def analyze(
        self,
        batch_size: int = 32,
        prefill_length: int = 2048,
        decode_length: int = 512,
        kv_cache_enabled: bool = True,
    ) -> MemoryBreakdown:
        """估算推理峰值显存。

        Args:
            batch_size: 推理批次大小
            prefill_length: prefill 序列长度
            decode_length: decode 生成长度
            kv_cache_enabled: 是否启用 KV Cache

        Returns:
            MemoryBreakdown
        """
        result = MemoryBreakdown()
        m = self._model
        tp = max(self._tp, 1)

        # 1. 模型参数
        shared_per_gpu, expert_per_gpu = self._count_params_per_gpu()
        total_params = self.count_total_params()
        result.params_bytes = shared_per_gpu * self._bpe
        result.expert_params_bytes = expert_per_gpu * self._bpe
        result.num_params_total = total_params
        result.num_params_per_gpu = shared_per_gpu + expert_per_gpu

        # 2. KV Cache
        total_seq = prefill_length + decode_length
        if kv_cache_enabled:
            kv_per_req = self._kv_cache_per_request(total_seq)
            result.kv_cache_bytes = kv_per_req * batch_size
            result.kv_cache_per_request_bytes = kv_per_req
        else:
            # 无 KV Cache: 只需当前步激活
            result.kv_cache_bytes = 0

        # 3. 激活 (当前步中间结果, 推理无需保存历史)
        # 每层约 10 个张量: batch * seq * h * bpe
        # 推理时只需当前步, 不需要 num_layers 层同时活跃
        # 但 PP 情况下需要多层, 简化为全层
        activation_per_step = (
            batch_size * self._bpe * m.embedding_dim * 4  # 约 4 个临时张量
        )
        result.activations_bytes = activation_per_step

        # 4. 通信缓冲区 (TP all-reduce 需要)
        result.comm_buffers_bytes = (
            batch_size * m.embedding_dim * self._bpe
        )

        # 5. MoE 路由缓冲区
        if self._is_moe:
            result.routing_buffers_bytes = (
                batch_size * m.num_experts * 4  # float32 路由权重
            )

        # 6. 临时缓冲区
        result.temp_buffers_bytes = activation_per_step // 2

        # 7. 峰值
        result.peak_allocated_bytes = (
            result.params_bytes
            + result.expert_params_bytes
            + result.kv_cache_bytes
            + result.activations_bytes
            + result.comm_buffers_bytes
            + result.routing_buffers_bytes
            + result.temp_buffers_bytes
        )
        result.peak_reserved_bytes = int(result.peak_allocated_bytes * 1.1)

        # 8. GPU 容量与 OOM
        result.gpu_capacity_bytes = int(self._device.memory_gb * 1024 ** 3)
        result.is_oom = result.peak_reserved_bytes > result.gpu_capacity_bytes
        if result.gpu_capacity_bytes > 0:
            result.memory_utilization = (
                result.peak_reserved_bytes / result.gpu_capacity_bytes
            )

        # 9. 最大不 OOM 的 batch size
        fixed_mem = (
            result.params_bytes
            + result.expert_params_bytes
            + result.activations_bytes
            + result.comm_buffers_bytes
            + result.routing_buffers_bytes
            + result.temp_buffers_bytes
        )
        available = int(result.gpu_capacity_bytes * 0.9) - fixed_mem
        if kv_cache_enabled:
            kv_per_req = self._kv_cache_per_request(total_seq)
            if kv_per_req > 0 and available > 0:
                result.max_batch_before_oom = max(1, available // kv_per_req)
            else:
                result.max_batch_before_oom = 0 if available <= 0 else batch_size
        else:
            result.max_batch_before_oom = batch_size

        # 10. 分解比例
        total = max(result.peak_allocated_bytes, 1)
        result.breakdown_pct = {
            "params": result.params_bytes / total * 100,
            "expert_params": result.expert_params_bytes / total * 100,
            "kv_cache": result.kv_cache_bytes / total * 100,
            "activations": result.activations_bytes / total * 100,
            "comm_buffers": result.comm_buffers_bytes / total * 100,
            "routing_buffers": result.routing_buffers_bytes / total * 100,
            "temp_buffers": result.temp_buffers_bytes / total * 100,
        }

        return result
