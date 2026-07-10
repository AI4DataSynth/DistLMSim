"""Acceptance Profile 模块

封装 DFlash / DSpark 在不同 workload domain 下的位置级接受率。
数据来源于 DSpark 论文 Table 1 和 Figure 2。

依赖层次: Layer 4
  输入: config (DisaggregatedConfig)
  输出: AcceptanceProfile (被 speculative_decoder, prefix_scheduler 消费)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List

from distlmsim.config import DisaggregatedConfig


@dataclass
class PositionAcceptance:
    """单个 drafter 在单个 domain 下的位置级接受率曲线。"""
    conditional_rates: List[float]  # 每位置的条件接受率 c_k
    block_size: int                 # γ


# ─── 内置 Profiles (论文 Table 1 + Figure 2 提取) ────────────────────────

# DFlash: 并行 drafter, suffix decay 明显 (多模态碰撞)
_BUILTIN_DFLASH: Dict[str, PositionAcceptance] = {
    "math": PositionAcceptance(
        conditional_rates=[0.88, 0.86, 0.84, 0.82, 0.80, 0.79, 0.78],
        block_size=7,
    ),
    "code": PositionAcceptance(
        conditional_rates=[0.82, 0.80, 0.79, 0.78, 0.77, 0.76, 0.75],
        block_size=7,
    ),
    "chat": PositionAcceptance(
        conditional_rates=[0.72, 0.69, 0.67, 0.65, 0.64, 0.63, 0.62],
        block_size=7,
    ),
    "mixed": PositionAcceptance(
        conditional_rates=[0.80, 0.78, 0.76, 0.74, 0.73, 0.72, 0.71],
        block_size=7,
    ),
}

# DSpark: 半自回归, suffix decay 被 Markov head 缓解
# 数据校准: 论文 Figure 2 显示 DSpark Math pos1=0.93, 所有 domain 的 pos1 调高 ~0.05
_BUILTIN_DSPARK: Dict[str, PositionAcceptance] = {
    "math": PositionAcceptance(
        conditional_rates=[0.93, 0.91, 0.90, 0.89, 0.88, 0.88, 0.87],
        block_size=7,
    ),
    "code": PositionAcceptance(
        conditional_rates=[0.87, 0.85, 0.84, 0.83, 0.83, 0.82, 0.82],
        block_size=7,
    ),
    "chat": PositionAcceptance(
        conditional_rates=[0.76, 0.74, 0.73, 0.72, 0.71, 0.71, 0.70],
        block_size=7,
    ),
    "mixed": PositionAcceptance(
        conditional_rates=[0.85, 0.83, 0.82, 0.81, 0.81, 0.80, 0.80],
        block_size=7,
    ),
}


class AcceptanceProfile:
    """Drafter 的位置级接受率 profile 管理器。

    根据 speculative_mode (dflash/dspark) 和 workload domain
    返回对应的位置级条件接受率曲线。

    支持三种数据源:
    1. 内置 profile (论文 Table 1 / Figure 2)
    2. 外部 JSON 文件 (用户自定义 profiling)
    3. 参数化合成 (从 config 的 dflash/dspark 参数生成)
    """

    def __init__(self, config: DisaggregatedConfig):
        self._mode = config.speculative_mode
        self._block_size = config.block_size
        self._config = config
        self._profiles: Dict[str, Dict[str, PositionAcceptance]] = {}

        if config.acceptance_profile_path:
            self._load_from_file(config.acceptance_profile_path)
        else:
            self._load_builtin()

    def _load_builtin(self):
        """加载内置 profile。"""
        if self._mode == "dflash":
            self._profiles = {"dflash": _BUILTIN_DFLASH}
        elif self._mode == "dspark":
            self._profiles = {"dspark": _BUILTIN_DSPARK}
        else:
            self._profiles = {"standard": _BUILTIN_DFLASH}

    def _load_from_file(self, path: str):
        """从 JSON 文件加载自定义 profile。

        JSON 格式::

            {
              "dflash": {
                "math": {"conditional_rates": [0.88, 0.86, ...], "block_size": 7}
              },
              "dspark": { ... }
            }
        """
        if not os.path.exists(path):
            self._load_builtin()
            return
        with open(path) as f:
            data = json.load(f)
        for mode, domains in data.items():
            self._profiles[mode] = {}
            for domain, pdata in domains.items():
                self._profiles[mode][domain] = PositionAcceptance(
                    conditional_rates=pdata["conditional_rates"],
                    block_size=pdata.get("block_size", 7),
                )

    def _synthesize_profile(self) -> List[float]:
        """从 config 参数合成接受率曲线。"""
        if self._mode == "dflash":
            alpha = self._config.dflash_pos1_alpha
            decay = self._config.dflash_decay_rate
        elif self._mode == "dspark":
            alpha = self._config.dspark_pos1_alpha
            decay = self._config.dspark_decay_rate
        else:
            alpha = self._config.acceptance_rate
            decay = 0.05
        rates = []
        for k in range(self._block_size):
            rates.append(max(0.1, min(1.0, alpha * (1.0 - decay * k))))
        return rates

    def get_conditional_rates(self, domain: str = "mixed") -> List[float]:
        """获取指定 domain 的位置级条件接受率。

        Args:
            domain: 工作负载域 (math/code/chat/mixed)

        Returns:
            每位置的条件接受率 [c_1, c_2, ..., c_γ]
        """
        # 对于 "standard" 模式，使用 acceptance_rate 参数合成 profile
        # 而不是使用内置的 DFlash profile
        if self._mode == "standard" and not self._config.acceptance_profile_path:
            return self._synthesize_profile()
        
        mode_profiles = self._profiles.get(self._mode, {})
        if not mode_profiles:
            mode_profiles = self._profiles.get("standard", {})
        profile = mode_profiles.get(domain)
        if profile is None:
            profile = mode_profiles.get("mixed")
        if profile is None:
            return self._synthesize_profile()

        rates = list(profile.conditional_rates)
        if len(rates) < self._block_size:
            last_decay = rates[-2] - rates[-1] if len(rates) >= 2 else 0.02
            while len(rates) < self._block_size:
                rates.append(max(0.1, rates[-1] - last_decay))
        return rates[:self._block_size]

    def get_prefix_survival(self, domain: str = "mixed") -> List[float]:
        """获取前缀存活概率 (cumulative product)。

        a_j = Π_{i=1}^{j} c_i
        """
        rates = self.get_conditional_rates(domain)
        survival = []
        cumprod = 1.0
        for c in rates:
            cumprod *= c
            survival.append(cumprod)
        return survival

    def get_expected_accepted(self, domain: str = "mixed") -> float:
        """获取期望接受 token 数 (含 bonus token)。

        τ = 1 + Σ_{j=1}^{γ} a_j
        """
        survival = self.get_prefix_survival(domain)
        return 1.0 + sum(survival)
