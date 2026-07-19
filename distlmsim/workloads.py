"""Vidur-Bench workload definitions for DistLMSim experiments.

Based on Vidur paper (Agrawal et al., MLSys 2024) Section 7.1:
  - Chat-1M:  LMSYS-Chat-1M multi-round conversations (short prefill, high variance)
  - Arxiv-4K: ArXiv summarization (long prefill, short decode)
  - BWB-4K:   Bilingual Web Book translation (medium prefill, very long decode)

Statistics computed from trace files in:
  vidur-main/vidur-main/data/processed_traces/

Usage:
  from distlmsim.workloads import VIDUR_WORKLOADS, apply_workload
  wl = VIDUR_WORKLOADS["chat-1m"]
  apply_workload(config, "chat-1m")
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class WorkloadProfile:
    """Workload 统计量定义。"""
    name: str
    prefill_mean: int
    decode_mean: int
    prefill_cv: float       # coefficient of variation (std/mean) for prefill
    decode_cv: float        # coefficient of variation (std/mean) for decode
    description: str
    source: str             # Vidur trace file name

    @property
    def pd_ratio(self) -> float:
        return self.prefill_mean / self.decode_mean

    @property
    def prefill_std(self) -> float:
        return self.prefill_mean * self.prefill_cv

    @property
    def decode_std(self) -> float:
        return self.decode_mean * self.decode_cv


VIDUR_WORKLOADS: Dict[str, WorkloadProfile] = {
    "chat-1m": WorkloadProfile(
        name="Chat-1M",
        prefill_mean=462,
        decode_mean=210,
        prefill_cv=1.42,     # std=657
        decode_cv=0.86,      # std=180
        description="LMSYS-Chat-1M: multi-round chat, short prefill (high variance), short decode",
        source="lmsys_chat_1m_conversation_stats_llama2_tokenizer.csv",
    ),
    "arxiv-4k": WorkloadProfile(
        name="Arxiv-4K",
        prefill_mean=2588,
        decode_mean=291,
        prefill_cv=0.36,     # std=945
        decode_cv=1.82,      # std=531
        description="ArXiv Summarization: long prefill, short decode",
        source="arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv",
    ),
    "bwb-4k": WorkloadProfile(
        name="BWB-4K",
        prefill_mean=1072,
        decode_mean=1602,
        prefill_cv=0.43,     # std=462
        decode_cv=0.17,      # std=267
        description="Bilingual Web Book: medium prefill, very long decode",
        source="bwb_stats_llama2_tokenizer_filtered_v2.csv",
    ),
    "default": WorkloadProfile(
        name="Default",
        prefill_mean=512,
        decode_mean=128,
        prefill_cv=0.5,
        decode_cv=0.5,
        description="DistLMSim default synthetic workload",
        source="synthetic",
    ),
}


def apply_workload(config, workload_name: str) -> None:
    """Apply Vidur workload parameters to a SimulationConfig's request config.

    Modifies config.request in-place:
      - prefill_length, decode_length (mean values)
      - prefill_length_cv, decode_length_cv (per-phase CV)
      - length_distribution = "normal"

    Args:
        config: SimulationConfig instance
        workload_name: one of "chat-1m", "arxiv-4k", "bwb-4k", "default"

    Raises:
        ValueError: if workload_name is not recognized
    """
    if workload_name not in VIDUR_WORKLOADS:
        raise ValueError(
            f"Unknown workload '{workload_name}'. "
            f"Available: {list(VIDUR_WORKLOADS.keys())}"
        )

    wl = VIDUR_WORKLOADS[workload_name]
    req = config.request
    req.prefill_length = wl.prefill_mean
    req.decode_length = wl.decode_mean
    req.prefill_length_cv = wl.prefill_cv
    req.decode_length_cv = wl.decode_cv
    req.length_distribution = "normal"


def print_workload_summary() -> None:
    """Print a summary table of all workload profiles."""
    print(f"  {'Workload':<12} {'PF mean':>8} {'PF std':>8} {'DC mean':>8} "
          f"{'DC std':>8} {'P:D':>6}  Description")
    print(f"  {'─' * 80}")
    for key, wl in VIDUR_WORKLOADS.items():
        print(f"  {key:<12} {wl.prefill_mean:>8} {wl.prefill_std:>8.0f} "
              f"{wl.decode_mean:>8} {wl.decode_std:>8.0f} {wl.pd_ratio:>5.2f}  "
              f"{wl.description}")
