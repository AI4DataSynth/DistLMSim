"""Chrome Trace 时间线分析模块

从模拟结果生成 Chrome Trace Viewer 兼容的 JSON 时间线。
从 Charon 项目移植，适配 DistLMSim 的推理场景。

输出格式: Chrome Trace JSON (list of event dicts)
每个事件: {name, cat, ph, ts, dur, pid, tid, args}
  - ph: "X" (完整事件), "M" (metadata)
  - ts/dur: 微秒
  - pid: 节点 ID
  - tid: 线程 ID (Prefill/Decode/KV Transfer 流)

可视化:
  - X 轴: 时间
  - Y 轴: 节点 (pid)
  - 每节点内: Prefill 流、Decode 流、KV 传输流
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from distlmsim.metrics.metrics_store import MetricsStore, RequestMetrics


@dataclass
class TraceEvent:
    """单个 Trace 事件 (Chrome Trace 格式)"""
    name: str
    ph: str
    cat: str = ""
    ts: float = 0.0    # 微秒
    dur: float = 0.0   # 微秒
    pid: int = 0
    tid: int = 0
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "cat": self.cat,
            "ph": self.ph,
            "ts": round(self.ts, 3),
            "pid": self.pid,
            "tid": self.tid,
        }
        if self.ph == "X":
            d["dur"] = round(max(self.dur, 0.001), 3)
        if self.args:
            d["args"] = self.args
        return d


# 线程 ID 常量
TID_PREFILL = 0
TID_DECODE = 1
TID_KV_TRANSFER = 2
TID_SCHEDULING = 3


class TimelineAnalyzer:
    """推理时间线分析器

    从 MetricsStore 生成 Chrome Trace 格式的执行时间线。

    使用方法:
        analyzer = TimelineAnalyzer()
        events = analyzer.generate_trace(metrics_store)
        analyzer.save_trace(events, "results/timeline.json")
    """

    def generate_trace(
        self,
        metrics_store: MetricsStore,
        disaggregated: bool = False,
    ) -> List[Dict[str, Any]]:
        """从 MetricsStore 生成 Chrome Trace 事件列表。

        Args:
            metrics_store: 已完成模拟的指标存储
            disaggregated: 是否为存算分离模式

        Returns:
            Chrome Trace 事件字典列表
        """
        events: List[TraceEvent] = []

        # 收集所有请求
        request_metrics = metrics_store._request_metrics
        if not request_metrics:
            return []

        # 确定节点集合
        node_ids = set()
        for m in request_metrics.values():
            if m.prefill_node_id >= 0:
                node_ids.add(m.prefill_node_id)
            if m.decode_node_id >= 0:
                node_ids.add(m.decode_node_id)

        # 生成 metadata 事件
        events.extend(self._generate_metadata(node_ids, disaggregated))

        # 为每个完成请求生成事件
        for req_id, m in sorted(request_metrics.items()):
            if m.decode_end_time <= 0:
                continue
            events.extend(self._generate_request_events(m, disaggregated))

        return [e.to_dict() for e in events]

    def _generate_metadata(
        self,
        node_ids: set,
        disaggregated: bool,
    ) -> List[TraceEvent]:
        """生成 metadata 事件 (进程名和线程名)"""
        events: List[TraceEvent] = []
        for nid in sorted(node_ids):
            if disaggregated:
                # 根据角色命名
                events.append(TraceEvent(
                    name="process_name", ph="M", pid=nid, tid=0,
                    args={"name": f"Node {nid}"},
                ))
            else:
                events.append(TraceEvent(
                    name="process_name", ph="M", pid=nid, tid=0,
                    args={"name": f"Node {nid}"},
                ))

            # 线程名
            events.append(TraceEvent(
                name="thread_name", ph="M", pid=nid, tid=TID_PREFILL,
                args={"name": "Prefill"},
            ))
            events.append(TraceEvent(
                name="thread_name", ph="M", pid=nid, tid=TID_DECODE,
                args={"name": "Decode"},
            ))
            events.append(TraceEvent(
                name="thread_name", ph="M", pid=nid, tid=TID_KV_TRANSFER,
                args={"name": "KV Transfer"},
            ))
            events.append(TraceEvent(
                name="thread_name", ph="M", pid=nid, tid=TID_SCHEDULING,
                args={"name": "Scheduling"},
            ))

        return events

    def _generate_request_events(
        self,
        m: RequestMetrics,
        disaggregated: bool,
    ) -> List[TraceEvent]:
        """为单个请求生成 trace 事件。"""
        events: List[TraceEvent] = []
        rid = m.request_id
        # 时间单位: ms → μs
        scale = 1000.0

        # 调度延迟
        if m.scheduled_time > m.arrival_time:
            events.append(TraceEvent(
                name=f"sched_r{rid}",
                cat="scheduling",
                ph="X",
                ts=m.arrival_time * scale,
                dur=(m.scheduled_time - m.arrival_time) * scale,
                pid=m.prefill_node_id if m.prefill_node_id >= 0 else 0,
                tid=TID_SCHEDULING,
                args={"request_id": rid, "type": "scheduling_delay"},
            ))

        # Prefill
        if m.prefill_end_time > m.prefill_start_time:
            pnode = m.prefill_node_id if m.prefill_node_id >= 0 else 0
            events.append(TraceEvent(
                name=f"prefill_r{rid}",
                cat="compute",
                ph="X",
                ts=m.prefill_start_time * scale,
                dur=(m.prefill_end_time - m.prefill_start_time) * scale,
                pid=pnode,
                tid=TID_PREFILL,
                args={
                    "request_id": rid,
                    "prefill_tokens": m.prefill_tokens,
                    "type": "prefill",
                },
            ))

        # KV Cache 传输
        if m.kv_cache_transfer_end > m.kv_cache_transfer_start:
            src = m.prefill_node_id if m.prefill_node_id >= 0 else 0
            events.append(TraceEvent(
                name=f"kv_xfer_r{rid}",
                cat="communication",
                ph="X",
                ts=m.kv_cache_transfer_start * scale,
                dur=(m.kv_cache_transfer_end - m.kv_cache_transfer_start) * scale,
                pid=src,
                tid=TID_KV_TRANSFER,
                args={
                    "request_id": rid,
                    "src_node": src,
                    "dst_node": m.decode_node_id,
                    "type": "kv_cache_transfer",
                },
            ))

        # Decode
        if m.decode_end_time > m.decode_start_time:
            dnode = m.decode_node_id if m.decode_node_id >= 0 else 0
            events.append(TraceEvent(
                name=f"decode_r{rid}",
                cat="compute",
                ph="X",
                ts=m.decode_start_time * scale,
                dur=(m.decode_end_time - m.decode_start_time) * scale,
                pid=dnode,
                tid=TID_DECODE,
                args={
                    "request_id": rid,
                    "decode_tokens": m.decode_tokens,
                    "type": "decode",
                },
            ))

        return events

    def save_trace(
        self,
        events: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        """保存 Chrome Trace JSON 文件。

        Args:
            events: trace 事件列表
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, ensure_ascii=False)

    def compute_stats(
        self,
        events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """从 trace 事件计算统计信息。

        Returns:
            dict 包含总时间、按类别分解、气泡比例等
        """
        if not events:
            return {}

        # 过滤出完整事件 (ph="X")
        complete = [e for e in events if e.get("ph") == "X"]
        if not complete:
            return {}

        # 总时间
        min_ts = min(e["ts"] for e in complete)
        max_ts = max(e["ts"] + e.get("dur", 0) for e in complete)
        total_us = max_ts - min_ts

        # 按类别分解
        by_cat: Dict[str, float] = {}
        for e in complete:
            cat = e.get("cat", "unknown")
            dur = e.get("dur", 0)
            by_cat[cat] = by_cat.get(cat, 0) + dur

        # 按节点分解
        by_pid: Dict[int, float] = {}
        for e in complete:
            pid = e.get("pid", 0)
            dur = e.get("dur", 0)
            by_pid[pid] = by_pid.get(pid, 0) + dur

        return {
            "total_time_us": total_us,
            "total_time_ms": total_us / 1000.0,
            "num_events": len(complete),
            "by_category": {k: round(v, 2) for k, v in by_cat.items()},
            "by_node": {str(k): round(v, 2) for k, v in by_pid.items()},
        }
