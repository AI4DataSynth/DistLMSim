"""
高级 Replica 调度器 - 从 TRADIOS 迁移

包含:
- MLFQ: 多级反馈队列
- PO: 优先级排序 (短作业 FCFS + 长作业 SJF)
- OPT: 最优调度 (Score = remaining_tokens × noise_factor)
- LightLLM: 分离 prefill/decode batch
"""

import numpy as np
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from distlmsim.entities import Request


@dataclass
class MLFQState:
    """MLFQ 调度器状态"""
    num_queues: int = 8
    base_quantum: int = 16  # 基础时间片 (tokens)
    priority_threshold: float = 2.0  # 优先级阈值
    starvation_threshold: int = 1000  # 饥饿预防阈值 (ms)
    
    # 每个请求的状态
    request_queues: Dict[int, int] = field(default_factory=dict)  # request_id -> queue_level
    request_wait_time: Dict[int, float] = field(default_factory=dict)  # request_id -> wait_time_ms
    request_service_time: Dict[int, float] = field(default_factory=dict)  # request_id -> service_time_ms
    
    def get_queue_level(self, request_id: int) -> int:
        return self.request_queues.get(request_id, 0)
    
    def update_after_service(self, request_id: int, service_time: float):
        """服务后更新队列级别"""
        current_level = self.get_queue_level(request_id)
        self.request_service_time[request_id] = self.request_service_time.get(request_id, 0) + service_time
        
        # 根据服务时间提升到更低的优先级队列
        if service_time > self.base_quantum * (2 ** current_level):
            new_level = min(current_level + 1, self.num_queues - 1)
            self.request_queues[request_id] = new_level
    
    def check_starvation(self, request_id: int, current_time: float) -> bool:
        """检查是否饥饿"""
        wait_time = current_time - self.request_wait_time.get(request_id, current_time)
        return wait_time > self.starvation_threshold
    
    def promote_starved(self, current_time: float) -> List[int]:
        """提升饥饿的请求到最高优先级"""
        promoted = []
        for request_id in self.request_queues:
            if self.check_starvation(request_id, current_time):
                self.request_queues[request_id] = 0
                promoted.append(request_id)
        return promoted


@dataclass
class POState:
    """Priority Ordering 调度器状态"""
    # 对数正态分布参数 (从 TRADIOS 迁移)
    mu: float = 5.2355
    sigma: float = 1.0224
    threshold_percentile: float = 0.5  # 50% 分位数作为短/长作业阈值
    
    def __post_init__(self):
        # 计算阈值
        self.threshold = math.exp(self.mu + self.sigma * self._norm_ppf(self.threshold_percentile))
    
    def _norm_ppf(self, p: float) -> float:
        """标准正态分布的百分位点函数 (近似)"""
        # 使用 Abramowitz and Stegun 近似
        if p <= 0 or p >= 1:
            return 0
        if p < 0.5:
            return -self._norm_ppf(1 - p)
        
        t = math.sqrt(-2 * math.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t)
    
    def is_short_job(self, decode_tokens: int) -> bool:
        """判断是否为短作业"""
        return decode_tokens < self.threshold
    
    def predict_decode_length(self, prefill_tokens: int) -> float:
        """预测 decode 长度 (简化版，实际应该用更复杂的模型)"""
        # 简化：假设 decode 长度与 prefill 长度正相关
        return prefill_tokens * 0.3


@dataclass
class OPTState:
    """OPT 调度器状态 (true oracle: 完美预知剩余 token 数)"""
    prediction_error_std: float = 0.0  # oracle 无预测误差
    starvation_limit: int = 100  # 饥饿限制 (迭代次数)
    promotion_period: int = 10  # 提升周期
    
    # 每个请求的状态
    request_scores: Dict[int, float] = field(default_factory=dict)
    request_iterations: Dict[int, int] = field(default_factory=dict)  # 等待的迭代次数
    
    def compute_score(self, request: Request, rng: np.random.Generator) -> float:
        """计算请求分数: remaining_tokens × noise_factor"""
        remaining_tokens = request.decode_tokens - request.num_generated_tokens
        
        # 添加预测误差噪声
        noise = rng.normal(1.0, self.prediction_error_std)
        noise = max(0.1, noise)  # 确保噪声为正
        
        score = remaining_tokens * noise
        return score
    
    def check_starvation(self, request_id: int) -> bool:
        """检查是否饥饿"""
        iterations = self.request_iterations.get(request_id, 0)
        return iterations >= self.starvation_limit
    
    def update_iterations(self):
        """更新所有请求的等待迭代次数"""
        for request_id in self.request_iterations:
            self.request_iterations[request_id] += 1
    
    def promote_starved(self) -> List[int]:
        """提升饥饿的请求"""
        promoted = []
        for request_id, iterations in self.request_iterations.items():
            if iterations >= self.starvation_limit:
                promoted.append(request_id)
                self.request_iterations[request_id] = 0
        return promoted


@dataclass
class LightLLMState:
    """LightLLM 调度器状态"""
    max_waiting_iters: int = 10  # 最大等待迭代次数
    max_prefill_batch_size: int = 8
    max_decode_batch_size: int = 32
    
    # 分离的队列
    prefill_queue: List[int] = field(default_factory=list)  # request_ids
    decode_queue: List[int] = field(default_factory=list)  # request_ids
    
    # 等待迭代计数
    prefill_wait_iters: Dict[int, int] = field(default_factory=dict)
    
    def add_to_prefill_queue(self, request_id: int):
        self.prefill_queue.append(request_id)
        self.prefill_wait_iters[request_id] = 0
    
    def move_to_decode_queue(self, request_id: int):
        if request_id in self.prefill_queue:
            self.prefill_queue.remove(request_id)
        self.decode_queue.append(request_id)
        if request_id in self.prefill_wait_iters:
            del self.prefill_wait_iters[request_id]
    
    def update_wait_iters(self):
        """更新 prefill 队列的等待迭代次数"""
        for request_id in self.prefill_queue:
            self.prefill_wait_iters[request_id] = self.prefill_wait_iters.get(request_id, 0) + 1
    
    def get_urgent_prefill(self) -> List[int]:
        """获取等待时间过长的 prefill 请求"""
        urgent = []
        for request_id in self.prefill_queue:
            if self.prefill_wait_iters.get(request_id, 0) >= self.max_waiting_iters:
                urgent.append(request_id)
        return urgent


class AdvancedSchedulers:
    """高级调度器集合"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.mlfq_state = MLFQState()
        self.po_state = POState()
        self.opt_state = OPTState()
        self.lightllm_state = LightLLMState()
    
    def select_mlfq(
        self,
        waiting_queue: List[Request],
        batch_size: int,
        current_time: float
    ) -> List[Request]:
        """MLFQ: 多级反馈队列调度"""
        if not waiting_queue:
            return []
        
        # 初始化新请求
        for req in waiting_queue:
            if req.id not in self.mlfq_state.request_queues:
                self.mlfq_state.request_queues[req.id] = 0
                self.mlfq_state.request_wait_time[req.id] = current_time
        
        # 饥饿预防：提升饥饿的请求
        self.mlfq_state.promote_starved(current_time)
        
        # 按队列级别排序 (级别越低优先级越高)
        sorted_requests = sorted(
            waiting_queue,
            key=lambda r: (
                self.mlfq_state.get_queue_level(r.id),
                r.arrival_time  # 同级别内按到达时间
            )
        )
        
        return sorted_requests[:batch_size]
    
    def select_po(
        self,
        waiting_queue: List[Request],
        batch_size: int
    ) -> List[Request]:
        """PO: 优先级排序 (短作业 FCFS + 长作业 SJF)"""
        if not waiting_queue:
            return []
        
        # 分类短作业和长作业
        short_jobs = []
        long_jobs = []
        
        for req in waiting_queue:
            predicted_decode = self.po_state.predict_decode_length(req.prefill_tokens)
            if self.po_state.is_short_job(int(predicted_decode)):
                short_jobs.append(req)
            else:
                long_jobs.append(req)
        
        # 短作业按 FCFS (到达时间)
        short_jobs.sort(key=lambda r: r.arrival_time)
        
        # 长作业按 SJF (decode tokens 升序)
        long_jobs.sort(key=lambda r: r.decode_tokens)
        
        # 合并：短作业优先
        selected = short_jobs + long_jobs
        return selected[:batch_size]
    
    def select_opt(
        self,
        waiting_queue: List[Request],
        batch_size: int,
        current_time: float
    ) -> List[Request]:
        """OPT: 最优调度 (Score = remaining_tokens × noise_factor)"""
        if not waiting_queue:
            return []
        
        # 更新等待迭代次数
        self.opt_state.update_iterations()
        
        # 初始化新请求
        for req in waiting_queue:
            if req.id not in self.opt_state.request_iterations:
                self.opt_state.request_iterations[req.id] = 0
        
        # 饥饿预防：提升饥饿的请求
        starved = self.opt_state.promote_starved()
        
        # 计算所有请求的分数
        scored_requests = []
        for req in waiting_queue:
            if req.id in starved:
                # 饥饿的请求给予最高优先级
                score = -float('inf')
            else:
                score = self.opt_state.compute_score(req, self.rng)
            scored_requests.append((score, req))
        
        # 按分数升序排序 (分数越低优先级越高)
        scored_requests.sort(key=lambda x: x[0])
        
        selected = [req for _, req in scored_requests[:batch_size]]
        return selected
    
    def select_lightllm_prefill(
        self,
        waiting_queue: List[Request],
        batch_size: int,
        current_time: float
    ) -> List[Request]:
        """LightLLM: Prefill 阶段调度"""
        if not waiting_queue:
            return []
        
        # 更新等待迭代次数
        self.lightllm_state.update_wait_iters()
        
        # 获取紧急的 prefill 请求 (等待时间过长)
        urgent_ids = self.lightllm_state.get_urgent_prefill()
        urgent_requests = [req for req in waiting_queue if req.id in urgent_ids]
        
        # 剩余请求按到达时间排序
        remaining = [req for req in waiting_queue if req.id not in urgent_ids]
        remaining.sort(key=lambda r: r.arrival_time)
        
        # 合并：紧急请求优先
        selected = urgent_requests + remaining
        
        # 限制 prefill batch size
        max_bs = min(batch_size, self.lightllm_state.max_prefill_batch_size)
        return selected[:max_bs]
    
    def select_lightllm_decode(
        self,
        waiting_queue: List[Request],
        batch_size: int
    ) -> List[Request]:
        """LightLLM: Decode 阶段调度"""
        if not waiting_queue:
            return []
        
        # Decode 按 FCFS
        sorted_requests = sorted(waiting_queue, key=lambda r: r.arrival_time)
        
        # 限制 decode batch size
        max_bs = min(batch_size, self.lightllm_state.max_decode_batch_size)
        return sorted_requests[:max_bs]
