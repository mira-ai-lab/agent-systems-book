"""优化过程与结果的统一数据结构。

local_prompt、textgrad_lib 等 Optimizer 在每一步结束后都应写入
``OptimizationStepRecord``，全部步骤收敛后汇总为 ``OptimizationResult``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class OptimizationStepRecord:
    """单步优化记录（一步 = 一次「跑 benchmark → 改 prompt → dev 验证」）。"""

    step: int  # 当前步序号，从 1 开始
    train_average: float  # 本步在 train split 上的平均得分
    dev_average: float  # 截至本步为止，已接受的最优 prompt 在 dev 上的得分
    candidate_dev_average: float  # 本步候选 prompt 在 dev 上的得分（用于 rollback 判断）
    accepted: bool  # 候选 prompt 是否被接受（替换当前最优）
    failure_count: int  # 本步 train 上低于 failure_threshold 的 case 数量
    prompt_preview: str = ""  # 候选 prompt 前 160 字符，便于日志与报告预览
    optimizer: str = "local"  # 产生本步记录的 Optimizer 标识，如 local_prompt / textgrad_lib

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可写入 JSON 报告的字典。"""
        return {
            "step": self.step,
            "train_average": round(self.train_average, 4),
            "dev_average": round(self.dev_average, 4),
            "candidate_dev_average": round(self.candidate_dev_average, 4),
            "accepted": self.accepted,
            "failure_count": self.failure_count,
            "prompt_preview": self.prompt_preview,
            "optimizer": self.optimizer,
        }


@dataclass
class OptimizationResult:
    """一次完整 prompt 优化的最终结果。"""

    best_prompt: str  # rollback 后保留的最优 prompt 全文
    baseline_dev_score: float  # 优化开始前 baseline prompt 在 dev 上的得分
    best_dev_score: float  # 优化结束后 best_prompt 在 dev 上的得分
    steps: List[OptimizationStepRecord] = field(default_factory=list)  # 逐步记录
    optimizer: str = "local"  # 本次优化使用的 Optimizer 标识

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可写入 JSON 报告的字典（不含 best_prompt 全文，由 save 层单独持久化）。"""
        return {
            "optimizer": self.optimizer,
            "baseline_dev_score": round(self.baseline_dev_score, 4),
            "best_dev_score": round(self.best_dev_score, 4),
            "steps": [item.to_dict() for item in self.steps],
        }
