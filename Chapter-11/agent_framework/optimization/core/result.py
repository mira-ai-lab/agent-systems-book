"""Shared optimization result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class OptimizationStepRecord:
    step: int
    train_average: float
    dev_average: float
    candidate_dev_average: float
    accepted: bool
    failure_count: int
    prompt_preview: str = ""
    optimizer: str = "local"

    def to_dict(self) -> Dict[str, Any]:
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
    best_prompt: str
    baseline_dev_score: float
    best_dev_score: float
    steps: List[OptimizationStepRecord] = field(default_factory=list)
    optimizer: str = "local"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "baseline_dev_score": round(self.baseline_dev_score, 4),
            "best_dev_score": round(self.best_dev_score, 4),
            "steps": [item.to_dict() for item in self.steps],
        }
