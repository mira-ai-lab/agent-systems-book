"""Mini-pipeline benchmark 评测。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from langchain_openai import ChatOpenAI

from .fixtures import MiniPipelineCase, MiniPipelineFixtures, load_mini_pipeline_cases
from .runtime import MiniPipelineRunner
from .scorer import MiniPipelineScore, score_mini_pipeline_run


@dataclass
class MiniPipelineCaseResult:
    case_id: str
    user_query: str
    score: MiniPipelineScore
    step_results: Dict[str, Any] = field(default_factory=dict)
    final_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "user_query": self.user_query,
            "score": self.score.to_dict(),
            "step_results": dict(self.step_results),
            "final_response": self.final_response,
        }


@dataclass
class MiniPipelineBenchmarkReport:
    locale: str
    split: str
    case_count: int
    average_score: float
    cases: List[MiniPipelineCaseResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "locale": self.locale,
            "split": self.split,
            "case_count": self.case_count,
            "average_score": round(self.average_score, 4),
            "cases": [item.to_dict() for item in self.cases],
        }


def evaluate_mini_pipeline_case(
    runner: MiniPipelineRunner,
    case: MiniPipelineCase,
    *,
    prompt_templates: Mapping[str, str],
) -> MiniPipelineCaseResult:
    """评测单条 mini-pipeline case。"""
    result = runner.run_case(case, prompt_templates=prompt_templates)
    score = score_mini_pipeline_run(result, case)
    return MiniPipelineCaseResult(
        case_id=case.case_id,
        user_query=case.user_query,
        score=score,
        step_results=dict(result.get("step_results") or {}),
        final_response=str(result.get("final_response") or ""),
    )


async def evaluate_mini_pipeline_benchmark(
    runner: MiniPipelineRunner,
    *,
    fixtures: Optional[MiniPipelineFixtures] = None,
    split: str = "dev",
    prompt_templates: Mapping[str, str],
) -> MiniPipelineBenchmarkReport:
    """批量评测 mini-pipeline 在指定 split 上的平均得分。"""
    loaded = fixtures or load_mini_pipeline_cases()
    selected = loaded.cases_for_split(split)

    results: List[MiniPipelineCaseResult] = []
    for case in selected:
        results.append(
            evaluate_mini_pipeline_case(
                runner,
                case,
                prompt_templates=prompt_templates,
            )
        )

    average = sum(item.score.total for item in results) / len(results) if results else 0.0
    return MiniPipelineBenchmarkReport(
        locale=loaded.locale,
        split=split,
        case_count=len(results),
        average_score=average,
        cases=results,
    )
