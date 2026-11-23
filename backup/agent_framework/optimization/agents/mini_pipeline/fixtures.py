"""Mini-pipeline benchmark fixtures（固定 subtask 串联，不跑 Planner）。

Agent-B3：用 fixture 中的 subtask 文本依次调用各子 Agent，模拟简化版执行链。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_framework.config import PROJECT_ROOT
from agent_framework.optimization.agents.fixtures import SingleAgentCase

DEFAULT_MINI_PIPELINE_CASES_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "travel_mini_pipeline_cases.json"
)


@dataclass(frozen=True)
class MiniPipelineStep:
    """串联 pipeline 中的单步：固定 subtask → 指定 Agent。"""

    step_id: str
    agent_name: str
    subtask: str
    tool: str
    tool_args: Dict[str, Any]
    response_keywords: List[str] = field(default_factory=list)

    def to_single_agent_case(self, *, case_id: str) -> SingleAgentCase:
        """转为单 Agent case，供 B1 graph 与 scorer 复用。"""
        return SingleAgentCase(
            case_id=f"{case_id}-{self.step_id}",
            agent_name=self.agent_name,
            user_query=self.subtask,
            tool=self.tool,
            tool_args=dict(self.tool_args),
            response_keywords=list(self.response_keywords),
        )


@dataclass(frozen=True)
class MiniPipelineExpect:
    """Pipeline 级期望（类似 E2E expect，但针对固定 subtask 链）。"""

    min_completed_steps: int = 1
    required_agents: List[str] = field(default_factory=list)
    response_keywords: List[str] = field(default_factory=list)
    min_step_score: float = 0.8


@dataclass(frozen=True)
class MiniPipelineCase:
    """一条 mini-pipeline benchmark：user_query + 有序 steps + pipeline 期望。"""

    case_id: str
    user_query: str
    steps: List[MiniPipelineStep]
    expect: MiniPipelineExpect

    def steps_for_agent(self, agent_name: str) -> List[MiniPipelineStep]:
        return [step for step in self.steps if step.agent_name == agent_name]


@dataclass(frozen=True)
class MiniPipelineFixtures:
    locale: str
    cases: List[MiniPipelineCase]

    def cases_for_split(self, split: str) -> List[MiniPipelineCase]:
        """train=第 1 条 case，dev=第 2 条，all=全部。"""
        normalized = (split or "all").strip().lower()
        if normalized == "all":
            return list(self.cases)
        if normalized not in ("train", "dev"):
            raise ValueError(f"不支持的 split='{split}'，可选: train, dev, all")
        index = 0 if normalized == "train" else 1
        if len(self.cases) <= index:
            return []
        return [self.cases[index]]


def default_mini_pipeline_cases_path() -> Path:
    return DEFAULT_MINI_PIPELINE_CASES_PATH


def load_mini_pipeline_cases(path: Optional[Path] = None) -> MiniPipelineFixtures:
    """加载 travel_mini_pipeline_cases.json。"""
    fixture_path = path or default_mini_pipeline_cases_path()
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    locale = str(payload.get("locale") or "zh").strip() or "zh"
    raw_cases = payload.get("cases") or []
    if not isinstance(raw_cases, list):
        raise ValueError("travel_mini_pipeline_cases.json: cases 必须是数组")

    cases: List[MiniPipelineCase] = []
    for item in raw_cases:
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("case_id") or "").strip()
        user_query = str(item.get("user_query") or "").strip()
        if not case_id or not user_query:
            raise ValueError("mini-pipeline case 缺少 case_id / user_query")

        steps_raw = item.get("steps") or []
        steps: List[MiniPipelineStep] = []
        for idx, step_item in enumerate(steps_raw):
            if not isinstance(step_item, dict):
                continue
            step_id = str(step_item.get("step_id") or f"S{idx + 1}").strip()
            agent_name = str(step_item.get("agent_name") or "").strip()
            subtask = str(step_item.get("subtask") or "").strip()
            tool = str(step_item.get("tool") or "").strip()
            tool_args = step_item.get("tool_args") or {}
            if not agent_name or not subtask or not tool or not isinstance(tool_args, dict):
                raise ValueError(f"{case_id}/{step_id}: 缺少 agent_name / subtask / tool / tool_args")
            keywords = [
                str(token).strip()
                for token in step_item.get("response_keywords") or []
                if str(token).strip()
            ]
            steps.append(
                MiniPipelineStep(
                    step_id=step_id,
                    agent_name=agent_name,
                    subtask=subtask,
                    tool=tool,
                    tool_args=dict(tool_args),
                    response_keywords=keywords,
                )
            )

        if not steps:
            raise ValueError(f"{case_id}: steps 不能为空")

        expect_raw = item.get("expect") or {}
        expect = MiniPipelineExpect(
            min_completed_steps=int(expect_raw.get("min_completed_steps", 1)),
            required_agents=[
                str(name).strip()
                for name in expect_raw.get("required_agents") or []
                if str(name).strip()
            ],
            response_keywords=[
                str(token).strip()
                for token in expect_raw.get("response_keywords") or []
                if str(token).strip()
            ],
            min_step_score=float(expect_raw.get("min_step_score", 0.8)),
        )
        cases.append(
            MiniPipelineCase(
                case_id=case_id,
                user_query=user_query,
                steps=steps,
                expect=expect,
            )
        )

    if not cases:
        raise ValueError("travel_mini_pipeline_cases.json: 未加载到任何 case")
    return MiniPipelineFixtures(locale=locale, cases=cases)
