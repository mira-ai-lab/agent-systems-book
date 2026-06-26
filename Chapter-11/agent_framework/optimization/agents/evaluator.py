"""子 Agent benchmark 评测与批量报告。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

CaseEvalProgressCallback = Callable[[str, "SingleAgentCaseResult"], None]

from langchain_openai import ChatOpenAI

from .fixtures import SingleAgentCase, SingleAgentCaseFixtures, load_single_agent_cases
from .runtime import AgentSyncBridge, default_agent_prompt_template, default_flight_prompt_template
from .scorer import SingleAgentScore, score_single_agent_run


@dataclass
class SingleAgentCaseResult:
    case_id: str
    agent_name: str
    query: str
    score: SingleAgentScore
    raw_response: str = ""
    invoked_tools: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "agent_name": self.agent_name,
            "query": self.query,
            "score": self.score.to_dict(),
            "raw_response": self.raw_response,
            "invoked_tools": list(self.invoked_tools),
        }


@dataclass
class SingleAgentBenchmarkReport:
    agent_name: str
    locale: str
    split: str
    case_count: int
    average_score: float
    cases: List[SingleAgentCaseResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "locale": self.locale,
            "split": self.split,
            "case_count": self.case_count,
            "average_score": round(self.average_score, 4),
            "cases": [item.to_dict() for item in self.cases],
        }


def make_case_eval_progress_printer(agent_name: str) -> CaseEvalProgressCallback:
    """构造 ``--verbose`` 时每条 case 评测完成后的打印回调。"""

    def _on(phase: str, result: SingleAgentCaseResult) -> None:
        tools = result.invoked_tools or ["(none)"]
        print(
            f"[{agent_name}] {phase} {result.case_id} "
            f"score={result.score.total:.3f} tools={tools}",
            flush=True,
        )

    return _on


async def evaluate_single_agent_case(
    bridge: AgentSyncBridge,
    case: SingleAgentCase,
    *,
    system_prompt_template: str,
    phase: str = "eval",
    on_case_evaluated: Optional[CaseEvalProgressCallback] = None,
) -> SingleAgentCaseResult:
    """评测单条 case（bridge 内部 sync，外层保持 async 接口与 planner 优化器一致）。"""
    state = bridge.invoke(
        system_prompt_template=system_prompt_template,
        user_query=case.user_query,
        thread_id=case.case_id,
    )
    score = score_single_agent_run(state, case)
    from .scorer import extract_ai_text, extract_invoked_tool_names

    result = SingleAgentCaseResult(
        case_id=case.case_id,
        agent_name=case.agent_name,
        query=case.user_query,
        score=score,
        raw_response=extract_ai_text(state),
        invoked_tools=extract_invoked_tool_names(state),
    )
    if on_case_evaluated is not None:
        on_case_evaluated(phase, result)
    return result


async def evaluate_single_agent_benchmark(
    bridge: AgentSyncBridge,
    *,
    fixtures: Optional[SingleAgentCaseFixtures] = None,
    agent_name: str = "FlightAgent",
    split: str = "dev",
    system_prompt_template: Optional[str] = None,
    phase: str = "eval",
    on_case_evaluated: Optional[CaseEvalProgressCallback] = None,
) -> SingleAgentBenchmarkReport:
    """批量评测某 Agent 在指定 split 上的平均得分。"""
    loaded = fixtures or load_single_agent_cases()
    template = system_prompt_template or default_agent_prompt_template(
        agent_name, locale=loaded.locale
    )
    selected = loaded.cases_for_split(split, agent_name=agent_name)

    results: List[SingleAgentCaseResult] = []
    for case in selected:
        results.append(
            await evaluate_single_agent_case(
                bridge,
                case,
                system_prompt_template=template,
                phase=phase,
                on_case_evaluated=on_case_evaluated,
            )
        )

    average = sum(item.score.total for item in results) / len(results) if results else 0.0
    return SingleAgentBenchmarkReport(
        agent_name=agent_name,
        locale=loaded.locale,
        split=split,
        case_count=len(results),
        average_score=average,
        cases=results,
    )


def create_agent_bridge(
    llm: ChatOpenAI,
    *,
    agent_name: str = "FlightAgent",
    locale: str = "zh",
) -> AgentSyncBridge:
    """创建指定子 Agent 的 bridge。"""
    return AgentSyncBridge(llm=llm, locale=locale, agent_name=agent_name)


def create_flight_agent_bridge(llm: ChatOpenAI, *, locale: str = "zh") -> AgentSyncBridge:
    """B1 兼容：创建 FlightAgent 专用 bridge。"""
    return create_agent_bridge(llm, agent_name="FlightAgent", locale=locale)
