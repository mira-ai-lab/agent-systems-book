"""旅行子 Agent 单测 / 优化 benchmark fixtures 加载器。

数据来源：``tests/fixtures/travel_single_agent_cases.json``（5 Agent × 2 case）。
Agent-B1 从 FlightAgent 起步，fixtures 已覆盖全部子 Agent 便于后续 Agent-B2 扩展。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_framework.config import PROJECT_ROOT

# 与 tests/fixtures 共用同一份 JSON，避免重复维护
DEFAULT_CASES_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "travel_single_agent_cases.json"
)
# 兼容 tests.travel_agents.cases.CASES_PATH
CASES_PATH = DEFAULT_CASES_PATH


@dataclass(frozen=True)
class SingleAgentCase:
    """单 Agent benchmark 用例。"""

    case_id: str
    agent_name: str
    user_query: str
    tool: str  # 期望调用的工具名
    tool_args: Dict[str, Any]
    response_keywords: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SingleAgentCaseFixtures:
    locale: str
    cases: List[SingleAgentCase]

    def cases_for_agent(self, agent_name: str) -> List[SingleAgentCase]:
        """按 Agent 名筛选 case（如 FlightAgent）。"""
        return [case for case in self.cases if case.agent_name == agent_name]

    def cases_for_split(self, split: str, *, agent_name: Optional[str] = None) -> List[SingleAgentCase]:
        """简易 split：train=每个 Agent 第 1 条，dev=每个 Agent 第 2 条，all=全部。"""
        normalized = (split or "all").strip().lower()
        pool = self.cases_for_agent(agent_name) if agent_name else list(self.cases)
        if normalized == "all":
            return pool
        if normalized not in ("train", "dev"):
            raise ValueError(f"不支持的 split='{split}'，可选: train, dev, all")

        by_agent: Dict[str, List[SingleAgentCase]] = {}
        for case in pool:
            by_agent.setdefault(case.agent_name, []).append(case)

        index = 0 if normalized == "train" else 1
        selected: List[SingleAgentCase] = []
        for agent_cases in by_agent.values():
            if len(agent_cases) > index:
                selected.append(agent_cases[index])
        return selected


def default_cases_path() -> Path:
    return DEFAULT_CASES_PATH


def load_single_agent_cases(path: Optional[Path] = None) -> SingleAgentCaseFixtures:
    """加载 travel_single_agent_cases.json。"""
    fixture_path = path or default_cases_path()
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    locale = str(payload.get("locale") or "zh").strip() or "zh"
    agents_raw = payload.get("agents") or {}
    if not isinstance(agents_raw, dict):
        raise ValueError("travel_single_agent_cases.json: agents 必须是对象")

    cases: List[SingleAgentCase] = []
    for agent_name, items in agents_raw.items():
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            case_id = str(item.get("case_id") or f"{agent_name.lower()}-{idx + 1}").strip()
            user_query = str(item.get("user_query") or "").strip()
            tool = str(item.get("tool") or "").strip()
            tool_args = item.get("tool_args") or {}
            if not user_query or not tool or not isinstance(tool_args, dict):
                raise ValueError(f"{case_id}: 缺少 user_query / tool / tool_args")
            keywords = [
                str(token).strip()
                for token in item.get("response_keywords") or []
                if str(token).strip()
            ]
            cases.append(
                SingleAgentCase(
                    case_id=case_id,
                    agent_name=str(agent_name).strip(),
                    user_query=user_query,
                    tool=tool,
                    tool_args=dict(tool_args),
                    response_keywords=keywords,
                )
            )

    if not cases:
        raise ValueError("travel_single_agent_cases.json: 未加载到任何 case")
    return SingleAgentCaseFixtures(locale=locale, cases=cases)
