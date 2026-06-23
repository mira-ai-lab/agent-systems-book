"""Load travel single-agent test cases from JSON fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

CASES_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "travel_single_agent_cases.json"


@dataclass(frozen=True)
class SingleAgentCase:
    case_id: str
    agent_name: str
    user_query: str
    tool: str
    tool_args: Dict[str, Any]
    response_keywords: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SingleAgentCaseFixtures:
    locale: str
    cases: List[SingleAgentCase]

    def cases_for_agent(self, agent_name: str) -> List[SingleAgentCase]:
        return [case for case in self.cases if case.agent_name == agent_name]


def default_cases_path() -> Path:
    return CASES_PATH


def load_single_agent_cases(path: Optional[Path] = None) -> SingleAgentCaseFixtures:
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
