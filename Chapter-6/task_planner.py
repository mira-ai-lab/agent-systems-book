"""Planning pipeline for the Chapter 6 central controller.

This module combines pre-survey, Chinese task decomposition, dependency
validation, and city consistency checks. Domain-agent routing is delegated to
``agent_routing.AgentRouter``.

:author: zhanghong, liuhui
:date: 2026-11-17
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI

from agent_routing import AgentRouter
from config import BOOK_ROOT
from llm import invoke_chat


CHAPTER4_DIR = BOOK_ROOT / "Chapter-4"
if not CHAPTER4_DIR.is_dir():
    raise FileNotFoundError(f"Could not locate {CHAPTER4_DIR}")
if str(CHAPTER4_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER4_DIR))

from dependency_analysis import analyze_task_dependencies
from task_decomposition import decompose_task


FACTS_PROMPT = """在我提出正式请求之前，先认真回答以下预调查问题，务必做到知无不言、言无不尽。你具备最顶尖的常识问答水平和大师级的谜题破解能力，知识体系完整且深厚，请毫无保留地运用你的全部知识储备。

【系统参考】今天是 {today}（本地时间）。
- 用户若说「今天」「明天」「后天」「下周」「下礼拜」等相对日期，请基于 {today} 推算具体 YYYY-MM-DD，并写入「已给出或已验证的事实」或「需要推导的事实」。
- 不要使用 2023、2024 等过时年份作为示例日期；若需举例，年份必须与 {today} 一致。
- 若用户消息末尾含「[系统日期锚定：…]」，以该锚定块中的日期为准。

用户输入任务：

{task}

以下是预调查问题：

    1. 请列出请求中明确给出的任何具体事实或数据。可能没有任何此类信息。
    2. 请列出可能需要查阅的任何事实，以及具体可以在哪里找到这些信息。在某些情况下，请求本身会提及权威来源。
    3. 请列出可能需要推导的任何事实（例如，通过逻辑演绎、模拟或计算得出）。
    4. 请列出从记忆中回忆出的任何事实、直觉、经过充分推理的猜测等。

在回答此调查时，请记住，"事实"通常是具体的名称、日期、统计数据等。您的回答应使用以下标题：

    1. 已给出或已验证的事实
    2. 需要查阅的事实
    3. 需要推导的事实
    4. 有根据的猜测

不要在你的回复中包含其他任何标题或部分。在被要求之前，不要列出下一步行动或计划。

在以上四个部分写完后，**另起一行**仅输出如下 JSON 代码块（用于下游编排，不要省略）：
```json
{{"trip_cities": ["城市名1", "城市名2"], "trip_dates": ["YYYY-MM-DD"]}}
```
- trip_cities：用户明确要去的全部城市（去掉「市」后缀），多城旅行须列全
- trip_dates：用户明确出行日期（YYYY-MM-DD），相对日期须按 {today} 换算后写入
- 若无法确定某字段，用空数组 []，不要编造
"""

PRE_SURVEY_SECTION_KEYS = {
    "已给出或已验证的事实": "given_facts",
    "需要查阅的事实": "facts_to_lookup",
    "需要推导的事实": "facts_to_derive",
    "有根据的猜测": "educated_guesses",
}

def parse_pre_survey(text: str) -> Dict[str, Any]:
    """解析 Chapter-2 预调查四段式输出"""
    result: Dict[str, Any] = {
        "given_facts": [],
        "facts_to_lookup": [],
        "facts_to_derive": [],
        "educated_guesses": [],
        "trip_cities": [],
        "trip_dates": [],
        "raw_text": text,
    }
    section_patterns = [
        (re.compile(r"1[\.\、].*已给出"), "given_facts"),
        (re.compile(r"2[\.\、].*需要查阅"), "facts_to_lookup"),
        (re.compile(r"3[\.\、].*需要推导"), "facts_to_derive"),
        (re.compile(r"4[\.\、].*有根据"), "educated_guesses"),
    ]
    current_key: Optional[str] = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        matched = False
        for title, key in PRE_SURVEY_SECTION_KEYS.items():
            if stripped.startswith(title) or stripped.startswith(f"{title}:"):
                current_key = key
                matched = True
                rest = stripped.split(":", 1)[-1].strip()
                if rest and rest != title:
                    result[key].append(rest)
                break
        if not matched:
            for pattern, key in section_patterns:
                if pattern.search(stripped):
                    current_key = key
                    matched = True
                    break
        if matched:
            continue
        if current_key and re.match(r"^[\d\-•*\.]+\s*", stripped):
            item = re.sub(r"^[\d\-•*\.]+\s*", "", stripped)
            if item:
                result[current_key].append(item)
        elif current_key and not stripped.startswith("#"):
            result[current_key].append(stripped)

    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            structured = json.loads(match.group(1))
            if isinstance(structured, dict):
                cities = structured.get("trip_cities")
                if isinstance(cities, list):
                    result["trip_cities"] = [
                        _normalize_city_token(str(c)) for c in cities
                        if _normalize_city_token(str(c))
                    ]
                dates = structured.get("trip_dates")
                if isinstance(dates, list):
                    result["trip_dates"] = [str(d).strip() for d in dates if str(d).strip()]
        except (json.JSONDecodeError, TypeError):
            pass
    return result


def collect_cities_from_subtasks(subtasks: List[Dict[str, Any]]) -> List[str]:
    """从路由 LLM 写入的 params 汇总城市列表（去重、保序）。"""
    found: List[str] = []
    for st in subtasks:
        params = st.get("params") or {}
        candidates: List[Any] = []
        if isinstance(params.get("cities"), list):
            candidates.extend(params["cities"])
        for key in ("city", "location", "departure_city", "destination_city"):
            if params.get(key):
                candidates.append(params[key])
        for c in candidates:
            name = str(c).strip()
            if name and name not in found:
                found.append(name)
    return found


_CITY_ENUM_SEP = re.compile(r"[、，,和及/\s]+")


def _normalize_city_token(token: str) -> str:
    name = token.strip().strip("「」\"'()（）[]")
    name = re.sub(r"(等(几)?个?(地方|城市)?|这几个?(地方|城市)?|三地|三城|几城)$", "", name).strip()
    if name.endswith("市") and len(name) > 1:
        name = name[:-1]
    if not (2 <= len(name) <= 8):
        return ""
    if any(ch in name for ch in "0123456789*＊"):
        return ""
    return name


def extract_cities_from_given_facts(given_facts: List[str]) -> List[str]:
    """从 Ch2 given_facts 解析目标城市（解析 LLM 输出结构，不维护城市白名单）。"""
    if not given_facts:
        return []

    found: List[str] = []
    priority_lines = [
        s for s in given_facts
        if s and any(k in s for k in ("目标城市", "城市", "目的地", "地点", "地方", "前往"))
    ]
    lines = priority_lines or [s for s in given_facts if s]

    for line in lines:
        for match in re.finditer(r"[：:]\s*([^。\n；;（(]+)", line):
            segment = match.group(1)
            for part in _CITY_ENUM_SEP.split(segment):
                name = _normalize_city_token(part)
                if name and name not in found:
                    found.append(name)

    if found:
        return found

    for line in lines:
        for match in re.finditer(
            r"([\u4e00-\u9fff]{2,8})[、，,和及/]+([\u4e00-\u9fff]{2,8})[、，,和及/]+([\u4e00-\u9fff]{2,8})",
            line,
        ):
            for g in match.groups():
                name = _normalize_city_token(g)
                if name and name not in found:
                    found.append(name)
    return found


class TaskPlanner:
    """Build a validated execution plan for the central controller."""

    def __init__(self, llm: ChatOpenAI, agent_registry: Any):
        """Initialize the planner and its domain-agent router.

        Args:
            llm: Model used for pre-survey, decomposition, and dependency work.
            agent_registry: Registry exposing domain-agent capabilities.
        """
        self.llm = llm
        self.agent_registry = agent_registry
        self.agent_router = AgentRouter(llm, agent_registry)

    async def run_pre_survey(self, user_query: str) -> Dict[str, Any]:
        """Run the Chapter 2 pre-survey for a user request."""
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = FACTS_PROMPT.format(today=today, task=user_query.strip())
        response = await invoke_chat(self.llm, prompt)
        return parse_pre_survey(response)

    def _validate_cities(
        self,
        subtasks: List[Dict[str, Any]],
        pre_survey: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Validate and patch multi-city parameters when an agent accepts them."""
        expected = list(pre_survey.get("trip_cities") or [])
        if not expected:
            expected = extract_cities_from_given_facts(
                pre_survey.get("given_facts") or []
            )
        routed = collect_cities_from_subtasks(subtasks)
        missing = [city for city in expected if city not in routed]
        report: Dict[str, Any] = {
            "expected_cities": expected,
            "routed_cities": routed,
            "missing_cities": missing,
            "aligned": not missing,
            "patched": False,
        }

        if len(expected) < 2 or not missing:
            return subtasks, report

        for subtask in subtasks:
            agent_name = subtask.get("agent")
            if not agent_name or subtask.get("routing_error"):
                continue
            if "cities" not in self.agent_registry.get_agent_input_fields(agent_name):
                continue
            params = dict(subtask.get("params") or {})
            params["cities"] = list(expected)
            subtask["params"] = params
            subtask["city_validation"] = "patched_missing_cities"
            report["patched"] = True

        report["routed_cities_after_patch"] = collect_cities_from_subtasks(subtasks)
        return subtasks, report

    async def build_execution_plan(
        self,
        user_query: str,
        pre_survey: Dict[str, Any],
        memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the complete execution plan consumed by the central controller.

        Args:
            user_query: User's original travel request.
            pre_survey: Structured pre-survey facts.
            memories: Retrieved long-term memory records.

        Returns:
            A dictionary containing goals, routed subtasks, dependencies, and
            execution order.
        """
        background_parts = [
            "【思维链预调查】",
            json.dumps(
                {k: v for k, v in pre_survey.items() if k != "raw_text"},
                ensure_ascii=False,
            ),
        ]
        if memories:
            background_parts.extend(
                [
                    "【检索到的长期记忆】",
                    json.dumps(memories, ensure_ascii=False),
                ]
            )

        try:
            decomposition = await decompose_task(
                self.llm,
                user_query,
                background_info="\n".join(background_parts),
                capability_catalog=self.agent_registry.get_all_agents_text(),
            )
        except ValueError:
            decomposition = {
                "total_goal": user_query.strip(),
                "sub_steps": [user_query.strip()],
                "fallback": "single_task",
            }
        sub_steps = decomposition["sub_steps"]
        id_to_task = {
            f"T{index + 1}": task
            for index, task in enumerate(sub_steps)
        }
        execution_order, depends_map = await analyze_task_dependencies(
            self.llm,
            id_to_task,
            self.agent_registry.get_agent_parameters_text(),
        )
        subtasks = await self.agent_router.route_to_agents(
            sub_steps,
            execution_order,
            depends_map,
            pre_survey,
        )
        subtasks, city_validation = self._validate_cities(subtasks, pre_survey)
        return {
            "pre_survey": {k: v for k, v in pre_survey.items() if k != "raw_text"},
            "pre_survey_raw": pre_survey.get("raw_text", ""),
            "retrieved_memories": memories,
            "total_goal": decomposition["total_goal"],
            "planner_fallback": decomposition.get("fallback"),
            "subtasks": subtasks,
            "execution_order": execution_order,
            "city_validation": city_validation,
        }
