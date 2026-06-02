"""
Chapter-6: 任务规划模块 — 整合 Chapter-2 预调查 + Chapter-4 拆解与依赖分析
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from prompts import (
    AGENT_ROUTING_PROMPT,
    DEPENDENCY_SYSTEM_PROMPT_ZH,
    DEPENDENCY_USER_PROMPT_ZH,
    FACTS_PROMPT,
    PRE_SURVEY_SECTION_KEYS,
    PROMPT_TP_ZH,
)


def parse_pre_survey(text: str) -> Dict[str, Any]:
    """解析 Chapter-2 预调查四段式输出"""
    result: Dict[str, Any] = {
        "given_facts": [],
        "facts_to_lookup": [],
        "facts_to_derive": [],
        "educated_guesses": [],
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
    return result


def parse_decomposition_response(response: str, lang: str = "zh") -> Dict[str, Any]:
    """解析 Chapter-4 任务拆解输出"""
    totalgoal_key = "# 目标" if lang == "zh" else "# Goal"
    substep_key = "# 任务拆解" if lang == "zh" else "# Subtasks"

    total_goal = ""
    sub_steps: List[str] = []
    goal_lines: List[str] = []
    reached_tasks = False

    for line in response.split("\n"):
        line = line.strip()
        if not reached_tasks:
            if line.startswith(substep_key):
                reached_tasks = True
                continue
            if line.startswith(totalgoal_key):
                continue
            if line:
                goal_lines.append(line)
        elif line.startswith("-"):
            task = line.replace("- ", "").strip()
            if task and task != "NULL":
                sub_steps.append(task)

    total_goal = " ".join(goal_lines)
    if not sub_steps:
        sub_steps = ["NULL"]
    return {"totalGoal": total_goal, "subSteps": sub_steps}


def parse_json_from_llm(text: str) -> Any:
    """从 LLM 输出中提取 JSON"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"无法解析 JSON: {text[:500]}")


def order_from_dependency_json(order_json: Dict[str, str], num_tasks: int) -> List[str]:
    """将依赖分析 JSON 转为 task_id 列表"""
    if not order_json:
        return [f"T{i + 1}" for i in range(num_tasks)]
    ordered = []
    for i in range(1, num_tasks + 1):
        tid = order_json.get(str(i)) or order_json.get(i)
        if tid:
            ordered.append(tid)
    if len(ordered) != num_tasks:
        ordered = [f"T{i + 1}" for i in range(num_tasks)]
    return ordered


def infer_depends_on(subtasks: List[Dict[str, Any]]) -> None:
    """根据 execution_order 与描述补全 depends_on（若路由未给出）"""
    pass


class TaskPlanner:
    """整合 Ch2 + Ch4 + 子智能体路由的任务规划器"""

    def __init__(self, llm: ChatOpenAI, agent_registry: Any):
        self.llm = llm
        self.agent_registry = agent_registry

    async def run_pre_survey(self, user_query: str) -> Dict[str, Any]:
        prompt = FACTS_PROMPT.format(task=user_query.strip())
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return parse_pre_survey(response.content or "")

    async def run_decomposition(
        self,
        user_query: str,
        pre_survey: Dict[str, Any],
        memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        background_parts = [
            "【思维链预调查】",
            json.dumps({k: v for k, v in pre_survey.items() if k != "raw_text"}, ensure_ascii=False),
        ]
        if memories:
            background_parts.append("【检索到的长期记忆】")
            background_parts.append(json.dumps(memories, ensure_ascii=False))

        prompt = PROMPT_TP_ZH.format(
            background_info="\n".join(background_parts),
            agent_team=self.agent_registry.get_all_agents_text(),
            user_input=user_query.strip(),
        )
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return parse_decomposition_response(response.content or "", lang="zh")

    async def run_dependency_analysis(self, sub_steps: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
        id_to_task = {f"T{i + 1}": task for i, task in enumerate(sub_steps)}
        user_prompt = DEPENDENCY_USER_PROMPT_ZH.format(
            subtasks=id_to_task,
            agents=self.agent_registry.get_agent_parameters_text(),
        )
        response = await self.llm.ainvoke([
            SystemMessage(content=DEPENDENCY_SYSTEM_PROMPT_ZH),
            HumanMessage(content=user_prompt),
        ])
        order_json = parse_json_from_llm(response.content or "{}")
        execution_order = order_from_dependency_json(order_json, len(sub_steps))

        depends_map: Dict[str, List[str]] = {tid: [] for tid in id_to_task}
        return execution_order, depends_map

    async def route_to_agents(
        self,
        sub_steps: List[str],
        execution_order: List[str],
        depends_map: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        id_to_desc = {f"T{i + 1}": desc for i, desc in enumerate(sub_steps)}
        subtasks_for_prompt = [
            {"task_id": tid, "description": id_to_desc.get(tid, ""), "depends_on": depends_map.get(tid, [])}
            for tid in execution_order
        ]
        prompt = AGENT_ROUTING_PROMPT.format(
            agent_team=self.agent_registry.get_all_agents_text()
            + "\n"
            + self.agent_registry.get_agent_parameters_text(),
            subtasks_json=json.dumps(subtasks_for_prompt, ensure_ascii=False, indent=2),
        )
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        routed = parse_json_from_llm(response.content or "[]")
        if not isinstance(routed, list):
            routed = routed.get("subtasks", [])
        by_id = {t["task_id"]: t for t in routed if isinstance(t, dict) and t.get("task_id")}
        subtasks = []
        for tid in execution_order:
            if tid in by_id:
                item = by_id[tid]
                subtasks.append({
                    "task_id": tid,
                    "description": item.get("description") or id_to_desc.get(tid, ""),
                    "agent": item.get("agent", "ItineraryAgent"),
                    "params": item.get("params") or {},
                    "depends_on": item.get("depends_on") or depends_map.get(tid, []),
                })
            else:
                subtasks.append({
                    "task_id": tid,
                    "description": id_to_desc.get(tid, ""),
                    "agent": self._guess_agent(id_to_desc.get(tid, "")),
                    "params": {},
                    "depends_on": depends_map.get(tid, []),
                })
        return subtasks

    @staticmethod
    def _guess_agent(description: str) -> str:
        desc = description.lower()
        if any(k in desc for k in ("天气", "weather", "气温", "降水")):
            return "WeatherAgent"
        if any(k in desc for k in ("酒店", "hotel", "住宿", "民宿")):
            return "HotelAgent"
        if any(k in desc for k in ("景点", "attraction", "打卡", "景区")):
            return "AttractionAgent"
        if any(k in desc for k in ("餐厅", "美食", "restaurant", "菜")):
            return "RestaurantAgent"
        if any(k in desc for k in ("航班", "flight", "飞机")):
            return "FlightAgent"
        return "ItineraryAgent"

    async def build_execution_plan(
        self,
        user_query: str,
        pre_survey: Dict[str, Any],
        memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        decomposition = await self.run_decomposition(user_query, pre_survey, memories)
        execution_order, depends_map = await self.run_dependency_analysis(decomposition["subSteps"])
        subtasks = await self.route_to_agents(decomposition["subSteps"], execution_order, depends_map)
        return {
            "pre_survey": {k: v for k, v in pre_survey.items() if k != "raw_text"},
            "pre_survey_raw": pre_survey.get("raw_text", ""),
            "retrieved_memories": memories,
            "total_goal": decomposition["totalGoal"],
            "subtasks": subtasks,
            "execution_order": execution_order,
        }
