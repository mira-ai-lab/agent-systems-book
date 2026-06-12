"""纯函数：预调查 / 任务拆解 / JSON 解析（无 LangChain 依赖）。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from travel_multi_agent.domain.prompts import PRE_SURVEY_SECTION_KEYS


def parse_pre_survey(text: str) -> Dict[str, Any]:
    """解析预调查四段式输出"""
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
    """解析  任务拆解输出"""
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


def guess_agent(description: str) -> str:
    """根据子任务描述启发式匹配 Agent 名称。"""
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
