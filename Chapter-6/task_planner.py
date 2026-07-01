"""
Chapter-6: 任务规划模块 — 整合 Chapter-2 预调查 + Chapter-4 拆解与依赖分析
"""

from __future__ import annotations

import json
import re
from datetime import datetime
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


def validate_cities_against_pre_survey(
    subtasks: List[Dict[str, Any]],
    pre_survey: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """校验路由 params.cities 是否覆盖 given_facts 中的目标城市；缺失则补全。"""
    expected = list(pre_survey.get("trip_cities") or [])
    if not expected:
        expected = extract_cities_from_given_facts(pre_survey.get("given_facts") or [])
    routed = collect_cities_from_subtasks(subtasks)
    report: Dict[str, Any] = {
        "expected_cities": expected,
        "routed_cities": routed,
        "missing_cities": [],
        "aligned": True,
        "patched": False,
    }

    if len(expected) < 2:
        return subtasks, report

    missing = [c for c in expected if c not in routed]
    report["missing_cities"] = missing
    report["aligned"] = not missing
    if not missing:
        return subtasks, report

    for st in subtasks:
        if st.get("routing_error") or not st.get("agent"):
            continue
        if st.get("agent") not in (
            "WeatherAgent", "HotelAgent", "RestaurantAgent", "ItineraryAgent", "AttractionAgent",
        ):
            continue
        params = dict(st.get("params") or {})
        params["cities"] = list(expected)
        st["params"] = params
        st["city_validation"] = "patched_missing_cities"

    report["patched"] = True
    report["routed_cities_after_patch"] = collect_cities_from_subtasks(subtasks)
    return subtasks, report


def infer_depends_on(subtasks: List[Dict[str, Any]]) -> None:
    """根据 Agent 类型补全 depends_on：行程任务依赖所有上游采集类任务。"""
    upstream_ids = [
        t["task_id"] for t in subtasks
        if t.get("agent") and t.get("agent") != "ItineraryAgent"
    ]
    for task in subtasks:
        if task.get("agent") != "ItineraryAgent":
            continue
        deps = list(task.get("depends_on") or [])
        for uid in upstream_ids:
            if uid != task["task_id"] and uid not in deps:
                deps.append(uid)
        task["depends_on"] = deps


def expand_multi_city_subtasks(
    subtasks: List[Dict[str, Any]],
    *,
    authoritative_cities: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """若多城旅行：按 authoritative_cities 或 params 汇总结果，补齐各城酒店/美食子任务。"""
    cities = list(authoritative_cities) if authoritative_cities else collect_cities_from_subtasks(subtasks)
    if len(cities) < 2:
        return subtasks

    hotel_tasks = [st for st in subtasks if st.get("agent") == "HotelAgent"]
    restaurant_tasks = [st for st in subtasks if st.get("agent") == "RestaurantAgent"]
    hotel_cities = {(st.get("params") or {}).get("city") for st in hotel_tasks}
    rest_cities = {(st.get("params") or {}).get("location") for st in restaurant_tasks}
    need_hotel = bool(hotel_tasks) and (
        len(hotel_tasks) == 1 or any(c not in hotel_cities for c in cities)
    )
    need_rest = bool(restaurant_tasks) and (
        len(restaurant_tasks) == 1 or any(c not in rest_cities for c in cities)
    )

    expanded: List[Dict[str, Any]] = []
    hotel_done = rest_done = False

    for st in subtasks:
        agent = st.get("agent", "")
        params_base = dict(st.get("params") or {})

        if agent == "HotelAgent" and need_hotel:
            if hotel_done:
                continue
            hotel_done = True
            tmpl = hotel_tasks[0]
            params_base = dict(tmpl.get("params") or {})
            budget = params_base.get("budget_cny_per_night_max", 800)
            prefs = params_base.get("preferences", "安静")
            for city in cities:
                expanded.append({
                    **tmpl,
                    "description": (
                        f"HotelAgent：为{city}推荐{prefs}型酒店，"
                        f"预算每晚≤{budget}元人民币"
                    ),
                    "params": {**params_base, "city": city, "cities": cities},
                })
        elif agent == "RestaurantAgent" and need_rest:
            if rest_done:
                continue
            rest_done = True
            tmpl = restaurant_tasks[0]
            params_base = dict(tmpl.get("params") or {})
            budget = params_base.get("budget_cny_per_person", 150)
            for city in cities:
                cuisine = params_base.get("cuisine") or "本地菜"
                expanded.append({
                    **tmpl,
                    "description": (
                        f"RestaurantAgent：为{city}推荐{cuisine}及特色餐厅，"
                        f"人均≤{budget}元"
                    ),
                    "params": {**params_base, "location": city, "cities": cities},
                })
        elif agent == "WeatherAgent":
            expanded.append({
                **st,
                "params": {
                    **params_base,
                    "cities": cities,
                    "city": params_base.get("city") or cities[0],
                },
            })
        else:
            if agent == "HotelAgent" and hotel_done:
                continue
            if agent == "RestaurantAgent" and rest_done:
                continue
            if agent in ("HotelAgent", "RestaurantAgent", "WeatherAgent"):
                params = dict(st.get("params") or {})
                params["cities"] = cities
                expanded.append({**st, "params": params})
            else:
                expanded.append(st)

    for i, st in enumerate(expanded):
        st["task_id"] = f"T{i + 1}"
    return expanded


def rebuild_execution_order(subtasks: List[Dict[str, Any]]) -> List[str]:
    """非行程任务保持相对顺序；ItineraryAgent 置后以便依赖生效。"""
    non_itin = [t["task_id"] for t in subtasks if t.get("agent") != "ItineraryAgent"]
    itin = [t["task_id"] for t in subtasks if t.get("agent") == "ItineraryAgent"]
    return non_itin + itin


def ensure_attraction_subtasks(
    subtasks: List[Dict[str, Any]],
    pre_survey: Dict[str, Any],
    total_goal: str,
) -> List[Dict[str, Any]]:
    """多城旅行且缺景点子任务时，按城补 AttractionAgent（置于 Itinerary 之前）。"""
    if any(st.get("agent") == "AttractionAgent" for st in subtasks):
        return subtasks

    cities = list(pre_survey.get("trip_cities") or [])
    if len(cities) < 2:
        cities = extract_cities_from_given_facts(pre_survey.get("given_facts") or [])
    if len(cities) < 2:
        return subtasks

    goal = total_goal or ""
    if not any(k in goal for k in ("行程", "路线", "景点", "旅游", "旅行", "攻略")):
        return subtasks

    pref = "安静"
    for line in pre_survey.get("given_facts") or []:
        if "安静" in line:
            pref = "安静"
            break

    new_tasks: List[Dict[str, Any]] = []
    for st in subtasks:
        if st.get("agent") == "ItineraryAgent":
            for city in cities:
                new_tasks.append({
                    "task_id": "TMP",
                    "description": (
                        f"AttractionAgent：为{city}推荐符合「{pref}」偏好的核心景点，"
                        f"说明开放时间与是否需预约"
                    ),
                    "agent": "AttractionAgent",
                    "params": {"city": city, "preferences": pref, "limit": 5, "cities": cities},
                    "depends_on": [],
                })
        new_tasks.append(st)

    for i, st in enumerate(new_tasks):
        st["task_id"] = f"T{i + 1}"
    return new_tasks


class TaskPlanner:
    """整合 Ch2 + Ch4 + 子智能体路由的任务规划器"""

    def __init__(self, llm: ChatOpenAI, agent_registry: Any):
        self.llm = llm
        self.agent_registry = agent_registry

    async def run_pre_survey(self, user_query: str) -> Dict[str, Any]:
        today = datetime.now().strftime("%Y-%m-%d")
        prompt = FACTS_PROMPT.format(today=today, task=user_query.strip())
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

    def _valid_agent_names(self) -> set[str]:
        agents = getattr(self.agent_registry, "agents", None)
        if isinstance(agents, dict) and agents:
            return set(agents.keys())
        return {
            "WeatherAgent", "AttractionAgent", "HotelAgent",
            "RestaurantAgent", "FlightAgent", "ItineraryAgent",
        }

    @staticmethod
    def _merge_routed_item(
        tid: str,
        item: Dict[str, Any],
        *,
        id_to_desc: Dict[str, str],
        depends_map: Dict[str, List[str]],
        valid_agents: set[str],
    ) -> Dict[str, Any]:
        agent = (item.get("agent") or "").strip()
        if agent not in valid_agents:
            return {
                "task_id": tid,
                "description": item.get("description") or id_to_desc.get(tid, ""),
                "agent": None,
                "params": item.get("params") or {},
                "depends_on": item.get("depends_on") or depends_map.get(tid, []),
                "routing_error": "invalid_agent",
            }
        return {
            "task_id": tid,
            "description": item.get("description") or id_to_desc.get(tid, ""),
            "agent": agent,
            "params": item.get("params") or {},
            "depends_on": item.get("depends_on") or depends_map.get(tid, []),
        }

    @staticmethod
    def _unrouted_subtask(
        tid: str,
        *,
        id_to_desc: Dict[str, str],
        depends_map: Dict[str, List[str]],
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "task_id": tid,
            "description": id_to_desc.get(tid, ""),
            "agent": None,
            "params": {},
            "depends_on": depends_map.get(tid, []),
            "routing_error": reason,
        }

    async def _invoke_routing_llm(
        self,
        subtasks_for_prompt: List[Dict[str, Any]],
        pre_survey: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        given_facts = pre_survey.get("given_facts") or []
        prompt = AGENT_ROUTING_PROMPT.format(
            agent_team=self.agent_registry.get_all_agents_text()
            + "\n"
            + self.agent_registry.get_agent_parameters_text(),
            given_facts_json=json.dumps(given_facts, ensure_ascii=False, indent=2),
            subtasks_json=json.dumps(subtasks_for_prompt, ensure_ascii=False, indent=2),
        )
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        routed = parse_json_from_llm(response.content or "[]")
        if not isinstance(routed, list):
            routed = routed.get("subtasks", [])
        return [t for t in routed if isinstance(t, dict) and t.get("task_id")]

    async def route_to_agents(
        self,
        sub_steps: List[str],
        execution_order: List[str],
        depends_map: Dict[str, List[str]],
        pre_survey: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        valid_agents = self._valid_agent_names()
        id_to_desc = {f"T{i + 1}": desc for i, desc in enumerate(sub_steps)}

        def _prompt_batch(tids: List[str]) -> List[Dict[str, Any]]:
            return [
                {
                    "task_id": tid,
                    "description": id_to_desc.get(tid, ""),
                    "depends_on": depends_map.get(tid, []),
                }
                for tid in tids
            ]

        by_id: Dict[str, Dict[str, Any]] = {}
        for item in await self._invoke_routing_llm(_prompt_batch(execution_order), pre_survey):
            by_id[item["task_id"]] = item

        missing = [tid for tid in execution_order if tid not in by_id]
        if missing:
            for item in await self._invoke_routing_llm(_prompt_batch(missing), pre_survey):
                by_id[item["task_id"]] = item

        subtasks: List[Dict[str, Any]] = []
        for tid in execution_order:
            if tid not in by_id:
                subtasks.append(TaskPlanner._unrouted_subtask(
                    tid, id_to_desc=id_to_desc, depends_map=depends_map, reason="llm_routing_missing",
                ))
            else:
                subtasks.append(self._merge_routed_item(
                    tid, by_id[tid],
                    id_to_desc=id_to_desc,
                    depends_map=depends_map,
                    valid_agents=valid_agents,
                ))
        return subtasks

    async def build_execution_plan(
        self,
        user_query: str,
        pre_survey: Dict[str, Any],
        memories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        decomposition = await self.run_decomposition(user_query, pre_survey, memories)
        execution_order, depends_map = await self.run_dependency_analysis(decomposition["subSteps"])
        subtasks = await self.route_to_agents(
            decomposition["subSteps"], execution_order, depends_map, pre_survey,
        )
        subtasks, city_validation = validate_cities_against_pre_survey(subtasks, pre_survey)
        authoritative = city_validation.get("expected_cities") or None
        subtasks = expand_multi_city_subtasks(subtasks, authoritative_cities=authoritative)
        subtasks = ensure_attraction_subtasks(subtasks, pre_survey, decomposition["totalGoal"])
        infer_depends_on(subtasks)
        execution_order = rebuild_execution_order(subtasks)
        return {
            "pre_survey": {k: v for k, v in pre_survey.items() if k != "raw_text"},
            "pre_survey_raw": pre_survey.get("raw_text", ""),
            "retrieved_memories": memories,
            "total_goal": decomposition["totalGoal"],
            "subtasks": subtasks,
            "execution_order": execution_order,
            "city_validation": city_validation,
        }
