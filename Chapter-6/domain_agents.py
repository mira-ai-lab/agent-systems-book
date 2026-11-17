"""Domain agents and tools used by the Chapter 6 central controller.

The module defines weather, attraction, hotel, restaurant, flight, and itinerary
capabilities. Each domain agent combines a system prompt, a LangChain tool, and
short-term checkpoint memory.

:author: zhanghong, liuhui
:date: 2026-11-17
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from config import load_project_dotenv
from attraction_search import is_valid_attraction_poi, search_attractions
from hotel_search import is_valid_hotel_poi, search_hotels
from llm import create_llm, invoke_chat
from map_api import normalize_city_name, require_non_empty
from restaurant_search import is_valid_restaurant_poi, search_restaurants
from travel_planning import (
    resolve_relative_date,
    build_itinerary_from_candidates,
    build_multi_city_itinerary_from_context,
    enrich_itinerary_routes_with_baidu,
)

load_project_dotenv()

QUALITY_OK = "ok"
QUALITY_EMPTY = "empty"
QUALITY_ERROR = "error"
QUALITY_INVALID_POI = "invalid_poi"

ItemValidator = Callable[[Dict[str, Any]], bool]


@dataclass(frozen=True)
class ResultQualityPolicy:
    """Declare how one domain tool result should be classified."""

    items_field: str | None = None
    item_validator: ItemValidator | None = None
    required_field: str | None = None


def evaluate_result_quality(
    tool_data: Any,
    policy: ResultQualityPolicy,
) -> str:
    """Evaluate a tool result using its declarative quality policy.

    Args:
        tool_data: Raw tool payload returned by a domain agent.
        policy: Quality rule declared by the agent specification.

    Returns:
        One of ``ok``, ``empty``, ``error``, or ``invalid_poi``.
    """
    if not isinstance(tool_data, dict) or tool_data.get("error"):
        return QUALITY_ERROR

    if policy.items_field:
        items = tool_data.get(policy.items_field)
        if not isinstance(items, list) or not items:
            return QUALITY_EMPTY
        validator = policy.item_validator or (lambda _item: True)
        if not any(
            isinstance(item, dict) and validator(item)
            for item in items
        ):
            return QUALITY_INVALID_POI
        return QUALITY_OK

    if policy.required_field:
        return (
            QUALITY_OK
            if tool_data.get(policy.required_field)
            else QUALITY_EMPTY
        )
    return QUALITY_OK


@dataclass(frozen=True)
class ToolSpec:
    """Declare one agent-callable tool contract."""

    tool: Any
    parameters: List[str]
    returns: List[str]

    @property
    def name(self) -> str:
        """Return the callable tool's registered name."""
        return self.tool.name

    @property
    def description(self) -> str:
        """Return the callable tool's registered description."""
        return self.tool.description


@dataclass(frozen=True)
class AgentSpec:
    """Declare one domain agent's identity, tools, and system prompt."""

    name: str
    description: str
    tools: List[ToolSpec]
    system_prompt: str
    quality_policy: ResultQualityPolicy
    parallel_safe: bool = True
    deterministic: bool = False


def build_domain_agent_user_message(
    task: Dict[str, Any],
    prior_results: Dict[str, Any],
    *,
    dep_json_limit: int = 2000,
) -> str:
    """Build the user message sent from the controller to one domain agent.

    Args:
        task: Routed task with description, params, and dependencies.
        prior_results: Results of previously executed dependency tasks.
        dep_json_limit: Maximum serialized dependency context length.

    Returns:
        A controller-authored user message for the domain agent.
    """
    task_id = task.get("task_id", "?")
    agent_name = task.get("agent", "DomainAgent")
    description = (task.get("description") or "").strip()
    if "：" in description:
        head, tail = description.split("：", 1)
        if head.endswith("Agent"):
            description = tail.strip()

    lines = [
        f"【中心编排子任务 {task_id} · {agent_name}】",
        "请根据下方「结构化参数」调用你的工具，再用中文总结结果。",
        "即使任务说明很长或提到多城市/其他模块，也只完成本子任务；不要回复「我只能协助…」拒答。",
    ]
    if description:
        short = description if len(description) <= 600 else description[:600] + "…"
        lines.append(f"任务说明：{short}")

    params = task.get("params")
    if params:
        lines.append(f"结构化参数：{json.dumps(params, ensure_ascii=False)}")

    for dep_id in task.get("depends_on") or []:
        if dep_id not in prior_results:
            continue
        dep = prior_results[dep_id]
        if isinstance(dep, dict):
            brief = {
                "agent": dep.get("agent"),
                "tool_data": dep.get("tool_data"),
                "agent_summary": (dep.get("agent_summary") or "")[:400],
            }
        else:
            brief = dep
        dep_json = json.dumps(brief, ensure_ascii=False)
        if len(dep_json) > dep_json_limit:
            dep_json = dep_json[:dep_json_limit] + "…"
        lines.append(f"依赖 {dep_id} 摘要：{dep_json}")

    return "\n".join(lines)


def parse_domain_agent_invoke_result(
    state: Dict[str, Any],
    *,
    task_id: str,
    agent_name: str,
) -> Dict[str, Any]:
    """Extract structured tool data and final text from an agent invocation.

    Args:
        state: Agent state returned by ``ainvoke``.
        task_id: Task identifier assigned by the controller.
        agent_name: Domain-agent name used for the task.

    Returns:
        A normalized subtask result dictionary.
    """
    direct_result = state.get("direct_result")
    if isinstance(direct_result, dict):
        return {
            "task_id": task_id,
            "agent": agent_name,
            "tool_data": direct_result.get("tool_data"),
            "agent_summary": direct_result.get("agent_summary", ""),
        }

    tool_outputs: List[Any] = []
    agent_text = ""
    last_ai_after_tool = ""

    for msg in state.get("messages") or []:
        msg_type = getattr(msg, "type", None)
        if msg_type == "tool" and hasattr(msg, "content"):
            try:
                tool_outputs.append(json.loads(msg.content))
            except (json.JSONDecodeError, TypeError):
                tool_outputs.append(msg.content)
        elif msg_type == "ai" and getattr(msg, "content", None):
            if getattr(msg, "tool_calls", None):
                continue
            agent_text = msg.content
            if tool_outputs:
                last_ai_after_tool = msg.content

    if last_ai_after_tool:
        agent_text = last_ai_after_tool

    tool_data = tool_outputs[-1] if tool_outputs else None

    return {
        "task_id": task_id,
        "agent": agent_name,
        "tool_data": tool_data,
        "agent_summary": agent_text,
    }


# ============================================================================
# 1. WeatherAgent - 天气查询领域智能体
# ============================================================================

@tool
async def get_weather(city: str, date: str) -> Dict[str, Any]:
    """Return the Chapter 6 weather interface with an intentionally empty result.

    Args:
        city: Target city name.
        date: Exact ``YYYY-MM-DD`` value or a relative expression.

    Returns:
        A normalized empty weather response. Chapter 6 does not integrate a
        live weather service; later chapters provide concrete weather sources.
    """
    ok, err = require_non_empty(city, "city")
    if not ok:
        return {"error": err}

    norm_date, derr = resolve_relative_date(date)
    if derr:
        return {"error": derr}

    return {
        "city": city,
        "date": norm_date,
        "forecast": {},
        "count": 0,
        "data_source": "not_integrated",
        "note": "第 6 章尚未接入实时天气数据，后续章节会补充。",
    }


def _weather_agent_system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""你是专业的天气查询助手。

当前日期（本地时间）：{today}

职责：
1. 只能使用 get_weather 工具查询天气
2. city 和 date 参数必填
3. 工具返回空的 forecast 时，明确说明第 6 章尚未接入实时天气数据
4. 如果用户询问多日天气，可以分别调用工具，但不得补造缺失数据
5. 若当前消息已是明确的天气子任务（含具体城市与日期），专注完成查询；不要因为原始需求里曾提到机票/行程而拒绝

注意：
- date 可传 YYYY-MM-DD，或直接传「今天」「明天」「后天」（工具会自动换算）
- 用户说「今天」时必须对应当前日期 {today}，不要臆造其他日期
- **禁止**输出具体气温、降水概率、气候统计等编造内容
- 不得声称调用了外部天气 API 或 MCP 工具
"""


# ============================================================================
# 2. AttractionAgent - 景点推荐领域智能体
# ============================================================================

@tool
async def recommend_attractions(
    city: str, 
    preferences: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """Query attraction candidates for a city.

    Args:
        city: Target city name.
        preferences: Optional interest or landmark preference.
        limit: Maximum number of candidates.

    Returns:
        A dictionary containing attraction candidates and data source.
    """
    ok, err = require_non_empty(city, "city")
    if not ok:
        return {"error": err}
    
    try:
        result = await search_attractions(
            normalize_city_name(city),
            preferences=preferences,
            limit=limit,
        )
        if result.error:
            return {
                "error": result.error,
                "message": result.message,
                "data_source": result.data_source,
            }

        attractions = [
            attraction.model_dump()
            for attraction in result.attractions
        ]
        return {
            "city": city,
            "preferences": preferences,
            "attractions": attractions[:limit],
            "count": len(attractions),
            "data_source": result.data_source,
        }
    except Exception as e:
        return {"error": f"attraction_query_failed: {str(e)}"}


ATTRACTION_AGENT_SYSTEM_PROMPT = """你是专业的景点推荐助手。

职责：
1. 只能使用 recommend_attractions 工具查询景点
2. city 参数必填，preferences 可选
3. 只能从 tool_data.attractions 中挑选并推荐，不得补充未返回的景点
4. 推荐理由只能基于名称、地址、区域、评分和类型等已有字段，不得编造简介、开放时间或门票
5. 非景点相关问题，回复：我只能协助景点推荐

注意：
- preferences 可以是：历史文化、自然风光、现代建筑、亲子游、拍照打卡等
- 如果用户有特殊要求（如"必去XXX"），在推荐时优先考虑
- tool_data 含 error 或 attractions 为空时，只说明未检索到有效景点，不得凭常识补造名单
"""




@tool
async def recommend_hotel(
    city: str,
    preferences: Optional[str] = None,
    budget_cny_per_night_max: Optional[int] = None,
) -> Dict[str, Any]:
    """Query hotel candidates and apply an optional budget filter.

    Args:
        city: Target city name.
        preferences: Optional map-searchable area, landmark, or brand.
        budget_cny_per_night_max: Optional nightly budget ceiling in CNY.

    Returns:
        A dictionary containing normalized hotel candidates.
    """
    ok, err = require_non_empty(city, "city")
    if not ok:
        return {"error": err}

    pref = (preferences or "").strip()
    hotels: List[Dict[str, Any]] = []
    source = "none"
    search_query = f"{city} 酒店"
    try:
        result = await search_hotels(
            normalize_city_name(city),
            limit=10,
            keyword=pref,
        )
        search_query = result.search_query or search_query
        if not result.error:
            source = result.data_source
            for hotel in result.hotels:
                item = hotel.model_dump()
                if not is_valid_hotel_poi(item):
                    continue
                if (
                    budget_cny_per_night_max
                    and item.get("avg_price_cny")
                    and item["avg_price_cny"] > budget_cny_per_night_max
                ):
                    continue
                hotels.append(item)
        elif not hotels:
            return {
                "city": city,
                "error": result.error,
                "message": result.message,
                "search_query": search_query,
                "data_source": result.data_source,
            }
    except Exception as e:
        return {"error": f"hotel_query_failed: {type(e).__name__}: {e or 'unknown'}"}

    return {
        "city": city,
        "search_query": search_query,
        "preferences": pref or None,
        "budget_cny_per_night_max": budget_cny_per_night_max,
        "hotels": hotels,
        "count": len(hotels),
        "data_source": source,
        "note": "请结合 preferences 与 hotels 推荐 3–5 家最合适的，并说明各自特点。",
    }


HOTEL_AGENT_SYSTEM_PROMPT = """你是酒店推荐助手，只能通过工具 recommend_hotel 查酒店。

规则：
1. city 必填；有预算时传入 budget_cny_per_night_max。
2. preferences 只填「地图能搜的关键词」：区名、景点、地标、酒店品牌（如 西湖、平江路、希尔顿）。
   主观感受（安静、舒适）不要作为地图关键词；工具也不提供安静程度等主观属性，不得声称已核验。
3. 工具返回 hotels 后，只能基于区域、地址、参考价格、评分和类型推荐 3–5 家酒店。
   每家需包含：名称、地址/位置、参考价格、评分（如有）、1 句有字段依据的推荐理由。
   若用户明确要求「只推荐一家」，则只输出 1 家。
4. 若 hotels 为空或不足 3 家，如实说明并给出已有选项或调整建议（放宽预算/扩大范围）。
5. 非酒店问题，回复：我只能协助酒店推荐。
6. 中心编排器子任务（消息含「结构化参数」）时：务必按 params 中的 city 等字段调用 recommend_hotel；
   说明里提到多城市时，先完成 params 指定城市；可说明其他城市需另行子任务。
"""




# ============================================================================
# 4. RestaurantAgent - 美食推荐领域智能体
# ============================================================================

@tool
async def recommend_restaurant(
    location: str,
    cuisine: Optional[str] = None,
    budget_cny_per_person: Optional[int] = None
) -> Dict[str, Any]:
    """Query restaurant candidates for a city or district.

    Args:
        location: Target city or district.
        cuisine: Optional cuisine preference.
        budget_cny_per_person: Optional per-person budget ceiling in CNY.

    Returns:
        A dictionary containing normalized restaurant candidates.
    """
    ok, err = require_non_empty(location, "location")
    if not ok:
        return {"error": err}
    
    try:
        result = await search_restaurants(
            location,
            cuisine=cuisine,
            limit=10,
        )
        restaurants = [
            restaurant.model_dump()
            for restaurant in result.restaurants
        ]
        
        # 如果有预算限制，过滤超出预算的餐厅
        if budget_cny_per_person:
            restaurants = [
                r for r in restaurants 
                if not r.get("avg_price_cny") or r["avg_price_cny"] <= budget_cny_per_person
            ]
        
        if not restaurants:
            return {
                "location": location,
                "cuisine": cuisine,
                "budget_cny_per_person": budget_cny_per_person,
                "error": result.error or "no_valid_restaurants",
                "message": result.message or "未检索到有效餐厅 POI",
                "restaurants": [],
                "data_source": result.data_source,
            }
        
        return {
            "location": location,
            "cuisine": cuisine,
            "budget_cny_per_person": budget_cny_per_person,
            "restaurants": restaurants[:10],
            "count": len(restaurants),
            "data_source": result.data_source,
        }
    except Exception as e:
        return {"error": f"restaurant_query_failed: {str(e)}"}


RESTAURANT_AGENT_SYSTEM_PROMPT = """你是专业的美食推荐助手。

职责：
1. 只能使用 recommend_restaurant 工具查询餐厅
2. location 参数必填，cuisine 和 budget 可选
3. 返回餐厅列表后，根据用户偏好从 **tool_data 中的 restaurants** 推荐最合适的3-5家
4. 推荐理由只能基于菜系、区域、地址、人均价格和评分等已有字段，不得编造特色菜
5. 非美食相关问题，回复：我只能协助餐厅推荐

注意：
- cuisine 可以是：本地菜、海鲜、川菜、粤菜、日料、西餐等
- 如果用户有特殊要求（如"适合聚餐"、"环境好"），在推荐时考虑
- **tool_data 含 error 或 restaurants 为空时**：只说明检索失败，**禁止**编造店名、人均、评分
- 仅推荐 tool_data.restaurants 里出现的门店，不得引用未在工具结果中的餐厅
"""




# ============================================================================
# 5. FlightAgent - 航班查询领域智能体
# ============================================================================

@tool
async def search_flights(
    departure: str,
    arrival: str,
    date: str
) -> Dict[str, Any]:
    """Return the Chapter 6 flight interface with an intentionally empty result.

    Args:
        departure: Origin city.
        arrival: Destination city.
        date: Exact flight date in ``YYYY-MM-DD`` format.

    Returns:
        A normalized empty flight response. Chapter 6 demonstrates orchestration
        and does not integrate a live flight service or MCP tool.
    """
    ok1, err1 = require_non_empty(departure, "departure")
    if not ok1:
        return {"error": err1}
    
    ok2, err2 = require_non_empty(arrival, "arrival")
    if not ok2:
        return {"error": err2}

    norm_date, date_error = resolve_relative_date(date)
    if date_error:
        return {"error": date_error}
    
    return {
        "departure": departure,
        "arrival": arrival,
        "date": norm_date,
        "flights": [],
        "count": 0,
        "data_source": "not_integrated",
        "note": "第 6 章未接入实时航班数据或 MCP 工具，本工具仅保留编排接口。",
    }


def _flight_agent_system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""你是专业的航班查询助手。

当前日期（本地时间）：{today}

职责：
1. 只能使用 search_flights 工具查询航班
2. departure、arrival、date 三个参数必填
3. 工具返回空列表时，明确说明第 6 章暂无实时航班数据，不得编造航班号、时刻或价格
4. 不得声称调用了外部航班 API 或 MCP 工具
5. 非航班相关问题，回复：我只能协助航班查询

注意：
- date 必须是 YYYY-MM-DD 格式
- 用户说「今天」时必须对应当前日期 {today}，不要臆造其他日期
"""




# ============================================================================
# 6. ItineraryAgent - 行程规划领域智能体
# ============================================================================

def extract_attraction_context(prior_results: Dict[str, Any]) -> Dict[str, Any]:
    """Collect validated attraction candidates from prior tool results.

    Args:
        prior_results: Completed dependency task results.

    Returns:
        A dictionary containing a flat attraction list and city groupings.
    """
    attractions: List[Dict[str, Any]] = []
    attractions_by_city: Dict[str, List[Dict[str, Any]]] = {}

    for result in prior_results.values():
        if not isinstance(result, dict):
            continue
        tool_data = result.get("tool_data")
        if not isinstance(tool_data, dict) or tool_data.get("error"):
            continue
        candidates = tool_data.get("attractions")
        if not isinstance(candidates, list):
            continue

        city = normalize_city_name(tool_data.get("city"))
        for candidate in candidates:
            if not isinstance(candidate, dict) or not is_valid_attraction_poi(candidate):
                continue
            item = {
                "city": city,
                "name": candidate.get("name"),
                "address": candidate.get("address"),
                "district": candidate.get("district"),
                "location": candidate.get("location"),
                "rating": candidate.get("rating"),
                "type": candidate.get("type"),
            }
            attractions.append(item)
            if city:
                attractions_by_city.setdefault(city, []).append(item)

    return {
        "attraction_list": attractions[:15],
        "attractions_by_city": attractions_by_city,
    }


def inject_itinerary_context(
    task: Dict[str, Any],
    prior_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach validated upstream attractions to an itinerary task.

    Args:
        task: Routed subtask definition.
        prior_results: Completed dependency task results.

    Returns:
        The original task, or an itinerary task with attraction parameters.
    """
    if task.get("agent") != "ItineraryAgent":
        return task

    context = extract_attraction_context(prior_results)
    params = dict(task.get("params") or {})
    if context["attraction_list"]:
        params["attraction_list"] = context["attraction_list"]
    if context["attractions_by_city"]:
        params["attractions_by_city"] = context["attractions_by_city"]
    return {**task, "params": params}


def _coerce_attraction_list(items: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """将 LLM 传入的字符串景点名规范为 dict 列表。"""
    if not items:
        return []
    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str) and item.strip():
            out.append({"name": item.strip()})
    return out


def _coerce_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        parts = [str(v).strip() for v in value if str(v).strip()]
        return "；".join(parts) if parts else None
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    text = str(value).strip()
    return text or None


def _normalize_itinerary_tool_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """ItineraryAgent 内部归一化 planner 传入的结构化参数。"""
    out = dict(params or {})

    cities = out.get("cities")
    if isinstance(cities, str):
        cities = [c.strip() for c in cities.replace("，", ",").split(",") if c.strip()]
    elif not isinstance(cities, list):
        cities = []
    cities = [str(c).strip() for c in cities if str(c).strip()]
    if not cities:
        grouped = out.get("attractions_by_city")
        if isinstance(grouped, dict) and grouped:
            cities = [str(c).strip() for c in grouped.keys() if str(c).strip()]

    destination = (
        out.get("destination_city")
        or out.get("destination")
        or out.get("city")
        or ("、".join(cities) if cities else "")
    )
    if isinstance(destination, list):
        cities = cities or [str(c).strip() for c in destination if str(c).strip()]
        destination = "、".join(cities)

    departure = (
        out.get("departure_city")
        or out.get("origin_city")
        or out.get("from_city")
        or (cities[0] if cities else destination)
    )

    days = out.get("days") or out.get("duration_days")
    if not days:
        days = len(out.get("dates") or []) or 1
    try:
        day_count = int(days)
    except (TypeError, ValueError):
        day_count = len(out.get("dates") or []) or 1

    out["departure_city"] = _coerce_optional_text(departure) or ""
    out["destination_city"] = _coerce_optional_text(destination) or ""
    out["days"] = day_count
    out["cities"] = cities
    out["preferences"] = _coerce_optional_text(out.get("preferences"))
    supported_fields = {
        "departure_city",
        "destination_city",
        "days",
        "cities",
        "dates",
        "preferences",
        "attraction_list",
        "attractions_by_city",
    }
    return {key: value for key, value in out.items() if key in supported_fields}


@tool
async def plan_itinerary(
    departure_city: str,
    destination_city: str,
    days: int,
    cities: Optional[List[str]] = None,
    dates: Optional[List[str]] = None,
    preferences: Optional[Any] = None,
    attraction_list: Optional[List[Dict[str, Any]]] = None,
    attractions_by_city: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Build a structured multi-day itinerary from travel context.

    Args:
        departure_city: Origin city.
        destination_city: Destination city or route label.
        days: Number of travel days.
        cities: Optional ordered multi-city route.
        dates: Optional authoritative travel dates.
        preferences: Optional user preferences.
        attraction_list: Optional single-city attraction candidates.
        attractions_by_city: Optional attraction candidates grouped by city.

    Returns:
        A structured itinerary skeleton or an error dictionary.
    """
    ok1, err1 = require_non_empty(departure_city, "departure_city")
    if not ok1:
        return {"error": err1}
    
    ok2, err2 = require_non_empty(destination_city, "destination_city")
    if not ok2:
        return {"error": err2}
    
    try:
        preferences_text = _coerce_optional_text(preferences)
        city_route = [str(c).strip() for c in (cities or []) if str(c).strip()]
        if not city_route:
            if isinstance(attractions_by_city, dict) and attractions_by_city:
                city_route = [str(c).strip() for c in attractions_by_city.keys() if str(c).strip()]
        if len(city_route) >= 2 or attractions_by_city:
            itinerary = build_multi_city_itinerary_from_context(
                departure_city=departure_city,
                cities=city_route or [destination_city],
                dates=dates,
                preferences=preferences_text,
                attractions_by_city=attractions_by_city,
            )
            return await enrich_itinerary_routes_with_baidu(itinerary)

        attraction_list = _coerce_attraction_list(attraction_list)
        # 如果没有提供景点列表，先获取
        if not attraction_list:
            attr_result = await search_attractions(
                normalize_city_name(destination_city),
                preferences=preferences_text,
                limit=15,
            )
            attraction_list = [
                attraction.model_dump()
                for attraction in attr_result.attractions
            ]
        
        # 构建行程
        itinerary = build_itinerary_from_candidates(
            departure_city=departure_city,
            destination_city=destination_city,
            days=days,
            preferences=preferences_text,
            attractions=attraction_list,
        )

        if itinerary.get("error"):
            return itinerary
        
        return await enrich_itinerary_routes_with_baidu(itinerary)
    except Exception as e:
        return {"error": f"itinerary_planning_failed: {str(e)}"}


def _build_itinerary_summary_prompt(tool_data: Dict[str, Any]) -> str:
    """把结构化行程骨架交给 LLM 润色，但禁止补造工具数据外的事实。"""
    return f"""你是 ItineraryAgent，负责把结构化行程骨架写成用户可读的每日旅游计划。

要求：
1. 只能依据下方 JSON 中的 plan、local_route、candidates、transportation 写作
2. 不得新增 JSON 中不存在的景点、天气数值、酒店、餐厅或车次号
3. 不得补充 JSON 中没有的游览时长、价格、评分、口味评价、预约规则
4. 本 Agent 只做景点路线型行程，不输出酒店、餐饮、天气建议
5. 输出中文，按“第1天/第2天/第3天”组织
6. 重点说明城市切换、上午/下午景点安排、local_route 中的游览顺序和路段估算
7. 语言要像旅行计划，但事实必须保守；可以说“安排前往 X”，不要说“市中心区域/值得一去/菜品精致/环境优雅”等未由 JSON 支撑的评价
8. local_route.routing_status 为 baidu_direction_v2 时，可表述为“百度地图路线规划”；为 estimated_by_coordinates 时，必须表述为“地图坐标估算”，不要说成真实导航结果
9. 路段字段 duration_min 是路线 API 返回的分钟数；duration_min_est 是坐标估算分钟数，二者不要混用

结构化行程 JSON：
{json.dumps(tool_data, ensure_ascii=False, indent=2)}
"""


async def run_itinerary_agent(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    ItineraryAgent 的确定性执行入口：
    先调用 plan_itinerary 生成结构化骨架，再由 LLM 仅基于骨架生成人话摘要。
    """
    params = _normalize_itinerary_tool_params(params)
    if not params.get("attractions_by_city"):
        cities = params.get("cities") or [params.get("destination_city")]
        attractions_by_city: Dict[str, List[Dict[str, Any]]] = {}
        for city in [str(c).strip() for c in cities if str(c).strip()]:
            result = await search_attractions(
                normalize_city_name(city),
                preferences=params.get("preferences"),
                limit=8,
            )
            attractions = [
                attraction.model_dump()
                for attraction in result.attractions
            ]
            if attractions:
                attractions_by_city[city] = attractions
        if attractions_by_city:
            params["attractions_by_city"] = attractions_by_city

    try:
        tool_data = await plan_itinerary.ainvoke(params)
    except Exception as exc:
        tool_data = {
            "error": "itinerary_tool_validation_failed",
            "message": f"{type(exc).__name__}: {exc}",
        }

    if not isinstance(tool_data, dict):
        tool_data = {
            "error": "itinerary_tool_invalid_output",
            "message": "plan_itinerary 未返回字典结果",
            "raw": str(tool_data),
        }

    if tool_data.get("error"):
        return {
            "tool_data": tool_data,
            "agent_summary": f"行程生成失败：{tool_data.get('message') or tool_data.get('error')}",
        }

    try:
        llm = create_llm()
        summary = (
            await invoke_chat(llm, _build_itinerary_summary_prompt(tool_data))
        ).strip()
    except Exception as exc:
        cities = tool_data.get("cities") or [tool_data.get("destination_city")]
        cities_text = "、".join(str(c) for c in cities if c)
        days = tool_data.get("days") or len(tool_data.get("plan") or [])
        summary = (
            f"已生成{days}天结构化行程骨架"
            + (f"（{cities_text}）" if cities_text else "")
            + f"；LLM 润色失败：{type(exc).__name__}: {exc}"
        )

    return {
        "tool_data": tool_data,
        "agent_summary": summary,
    }


def _extract_structured_params_from_messages(messages: List[Any]) -> Dict[str, Any]:
    """从中心编排器下发的用户消息中读取「结构化参数」JSON。"""
    for msg in reversed(messages or []):
        content = ""
        if isinstance(msg, tuple) and len(msg) >= 2:
            content = str(msg[1])
        elif hasattr(msg, "content"):
            content = str(getattr(msg, "content") or "")
        else:
            content = str(msg)
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped.startswith("结构化参数："):
                continue
            raw = stripped.split("结构化参数：", 1)[1].strip()
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", raw)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                        return parsed if isinstance(parsed, dict) else {}
                    except json.JSONDecodeError:
                        return {}
    return {}


class DeterministicItineraryAgent:
    """
    与其他领域 Agent 一样暴露 ainvoke 接口；
    内部固定执行 plan_itinerary，再用 LLM 生成可读摘要。
    """

    async def ainvoke(self, input: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the deterministic itinerary workflow from routed parameters."""
        del config
        params = _extract_structured_params_from_messages(input.get("messages") or [])
        return {"direct_result": await run_itinerary_agent(params)}


def _itinerary_agent_system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""你是专业的行程规划助手。

当前日期（本地时间）：{today}

职责：
1. 只能使用 plan_itinerary 工具生成行程
2. departure_city、destination_city、days 三个参数必填
3. 只基于景点候选、城市顺序、日期和用户偏好生成“景点 + 本地路线”型行程
4. 不负责天气、酒店或餐饮推荐，也不得补造这些领域的数据
5. attraction_list / attractions_by_city 中的景点必须是对象数组，禁止传纯字符串列表
6. plan_itinerary 返回 plan、local_route、cities 和 days；若返回 error，如实说明原因
7. 非行程规划问题，回复：我只能协助行程规划
8. 中心编排器子任务（消息含「结构化参数」）时，务必调用 plan_itinerary

注意：
- 每天安排 2-3 个主要景点，避免过度紧凑
- 路线优先采用工具返回的 local_route，不得凭空补写导航距离或时长
- 不得编造开放时间、门票、评分或景点介绍
- 用户说「今天」时必须对应当前日期 {today}，不要臆造其他日期
"""


# ============================================================================
# 领域智能体定义目录
# ============================================================================

DOMAIN_AGENT_SPECS: Dict[str, AgentSpec] = {
    "WeatherAgent": AgentSpec(
        name="WeatherAgent",
        description="天气查询工具接口；第 6 章返回空结果，不提供实时天气",
        tools=[
            ToolSpec(
                tool=get_weather,
                parameters=["city", "date"],
                returns=["city", "date", "forecast", "count", "data_source", "note"],
            )
        ],
        system_prompt=_weather_agent_system_prompt(),
        quality_policy=ResultQualityPolicy(required_field="forecast"),
    ),
    "AttractionAgent": AgentSpec(
        name="AttractionAgent",
        description="按城市和偏好检索景点 POI 候选",
        tools=[
            ToolSpec(
                tool=recommend_attractions,
                parameters=["city", "preferences", "limit"],
                returns=["city", "attractions", "count", "data_source"],
            )
        ],
        system_prompt=ATTRACTION_AGENT_SYSTEM_PROMPT,
        quality_policy=ResultQualityPolicy(
            items_field="attractions",
            item_validator=is_valid_attraction_poi,
        ),
    ),
    "HotelAgent": AgentSpec(
        name="HotelAgent",
        description="按城市、地图关键词和预算检索酒店候选",
        tools=[
            ToolSpec(
                tool=recommend_hotel,
                parameters=["city", "preferences", "budget_cny_per_night_max"],
                returns=["city", "hotels", "count", "data_source"],
            )
        ],
        system_prompt=HOTEL_AGENT_SYSTEM_PROMPT,
        quality_policy=ResultQualityPolicy(
            items_field="hotels",
            item_validator=is_valid_hotel_poi,
        ),
        parallel_safe=False,
    ),
    "RestaurantAgent": AgentSpec(
        name="RestaurantAgent",
        description="按区域、菜系和预算检索餐厅候选",
        tools=[
            ToolSpec(
                tool=recommend_restaurant,
                parameters=["location", "cuisine", "budget_cny_per_person"],
                returns=["location", "restaurants", "count", "data_source"],
            )
        ],
        system_prompt=RESTAURANT_AGENT_SYSTEM_PROMPT,
        quality_policy=ResultQualityPolicy(
            items_field="restaurants",
            item_validator=is_valid_restaurant_poi,
        ),
    ),
    "FlightAgent": AgentSpec(
        name="FlightAgent",
        description="航班查询工具接口；第 6 章返回空结果，不提供实时航班",
        tools=[
            ToolSpec(
                tool=search_flights,
                parameters=["departure", "arrival", "date"],
                returns=[
                    "departure",
                    "arrival",
                    "date",
                    "flights",
                    "count",
                    "data_source",
                    "note",
                ],
            )
        ],
        system_prompt=_flight_agent_system_prompt(),
        quality_policy=ResultQualityPolicy(required_field="flights"),
    ),
    "ItineraryAgent": AgentSpec(
        name="ItineraryAgent",
        description="根据景点候选、城市顺序和日期生成每日路线型行程",
        tools=[
            ToolSpec(
                tool=plan_itinerary,
                parameters=[
                    "departure_city",
                    "destination_city",
                    "days",
                    "cities",
                    "dates",
                    "attraction_list",
                    "attractions_by_city",
                    "preferences",
                ],
                returns=[
                    "plan",
                    "cities",
                    "dates",
                    "days",
                    "transportation",
                    "candidates",
                    "data_source",
                ],
            )
        ],
        system_prompt=_itinerary_agent_system_prompt(),
        quality_policy=ResultQualityPolicy(required_field="plan"),
        deterministic=True,
    ),
}


class DomainAgentRegistry:
    """Capability catalog derived from the domain-agent specifications."""

    def __init__(self) -> None:
        self.agents = {
            name: {
                "name": spec.name,
                "description": spec.description,
                "skills": [
                    {
                        "name": tool.name,
                        "inputSchema": tool.parameters,
                        "outputSchema": tool.returns,
                    }
                    for tool in spec.tools
                ],
            }
            for name, spec in DOMAIN_AGENT_SPECS.items()
        }

    def get_all_agents_text(self) -> str:
        """Render compact agent descriptions for planning prompts."""
        return "\n".join(
            f"- {agent['name']}: {agent['description']}"
            for agent in self.agents.values()
        )

    def get_agent_parameters_text(self) -> str:
        """Render skill input and output schemas for dependency analysis."""
        lines: List[str] = []
        for info in self.agents.values():
            lines.append(info["name"])
            for skill in info["skills"]:
                lines.append(
                    f"\t{skill['name']}, inputSchema:{skill['inputSchema']}, "
                    f"outputSchema:{skill['outputSchema']}"
                )
        return "\n".join(lines)

    def get_agent_input_fields(self, agent_name: str) -> set[str]:
        """Return all input-field names declared by an agent's skills."""
        info = self.agents.get(agent_name) or {}
        return {
            str(field)
            for skill in info.get("skills", [])
            for field in skill.get("inputSchema", [])
        }

    def is_parallel_safe(self, agent_name: str) -> bool:
        """Return whether instances of this agent may execute concurrently."""
        spec = DOMAIN_AGENT_SPECS.get(agent_name)
        return bool(spec and spec.parallel_safe)

    def evaluate_result(self, agent_name: str, tool_data: Any) -> str:
        """Evaluate one tool result through the agent's quality policy."""
        spec = DOMAIN_AGENT_SPECS.get(agent_name)
        if not spec:
            return QUALITY_ERROR
        return evaluate_result_quality(tool_data, spec.quality_policy)


@lru_cache(maxsize=None)
def get_domain_agent(agent_name: str) -> Any:
    """Return one cached domain-agent instance.

    Args:
        agent_name: Registered domain-agent name.

    Returns:
        The initialized LangChain agent or deterministic itinerary agent.

    Raises:
        ValueError: If the agent name is not declared in ``DOMAIN_AGENT_SPECS``.
    """
    spec = DOMAIN_AGENT_SPECS.get(agent_name)
    if not spec:
        raise ValueError(f"未知的领域智能体: {agent_name}")

    print(f"🔧 创建领域智能体: {agent_name}")
    if spec.deterministic:
        return DeterministicItineraryAgent()
    return create_agent(
        create_llm(),
        tools=[tool_spec.tool for tool_spec in spec.tools],
        system_prompt=spec.system_prompt,
        checkpointer=MemorySaver(),
    )
