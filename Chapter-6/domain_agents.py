"""Define domain agents, tools, capability metadata, and agent creation.

This module supplies the specialized roles used by the Chapter 6 central
controller. It declares weather, attraction, hotel, restaurant, flight, and
itinerary agents, each with a focused prompt and an optional LangChain tool.
The registry derives a capability catalog from the same specifications so task
planning, dependency analysis, routing, and execution share one source of truth.

Main capabilities:

- Define tool schemas and the six domain-agent specifications.
- Normalize tool outputs and extract structured results from agent messages.
- Render agent descriptions and input/output schemas for planning and routing.
- Create cached LangChain agents with short-term checkpoint memory.

Weather and flight tools intentionally return empty results in Chapter 6. The
chapter uses local functions and ordinary HTTPS interfaces only; MCP-based tools
and remote agent collaboration are introduced in later chapters.

:author: zhanghong, liuhui
:date: 2026-11-18
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from config import load_project_dotenv
from attraction_search import search_attractions
from hotel_search import is_valid_hotel_poi, search_hotels
from llm import create_llm
from map_api import normalize_city_name
from restaurant_search import search_restaurants

load_project_dotenv()

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
    input_fields: List[str] = field(default_factory=list)
    output_fields: List[str] = field(default_factory=list)
    parallel_safe: bool = True


def build_domain_agent_user_message(
    task: Dict[str, Any],
    prior_results: Dict[str, Any],
) -> str:
    """Build the user message sent from the controller to one domain agent.

    Args:
        task: Routed task with description, params, and dependencies.
        prior_results: Results of previously executed dependency tasks.

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
        f"【子任务 {task_id} · {agent_name}】",
        "请完成下方子任务，并在调用工具后用中文总结结果。",
        "即使任务说明很长或提到多城市/其他模块，也只完成本子任务；不要回复「我只能协助…」拒答。",
    ]
    if description:
        short = description if len(description) <= 600 else description[:600] + "…"
        lines.append(f"子任务：{short}")

    params = task.get("params")
    if params:
        rendered_params = "，".join(
            f"{key}={value}"
            for key, value in params.items()
        )
        lines.append(f"工具参数：{rendered_params}")

    dependencies = task.get("depends_on") or []
    if dependencies:
        lines.append("前序依赖结果（作为本任务输入）：")
    for dep_id in dependencies:
        if dep_id not in prior_results:
            continue
        dep = prior_results[dep_id]
        if not isinstance(dep, dict):
            lines.append(f"- 前序任务 {dep_id} 的输出：{dep}")
            continue

        task_description = dep.get("task_description") or "（未记录）"
        dep_agent = dep.get("agent") or "（未记录）"
        dep_summary = (dep.get("agent_summary") or "（无文字回复）").strip()
        lines.extend(
            [
                f"- 前序任务 {dep_id}：{task_description}",
                f"  下发给智能体：{dep_agent}",
                f"  智能体回复：{dep_summary}",
            ]
        )

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
        date: Exact ``YYYY-MM-DD`` value produced by fact survey.

    Returns:
        A normalized empty weather response. Chapter 6 does not integrate a
        live weather service; later chapters provide concrete weather sources.
    """
    if not (city or "").strip():
        return {"error": "city is required."}

    try:
        norm_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return {"error": "date must be an exact YYYY-MM-DD string."}

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
    return f"""你是天气查询领域智能体。

当前日期（本地时间）：{today}

职责：
1. 使用可用天气工具完成当前天气子任务
2. 工具结果为空时，明确说明当前数据源未返回有效天气数据
3. 多日查询应基于工具实际返回的数据回答，不得补造缺失日期
4. 当前消息是明确天气子任务时，专注完成查询，不因原始需求还包含机票或行程而拒答

注意：
- 日期必须使用事实调查阶段换算后的 YYYY-MM-DD
- 禁止输出工具结果之外的气温、降水概率或气候统计
- 只根据实际工具结果描述数据来源，不得编造工具调用或天气数据
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
        preferences: Optional interest or landmark preference, such as
            ``历史文化``, ``自然风光``, ``亲子游``, or a named landmark.
        limit: Maximum number of candidates.

    Returns:
        A dictionary containing attraction candidates and data source.
    """
    if not (city or "").strip():
        return {"error": "city is required."}
    
    try:
        result = await search_attractions(
            normalize_city_name(city), preferences=preferences, limit=limit
        )
        if result.error:
            return {
                "error": result.error,
                "message": result.message,
                "data_source": result.data_source,
            }

        attractions = [attraction.model_dump() for attraction in result.attractions]
        return {
            "city": city,
            "preferences": preferences,
            "attractions": attractions[:limit],
            "count": len(attractions),
            "data_source": result.data_source,
        }
    except Exception as e:
        return {"error": f"attraction_query_failed: {str(e)}"}


ATTRACTION_AGENT_SYSTEM_PROMPT = """你是景点推荐领域智能体。

职责：
1. 使用可用景点工具完成当前子任务
2. 只能从工具返回的景点候选中推荐，不得补充未返回的景点
3. 推荐理由只能基于名称、地址、区域、评分和类型等已有字段，不得编造简介、开放时间或门票
4. 非景点相关问题，回复：我只能协助景点推荐

注意：
- 用户明确要求的必去景点或兴趣方向应在候选中优先匹配
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
        preferences: Optional map-searchable area, landmark, or hotel brand.
            Subjective words such as ``安静`` or ``舒适`` are not map-searchable
            attributes and should be left for post-selection judgement.
        budget_cny_per_night_max: Optional nightly budget ceiling in CNY.

    Returns:
        A dictionary containing normalized hotel candidates.
    """
    if not (city or "").strip():
        return {"error": "city is required."}

    pref = (preferences or "").strip()
    hotels: List[Dict[str, Any]] = []
    source = "none"
    search_query = f"{city} 酒店"
    try:
        result = await search_hotels(normalize_city_name(city), limit=10, keyword=pref)
        search_query = result.search_query or search_query
        if not result.error:
            source = result.data_source
            for hotel in result.hotels:
                item = hotel.model_dump()
                if not is_valid_hotel_poi(item):
                    continue
                price = item.get("avg_price_cny")
                if price and budget_cny_per_night_max and price > budget_cny_per_night_max:
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


HOTEL_AGENT_SYSTEM_PROMPT = """你是酒店推荐领域智能体。

规则：
1. 使用可用酒店工具完成当前子任务。
2. 安静、舒适等主观属性不由工具提供，不得声称已经核验。
3. 工具返回酒店候选后，只能基于区域、地址、参考价格、评分和类型推荐 3–5 家酒店。
   每家需包含：名称、地址/位置、参考价格、评分（如有）、1 句有字段依据的推荐理由。
   若用户明确要求「只推荐一家」，则只输出 1 家。
4. 若 hotels 为空或不足 3 家，如实说明并给出已有选项或调整建议（放宽预算/扩大范围）。
5. 非酒店问题，回复：我只能协助酒店推荐。
6. 中心编排器下发结构化参数时，只完成当前子任务指定范围，不扩展到其他城市。
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
        cuisine: Optional cuisine preference, such as ``本地菜``, ``海鲜``,
            ``川菜``, ``粤菜``, ``日料``, or ``西餐``.
        budget_cny_per_person: Optional per-person budget ceiling in CNY.

    Returns:
        A dictionary containing normalized restaurant candidates.
    """
    if not (location or "").strip():
        return {"error": "location is required."}
    
    try:
        result = await search_restaurants(location, cuisine=cuisine, limit=10)
        restaurants = [restaurant.model_dump() for restaurant in result.restaurants]
        
        # 如果有预算限制，过滤超出预算的餐厅
        if budget_cny_per_person:
            restaurants = [
                r for r in restaurants
                if not r.get("avg_price_cny")
                or r["avg_price_cny"] <= budget_cny_per_person
            ]
        
        if not restaurants:
            return {
                "location": location,
                "cuisine": cuisine,
                "budget_cny_per_person": budget_cny_per_person,
                "error": result.error or "no_valid_restaurants",
                "message": result.message or "未检索到有效餐厅",
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


RESTAURANT_AGENT_SYSTEM_PROMPT = """你是美食推荐领域智能体。

职责：
1. 使用可用餐厅工具完成当前子任务
2. 只能从工具返回的餐厅候选中推荐 3–5 家
3. 推荐理由只能基于菜系、区域、地址、人均价格和评分等已有字段，不得编造特色菜
4. 非美食相关问题，回复：我只能协助餐厅推荐

注意：
- 用户提出的聚餐场景或环境偏好只能作为排序参考，不得当成工具已核验的事实
- tool_data 含 error 或 restaurants 为空时，只说明检索失败，禁止编造店名、人均或评分
- 仅推荐工具结果中出现的门店，不得引用未返回的餐厅
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
        date: ``YYYY-MM-DD`` date or a supported relative expression such as
            ``今天`` or ``明天``.

    Returns:
        A normalized empty flight response. Chapter 6 demonstrates orchestration
        and does not integrate a live flight service or MCP tool.
    """
    if not (departure or "").strip():
        return {"error": "departure is required."}
    
    if not (arrival or "").strip():
        return {"error": "arrival is required."}

    try:
        norm_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return {"error": "date must be an exact YYYY-MM-DD string."}
    
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
    return f"""你是航班查询领域智能体。

当前日期（本地时间）：{today}

职责：
1. 使用可用航班工具完成当前子任务
2. 工具返回空列表时，明确说明第 6 章暂无实时航班数据，不得编造航班号、时刻或价格
3. 不得声称调用了外部航班 API 或 MCP 工具
4. 非航班相关问题，回复：我只能协助航班查询

注意：
- 日期必须使用事实调查阶段换算后的 YYYY-MM-DD
"""




# ============================================================================
# 6. ItineraryAgent - 行程规划领域智能体
# ============================================================================

def _itinerary_agent_system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""你是最终的行程规划与总结领域智能体。

当前日期（本地时间）：{today}

职责：
1. 综合依赖任务返回的天气、景点、酒店、餐厅、航班和用户偏好，生成最终日程与答复
2. 不调用工具，只依据依赖摘要中的事实规划日程
3. 数据缺失或冲突时如实说明，不补造工具未返回的信息
4. 用户只询问单一领域时，不扩展为完整旅行规划
5. 非旅行规划问题，回复：我只能协助旅行规划与总结

注意：
- 优先满足用户明确的城市顺序、日期、预算和偏好
- 日程安排应清晰区分每天的城市、景点、餐饮和住宿建议
- 不得编造依赖结果中不存在的景点、营业时间、价格、评分或交通班次
- 日期以事实调查阶段换算后的明确日期为准，不要重新解释相对日期
"""


# ============================================================================
# 领域智能体定义目录
# ============================================================================

DOMAIN_AGENT_SPECS: Dict[str, AgentSpec] = {
    "WeatherAgent": AgentSpec(
        name="WeatherAgent",
        description="根据城市和确切日期查询天气预报；每次请求只查询一个城市、一个日期的天气。",
        tools=[
            ToolSpec(
                tool=get_weather,
                parameters=["city", "date"],
                returns=["city", "date", "forecast", "count", "data_source", "note"],
            )
        ],
        system_prompt=_weather_agent_system_prompt(),
    ),
    "AttractionAgent": AgentSpec(
        name="AttractionAgent",
        description="根据城市和旅行偏好推荐景点；每次请求只检索一个城市的景点候选。",
        tools=[
            ToolSpec(
                tool=recommend_attractions,
                parameters=["city", "preferences", "limit"],
                returns=["city", "attractions", "count", "data_source"],
            )
        ],
        system_prompt=ATTRACTION_AGENT_SYSTEM_PROMPT,
    ),
    "HotelAgent": AgentSpec(
        name="HotelAgent",
        description="根据城市推荐酒店和住宿；每次请求只执行一次酒店推荐操作。",
        tools=[
            ToolSpec(
                tool=recommend_hotel,
                parameters=["city", "preferences", "budget_cny_per_night_max"],
                returns=["city", "hotels", "count", "data_source"],
            )
        ],
        system_prompt=HOTEL_AGENT_SYSTEM_PROMPT,
        parallel_safe=False,
    ),
    "RestaurantAgent": AgentSpec(
        name="RestaurantAgent",
        description="根据位置推荐餐厅；每次请求只查询一组餐厅推荐结果。",
        tools=[
            ToolSpec(
                tool=recommend_restaurant,
                parameters=["location", "cuisine", "budget_cny_per_person"],
                returns=["location", "restaurants", "count", "data_source"],
            )
        ],
        system_prompt=RESTAURANT_AGENT_SYSTEM_PROMPT,
    ),
    "FlightAgent": AgentSpec(
        name="FlightAgent",
        description="根据出发城市、到达城市和确切日期查询航班信息；每次请求只查询一组航线结果。",
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
    ),
    "ItineraryAgent": AgentSpec(
        name="ItineraryAgent",
        description="综合旅行出发时间和返回时间、天气、景点、酒店、餐厅和航班结果，生成最终旅行日程与用户答复。",
        tools=[],
        system_prompt=_itinerary_agent_system_prompt(),
        input_fields=[
            "cities",
            "dates",
            "preferences",
            "departure_city",
            "destination_city",
            "days",
        ],
        output_fields=["final_response"],
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
                ] if spec.tools else [
                    {
                        "name": f"{spec.name}Request",
                        "inputSchema": spec.input_fields,
                        "outputSchema": spec.output_fields,
                    }
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

    def get_capability_catalog_text(self) -> str:
        """Render task capabilities without domain-agent identifiers."""
        return "\n".join(
            f"- {agent['description']}"
            for agent in self.agents.values()
        )

    def get_agent_parameters_text(self) -> str:
        """Render skill input and output schemas for dependency analysis."""
        lines: List[str] = []
        for info in self.agents.values():
            lines.append(f"{info['name']}: {info['description']}")
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

@lru_cache(maxsize=None)
def get_domain_agent(agent_name: str) -> Any:
    """Return one cached domain-agent instance.

    Args:
        agent_name: Registered domain-agent name.

    Returns:
        The initialized LangChain agent.

    Raises:
        ValueError: If the agent name is not declared in ``DOMAIN_AGENT_SPECS``.
    """
    spec = DOMAIN_AGENT_SPECS.get(agent_name)
    if not spec:
        raise ValueError(f"未知的领域智能体: {agent_name}")

    print(f"🔧 创建领域智能体: {agent_name}")
    return create_agent(
        create_llm(),
        tools=[tool_spec.tool for tool_spec in spec.tools],
        system_prompt=spec.system_prompt,
        checkpointer=MemorySaver(),
    )
