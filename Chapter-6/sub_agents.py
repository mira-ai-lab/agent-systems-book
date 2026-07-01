"""
Chapter-6: 子智能体团队实现

基于 Chapter-5 的 hotel_recommendation_langchain.ipynb 扩展出多个专业子智能体：
- WeatherAgent: 天气查询
- AttractionAgent: 景点推荐  
- HotelAgent: 酒店推荐
- RestaurantAgent: 美食推荐
- ItineraryAgent: 行程规划
- FlightAgent: 航班查询

每个子智能体都使用 LangChain Agent + Tool 的模式实现
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from chapter6.paths import load_project_dotenv
from travel_common import (
    build_hotel_search_query,
    fetch_hotels_from_api,
    fetch_attractions_from_api,
    fetch_restaurants_from_api,
    fetch_flights_from_api,
    wttr_weather_by_city_and_date,
    amap_weather_by_city_and_date,
    norm_text,
    require_non_empty,
    resolve_relative_date,
    build_itinerary_from_candidates,
    build_multi_city_itinerary_from_context,
    enrich_itinerary_routes_with_baidu,
    _is_hotel_poi,
)
from weather_mcp import fetch_weather_via_mcp, get_last_mcp_error

load_project_dotenv()


# ============================================================================
# 通用 LLM 配置
# ============================================================================

def create_llm() -> ChatOpenAI:
    """创建统一的LLM客户端"""
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
    
    base_url = os.getenv(
        "DASHSCOPE_CHAT_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).rstrip("/")
    model = os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus")
    
    ssl_verify = os.getenv("OPENAI_SSL_VERIFY", "false").lower() not in (
        "0",
        "false",
        "no",
    )
    return ChatOpenAI(
        model=model,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(verify=ssl_verify),
        http_async_client=httpx.AsyncClient(verify=ssl_verify),
    )


def build_sub_agent_user_message(
    task: Dict[str, Any],
    prior_results: Dict[str, Any],
    *,
    dep_json_limit: int = 2000,
) -> str:
    """构造中心编排器下发给子 Agent 的用户消息（强调结构化 params）。"""
    task_id = task.get("task_id", "?")
    agent_name = task.get("agent", "SubAgent")
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


_WEATHER_FORECAST_KEYS = frozenset({
    "condition", "temp_high_c", "temp_low_c", "avg_humidity",
    "daily_chance_of_rain", "advice", "temp_c", "humidity", "wind_kph",
    "high", "low", "dayweather", "nightweather", "daytemp", "nighttemp", "text",
})


def _slim_forecast(forecast: Any) -> Dict[str, Any]:
    if not isinstance(forecast, dict):
        return {"text": str(forecast)}
    return {k: v for k, v in forecast.items() if k in _WEATHER_FORECAST_KEYS and v is not None}


def slim_weather_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """对外 tool_data 仅保留 city / date / forecast(s) / data_source / error。"""
    if data.get("error"):
        err_out: Dict[str, Any] = {"error": data["error"]}
        if data.get("attempts"):
            err_out["attempts"] = data["attempts"]
        return err_out

    slim: Dict[str, Any] = {}
    if data.get("city"):
        slim["city"] = data["city"]
    if data.get("date"):
        slim["date"] = data["date"]
    if data.get("forecast"):
        slim["forecast"] = _slim_forecast(data["forecast"])
    if data.get("forecasts"):
        slim["forecasts"] = data["forecasts"]
        slim["count"] = data.get("count", len(data["forecasts"]))
    if data.get("data_source"):
        slim["data_source"] = data["data_source"]
    return slim


def compact_weather_tool_data(tool_outputs: List[Any]) -> Optional[Dict[str, Any]]:
    """合并 WeatherAgent 多次 get_weather 调用为紧凑结构（单日 forecast 或多日 forecasts）。"""
    entries: List[Dict[str, Any]] = []
    data_source: Optional[str] = None

    for item in tool_outputs:
        if not isinstance(item, dict):
            continue
        if item.get("error"):
            continue
        fc = item.get("forecast")
        if not fc:
            continue
        entries.append({
            "date": item.get("date"),
            "forecast": _slim_forecast(fc),
        })
        data_source = item.get("data_source") or data_source

    if not entries:
        for item in reversed(tool_outputs):
            if isinstance(item, dict) and item.get("error"):
                return slim_weather_payload(item)
        last = tool_outputs[-1] if tool_outputs else None
        return slim_weather_payload(last) if isinstance(last, dict) else None

    city = next(
        (item.get("city") for item in tool_outputs if isinstance(item, dict) and item.get("city")),
        None,
    )
    if len(entries) == 1:
        return slim_weather_payload({
            "city": city,
            "date": entries[0]["date"],
            "forecast": entries[0]["forecast"],
            "data_source": data_source,
        })

    return slim_weather_payload({
        "city": city,
        "forecasts": entries,
        "count": len(entries),
        "data_source": data_source,
    })


def parse_sub_agent_invoke_result(
    state: Dict[str, Any],
    *,
    task_id: str,
    agent_name: str,
) -> Dict[str, Any]:
    """从子 Agent ainvoke 返回的 messages 中提取 tool_data 与最终总结。"""
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

    if agent_name == "WeatherAgent" and tool_outputs:
        tool_data = compact_weather_tool_data(tool_outputs)
    elif tool_outputs:
        tool_data = tool_outputs[-1]
        if agent_name == "WeatherAgent" and isinstance(tool_data, dict):
            tool_data = slim_weather_payload(tool_data)
    else:
        tool_data = None

    return {
        "task_id": task_id,
        "agent": agent_name,
        "tool_data": tool_data,
        "agent_summary": agent_text,
    }


# ============================================================================
# 1. WeatherAgent - 天气查询子智能体
# ============================================================================

@tool
async def get_weather(city: str, date: str) -> Dict[str, Any]:
    """
    查询指定城市、日期的天气预报
    
    Args:
        city: 城市名称（如"上海"、"北京"）
        date: 日期，支持 YYYY-MM-DD，或相对词：今天/明天/后天
    
    Returns:
        包含天气信息的字典
    """
    ok, err = require_non_empty(city, "city")
    if not ok:
        return {"error": err}

    norm_date, derr = resolve_relative_date(date)
    if derr:
        return {"error": derr}

    # 1) 优先 WeatherAPI MCP（weatherapi-Mcp / npx）
    mcp_result = await fetch_weather_via_mcp(city, norm_date)
    if mcp_result and not mcp_result.get("error"):
        return slim_weather_payload(mcp_result)

    failures: List[str] = []
    if mcp_result is None:
        mcp_detail = get_last_mcp_error() or "unavailable (see [weather-Mcp] stderr)"
        failures.append(f"mcp: {mcp_detail}")
    elif mcp_result.get("error"):
        failures.append(f"mcp: {mcp_result.get('error')}")

    # 2) 高德天气 API
    try:
        result = await amap_weather_by_city_and_date(city, norm_date)
        if not result.get("error") and result.get("forecast"):
            return slim_weather_payload({
                "city": city,
                "date": norm_date,
                "forecast": result["forecast"],
                "data_source": "amap_weather",
            })
        failures.append(f"amap: {result.get('error', 'no_forecast')}")
    except Exception as exc:
        failures.append(f"amap: {exc}")

    # 3) wttr.in 回退
    try:
        result = await wttr_weather_by_city_and_date(city, norm_date)
        if not result.get("error"):
            fc = result.get("forecast") or {}
            if result.get("text") and isinstance(fc, dict):
                fc = {**fc, "text": result["text"]}
            return slim_weather_payload({
                "city": city,
                "date": norm_date,
                "forecast": fc,
                "data_source": "wttr.in",
            })
        failures.append(f"wttr: {result.get('error')}")
    except Exception as e:
        failures.append(f"wttr: {e}")
    
    return {"error": "无法获取天气信息", "attempts": failures}


def _weather_agent_system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""你是专业的天气查询助手。

当前日期（本地时间）：{today}

职责：
1. 只能使用 get_weather 工具查询天气
2. city 和 date 参数必填
3. 返回天气信息后，用简洁友好的语言总结给用户
4. 如果用户询问多日天气，分别调用工具查询每一天
5. 若当前消息已是明确的天气子任务（含具体城市与日期），专注完成查询；不要因为原始需求里曾提到机票/行程而拒绝
6. 结构化参数含 cities 与 dates 时：须对**每个城市×每个日期**分别调用 get_weather（例如 3 城 3 日共 9 次），汇总后再回复

注意：
- date 可传 YYYY-MM-DD，或直接传「今天」「明天」「后天」（工具会自动换算）
- 用户说「今天」时必须对应当前日期 {today}，不要臆造其他日期
- 工具查询顺序：WeatherAPI MCP → 高德 → wttr.in（无需关心底层，只调用 get_weather）
- **工具返回 error 或 attempts 时**：只说明查询失败及 attempts 原因，**禁止**输出具体气温、降水概率、气候统计或 ECMWF 等编造内容
"""


def create_weather_agent() -> Any:
    """创建天气查询子智能体"""
    llm = create_llm()
    agent = create_agent(
        llm,
        tools=[get_weather],
        system_prompt=_weather_agent_system_prompt(),
        checkpointer=MemorySaver()
    )
    return agent


# ============================================================================
# 2. AttractionAgent - 景点推荐子智能体
# ============================================================================

@tool
async def recommend_attractions(
    city: str, 
    preferences: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    根据城市和偏好推荐旅游景点
    
    Args:
        city: 城市名称
        preferences: 偏好描述（如"历史文化"、"自然风光"、"亲子游"）
        limit: 返回数量限制
    
    Returns:
        包含景点列表的字典
    """
    ok, err = require_non_empty(city, "city")
    if not ok:
        return {"error": err}
    
    try:
        result = await fetch_attractions_from_api(city, preferences=preferences, limit=limit)
        if result.get("error"):
            return {"error": result["error"]}
        
        attractions = result.get("attractions", [])
        return {
            "city": city,
            "preferences": preferences,
            "attractions": attractions[:limit],
            "count": len(attractions),
            "data_source": result.get("data_source")
        }
    except Exception as e:
        return {"error": f"attraction_query_failed: {str(e)}"}


ATTRACTION_AGENT_SYSTEM_PROMPT = """你是专业的景点推荐助手。

职责：
1. 只能使用 recommend_attractions 工具查询景点
2. city 参数必填，preferences 可选
3. 返回景点列表后，根据用户偏好推荐最合适的3-5个
4. 提供每个景点的简要介绍和推荐理由
5. 非景点相关问题，回复：我只能协助景点推荐

注意：
- preferences 可以是：历史文化、自然风光、现代建筑、亲子游、拍照打卡等
- 如果用户有特殊要求（如"必去XXX"），在推荐时优先考虑
"""


def create_attraction_agent() -> Any:
    """创建景点推荐子智能体"""
    llm = create_llm()
    agent = create_agent(
        llm,
        tools=[recommend_attractions],
        system_prompt=ATTRACTION_AGENT_SYSTEM_PROMPT,
        checkpointer=MemorySaver()
    )
    return agent


# ============================================================================
# 3. HotelAgent - 酒店推荐子智能体（来自 Chapter-5）
# ============================================================================

import re

_DISTRICT_RE = re.compile(r"[\u4e00-\u9fff]{1,12}(?:区|县)")
_POI_HINT = re.compile(
    r"区|县|景区|景点|古城|公园|广场|山|湖|寺|庙|步行街|地铁|高铁|火车|机场|"
    r"希尔顿|万豪|洲际|如家|全季|亚朵|汉庭|民宿"
)
_SUBJECTIVE_HINT = re.compile(r"安静|吵闹|性价比|舒适|干净|卫生|便宜|奢华|亲子|早餐|贴心|便利|方便")


def poi_search_keyword(preferences: Optional[str], city: Optional[str] = None) -> str:
    """将用户偏好转换为地图搜索关键词（主观偏好映射为核心城区）。"""
    if city:
        return build_hotel_search_query(city, preferences)
    pref = norm_text(preferences)
    if not pref:
        return "酒店"
    if _POI_HINT.search(pref) or _DISTRICT_RE.search(pref):
        return pref if "酒店" in pref else f"{pref} 酒店"
    if _SUBJECTIVE_HINT.search(pref) or "," in pref or "，" in pref:
        return "酒店"
    return pref if "酒店" in pref else f"{pref} 酒店"


def _valid_hotel_poi(h: Dict[str, Any]) -> bool:
    """验证是否为有效的酒店POI"""
    name = (h.get("name") or "").strip()
    if not name:
        return False
    if name.endswith("市") and not h.get("address") and not h.get("district"):
        return False
    return _is_hotel_poi(h)


@tool
async def recommend_hotel(
    city: str,
    preferences: Optional[str] = None,
    budget_cny_per_night_max: Optional[int] = None,
) -> Dict[str, Any]:
    """
    查酒店列表。preferences 传区名/景点/品牌；安静等主观词由模型读 hotels 后判断。
    
    Args:
        city: 城市名称
        preferences: 偏好（如"近景区"、"希尔顿"、"安静"）
        budget_cny_per_night_max: 每晚预算上限（元）
    
    Returns:
        包含酒店列表的字典
    """
    ok, err = require_non_empty(city, "city")
    if not ok:
        return {"error": err}

    pref = norm_text(preferences)
    hotels: List[Dict[str, Any]] = []
    source = "none"
    search_query = build_hotel_search_query(city, pref)
    try:
        res = await fetch_hotels_from_api(city, limit=10, keyword=pref)
        search_query = res.get("search_query") or search_query
        if not res.get("error"):
            source = res.get("data_source", "api")
            for h in res.get("hotels") or []:
                if isinstance(h, dict):
                    item = {k: v for k, v in h.items() if k != "raw"}
                    if _valid_hotel_poi(item):
                        if budget_cny_per_night_max:
                            price = h.get("avg_price_cny")
                            if price and price > budget_cny_per_night_max:
                                continue
                        hotels.append(item)
        elif not hotels:
            return {
                "city": city,
                "error": res.get("error"),
                "message": res.get("message"),
                "search_query": search_query,
                "data_source": res.get("data_source"),
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
   主观感受（安静、舒适）无需填区名——工具会自动按核心城区检索；你在读 hotels 后再判断静音程度。
3. 工具返回 hotels 后，结合用户全部诉求，推荐 3–5 家最合适的酒店（按匹配度排序）。
   每家需包含：名称、地址/位置、参考价格、评分（如有）、1 句推荐理由。
   若用户明确要求「只推荐一家」，则只输出 1 家。
4. 若 hotels 为空或不足 3 家，如实说明并给出已有选项或调整建议（放宽预算/扩大范围）。
5. 非酒店问题，回复：我只能协助酒店推荐。
6. 中心编排器子任务（消息含「结构化参数」）时：务必按 params 中的 city 等字段调用 recommend_hotel；
   说明里提到多城市时，先完成 params 指定城市；可说明其他城市需另行子任务。
"""


def create_hotel_agent() -> Any:
    """创建酒店推荐子智能体"""
    llm = create_llm()
    agent = create_agent(
        llm,
        tools=[recommend_hotel],
        system_prompt=HOTEL_AGENT_SYSTEM_PROMPT,
        checkpointer=MemorySaver()
    )
    return agent


# ============================================================================
# 4. RestaurantAgent - 美食推荐子智能体
# ============================================================================

@tool
async def recommend_restaurant(
    location: str,
    cuisine: Optional[str] = None,
    budget_cny_per_person: Optional[int] = None
) -> Dict[str, Any]:
    """
    根据位置、菜系、预算推荐餐厅
    
    Args:
        location: 地点（城市或区域）
        cuisine: 菜系偏好（如"本地菜"、"海鲜"、"川菜"）
        budget_cny_per_person: 人均预算（元）
    
    Returns:
        包含餐厅列表的字典
    """
    ok, err = require_non_empty(location, "location")
    if not ok:
        return {"error": err}
    
    try:
        result = await fetch_restaurants_from_api(
            location, 
            cuisine=cuisine, 
            limit=10
        )
        if result.get("error"):
            return {"error": result["error"]}
        
        restaurants = result.get("restaurants", [])
        
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
                "error": result.get("error") or "no_valid_restaurants",
                "message": result.get("message") or "未检索到有效餐厅 POI",
                "restaurants": [],
                "data_source": result.get("data_source"),
            }
        
        return {
            "location": location,
            "cuisine": cuisine,
            "budget_cny_per_person": budget_cny_per_person,
            "restaurants": restaurants[:10],
            "count": len(restaurants),
            "data_source": result.get("data_source")
        }
    except Exception as e:
        return {"error": f"restaurant_query_failed: {str(e)}"}


RESTAURANT_AGENT_SYSTEM_PROMPT = """你是专业的美食推荐助手。

职责：
1. 只能使用 recommend_restaurant 工具查询餐厅
2. location 参数必填，cuisine 和 budget 可选
3. 返回餐厅列表后，根据用户偏好从 **tool_data 中的 restaurants** 推荐最合适的3-5家
4. 提供每家餐厅的特色菜和推荐理由
5. 非美食相关问题，回复：我只能协助餐厅推荐

注意：
- cuisine 可以是：本地菜、海鲜、川菜、粤菜、日料、西餐等
- 如果用户有特殊要求（如"适合聚餐"、"环境好"），在推荐时考虑
- **tool_data 含 error 或 restaurants 为空时**：只说明检索失败，**禁止**编造店名、人均、评分
- 仅推荐 tool_data.restaurants 里出现的门店，不得引用未在工具结果中的餐厅
"""


def create_restaurant_agent() -> Any:
    """创建美食推荐子智能体"""
    llm = create_llm()
    agent = create_agent(
        llm,
        tools=[recommend_restaurant],
        system_prompt=RESTAURANT_AGENT_SYSTEM_PROMPT,
        checkpointer=MemorySaver()
    )
    return agent


# ============================================================================
# 5. FlightAgent - 航班查询子智能体
# ============================================================================

@tool
async def search_flights(
    departure: str,
    arrival: str,
    date: str
) -> Dict[str, Any]:
    """
    查询出发地到目的地的航班信息
    
    Args:
        departure: 出发地（城市名或机场三字码，如"上海"或"PVG"）
        arrival: 目的地（城市名或机场三字码）
        date: 日期（格式：YYYY-MM-DD）
    
    Returns:
        包含航班列表的字典
    """
    ok1, err1 = require_non_empty(departure, "departure")
    if not ok1:
        return {"error": err1}
    
    ok2, err2 = require_non_empty(arrival, "arrival")
    if not ok2:
        return {"error": err2}
    
    # 尝试使用飞常准API，回退到aviationstack
    try:
        from travel_common import fetch_flights_from_variflight_api
        result = await fetch_flights_from_variflight_api(departure, arrival, date, limit=10)
        if not result.get("error"):
            return {
                "departure": departure,
                "arrival": arrival,
                "date": date,
                "flights": result.get("flights", []),
                "data_source": "variflight"
            }
    except Exception:
        pass
    
    # 回退方案
    try:
        result = await fetch_flights_from_api(departure, arrival, date, limit=10)
        if not result.get("error"):
            return {
                "departure": departure,
                "arrival": arrival,
                "date": date,
                "flights": result.get("flights", []),
                "data_source": result.get("data_source")
            }
    except Exception as e:
        return {"error": f"flight_query_failed: {str(e)}"}
    
    return {"error": "无法获取航班信息"}


def _flight_agent_system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""你是专业的航班查询助手。

当前日期（本地时间）：{today}

职责：
1. 只能使用 search_flights 工具查询航班
2. departure、arrival、date 三个参数必填
3. 返回航班列表后，推荐最合适的2-3个航班（考虑时间、价格）
4. 提供航班的起飞到达时间、航空公司、参考价格
5. 非航班相关问题，回复：我只能协助航班查询

注意：
- date 必须是 YYYY-MM-DD 格式
- 城市名会自动转换为机场代码（支持常见城市）
- 建议使用机场三字码（如PVG、PEK）以获得更准确的结果
- 用户说「今天」时必须对应当前日期 {today}，不要臆造其他日期
"""


def create_flight_agent() -> Any:
    """创建航班查询子智能体"""
    llm = create_llm()
    agent = create_agent(
        llm,
        tools=[search_flights],
        system_prompt=_flight_agent_system_prompt(),
        checkpointer=MemorySaver()
    )
    return agent


# ============================================================================
# 6. ItineraryAgent - 行程规划子智能体
# ============================================================================

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
    # ItineraryAgent 只基于景点和地图路线规划，不消费天气/酒店/餐饮上游数据。
    out.pop("weather_summary", None)
    out.pop("weather_by_city_date", None)
    out.pop("hotels", None)
    out.pop("hotels_by_city", None)
    out.pop("restaurants", None)
    out.pop("restaurants_by_city", None)
    return out


@tool
async def plan_itinerary(
    departure_city: str,
    destination_city: str,
    days: int,
    weather_summary: Optional[Any] = None,
    attraction_list: Optional[List[Dict[str, Any]]] = None,
    preferences: Optional[Any] = None,
    hotels: Optional[List[Dict[str, Any]]] = None,
    restaurants: Optional[List[Dict[str, Any]]] = None,
    cities: Optional[List[str]] = None,
    dates: Optional[List[str]] = None,
    weather_by_city_date: Optional[Dict[str, Dict[str, Any]]] = None,
    attractions_by_city: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    hotels_by_city: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    restaurants_by_city: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    综合各种信息生成详细的每日行程安排
    
    Args:
        departure_city: 出发城市
        destination_city: 目的地城市
        days: 旅行天数
        weather_summary: 天气概况（可选）
        attraction_list: 景点列表（可选）
        preferences: 用户偏好（可选）
    
    Returns:
        包含行程规划的字典
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
                weather_by_city_date=None,
                attractions_by_city=attractions_by_city,
                hotels_by_city=None,
                restaurants_by_city=None,
            )
            return await enrich_itinerary_routes_with_baidu(itinerary)

        attraction_list = _coerce_attraction_list(attraction_list)
        # 如果没有提供景点列表，先获取
        if not attraction_list:
            attr_result = await fetch_attractions_from_api(
                destination_city,
                preferences=preferences_text,
                limit=15,
            )
            attraction_list = attr_result.get("attractions", [])
        
        # 构建行程
        itinerary = build_itinerary_from_candidates(
            departure_city=departure_city,
            destination_city=destination_city,
            days=days,
            preferences=preferences_text,
            attractions=attraction_list,
            restaurants=[],
            hotels=[],
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
            result = await fetch_attractions_from_api(
                city,
                preferences=params.get("preferences"),
                limit=8,
            )
            attractions = result.get("attractions") if isinstance(result, dict) else None
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
        response = await llm.ainvoke([HumanMessage(content=_build_itinerary_summary_prompt(tool_data))])
        summary = (response.content or "").strip()
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
    与其他子 Agent 一样暴露 ainvoke 接口；
    内部固定执行 plan_itinerary，再用 LLM 生成可读摘要。
    """

    async def ainvoke(self, input: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
3. 综合考虑天气、景点、交通、住宿等因素
4. 生成详细的每日行程（上午、下午、晚上）
5. 提供交通建议、住宿推荐、注意事项
6. 非行程规划问题，回复：我只能协助行程规划
7. 中心编排器子任务（消息含「结构化参数」）时：务必调用 plan_itinerary；依赖摘要中的天气/酒店/美食供参考
8. attraction_list 必须是对象数组，如 [{{"name": "拙政园", "hours": "07:30-17:30"}}]，禁止传纯字符串列表
9. plan_itinerary 返回的 plan 为结构化 slots（含 poi 字段），由你写成自然语言每日攻略；若 tool 返回 error 字段，如实说明原因，禁止编造行程
10. 结构化参数含 weather_summary / hotels / restaurants / attraction_list 时，须传入 plan_itinerary 同名参数（勿留空）

注意：
- 行程安排要合理，考虑景点间的距离和游览时间
- 每天安排2-3个主要景点，避免过于紧凑
- 预留用餐时间和休息时间
- 考虑天气因素给出出行建议
- 用户说「今天」时必须对应当前日期 {today}，不要臆造其他日期
"""


def create_itinerary_agent() -> Any:
    """创建行程规划子智能体"""
    return DeterministicItineraryAgent()


# ============================================================================
# 子智能体工厂
# ============================================================================

class SubAgentFactory:
    """子智能体工厂，统一管理和创建所有子智能体"""
    
    _agents = {}
    
    @classmethod
    def get_agent(cls, agent_name: str) -> Any:
        """
        获取指定的子智能体实例（单例模式）
        
        Args:
            agent_name: 子智能体名称
        
        Returns:
            子智能体实例
        """
        if agent_name not in cls._agents:
            cls._agents[agent_name] = cls._create_agent(agent_name)
        return cls._agents[agent_name]
    
    @classmethod
    def _create_agent(cls, agent_name: str) -> Any:
        """创建指定的子智能体"""
        creators = {
            "WeatherAgent": create_weather_agent,
            "AttractionAgent": create_attraction_agent,
            "HotelAgent": create_hotel_agent,
            "RestaurantAgent": create_restaurant_agent,
            "FlightAgent": create_flight_agent,
            "ItineraryAgent": create_itinerary_agent,
        }
        
        creator = creators.get(agent_name)
        if not creator:
            raise ValueError(f"未知的子智能体: {agent_name}")
        
        print(f"🔧 创建子智能体: {agent_name}")
        return creator()
    
    @classmethod
    def get_all_agent_names(cls) -> List[str]:
        """获取所有可用的子智能体名称"""
        return [
            "WeatherAgent",
            "AttractionAgent", 
            "HotelAgent",
            "RestaurantAgent",
            "FlightAgent",
            "ItineraryAgent"
        ]


# ============================================================================
# 演示入口
# ============================================================================

async def demo_single_agent():
    """演示单个子智能体的使用"""
    from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

    def _content_to_str(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)
        return str(content) if content else ""

    print("=" * 80)
    print("演示：HotelAgent 酒店推荐子智能体")
    print("=" * 80)
    
    agent = SubAgentFactory.get_agent("HotelAgent")
    
    user_query = "我要去大同玩三天，给我推荐酒店，需要安静、近景区"
    
    print(f"\n用户请求: {user_query}\n")
    print("Agent响应: ", end="", flush=True)

    seen_tool_starts: set[str] = set()
    ai_text_buffer = ""
    config = {"configurable": {"thread_id": "hotel_demo"}}

    async for msg, _meta in agent.astream(
        {"messages": [("user", user_query)]},
        config,
        stream_mode="messages",
    ):
        if isinstance(msg, (AIMessage, AIMessageChunk)):
            for tc in msg.tool_calls or []:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                if name and name not in seen_tool_starts:
                    seen_tool_starts.add(name)
                    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                    print(f"\n\n>>> 调用工具: {name}")
                    print(f">>> 参数: {args}")
            if msg.tool_calls:
                continue
            text = _content_to_str(msg.content)
            if not text:
                continue
            if text.startswith(ai_text_buffer):
                delta = text[len(ai_text_buffer) :]
                ai_text_buffer = text
            else:
                delta = text
                ai_text_buffer += text
            if delta:
                print(delta, end="", flush=True)
        elif isinstance(msg, ToolMessage):
            print(f"\n>>> 工具返回: {msg.content}")
            print("\nAgent继续: ", end="", flush=True)
    
    print("\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_single_agent())
