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

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from chapter6.paths import load_project_dotenv
from travel_common import (
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
)
from weather_mcp import fetch_weather_via_mcp

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
        return mcp_result

    # 2) 高德天气 API
    try:
        result = await amap_weather_by_city_and_date(city, norm_date)
        if not result.get("error") and result.get("forecast"):
            return {
                "city": city,
                "date": norm_date,
                "forecast": result["forecast"],
                "data_source": "amap_weather",
            }
    except Exception:
        pass

    # 3) wttr.in 回退
    try:
        result = await wttr_weather_by_city_and_date(city, norm_date)
        if not result.get("error"):
            return {
                "city": city,
                "date": norm_date,
                "text": result.get("text"),
                "forecast": result.get("forecast"),
                "data_source": "wttr.in",
            }
    except Exception as e:
        return {"error": f"weather_query_failed: {str(e)}"}
    
    return {"error": "无法获取天气信息"}


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

注意：
- date 可传 YYYY-MM-DD，或直接传「今天」「明天」「后天」（工具会自动换算）
- 用户说「今天」时必须对应当前日期 {today}，不要臆造其他日期
- 工具查询顺序：WeatherAPI MCP → 高德 → wttr.in（无需关心底层，只调用 get_weather）
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
        result = await fetch_attractions_from_api(city, limit=limit)
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


def poi_search_keyword(preferences: Optional[str]) -> str:
    """将用户偏好转换为地图搜索关键词"""
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
    return True


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
    keyword = poi_search_keyword(pref)

    hotels: List[Dict[str, Any]] = []
    source = "none"
    try:
        res = await fetch_hotels_from_api(city, limit=10, keyword=keyword)
        if not res.get("error"):
            source = res.get("data_source", "api")
            for h in res.get("hotels") or []:
                if isinstance(h, dict):
                    item = {k: v for k, v in h.items() if k != "raw"}
                    if _valid_hotel_poi(item):
                        # 如果有预算限制，过滤超出预算的酒店
                        if budget_cny_per_night_max:
                            price = h.get("avg_price_cny")
                            if price and price > budget_cny_per_night_max:
                                continue
                        hotels.append(item)
    except Exception as e:
        return {"error": f"hotel_query_failed: {str(e)}"}

    return {
        "city": city,
        "search_query": keyword,
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
2. preferences 只填「地图能搜的关键词」：区名、景点、地标、酒店品牌（如 西湖、平城区、希尔顿）。
   不要填主观感受（安静、舒适、性价比）——这些留给你读 hotels 后再判断。
3. 工具返回 hotels 后，结合用户全部诉求，推荐 3–5 家最合适的酒店（按匹配度排序）。
   每家需包含：名称、地址/位置、参考价格、评分（如有）、1 句推荐理由。
   若用户明确要求「只推荐一家」，则只输出 1 家。
4. 若 hotels 为空或不足 3 家，如实说明并给出已有选项或调整建议（放宽预算/扩大范围）。
5. 非酒店问题，回复：我只能协助酒店推荐。
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
3. 返回餐厅列表后，根据用户偏好推荐最合适的3-5家
4. 提供每家餐厅的特色菜和推荐理由
5. 非美食相关问题，回复：我只能协助餐厅推荐

注意：
- cuisine 可以是：本地菜、海鲜、川菜、粤菜、日料、西餐等
- 如果用户有特殊要求（如"适合聚餐"、"环境好"），在推荐时考虑
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

@tool
async def plan_itinerary(
    departure_city: str,
    destination_city: str,
    days: int,
    weather_summary: Optional[str] = None,
    attraction_list: Optional[List[Dict[str, Any]]] = None,
    preferences: Optional[str] = None
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
        # 如果没有提供景点列表，先获取
        if not attraction_list:
            attr_result = await fetch_attractions_from_api(destination_city, limit=15)
            attraction_list = attr_result.get("attractions", [])
        
        # 获取餐厅候选
        rest_result = await fetch_restaurants_from_api(destination_city, limit=10)
        restaurant_list = rest_result.get("restaurants", [])
        
        # 获取酒店候选
        hotel_result = await fetch_hotels_from_api(destination_city, limit=5)
        hotel_list = hotel_result.get("hotels", [])
        
        # 构建行程
        itinerary = build_itinerary_from_candidates(
            departure_city=departure_city,
            destination_city=destination_city,
            days=days,
            preferences=preferences,
            attractions=attraction_list,
            restaurants=restaurant_list,
            hotels=hotel_list
        )
        
        # 添加天气信息
        if weather_summary:
            itinerary["weather_summary"] = weather_summary
        
        return itinerary
    except Exception as e:
        return {"error": f"itinerary_planning_failed: {str(e)}"}


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

注意：
- 行程安排要合理，考虑景点间的距离和游览时间
- 每天安排2-3个主要景点，避免过于紧凑
- 预留用餐时间和休息时间
- 考虑天气因素给出出行建议
- 用户说「今天」时必须对应当前日期 {today}，不要臆造其他日期
"""


def create_itinerary_agent() -> Any:
    """创建行程规划子智能体"""
    llm = create_llm()
    agent = create_agent(
        llm,
        tools=[plan_itinerary],
        system_prompt=_itinerary_agent_system_prompt(),
        checkpointer=MemorySaver()
    )
    return agent


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
    print("=" * 80)
    print("演示：HotelAgent 酒店推荐子智能体")
    print("=" * 80)
    
    agent = SubAgentFactory.get_agent("HotelAgent")
    
    user_query = "我要去大同玩三天，给我推荐酒店，需要安静、近景区"
    
    print(f"\n用户请求: {user_query}\n")
    print("Agent响应: ", end="", flush=True)
    
    async for ev in agent.astream_events(
        {"messages": [("user", user_query)]},
        {"configurable": {"thread_id": "hotel_demo"}},
        version="v2",
    ):
        kind = ev["event"]
        if kind == "on_chat_model_stream" and ev["data"]["chunk"].content:
            print(ev["data"]["chunk"].content, end="", flush=True)
        elif kind == "on_tool_start":
            print(f"\n\n>>> 调用工具: {ev.get('name')}")
            print(f">>> 参数: {ev['data'].get('input')}")
        elif kind == "on_tool_end":
            print(f"\n>>> 工具返回: {ev['data'].get('output')}")
            print("\nAgent继续: ", end="", flush=True)
    
    print("\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_single_agent())
