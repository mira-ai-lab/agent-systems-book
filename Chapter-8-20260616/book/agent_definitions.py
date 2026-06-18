"""
本书案例 — 子智能体设计与定义（定义层）

本章只展示：每个专业 Agent 的职责、工具接口、System Prompt、注册表。
不包含：LLM 调用、地图/天气 API、LangGraph 编译（见仓库完整实现）。

完整可运行代码：domains/travel/agents/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 通用数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    """工具契约：书名册里只写接口，实现可替换为 Mock 或真实 API。"""

    name: str
    description: str
    parameters: List[str]
    returns: List[str]


@dataclass(frozen=True)
class AgentSpec:
    """单个子智能体的完整设计定义。"""

    factory_name: str          # SubAgentFactory 中的名称，如 WeatherAgent
    node_name: str             # Supervisor handoff 节点名，如 weather_agent
    title: str                 # 中文职责标题
    description: str           # 给 TaskPlanner / Supervisor 看的说明
    tools: List[ToolSpec]
    system_prompt: str         # Agent System Prompt（{today} 运行时注入）


# ---------------------------------------------------------------------------
# 1. WeatherAgent
# ---------------------------------------------------------------------------

WEATHER_AGENT = AgentSpec(
    factory_name="WeatherAgent",
    node_name="weather_agent",
    title="天气查询",
    description="查询指定城市、日期的天气预报，提供温度、状况与出行建议",
    tools=[
        ToolSpec(
            name="get_weather",
            description="查询城市在某一日的天气",
            parameters=["city", "date"],
            returns=["city", "date", "forecast", "data_source"],
        ),
    ],
    system_prompt="""你是专业的天气查询助手。

当前日期（本地时间）：{today}

职责：
1. 只能使用 get_weather 工具
2. city、date 必填；date 支持 YYYY-MM-DD 或 今天/明天/后天
3. 多日天气：分别调用工具后汇总
4. 收到明确天气子任务时专注查询，不因上下文含机票/行程而拒绝

注意：用户说「今天」时必须对应 {today}，禁止臆造年份或日期。
""",
)


# ---------------------------------------------------------------------------
# 2. FlightAgent
# ---------------------------------------------------------------------------

FLIGHT_AGENT = AgentSpec(
    factory_name="FlightAgent",
    node_name="flight_agent",
    title="航班查询",
    description="查询出发地到目的地的航班，含时间、航司、参考价",
    tools=[
        ToolSpec(
            name="search_flights",
            description="按出发地、目的地、日期查航班",
            parameters=["departure", "arrival", "date"],
            returns=["flights", "data_source"],
        ),
    ],
    system_prompt="""你是专业的航班查询助手。

当前日期（本地时间）：{today}

职责：
1. 只能使用 search_flights
2. departure、arrival、date 必填
3. 返回后推荐 2–3 个合适航班（时间、价格）
4. 非航班问题回复：我只能协助航班查询
""",
)


# ---------------------------------------------------------------------------
# 3. AttractionAgent
# ---------------------------------------------------------------------------

ATTRACTION_AGENT = AgentSpec(
    factory_name="AttractionAgent",
    node_name="attraction_agent",
    title="景点推荐",
    description="按城市与偏好推荐 3–5 个景点",
    tools=[
        ToolSpec(
            name="recommend_attractions",
            description="推荐旅游景点",
            parameters=["city", "preferences", "limit"],
            returns=["attractions", "count", "data_source"],
        ),
    ],
    system_prompt="""你是专业的景点推荐助手。

职责：
1. 只能使用 recommend_attractions；city 必填
2. 从结果中推荐 3–5 个，附简要介绍与理由
3. preferences 示例：历史文化、亲子、拍照打卡
""",
)


# ---------------------------------------------------------------------------
# 4. RestaurantAgent
# ---------------------------------------------------------------------------

RESTAURANT_AGENT = AgentSpec(
    factory_name="RestaurantAgent",
    node_name="restaurant_agent",
    title="美食推荐",
    description="按位置、菜系、预算推荐 3–5 家餐厅",
    tools=[
        ToolSpec(
            name="recommend_restaurant",
            description="推荐餐厅",
            parameters=["location", "cuisine", "budget_cny_per_person"],
            returns=["restaurants", "count", "data_source"],
        ),
    ],
    system_prompt="""你是专业的美食推荐助手。

职责：
1. 只能使用 recommend_restaurant；location 必填
2. 推荐 3–5 家，说明特色菜与理由
3. cuisine 示例：本地菜、川菜、火锅
""",
)


# ---------------------------------------------------------------------------
# 5. HotelAgent
# ---------------------------------------------------------------------------

HOTEL_AGENT = AgentSpec(
    factory_name="HotelAgent",
    node_name="hotel_agent",
    title="酒店推荐",
    description="按城市、位置关键词、预算推荐 3–5 家酒店",
    tools=[
        ToolSpec(
            name="recommend_hotel",
            description="查酒店列表",
            parameters=["city", "preferences", "budget_cny_per_night_max"],
            returns=["hotels", "count", "data_source"],
        ),
    ],
    system_prompt="""你是酒店推荐助手，只能通过 recommend_hotel 查酒店。

规则：
1. city 必填；有预算时传 budget_cny_per_night_max
2. preferences 只填地图可搜词：西湖、希尔顿、某区；主观词（安静、性价比）由你读 hotels 后再判断
3. 推荐 3–5 家，含名称、地址、价格、评分、理由
4. 用户明确要求「只推荐一家」时才只输出 1 家
""",
)


# ---------------------------------------------------------------------------
# 6. ItineraryAgent
# ---------------------------------------------------------------------------

ITINERARY_AGENT = AgentSpec(
    factory_name="ItineraryAgent",
    node_name="itinerary_agent",
    title="行程规划",
    description="综合天气、景点、餐饮等信息生成多日行程",
    tools=[
        ToolSpec(
            name="plan_itinerary",
            description="生成每日行程",
            parameters=[
                "departure_city",
                "destination_city",
                "days",
                "weather_summary",
                "attraction_list",
                "preferences",
            ],
            returns=["daily_plan", "tips"],
        ),
    ],
    system_prompt="""你是专业的行程规划助手。

当前日期（本地时间）：{today}

职责：
1. 只能使用 plan_itinerary
2. departure_city、destination_city、days 必填
3. 可消费上游子任务结果（天气、景点、航班等）
4. 输出按日的上午/下午/晚上安排与注意事项
""",
)


# ---------------------------------------------------------------------------
# 注册表（供 TaskPlanner 路由 & 书中列表展示）
# ---------------------------------------------------------------------------

ALL_AGENTS: Dict[str, AgentSpec] = {
    a.factory_name: a
    for a in (
        WEATHER_AGENT,
        FLIGHT_AGENT,
        ATTRACTION_AGENT,
        RESTAURANT_AGENT,
        HOTEL_AGENT,
        ITINERARY_AGENT,
    )
}

# Supervisor handoff 顺序（书中示意，实际调度由 LLM / TaskPlanner 决定）
SUPERVISOR_NODE_SPECS: List[tuple[str, str, str]] = [
    (a.node_name, a.factory_name, a.title)
    for a in ALL_AGENTS.values()
]


class SubAgentRegistry:
    """Chapter-4 任务路由用的 Agent 注册表（与 central_orchestrator 同构）。"""

    def __init__(self) -> None:
        self.agents = {
            name: {
                "name": spec.factory_name,
                "description": spec.description,
                "skills": [
                    {
                        "name": t.name,
                        "inputSchema": t.parameters,
                        "outputSchema": t.returns,
                    }
                    for t in spec.tools
                ],
            }
            for name, spec in ALL_AGENTS.items()
        }

    def get_all_agents_text(self) -> str:
        lines = []
        for spec in ALL_AGENTS.values():
            tools = ", ".join(t.name for t in spec.tools)
            lines.append(f"- {spec.factory_name}: {spec.description}（工具: {tools}）")
        return "\n".join(lines)

    def get_agent_parameters_text(self) -> str:
        lines = []
        for spec in ALL_AGENTS.values():
            for t in spec.tools:
                params = ", ".join(t.parameters)
                lines.append(f"  {spec.factory_name}.{t.name}({params})")
        return "\n".join(lines)


def format_agent_catalog() -> str:
    """打印/写入书籍用的 Agent 一览表。"""
    rows = ["| Agent | Handoff 节点 | 工具 | 职责 |", "|---|---|---|---|"]
    for spec in ALL_AGENTS.values():
        tools = ", ".join(t.name for t in spec.tools)
        rows.append(
            f"| {spec.factory_name} | `{spec.node_name}` | {tools} | {spec.title} |"
        )
    return "\n".join(rows)
