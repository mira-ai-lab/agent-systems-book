"""
依赖分析与执行排序示例 - Task Dependency Order Demo

子任务 + Agent 技能 I/O → System/User Prompt → LLM(JSON) → 执行顺序
"""

import json
import os
import httpx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# 上一步分解得到的子任务（示例写死，书稿可替换为模型真实输出）
SUBTASKS = {
    "T1": "根据北京市和[具体出行日期]查询天气预报。",
    "T2": "根据北京市推荐适合带老人和儿童入住的酒店及住宿。",
    "T3": "根据故宫、长城和适合孩子的科技类场馆周边地点推荐餐厅。",
    "T4": "根据[出发城市]、北京市、3天2晚、预算8000元以内、必去景点（故宫、长城、适合孩子的科技类场馆）、每日步行不超过8000步及雨天自动调整为室内项目的要求，规划涵盖交通方式、每日行程、预算估计和备选方案的综合旅行行程。"
}

# Agent 技能输入/输出（依赖分析用）
# 注：实际工程中可从 Agent Card JSON 解析得到，这里为演示简化
AGENT_SKILLS  = [
    {
        "name": "WeatherAgent",
        "description": "根据用户输入的城市和具体日期查询天气预报。",
        "skills": [
            {
                "id": "weather_search_001",
                "name": "get_weather",
                "description": "根据城市和日期查询天气预报",
                "input_parameters": ["city", "date"],
                "output_parameters": ["weather_summary", "temperature"],
                "examples": ["查询上海5月1日的天气", "北京明天天气怎么样"],
            }
        ],
    },
    {
        "name": "HotelAgent",
        "description": "根据用户输入的城市推荐酒店及住宿。",
        "skills": [
            {
                "id": "hotel_rec_001",
                "name": "recommend_hotels",
                "description": "基于城市和偏好推荐酒店住宿",
                "input_parameters": ["city", "preferences", "budget"],
                "output_parameters": ["hotel_list"],
                "examples": ["推荐上海外滩附近的酒店", "北京有哪些性价比高的住宿"],
            }
        ],
    },
    {
        "name": "RestaurantAgent",
        "description": "根据用户输入的地点推荐餐厅。",
        "skills": [
            {
                "id": "restaurant_rec_001",
                "name": "recommend_restaurants",
                "description": "基于地点和口味推荐餐厅",
                "input_parameters": ["location", "cuisine_type", "budget"],
                "output_parameters": ["restaurant_list"],
                "examples": ["推荐杭州西湖附近的本帮菜餐厅", "上海有哪些适合约会的餐厅"],
            }
        ],
    },
    {
        "name": "ItineraryAgent",
        "description": "根据用户提供的出发地、目的地、天数等信息规划综合旅行行程，涵盖交通方式和必去景点等方面。",
        "skills": [
            {
                "id": "itinerary_plan_001",
                "name": "plan_itinerary",
                "description": "综合出发地、目的地、天数等信息规划旅行行程",
                "input_parameters": ["departure_city", "destination", "days", "preferences"],
                "output_parameters": ["itinerary"],
                "examples": ["为我规划上海到杭州的3日游行程", "设计北京5日深度游"],
            }
        ],
    },
    {
        "name": "FlightAgent",
        "description": "根据具体日期、出发城市和到达城市查询航班信息。",
        "skills": [
            {
                "id": "flight_search_001",
                "name": "search_flights",
                "description": "查询指定日期和航线的航班信息",
                "input_parameters": ["departure_city", "arrival_city", "date"],
                "output_parameters": ["flight_list", "price_range"],
                "examples": ["查询5月1日北京到上海的航班", "杭州到广州的机票价格"],
            }
        ],
    },
    {
        "name": "NewsAgent",
        "description": "根据用户输入的新闻标题搜索相关新闻内容。",
        "skills": [
            {
                "id": "news_search_001",
                "name": "search_news",
                "description": "根据新闻标题或关键词搜索相关新闻",
                "input_parameters": ["headline", "keywords"],
                "output_parameters": ["news_articles"],
                "examples": ["查看最新的NBA新闻", "搜索人工智能相关政策新闻"],
            }
        ],
    },
]
DEPENDENCY_SYSTEM_PROMPT = """
你是一个多智能体系统中的[任务依赖分析器], 负责根据子任务与候选 Agent 技能之间的输入输出参数依赖关系, 识别子任务之间的执行依赖后调整子任务的执行顺序

# 输入
1. subtasks: 已拆分的子任务列表, 每个子任务包含 task_id、task_description
2. agents: 候选 Agent 列表, 每个技能包含 skill_name、input_parameters、output_parameters

# 目标
分析子任务之间的必要依赖关系后调整子任务的执行顺序。
- 若子任务 A 的输入可从子任务 B 的输出获得，则 A 依赖 B，B 先于 A 执行。
- 无依赖的子任务保持原有相对顺序，不要多余重排。

# 输出
仅输出合法 JSON，格式如下：
{"1": "task_id", "2": "task_id", ...}
order 从 1 递增；task_id 必须与输入中的子任务 id 一致。
"""

DEPENDENCY_USER_PROMPT = """
subtasks:
{subtasks}

agents:
{agents}

请严格按照输出格式返回, 除此之外不要有其他输出
"""

user_prompt = PromptTemplate.from_template(DEPENDENCY_USER_PROMPT).format(
    subtasks=SUBTASKS, agents=AGENT_SKILLS
)

print("=" * 70)
print("【System Prompt】\n")
print(DEPENDENCY_SYSTEM_PROMPT.strip())
print("\n" + "-" * 70)
print("【User Prompt】\n")
print(user_prompt)
print("\n" + "-" * 70)

llm = ChatOpenAI(
    model=os.getenv("DASHSCOPE_CHAT_MODEL", "qwen3.6-plus"),
    temperature=0,
    base_url=os.getenv(
        "DASHSCOPE_CHAT_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).rstrip("/"),
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    http_client=httpx.Client(verify=False),
    model_kwargs={"response_format": {"type": "json_object"}},
)

print("\n【模型输出：执行顺序 JSON】\n")
response = llm.invoke(
    [
        SystemMessage(content=DEPENDENCY_SYSTEM_PROMPT.strip()),
        HumanMessage(content=user_prompt),
    ]
)
print(response.content)

order = json.loads(response.content)
print("\n【最终执行计划】\n")
for i in range(1, len(SUBTASKS) + 1):
    tid = order.get(str(i))
    print(f"  步骤 {i}: {SUBTASKS[tid]}")

# ======================================================================
# 【System Prompt】
#
# 你是一个多智能体系统中的[任务依赖分析器], 负责根据子任务与候选 Agent 技能之间的输入输出参数依赖关系, 识别子任务之间的执行依赖后调整子任务的执行顺序
#
# # 输入
# 1. subtasks: 已拆分的子任务列表, 每个子任务包含 task_id、task_description
# 2. agents: 候选 Agent 列表, 每个技能包含 skill_name、input_parameters、output_parameters
#
# # 目标
# 分析子任务之间的必要依赖关系后调整子任务的执行顺序。
# - 若子任务 A 的输入可从子任务 B 的输出获得，则 A 依赖 B，B 先于 A 执行。
# - 无依赖的子任务保持原有相对顺序，不要多余重排。
#
# # 输出
# 仅输出合法 JSON，格式如下：
# {"1": "task_id", "2": "task_id", ...}
# order 从 1 递增；task_id 必须与输入中的子任务 id 一致。
#
# ----------------------------------------------------------------------
# 【User Prompt】
#
#
# subtasks:
# {'T1': '根据北京市和[具体出行日期]查询天气预报。', 'T2': '根据北京市推荐适合带老人和儿童入住的酒店及住宿。', 'T3': '根据故宫、长城和适合孩子的科技类场馆周边地点推荐餐厅。', 'T4': '根据[出发城市]、北京市、3天2晚、预算8000元以内、必去景点（故宫、长城、适合孩子的科技类场馆）、每日步行不超过8000步及雨天自动调整为室内项目的要求，规划涵盖交通方式、每日行程、预算估计和备选方案的综合旅行行程。'}
#
# agents:
# [{'name': 'WeatherAgent', 'description': '根据用户输入的城市和具体日期查询天气预报。', 'skills': [{'id': 'weather_search_001', 'name': 'get_weather', 'description': '根据城市和日期查询天气预报', 'input_parameters': ['city', 'date'], 'output_parameters': ['weather_summary', 'temperature'], 'examples': ['查询上海5月1日的天气', '北京明天天气怎么样']}]}, {'name': 'HotelAgent', 'description': '根据用户输入的城市推荐酒店及住宿。', 'skills': [{'id': 'hotel_rec_001', 'name': 'recommend_hotels', 'description': '基于城市和偏好推荐酒店住宿', 'input_parameters': ['city', 'preferences', 'budget'], 'output_parameters': ['hotel_list'], 'examples': ['推荐上海外滩附近的酒店', '北京有哪些性价比高的住宿']}]}, {'name': 'RestaurantAgent', 'description': '根据用户输入的地点推荐餐厅。', 'skills': [{'id': 'restaurant_rec_001', 'name': 'recommend_restaurants', 'description': '基于地点和口味推荐餐厅', 'input_parameters': ['location', 'cuisine_type', 'budget'], 'output_parameters': ['restaurant_list'], 'examples': ['推荐杭州西湖附近的本帮菜餐厅', '上海有哪些适合约会的餐厅']}]}, {'name': 'ItineraryAgent', 'description': '根据用户提供的出发地、目的地、天数等信息规划综合旅行行程，涵盖交通方式和必去景点等方面。', 'skills': [{'id': 'itinerary_plan_001', 'name': 'plan_itinerary', 'description': '综合出发地、目的地、天数等信息规划旅行行程', 'input_parameters': ['departure_city', 'destination', 'days', 'preferences'], 'output_parameters': ['itinerary'], 'examples': ['为我规划上海到杭州的3日游行程', '设计北京5日深度游']}]}, {'name': 'FlightAgent', 'description': '根据具体日期、出发城市和到达城市查询航班信息。', 'skills': [{'id': 'flight_search_001', 'name': 'search_flights', 'description': '查询指定日期和航线的航班信息', 'input_parameters': ['departure_city', 'arrival_city', 'date'], 'output_parameters': ['flight_list', 'price_range'], 'examples': ['查询5月1日北京到上海的航班', '杭州到广州的机票价格']}]}, {'name': 'NewsAgent', 'description': '根据用户输入的新闻标题搜索相关新闻内容。', 'skills': [{'id': 'news_search_001', 'name': 'search_news', 'description': '根据新闻标题或关键词搜索相关新闻', 'input_parameters': ['headline', 'keywords'], 'output_parameters': ['news_articles'], 'examples': ['查看最新的NBA新闻', '搜索人工智能相关政策新闻']}]}]
#
# 请严格按照输出格式返回, 除此之外不要有其他输出
#
#
# ----------------------------------------------------------------------
#
# 【模型输出：执行顺序 JSON】
#
# {"1": "T1", "2": "T2", "3": "T3", "4": "T4"}
#
# 【最终执行计划】
#
#   步骤 1: 根据北京市和[具体出行日期]查询天气预报。
#   步骤 2: 根据北京市推荐适合带老人和儿童入住的酒店及住宿。
#   步骤 3: 根据故宫、长城和适合孩子的科技类场馆周边地点推荐餐厅。
#   步骤 4: 根据[出发城市]、北京市、3天2晚、预算8000元以内、必去景点（故宫、长城、适合孩子的科技类场馆）、每日步行不超过8000步及雨天自动调整为室内项目的要求，规划涵盖交通方式、每日行程、预算估计和备选方案的综合旅行行程。

