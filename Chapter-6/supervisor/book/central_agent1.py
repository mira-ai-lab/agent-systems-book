"""
书籍案例 — 旅行多智能体中心调度演示（离线可跑，无需 API）

流程：拆任务 → 选路线（Supervisor / 规划流水线）→ 派给子 Agent → 汇总

运行：
  cd Chapter-6/supervisor/book
  python central_agent1.py
  python central_agent1.py -q "成都今天天气怎么样？"
  python central_agent1.py --chat

与真实代码对照（本书案例 → Chapter-6 可运行项目）：

  ┌────────────────────────────┬─────────────────────────────────────────────┐
  │ 本书 central_agent1.py     │ Chapter-6 真实代码                          │
  ├────────────────────────────┼─────────────────────────────────────────────┤
  │ TravelCentralAgent         │ supervisor/local_supervisor.py              │
  │ classify_route             │ local_supervisor._dispatch_query            │
  │ TaskPlannerDemo            │ task_planner.py → TaskPlanner（LLM）        │
  │ PlannedPipelineDemo        │ supervisor/planned_pipeline.py              │
  │ SupervisorDemo             │ local_supervisor + create_supervisor        │
  │ SubAgentDemo               │ sub_agents.py → SubAgentFactory             │
  │ SubTask / ExecutionPlan    │ planned_pipeline 内 execution_plan 结构     │
  │ agent_definitions          │ book/agent_definitions.py（设计说明书）     │
  │ TripInfo 规则解析          │ TaskPlanner + LLM 完成                      │
  │ fixed_graph 固定图版       │ fixed_graph/orchestrator.py                 │
  └────────────────────────────┴─────────────────────────────────────────────┘

子 Agent 岗位说明书：book/agent_definitions.py
完整 Supervisor 入口：cd Chapter-6 && python -m supervisor.local_supervisor
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_BOOK_DIR = Path(__file__).resolve().parent
if str(_BOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_BOOK_DIR))

from agent_definitions import ALL_AGENTS, SUPERVISOR_NODE_SPECS  # noqa: E402

CITIES = ("上海", "北京", "成都", "杭州", "苏州", "广州", "深圳", "重庆", "西安", "南京")


# ── 数据结构 ────────────────────────────────────────────────────────────────
# 真实项目：planned_pipeline / fixed_graph 中 execution_plan 的 subtasks 字段结构相同

class RouteMode(str, Enum):
    SUPERVISOR = "supervisor"  # 单任务：经理直接派活
    PLANNED = "planned"        # 多任务：按清单流水线执行


@dataclass
class SubTask:
    task_id: str
    description: str
    agent: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    user_goal: str
    subtasks: List[SubTask]
    execution_order: List[str] = field(default_factory=list)


@dataclass
class SubTaskResult:
    task_id: str
    agent: str
    answer: str


@dataclass
class TripInfo:
    """从用户话里粗提取的行程信息（演示用规则，真实项目由 LLM 完成）。"""

    query: str
    from_city: str
    to_city: str
    cities: List[str]
    days: int
    date_labels: List[str]
    date_params: List[str]


# ── 子 Agent 演示（固定假数据）──────────────────────────────────────────────
# 真实代码：Chapter-6/sub_agents.py → SubAgentFactory.get_agent()

class SubAgentDemo:
    REPLIES = {
        "WeatherAgent": "【天气科】晴，28°C，适合出行。",
        "FlightAgent": "【机票科】MU5401 08:00-11:20，约 680 元。",
        "AttractionAgent": "【景点科】武侯祠、大熊猫基地、宽窄巷子。",
        "RestaurantAgent": "【美食科】陈麻婆豆腐、蜀大侠火锅、钟水饺。",
        "HotelAgent": "【酒店科】XX 酒店 399元/晚、YY 酒店 450元/晚。",
        "ItineraryAgent": "【行程科】Day1 市区+锦里；Day2 熊猫基地；Day3 青城山。",
    }

    def run(self, agent: str, instruction: str) -> str:
        preview = instruction.replace("\n", " ")[:100]
        print(f"      └─ [{agent}] {preview}…")
        return self.REPLIES.get(agent, f"【{agent}】已处理")


# ── 任务规划（规则模拟 LLM 拆解）──────────────────────────────────────────
# 真实代码：Chapter-6/task_planner.py → TaskPlanner（pre_survey + build_execution_plan）
# 本书用关键词+模板模拟 LLM 输出的 description / params / depends_on

class TaskPlannerDemo:
    """关键词匹配需要哪些 Agent，再为每个 Agent 生成明确的 description + params。"""

    KEYWORDS: Dict[str, List[str]] = {
        "WeatherAgent": ["天气", "气温", "下雨"],
        "FlightAgent": ["机票", "航班", "飞机"],
        "AttractionAgent": ["景点", "打卡", "景区"],
        "RestaurantAgent": ["美食", "餐厅", "吃"],
        "HotelAgent": ["酒店", "住宿", "民宿"],
        "ItineraryAgent": ["行程", "规划", "安排"],
    }

    def build_plan(self, user_query: str) -> ExecutionPlan:
        print("\n  📋 TaskPlanner：正在拆任务…")
        trip = self._parse_trip(user_query)

        subtasks: List[SubTask] = []
        for agent, words in self.KEYWORDS.items():
            if any(w in user_query for w in words):
                subtasks.append(self._make_task(f"T{len(subtasks) + 1}", agent, trip))

        if not subtasks:
            subtasks = [self._make_task("T1", "ItineraryAgent", trip)]

        # 行程科依赖前面所有子任务
        for t in subtasks:
            if t.agent == "ItineraryAgent" and len(subtasks) > 1:
                t.depends_on = [x.task_id for x in subtasks if x.agent != "ItineraryAgent"]

        plan = ExecutionPlan(
            user_goal=user_query,
            subtasks=subtasks,
            execution_order=[t.task_id for t in subtasks],
        )
        print(f"  ✓ 拆出 {len(subtasks)} 项：", " → ".join(t.agent for t in subtasks))
        print(json.dumps(self._to_json(plan), ensure_ascii=False, indent=2))
        return plan

    def _parse_trip(self, q: str) -> TripInfo:
        """粗解析：出发地、目的地、天数、日期列表。"""
        cities_in_q = [c for c in CITIES if c in q]
        from_city, to_city = "", cities_in_q[0] if cities_in_q else "目的地"

        m = re.search(r"从(.+?)去(.+?)(?:玩|旅游|，|。|$)", q)
        if m:
            from_city = m.group(1).strip()
            to_city = re.sub(r"玩\d+天.*", "", m.group(2)).strip()

        days_m = re.search(r"玩(\d+)天", q)
        days = int(days_m.group(1)) if days_m else (1 if "天气" in q else 3)

        today = date.today()
        start = today + timedelta(days=7) if "下周" in q else today
        date_labels, date_params = [], []
        for i in range(days):
            d = start + timedelta(days=i)
            date_labels.append(f"{d.year}年{d.month}月{d.day}日")
            date_params.append(d.strftime("%Y-%m-%d"))

        all_cities = cities_in_q if len(cities_in_q) >= 2 else [to_city]
        return TripInfo(q, from_city, to_city, all_cities, days, date_labels, date_params)

    def _make_task(self, task_id: str, agent: str, trip: TripInfo) -> SubTask:
        """按 Agent 类型生成明确子任务（对齐 execution_plan.subtasks 结构）。"""
        cities_label = "、".join(trip.cities) if len(trip.cities) > 1 else trip.to_city
        dates_label = "、".join(trip.date_labels)
        d0, d_last = trip.date_params[0], trip.date_params[-1]

        templates: Dict[str, Tuple[str, Dict[str, Any]]] = {
            "WeatherAgent": (
                f"WeatherAgent：查询{dates_label}{cities_label}逐日天气预报"
                "（最高温、最低温、降水概率、出行建议）",
                {"cities": trip.cities, "dates": trip.date_params, "city": trip.cities[0], "date": d0},
            ),
            "FlightAgent": (
                f"FlightAgent：查询{trip.from_city}→{trip.to_city}{trip.date_labels[0]}前后直飞航班"
                "（航班号、时间、票价，推荐 2–3 班）",
                {"departure": trip.from_city, "arrival": trip.to_city, "date": d0},
            ),
            "AttractionAgent": (
                f"AttractionAgent：为{trip.to_city}推荐{trip.days}天内 3–5 个代表性景点"
                "（类型、停留时长、门票、交通方式）",
                {"city": trip.to_city, "limit": 5},
            ),
            "RestaurantAgent": (
                f"RestaurantAgent：为{cities_label}推荐本地特色美食及餐厅"
                "（菜系、必点菜、人均、食客评价）",
                {"location": trip.cities[0], "cuisine": "本地菜", "budget_cny_per_person": 150},
            ),
            "HotelAgent": (
                f"HotelAgent：为{cities_label}推荐{d0}至{d_last}可订酒店"
                "（≤800元/晚、安静交通便利、含评价摘要）",
                {"city": trip.cities[0], "budget_cny_per_night_max": 800},
            ),
            "ItineraryAgent": (
                f"ItineraryAgent：基于{dates_label}、{trip.days}天、"
                f"{trip.from_city}→{trip.to_city}生成逐日行程"
                "（交通、景点顺序、餐饮酒店衔接、天气适配）",
                {
                    "departure_city": trip.from_city,
                    "destination_city": trip.to_city,
                    "days": trip.days,
                    "preferences": "避开人流高峰,交通顺畅",
                },
            ),
        }
        desc, params = templates.get(
            agent,
            (f"{agent}：处理与{trip.to_city}相关的专项任务", {}),
        )
        return SubTask(task_id=task_id, description=desc, agent=agent, params=params)

    @staticmethod
    def _to_json(plan: ExecutionPlan) -> Dict[str, Any]:
        return {
            "subtasks": [
                {
                    "task_id": t.task_id,
                    "description": t.description,
                    "agent": t.agent,
                    "params": t.params,
                    "depends_on": t.depends_on,
                }
                for t in plan.subtasks
            ],
            "execution_order": plan.execution_order,
        }


# ── 路由与执行 ──────────────────────────────────────────────────────────────
# classify_route：supervisor/local_supervisor.py → _dispatch_query（子任务>1 走流水线）
# PlannedPipelineDemo：supervisor/planned_pipeline.py → PlannedPipeline
# SupervisorDemo：supervisor/local_supervisor.py → create_supervisor + handoff
# build_instruction：fixed_graph/nodes.py → _invoke_sub_agent（拼装 description + params + 依赖结果）

def classify_route(plan: ExecutionPlan) -> RouteMode:
    return RouteMode.PLANNED if len(plan.subtasks) > 1 else RouteMode.SUPERVISOR


def build_instruction(task: SubTask, prior: Dict[str, SubTaskResult]) -> str:
    parts = [task.description]
    if task.params:
        parts.append(f"参数: {json.dumps(task.params, ensure_ascii=False)}")
    for dep_id in task.depends_on:
        if dep_id in prior:
            parts.append(f"参考 {dep_id}：{prior[dep_id].answer}")
    return "\n".join(parts)


class PlannedPipelineDemo:
    def __init__(self) -> None:
        self.planner = TaskPlannerDemo()
        self.workers = SubAgentDemo()

    def classify_and_plan(self, user_query: str) -> Tuple[RouteMode, ExecutionPlan]:
        plan = self.planner.build_plan(user_query)
        return classify_route(plan), plan

    def run(self, user_query: str, plan: ExecutionPlan) -> str:
        print("\n  🔀 进入【规划流水线】模式")
        results: List[SubTaskResult] = []
        done: Dict[str, SubTaskResult] = {}

        for task in plan.subtasks:
            print(f"\n  ▶ 执行 {task.task_id} → {task.agent}")
            answer = self.workers.run(task.agent, build_instruction(task, done))
            res = SubTaskResult(task.task_id, task.agent, answer)
            results.append(res)
            done[task.task_id] = res
            print(f"  ✓ {task.task_id} 完成")

        lines = [f"您好！针对「{user_query}」，整理如下：\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.answer}")
        lines.append("\n（以上为书籍演示数据。）")
        return "\n".join(lines)


class SupervisorDemo:
    def __init__(self) -> None:
        self.workers = SubAgentDemo()

    def run(self, user_query: str, plan: ExecutionPlan) -> str:
        print("\n  🔀 进入【Supervisor 派活】模式")
        task = plan.subtasks[0]
        node = next(
            (n for n, factory, _ in SUPERVISOR_NODE_SPECS if factory == task.agent),
            task.agent,
        )
        print(f"\n  ▶ Supervisor 派活 → {node}（{task.agent}）")
        answer = self.workers.run(task.agent, build_instruction(task, {}))
        return f"{answer}\n\n（Supervisor 模式：只派了一个专员。）"


class TravelCentralAgent:
    """中心智能体：拆任务 → 选路线 → 执行 → 返回最终回复。

    真实入口：supervisor/local_supervisor.py（Supervisor + 规划流水线双模式）
              fixed_graph/orchestrator.py（LangGraph 固定图版）
    """

    def __init__(self) -> None:
        self.pipeline = PlannedPipelineDemo()
        self.supervisor = SupervisorDemo()

    def handle(self, user_query: str) -> str:
        print("\n" + "=" * 60)
        print(f"📥 用户：{user_query}")
        print("=" * 60)

        route, plan = self.pipeline.classify_and_plan(user_query)
        if route is RouteMode.PLANNED:
            final = self.pipeline.run(user_query, plan)
            chain = " → ".join(t.agent for t in plan.subtasks)
            print(f"\n【调度链】规划流水线 → {chain}")
        else:
            final = self.supervisor.run(user_query, plan)
            print(f"\n【调度链】Supervisor → {plan.subtasks[0].agent}")
        return final


# ── 主程序 ──────────────────────────────────────────────────────────────────
# 对应 Chapter-6/supervisor/local_supervisor.py → main / run_interactive

DEMO_QUERIES = [
    "成都今天天气怎么样？",
    "我下周从上海去成都玩3天，帮我查天气、订机票、推荐景点和美食，最后出个行程",
]


def run_builtin_demos() -> None:
    central = TravelCentralAgent()
    for q in DEMO_QUERIES:
        print(central.handle(q))
        print("\n" + "-" * 60)
        if sys.stdin.isatty():
            input("按回车继续…")


def run_single_query(query: str) -> None:
    print(TravelCentralAgent().handle(query))


def run_chat() -> None:
    print("旅行中心智能体演示 · 输入 quit 退出")
    central = TravelCentralAgent()
    while True:
        try:
            q = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("quit", "exit", "q", "退出"):
            break
        print(central.handle(q))


def main() -> None:
    parser = argparse.ArgumentParser(description="旅行中心智能体书籍演示")
    parser.add_argument("-q", "--query", help="单条问题")
    parser.add_argument("--chat", action="store_true", help="对话模式")
    parser.add_argument("--list-agents", action="store_true", help="列出子 Agent")
    args = parser.parse_args()

    if args.list_agents:
        for name, spec in ALL_AGENTS.items():
            print(f"  {spec.title:6}  {name:18}  {spec.node_name}")
        return
    if args.query:
        run_single_query(args.query)
    elif args.chat:
        run_chat()
    else:
        run_builtin_demos()


if __name__ == "__main__":
    main()



#
# 下面是代码输出
# ============================================================
# 📥 用户：成都今天天气怎么样？
# ============================================================
#
#   📋 TaskPlanner：正在拆任务…
#   ✓ 拆出 1 项： WeatherAgent
# {
#   "subtasks": [
#     {
#       "task_id": "T1",
#       "description": "WeatherAgent：查询2026年6月11日成都逐日天气预报（最高温、最低温、降水概率、出行建议）",
#       "agent": "WeatherAgent",
#       "params": {
#         "cities": [
#           "成都"
#         ],
#         "dates": [
#           "2026-06-11"
#         ],
#         "city": "成都",
#         "date": "2026-06-11"
#       },
#       "depends_on": []
#     }
#   ],
#   "execution_order": [
#     "T1"
#   ]
# }
#
#   🔀 进入【Supervisor 派活】模式
#
#   ▶ Supervisor 派活 → weather_agent（WeatherAgent）
#       └─ [WeatherAgent] WeatherAgent：查询2026年6月11日成都逐日天气预报（最高温、最低温、降水概率、出行建议） 参数: {"cities": ["成都"], "dates": ["2026-06-11"],…
#
# 【调度链】Supervisor → WeatherAgent
# 【天气科】晴，28°C，适合出行。
#
# （Supervisor 模式：只派了一个专员。）
#
# ------------------------------------------------------------
#
# ============================================================
# 📥 用户：我下周从上海去成都玩3天，帮我查天气、订机票、推荐景点和美食，最后出个行程
# ============================================================
#
#   📋 TaskPlanner：正在拆任务…
#   ✓ 拆出 5 项： WeatherAgent → FlightAgent → AttractionAgent → RestaurantAgent → ItineraryAgent
# {
#   "subtasks": [
#     {
#       "task_id": "T1",
#       "description": "WeatherAgent：查询2026年6月18日、2026年6月19日、2026年6月20日上海、成都逐日天气预报（最高温、最低温、降水概率、出行建议）",
#       "agent": "WeatherAgent",
#       "params": {
#         "cities": [
#           "上海",
#           "成都"
#         ],
#         "dates": [
#           "2026-06-18",
#           "2026-06-19",
#           "2026-06-20"
#         ],
#         "city": "上海",
#         "date": "2026-06-18"
#       },
#       "depends_on": []
#     },
#     {
#       "task_id": "T2",
#       "description": "FlightAgent：查询上海→成都2026年6月18日前后直飞航班（航班号、时间、票价，推荐 2–3 班）",
#       "agent": "FlightAgent",
#       "params": {
#         "departure": "上海",
#         "arrival": "成都",
#         "date": "2026-06-18"
#       },
#       "depends_on": []
#     },
#     {
#       "task_id": "T3",
#       "description": "AttractionAgent：为成都推荐3天内 3–5 个代表性景点（类型、停留时长、门票、交通方式）",
#       "agent": "AttractionAgent",
#       "params": {
#         "city": "成都",
#         "limit": 5
#       },
#       "depends_on": []
#     },
#     {
#       "task_id": "T4",
#       "description": "RestaurantAgent：为上海、成都推荐本地特色美食及餐厅（菜系、必点菜、人均、食客评价）",
#       "agent": "RestaurantAgent",
#       "params": {
#         "location": "上海",
#         "cuisine": "本地菜",
#         "budget_cny_per_person": 150
#       },
#       "depends_on": []
#     },
#     {
#       "task_id": "T5",
#       "description": "ItineraryAgent：基于2026年6月18日、2026年6月19日、2026年6月20日、3天、上海→成都生成逐日行程（交通、景点顺序、餐饮酒店衔接、天气适配）",
#       "agent": "ItineraryAgent",
#       "params": {
#         "departure_city": "上海",
#         "destination_city": "成都",
#         "days": 3,
#         "preferences": "避开人流高峰,交通顺畅"
#       },
#       "depends_on": [
#         "T1",
#         "T2",
#         "T3",
#         "T4"
#       ]
#     }
#   ],
#   "execution_order": [
#     "T1",
#     "T2",
#     "T3",
#     "T4",
#     "T5"
#   ]
# }
#
#   🔀 进入【规划流水线】模式
#
#   ▶ 执行 T1 → WeatherAgent
#       └─ [WeatherAgent] WeatherAgent：查询2026年6月18日、2026年6月19日、2026年6月20日上海、成都逐日天气预报（最高温、最低温、降水概率、出行建议） 参数: {"cities": ["上海", …
#   ✓ T1 完成
#
#   ▶ 执行 T2 → FlightAgent
#       └─ [FlightAgent] FlightAgent：查询上海→成都2026年6月18日前后直飞航班（航班号、时间、票价，推荐 2–3 班） 参数: {"departure": "上海", "arrival": "成都", "da…
#   ✓ T2 完成
#
#   ▶ 执行 T3 → AttractionAgent
#       └─ [AttractionAgent] AttractionAgent：为成都推荐3天内 3–5 个代表性景点（类型、停留时长、门票、交通方式） 参数: {"city": "成都", "limit": 5}…
#   ✓ T3 完成
#
#   ▶ 执行 T4 → RestaurantAgent
#       └─ [RestaurantAgent] RestaurantAgent：为上海、成都推荐本地特色美食及餐厅（菜系、必点菜、人均、食客评价） 参数: {"location": "上海", "cuisine": "本地菜", "budget_c…
#   ✓ T4 完成
#
#   ▶ 执行 T5 → ItineraryAgent
#       └─ [ItineraryAgent] ItineraryAgent：基于2026年6月18日、2026年6月19日、2026年6月20日、3天、上海→成都生成逐日行程（交通、景点顺序、餐饮酒店衔接、天气适配） 参数: {"departur…
#   ✓ T5 完成
#
# 【调度链】规划流水线 → WeatherAgent → FlightAgent → AttractionAgent → RestaurantAgent → ItineraryAgent
# 您好！针对「我下周从上海去成都玩3天，帮我查天气、订机票、推荐景点和美食，最后出个行程」，整理如下：
#
# 1. 【天气科】晴，28°C，适合出行。
# 2. 【机票科】MU5401 08:00-11:20，约 680 元。
# 3. 【景点科】武侯祠、大熊猫基地、宽窄巷子。
# 4. 【美食科】陈麻婆豆腐、蜀大侠火锅、钟水饺。
# 5. 【行程科】Day1 市区+锦里；Day2 熊猫基地；Day3 青城山。
#
# （以上为书籍演示数据。）

