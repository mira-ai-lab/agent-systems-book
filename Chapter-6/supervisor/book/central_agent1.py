"""
================================================================================
  书籍案例  — 旅行多智能体「中心调度」伪代码（可运行演示版）
================================================================================

【这个文件是干什么的？】
    - 中心智能体怎么接用户问题
    - 怎么拆任务、选路线
    - 怎么派给 6 个子智能体
    - 怎么汇总成最终回复

【和真实代码的对应关系】
  ┌─────────────────────────────┬──────────────────────────────────┐
  │ 真实项目（要联网/API）       │ 本书案例（本文件，离线可跑）      │
  ├─────────────────────────────┼──────────────────────────────────┤
  │ local_supervisor.py         │ central_agent1.py                │
  │ planned_pipeline.py         │ PlannedPipelineDemo              │
  │ create_supervisor + handoff │ SupervisorDemo                   │
  │ TaskPlanner + 大模型        │ TaskPlannerDemo（规则模拟）       │
  │ sub_agents.py + 地图 API    │ SubAgentDemo（固定示例回复）      │
  └─────────────────────────────┴──────────────────────────────────┘

【怎么运行】
  cd Chapter-6/supervisor/book
  python central_agent1.py                    # 跑 2 个内置例子
  python central_agent1.py -q "北京今天天气"     # 单条问题
  python central_agent1.py --chat             # 简单对话（输入 quit 退出）

【生活比喻】
  用户 = 顾客
  中心智能体 = 旅行社前台经理
  子智能体 = 天气科 / 机票科 / 景点科 …（见 agent_definitions.py）


【流程图2】
    用户提问
        │
        ▼
    ┌─────────────────┐
    │  TaskPlanner    │  pre_survey + build_plan
    │  classify_route │
    └────────┬────────┘
             │
     ┌───────┴───────┐
     │ subtasks > 1? │
     └───────┬───────┘
          yes│    no
             │     └──────────────────┐
             ▼                        ▼
    ┌─────────────────┐    ┌─────────────────────┐
    │ execute_layer   │    │ Supervisor + handoff │
    │ (按依赖分层)     │    │ (动态选 1 个子 Agent) │
    └────────┬────────┘    └──────────┬──────────┘
             │                        │
             └──────────┬─────────────┘
                        ▼
                   聚合 / 最终回复

"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# 同目录下的 agent_definitions.py：六个专员的「岗位说明书」
_BOOK_DIR = Path(__file__).resolve().parent
if str(_BOOK_DIR) not in sys.path:
    sys.path.insert(0, str(_BOOK_DIR))

from agent_definitions import ALL_AGENTS, SUPERVISOR_NODE_SPECS  # noqa: E402


# =============================================================================
# 第 0 部分：共用数据结构（真实项目里也有类似结构）
# =============================================================================

class RouteMode(str, Enum):
    """走哪条路：简单派活 vs 按清单流水线。"""

    SUPERVISOR = "supervisor"   # 对应 local_supervisor 里的 Supervisor handoff
    PLANNED = "planned"         # 对应 planned_pipeline.py


@dataclass
class SubTask:
    """待办清单里的一行。"""

    task_id: str           # 如 T1, T2
    description: str       # 自然语言说明要做什么
    agent: str             # 交给哪个专员，如 WeatherAgent
    depends_on: List[str] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """TaskPlanner 拆完任务后的整张清单。"""

    user_goal: str
    subtasks: List[SubTask]


@dataclass
class SubTaskResult:
    """某个专员做完一件事的结果。"""

    task_id: str
    agent: str
    answer: str


# =============================================================================
# 第 1 部分：六个子智能体（伪实现）
# 真实代码：sub_agents.py + SubAgentFactory.get_agent(...)
# =============================================================================

class SubAgentDemo:
    """
    六个「专员」的演示版。

    真实项目里每个专员 = 大模型 + 工具（查天气 API、查酒店 API…）
    这里用固定文字模拟，方便没网也能上课演示。
    """

    def run(self, agent_name: str, instruction: str) -> str:
        """专员收到任务 instruction，返回一段文字结果。"""
        print(f"      └─ [{agent_name}] 收到任务：{instruction[:50]}…")

        # 下面都是「假数据」，仅用于演示流程
        if agent_name == "WeatherAgent":
            return "【天气科】成都 6月8日：晴，28°C，适合出行。"
        if agent_name == "FlightAgent":
            return "【机票科】上海→成都 6月8日：MU5401 08:00-11:20，约 680 元。"
        if agent_name == "AttractionAgent":
            return "【景点科】推荐：武侯祠、大熊猫基地、宽窄巷子（共 3 处）。"
        if agent_name == "RestaurantAgent":
            return "【美食科】推荐：陈麻婆豆腐、蜀大侠火锅、钟水饺（共 3 家）。"
        if agent_name == "HotelAgent":
            return "【酒店科】推荐：XX 酒店 399元/晚、YY 酒店 450元/晚（共 2 家）。"
        if agent_name == "ItineraryAgent":
            return "【行程科】Day1 市区+锦里；Day2 熊猫基地；Day3 青城山。"
        return f"【{agent_name}】（演示）已处理：{instruction[:30]}…"


# =============================================================================
# 第 2 部分：TaskPlanner — 把用户大问题拆成待办清单
# 真实代码：task_planner.py + 大模型
# =============================================================================

class TaskPlannerDemo:
    """
    任务规划器（演示版）。

    真实项目：pre_survey → 拆解 → 分析依赖 → 路由到 Agent（都靠大模型）
    演示版：用关键词判断用户问了哪几类事，拆成 T1、T2…
    """

    # 用户话里出现这些词，就认为需要对应专员
    KEYWORDS: Dict[str, List[str]] = {
        "WeatherAgent": ["天气", "气温", "下雨"],
        "FlightAgent": ["机票", "航班", "飞机"],
        "AttractionAgent": ["景点", "玩", "打卡", "旅游"],
        "RestaurantAgent": ["美食", "餐厅", "吃"],
        "HotelAgent": ["酒店", "住宿", "住"],
        "ItineraryAgent": ["行程", "规划", "安排"],
    }

    def build_plan(self, user_query: str) -> ExecutionPlan:
        print("\n  📋 TaskPlanner：正在拆任务…")

        matched: List[SubTask] = []
        for agent, words in self.KEYWORDS.items():
            if any(w in user_query for w in words):
                tid = f"T{len(matched) + 1}"
                matched.append(
                    SubTask(
                        task_id=tid,
                        description=f"处理与用户「{user_query[:20]}…」相关的{agent}任务",
                        agent=agent,
                        depends_on=[],
                    )
                )

        # 如果用户明确要「行程」，且前面已有别的任务，让行程依赖它们
        for task in matched:
            if task.agent == "ItineraryAgent" and len(matched) > 1:
                task.depends_on = [t.task_id for t in matched if t.agent != "ItineraryAgent"]

        if not matched:
            # 没匹配到关键词时，默认当作「只问一件事」，交给行程科兜底理解
            matched = [
                SubTask(
                    task_id="T1",
                    description=user_query,
                    agent="ItineraryAgent",
                )
            ]

        print(f"  ✓ 拆出 {len(matched)} 项：", " → ".join(t.agent for t in matched))
        return ExecutionPlan(user_goal=user_query, subtasks=matched)


# =============================================================================
# 第 3 部分：路由 — local_supervisor._dispatch_query 的第一段逻辑
# =============================================================================

def classify_route(plan: ExecutionPlan) -> RouteMode:
    """
    根据待办数量选路线（与 local_supervisor / planned_pipeline 相同规则）。

    记住一句：
      待办 > 1  → 规划流水线
      待办 ≤ 1  → Supervisor 现场派活
    """
    n = len(plan.subtasks)
    if n > 1:
        return RouteMode.PLANNED
    return RouteMode.SUPERVISOR


# =============================================================================
# 第 4 部分：规划流水线 — 对应 planned_pipeline.py
# =============================================================================

class PlannedPipelineDemo:
    """
    复杂问题走这条路：按清单逐项执行 → 汇总。

    真实步骤：pre_survey → build_plan → execute_layer（分层）→ aggregate
    演示版：按顺序调用 SubAgentDemo，最后拼成一段文字。
    """

    def __init__(self) -> None:
        self.planner = TaskPlannerDemo()
        self.workers = SubAgentDemo()

    def classify_and_plan(self, user_query: str) -> tuple[RouteMode, ExecutionPlan]:
        plan = self.planner.build_plan(user_query)
        return classify_route(plan), plan

    def run(self, user_query: str, plan: ExecutionPlan) -> str:
        print("\n  🔀 进入【规划流水线】模式")
        results: List[SubTaskResult] = []
        done: Dict[str, SubTaskResult] = {}

        # 演示版：简单按 T1→T2 顺序做（真实项目会按 depends_on 分层，同层可并行）
        for task in plan.subtasks:
            # 若有依赖，先把依赖结果塞进指令里（真实项目也会这样做）
            extra = ""
            for dep_id in task.depends_on:
                if dep_id in done:
                    extra += f"\n（参考 {dep_id}：{done[dep_id].answer}）"
            instruction = task.description + extra

            print(f"\n  ▶ 执行 {task.task_id} → {task.agent}")
            answer = self.workers.run(task.agent, instruction)
            res = SubTaskResult(task.task_id, task.agent, answer)
            results.append(res)
            done[task.task_id] = res
            print(f"  ✓ {task.task_id} 完成")

        print("\n  📝 汇总各科结果…")
        return self._aggregate(user_query, results)

    def _aggregate(self, user_query: str, results: List[SubTaskResult]) -> str:
        """把多条专员回复合成给用户的一段话（真实项目用大模型 + 聚合 Prompt）。"""
        lines = [f"您好！针对「{user_query}」，整理如下：\n"]
        for i, r in enumerate(results, start=1):
            lines.append(f"{i}. {r.answer}")
        lines.append("\n（以上为书籍演示数据，真实项目会调用在线 API。）")
        return "\n".join(lines)


# =============================================================================
# 第 5 部分：Supervisor 模式 — 对应 create_supervisor + handoff
# =============================================================================

class SupervisorDemo:
    """
    简单问题走这条路：经理选一个专员 → 收结果 → 回复用户。

    真实项目：langgraph_supervisor.create_supervisor，经理是大模型，用 handoff 工具派活。
    演示版：用关键词决定派谁（相当于「最笨的经理」，但能看懂流程）。
    """

    def __init__(self) -> None:
        self.workers = SubAgentDemo()

    def pick_agent(self, user_query: str) -> str:
        """经理决定：这个问题交给谁？（演示版用关键词）。"""
        for agent, words in TaskPlannerDemo.KEYWORDS.items():
            if any(w in user_query for w in words):
                return agent
        return "ItineraryAgent"

    def run(self, user_query: str) -> str:
        print("\n  🔀 进入【Supervisor 派活】模式")

        agent = self.pick_agent(user_query)
        # 找到 handoff 节点名（真实项目里是 weather_agent 这种名字）
        node_name = next(
            (node for node, factory, _ in SUPERVISOR_NODE_SPECS if factory == agent),
            agent,
        )

        print(f"\n  ▶ Supervisor 派活 → {node_name}（{agent}）")
        answer = self.workers.run(agent, user_query)
        print(f"  ✓ handoff 完成，经理整理回复")

        # 真实项目里经理可能再改写成更口语的话；演示版直接返回
        return f"{answer}\n\n（Supervisor 模式演示：只派了一个专员。）"


# =============================================================================
# 第 6 部分：中心调度入口 — 对应 local_supervisor._dispatch_query
# =============================================================================

class TravelCentralAgent:
    """
    ★ 全书主角：旅行中心智能体 ★

    它本身不查天气、不卖机票，只做三件事：
      1. 让 TaskPlanner 拆任务
      2. 决定走 Supervisor 还是规划流水线
      3. 把最终文字交给用户
    """

    def __init__(self) -> None:
        self.pipeline = PlannedPipelineDemo()
        self.supervisor = SupervisorDemo()

    def handle(self, user_query: str) -> str:
        """处理一条用户问题 — 对应 local_supervisor._dispatch_query。"""
        print("\n" + "=" * 60)
        print(f"📥 用户：{user_query}")
        print("=" * 60)

        # 第 1 步：拆任务（真实项目里这步总是要做的，用来决定路由）
        route, plan = self.pipeline.classify_and_plan(user_query)

        # 第 2 步：按路线执行
        if route is RouteMode.PLANNED:
            final = self.pipeline.run(user_query, plan)
            print("\n【调度链】规划流水线 → " + " → ".join(t.agent for t in plan.subtasks))
        else:
            final = self.supervisor.run(user_query)
            print("\n【调度链】Supervisor → " + plan.subtasks[0].agent)

        return final


# =============================================================================
# 第 7 部分：主程序 — 对应 local_supervisor.main / run_interactive
# =============================================================================

DEMO_QUERIES = [
    "成都今天天气怎么样？",  # 期望：1 项 → Supervisor
    "我下周从上海去成都玩3天，帮我查天气、订机票、推荐景点和美食，最后出个行程",  # 期望：多项 → 规划流水线
]


def run_builtin_demos() -> None:
    """跑两个内置例子，适合课堂一键演示。"""
    central = TravelCentralAgent()
    for q in DEMO_QUERIES:
        answer = central.handle(q)
        print("\n" + "-" * 60)
        print("Assistant：")
        print(answer)
        print("-" * 60)
        if sys.stdin.isatty():
            input("\n按回车继续下一个例子…")


def run_single_query(query: str) -> None:
    central = TravelCentralAgent()
    answer = central.handle(query)
    print("\n" + "-" * 60)
    print("Assistant：")
    print(answer)
    print("-" * 60)


def run_chat() -> None:
    """简单对话循环，对应 local_supervisor.run_interactive。"""
    print("=" * 60)
    print("旅行中心智能体 · 书籍演示版（伪代码，无需 API）")
    print("输入问题回车；输入 quit 退出")
    print("=" * 60)
    central = TravelCentralAgent()
    while True:
        try:
            q = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not q or q.lower() in ("quit", "exit", "q", "退出"):
            print("再见。")
            break
        try:
            answer = central.handle(q)
            print("\nAssistant：")
            print(answer)
        except Exception as e:
            print(f"\n出错：{e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="书籍案例：旅行中心智能体伪代码演示（central_agent1.py）",
    )
    parser.add_argument("-q", "--query", help="只跑一条问题")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="进入对话模式",
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="只打印六个专员名单",
    )
    args = parser.parse_args()

    if args.list_agents:
        print("六个子智能体（详见 agent_definitions.py）：\n")
        for name, spec in ALL_AGENTS.items():
            print(f"  · {spec.title:6}  {name:18}  handoff节点={spec.node_name}")
        return

    if args.query:
        run_single_query(args.query)
    elif args.chat:
        run_chat()
    else:
        run_builtin_demos()


if __name__ == "__main__":
    main()



# ============================================================
# 📥 用户：成都今天天气怎么样？
# ============================================================
#
#   📋 TaskPlanner：正在拆任务…
#   ✓ 拆出 1 项： WeatherAgent
#
#   🔀 进入【Supervisor 派活】模式
#
#   ▶ Supervisor 派活 → weather_agent（WeatherAgent）
#       └─ [WeatherAgent] 收到任务：成都今天天气怎么样？…
#   ✓ handoff 完成，经理整理回复
#
# 【调度链】Supervisor → WeatherAgent
#
# ------------------------------------------------------------
# Assistant：
# 【天气科】成都 6月8日：晴，28°C，适合出行。
#
# （Supervisor 模式演示：只派了一个专员。）
# ------------------------------------------------------------
#
# ============================================================
# 📥 用户：我下周从上海去成都玩3天，帮我查天气、订机票、推荐景点和美食，最后出个行程
# ============================================================
#
#   📋 TaskPlanner：正在拆任务…
#   ✓ 拆出 5 项： WeatherAgent → FlightAgent → AttractionAgent → RestaurantAgent → ItineraryAgent
#
#   🔀 进入【规划流水线】模式
#
#   ▶ 执行 T1 → WeatherAgent
#       └─ [WeatherAgent] 收到任务：处理与用户「我下周从上海去成都玩3天，帮我查天气、订…」相关的WeatherAgent任务…
#   ✓ T1 完成
#
#   ▶ 执行 T2 → FlightAgent
#       └─ [FlightAgent] 收到任务：处理与用户「我下周从上海去成都玩3天，帮我查天气、订…」相关的FlightAgent任务…
#   ✓ T2 完成
#
#   ▶ 执行 T3 → AttractionAgent
#       └─ [AttractionAgent] 收到任务：处理与用户「我下周从上海去成都玩3天，帮我查天气、订…」相关的AttractionAgent任务…
#   ✓ T3 完成
#
#   ▶ 执行 T4 → RestaurantAgent
#       └─ [RestaurantAgent] 收到任务：处理与用户「我下周从上海去成都玩3天，帮我查天气、订…」相关的RestaurantAgent任务…
#   ✓ T4 完成
#
#   ▶ 执行 T5 → ItineraryAgent
#       └─ [ItineraryAgent] 收到任务：处理与用户「我下周从上海去成都玩3天，帮我查天气、订…」相关的ItineraryAgent任务
# （参…
#   ✓ T5 完成
#
#   📝 汇总各科结果…
#
# 【调度链】规划流水线 → WeatherAgent → FlightAgent → AttractionAgent → RestaurantAgent → ItineraryAgent
#
# ------------------------------------------------------------
# Assistant：
# 您好！针对「我下周从上海去成都玩3天，帮我查天气、订机票、推荐景点和美食，最后出个行程」，整理如下：
#
# 1. 【天气科】成都 6月8日：晴，28°C，适合出行。
# 2. 【机票科】上海→成都 6月8日：MU5401 08:00-11:20，约 680 元。
# 3. 【景点科】推荐：武侯祠、大熊猫基地、宽窄巷子（共 3 处）。
# 4. 【美食科】推荐：陈麻婆豆腐、蜀大侠火锅、钟水饺（共 3 家）。
# 5. 【行程科】Day1 市区+锦里；Day2 熊猫基地；Day3 青城山。
#
# （以上为书籍演示数据，真实项目会调用在线 API。）
# ------------------------------------------------------------
#
