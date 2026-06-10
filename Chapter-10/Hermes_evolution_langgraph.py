"""
Hermes Agent 技能自学习 / 自进化闭环（LangGraph · 旅游场景 · 接法 A）

外层 Hermes：skill_view → 评估 → create/patch/Hub
内层业务：executor ReAct 通过 @tool 包装调用 Chapter-10/sub_agents.py 子智能体

Demo：丽江 3 日游 → 学技能 → 大理 3 日游 skill_view 复用
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

import operator
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

# ---------------------------------------------------------------------------
# 路径引导：允许从 Chapter-10 目录直接 python Hermes_evolution_langgraph.py
# ---------------------------------------------------------------------------
_CH8_DIR = Path(__file__).resolve().parent
_CH6_DIR = _CH8_DIR.parent / "Chapter-6"
if str(_CH8_DIR) not in sys.path:
    sys.path.insert(0, str(_CH8_DIR))
if str(_CH6_DIR) not in sys.path:
    sys.path.insert(0, str(_CH6_DIR))

from prompt_builder import PromptBuilder
from skill_manager_tool import SkillManager
from skills_hub import SkillsHub
from skills_tool import SkillLoader
from skills_tool_view import make_skill_view_tool

# 加载 LLM Key：优先书根 .env，其次 Chapter-10/.env
load_dotenv(_CH8_DIR.parent / ".env")
load_dotenv(_CH8_DIR / ".env")

# 兼容 OpenAI / OpenRouter / DashScope 多种环境变量名
_api_key = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
    or os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("CHAT_API_KEY")
    or ""
).strip()
if _api_key:
    os.environ["OPENAI_API_KEY"] = _api_key


def _make_chat_openai(*, json_mode: bool = False) -> ChatOpenAI:
    """创建 ChatOpenAI 客户端；json_mode=True 时强制 JSON 输出（评估/抽技能用）。"""
    kw: dict = {
        "model": os.getenv("OPENAI_MODEL") or os.getenv("DEPLOYMENT_NAME") or "gpt-4o-mini",
        "temperature": 0,
        "api_key": _api_key or "your-api-key-here",
    }
    base_url = (os.getenv("OPENAI_BASE_URL") or os.getenv("CHAT_ENDPOINT") or "").strip()
    if base_url:
        kw["base_url"] = base_url.rstrip("/")
    if json_mode:
        kw["model_kwargs"] = {"response_format": {"type": "json_object"}}
    # 内网自签名网关时可关闭 SSL 校验
    if os.getenv("OPENAI_SSL_VERIFY", "1").strip().lower() in ("0", "false", "no", "off"):
        import httpx

        kw["http_client"] = httpx.Client(verify=False)
    return ChatOpenAI(**kw)


# 普通对话 LLM（executor ReAct 循环）
LLM = _make_chat_openai()
# 结构化 JSON 输出 LLM（evaluator / skill_extractor / skill_patcher）
LLM_STRICT = _make_chat_openai(json_mode=True)

# ---------------------------------------------------------------------------
# Hermes 触发阈值（与 System Prompt SKILLS_GUIDANCE 对应）
# ---------------------------------------------------------------------------
MIN_TOOL_CALLS_FOR_SKILL = 5   # 复杂任务：≥5 次 tool call → 值得抽技能
MIN_SCORE_TO_CREATE = 7.0      # 兜底：高分且无已有技能 → 也可创建
MIN_STEPS_TO_CREATE = 3
MAX_SCORE_BEFORE_PATCH = 5.9   # 低分或有问题 → patch 已有技能
MIN_SCORE_TO_PUBLISH = 8.0     # 高分成熟技能 → 发布 Hub

# 演示用通用旅游技能（任务1 创建，任务2 skill_view 复用）
GENERIC_TRAVEL_SKILL_NAME = "domestic_3day_trip_planning"
GENERIC_TRAVEL_SKILL_TITLE = "国内三日游行程规划"
GENERIC_TRAVEL_TASK_TYPE = "travel_planning"

# SubAgent 调用线程池（同步 executor 内跑 async ainvoke）
_SUBAGENT_EXECUTOR = ThreadPoolExecutor(max_workers=2)


# ==============================
# 运行时上下文（全局单例，create 时初始化）
# ==============================


class HermesRuntime:
    """
    集中管理 Agent 运行所需的所有持久化组件。

    目录结构（storage_dir）：
        memory.md                      — L1 热记忆
        skills/*.md                    — SKILL.md 技能库
        skills/{name}.stats.json       — 使用统计 sidecar
        .skills_prompt_snapshot.json   — 索引 L2 快照
        hub_export/                    — 社区发布目录
    """

    def __init__(self, storage_dir: str = "./agent_memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.storage_dir / "memory.md"
        self.memory_char_limit = 2000  # L1 热记忆字符上限

        skills_dir = self.storage_dir / "skills"
        self.loader = SkillLoader(str(skills_dir))              # 技能加载 Tier 0–3
        self.skill_manager = SkillManager(str(self.storage_dir))  # create/patch/rollback
        self.prompt_builder = PromptBuilder(self.loader, str(self.storage_dir))  # 索引+Prompt
        self.hub = SkillsHub(str(self.storage_dir))               # 本地 Hub 发布

        self.memory = self.memory_path.read_text(encoding="utf-8") if self.memory_path.exists() else ""

    def get_memory_snapshot(self) -> str:
        """供 System Prompt 注入的 L1 热记忆快照。"""
        return self.memory

    def add_to_memory(self, entry: str) -> None:
        """追加一条 L1 记忆；超限则先 consolidate 再写入。"""
        new_memory = f"{self.memory}\n- {entry}".strip()
        if len(new_memory) > self.memory_char_limit:
            self.consolidate_memory()
            new_memory = f"{self.memory}\n- {entry}".strip()
        self.memory = new_memory
        self.memory_path.write_text(self.memory, encoding="utf-8")

    def consolidate_memory(self) -> None:
        """LLM 压缩 L1 热记忆，保留核心要点。"""
        if not self.memory:
            return
        prompt = (
            f"请将以下记忆整合为不超过 {self.memory_char_limit // 2} 字符的核心要点，"
            f"每行以 '- ' 开头：\n{self.memory}"
        )
        self.memory = LLM.invoke(prompt).content.strip()
        self.memory_path.write_text(self.memory, encoding="utf-8")


# 图节点通过 global runtime 访问上述组件（create_self_improving_agent 时赋值）
runtime: HermesRuntime


# ==============================
# 接法 A：旅游 SubAgent 包装为 Hermes executor 工具
# （底层实现见 Chapter-10/sub_agents.py + travel_common.py）
# ==============================


def _extract_sub_agent_text(state: Dict[str, Any]) -> str:
    """从 SubAgent invoke 返回的 messages 中提取可读文本。"""
    messages = state.get("messages") or []
    agent_text = ""
    last_tool = ""
    for msg in messages:
        msg_type = getattr(msg, "type", None)
        content = getattr(msg, "content", None)
        if msg_type == "ai" and content:
            agent_text = str(content)
        elif msg_type == "tool" and content:
            last_tool = str(content)
    if agent_text.strip():
        return agent_text
    if last_tool.strip():
        return last_tool
    return json.dumps(state, ensure_ascii=False)[:4000]


def _invoke_sub_agent_sync(agent_name: str, user_message: str) -> str:
    """同步包装：在 Hermes executor 的 ReAct 循环中调用 async SubAgent。"""
    from sub_agents import SubAgentFactory

    agent = SubAgentFactory.get_agent(agent_name)
    thread_id = f"hermes_{agent_name}_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    async def _run() -> Dict[str, Any]:
        return await agent.ainvoke(
            {"messages": [("user", user_message)]},
            config,
        )

    def _run_in_new_loop() -> Dict[str, Any]:
        return asyncio.run(_run())

    try:
        asyncio.get_running_loop()
        state = _SUBAGENT_EXECUTOR.submit(_run_in_new_loop).result(timeout=300)
    except RuntimeError:
        state = asyncio.run(_run())

    return _extract_sub_agent_text(state)


@tool
def call_itinerary_agent(request: str) -> str:
    """调用 ItineraryAgent：综合信息生成逐日行程、预算、交通与注意事项。request 为完整需求或上下文。"""
    print("  ▶ SubAgent ItineraryAgent", flush=True)
    return _invoke_sub_agent_sync("ItineraryAgent", request)


@tool
def call_attraction_agent(city: str, preferences: str = "") -> str:
    """调用 AttractionAgent：推荐目的地 POI/景点。city 如丽江、大理；preferences 为偏好说明。"""
    query = f"请推荐{city}的景点与游玩建议。"
    if preferences.strip():
        query += f" 用户偏好：{preferences}"
    print(f"  ▶ SubAgent AttractionAgent ({city})", flush=True)
    return _invoke_sub_agent_sync("AttractionAgent", query)


@tool
def call_hotel_agent(city: str, request: str) -> str:
    """调用 HotelAgent：推荐住宿区域与酒店。city 为目的地；request 含预算、天数、偏好等。"""
    query = f"目的地：{city}。{request}"
    print(f"  ▶ SubAgent HotelAgent ({city})", flush=True)
    return _invoke_sub_agent_sync("HotelAgent", query)


@tool
def call_weather_agent(city: str, date: str = "明天") -> str:
    """调用 WeatherAgent：查询目的地日期天气，供行程规划参考。"""
    query = f"请查询{city}在{date}的天气概况，并给出出行穿衣建议。"
    print(f"  ▶ SubAgent WeatherAgent ({city}, {date})", flush=True)
    return _invoke_sub_agent_sync("WeatherAgent", query)


@tool
def call_refine_itinerary(issue: str, context: str) -> str:
    """
    调用 ItineraryAgent 完善/修正行程（如预算超限、节奏过满）。
    触发 Hermes「行程完善」路径（had_error_recovery）。
    """
    query = (
        f"以下行程存在问题，请给出完善后的方案：\n"
        f"问题：{issue}\n"
        f"当前上下文：{context[:2000]}"
    )
    print("  ▶ SubAgent ItineraryAgent (refine)", flush=True)
    return _invoke_sub_agent_sync("ItineraryAgent", query)


EXECUTOR_TOOLS = [
    call_itinerary_agent,
    call_attraction_agent,
    call_hotel_agent,
    call_weather_agent,
    call_refine_itinerary,
]


def build_executor_tools(loader: SkillLoader):
    """
    组装 executor 可用工具。

    skill_view 放在首位：对齐 Hermes「先加载技能，再调领域工具」的顺序。
    """
    skill_view = make_skill_view_tool(loader)
    return [skill_view, *EXECUTOR_TOOLS]


# ==============================
# LangGraph State（节点间共享的运行时数据）
# ==============================


class AgentState(TypedDict):
    """图状态：每个节点读/写其中部分字段。"""

    messages: Annotated[Sequence[BaseMessage], operator.add]  # 对话消息（累加）
    task: str                          # 用户任务原文
    task_type: str                     # 任务类型标签
    matched_skill_name: Optional[str]  # executor 中 skill_view 加载的技能名
    skill_load_tier: int               # 渐进加载层级（默认 2 = 完整正文）
    execution_steps: List[str]         # 工具调用步骤记录
    result: str                        # executor 最终文本输出
    tool_call_count: int               # 工具调用总次数（Hermes 创建触发）
    had_error_recovery: bool           # 是否调用 call_refine_itinerary（行程完善触发）
    user_correction: Optional[str]     # 用户手动纠正（创建触发）
    evaluation_score: float            # 评估得分 1–10
    evaluation_comment: str            # 评估理由
    should_create_skill: bool          # 是否进入 skill_extractor 节点
    should_patch_skill: bool           # 是否进入 skill_patcher 节点
    should_update_memory: bool           # 是否进入 memory_updater 节点
    should_publish_skill: bool         # 是否进入 hub_publish 节点
    task_count: int                    # 累计完成任务数
    execution_had_issues: bool         # 执行是否有问题


def _empty_state(task: str, **extra) -> dict:
    """构造一次 invoke 的初始状态（demo / 测试用）。"""
    base = {
        "task": task,
        "messages": [],
        "task_type": "general",
        "matched_skill_name": None,
        "skill_load_tier": 2,
        "execution_steps": [],
        "result": "",
        "tool_call_count": 0,
        "had_error_recovery": False,
        "user_correction": None,
        "evaluation_score": 0.0,
        "evaluation_comment": "",
        "should_create_skill": False,
        "should_patch_skill": False,
        "should_update_memory": False,
        "should_publish_skill": False,
        "task_count": 0,
        "execution_had_issues": False,
    }
    base.update(extra)
    return base


# ==============================
# LangGraph 节点
# ==============================


def rebuild_index_node(_state: AgentState) -> Dict[str, Any]:
    """
    索引重建节点：skill create/patch 之后执行。

    - 清空 L1 内存索引 + 删除 L2 .skills_prompt_snapshot.json
    - reload SkillLoader，使 Tier 2/3 读到最新 SKILL.md
    """
    runtime.prompt_builder.invalidate_cache()
    runtime.loader.reload()
    return {}


def router_node(state: AgentState) -> Dict[str, Any]:
    """
    轻量路由节点（入口）。

    技能匹配已交给 executor 内 LLM + skill_view 工具（对齐 Hermes 生产路径）。
    System Prompt 已含 Skills (mandatory) 与 <available_skills> 索引。
    此处仅初始化 task_type / skill_load_tier，不做额外 LLM 预匹配。
    """
    _ = state["task"]
    return {
        "matched_skill_name": None,
        "task_type": "general",
        "skill_load_tier": 2,
    }


def executor_node(state: AgentState) -> Dict[str, Any]:
    """
    执行节点（核心）：ReAct 式 tool loop。

    流程：
    1. 构建可缓存 System Prompt（含 <available_skills> + Skills mandatory）
    2. LLM 扫描索引，相关则先 skill_view(name, tier=2)
    3. 再调用 call_itinerary_agent / call_attraction_agent 等 SubAgent 包装工具
    4. skill_view 结果进入 ToolMessage，不修改 System Prompt（保护 Prompt Cache）
    """
    task = state["task"]
    if state.get("user_correction"):
        task = f"{task}\n\n【用户纠正】\n{state['user_correction']}"

    travel_base = (
        "你是 Hermes 旅游行程规划 Agent。"
        "若 <available_skills> 中有行程规划相关技能，先用 skill_view(name, tier=2) 加载"
        "（name 必须与索引中的技能名完全一致，不要用 task_type 当 name）；"
        "若无可用技能则直接按顺序调用 SubAgent："
        "WeatherAgent → AttractionAgent → HotelAgent → ItineraryAgent；"
        "若行程有问题可 call_refine_itinerary 完善。"
    )
    system_prompt = runtime.prompt_builder.build_system_prompt(
        runtime.get_memory_snapshot(),
        base=travel_base,
    )
    lc_messages: List[Any] = [
        {"role": "system", "content": system_prompt},
        HumanMessage(
            content=(
                f"{task}\n\n"
                "Reminder: 查看 <available_skills>；有匹配技能则 skill_view(索引中的 name, tier=2)，"
                "无技能则跳过 skill_view，直接调用 "
                "call_weather_agent / call_attraction_agent / call_hotel_agent / "
                "call_itinerary_agent 完成规划。"
            )
        ),
    ]

    tools = build_executor_tools(runtime.loader)
    tools_by_name = {t.name: t for t in tools}
    llm_tools = LLM.bind_tools(tools)

    tool_call_count = 0
    had_error_recovery = False
    execution_steps: List[str] = []
    loaded_skill_name: Optional[str] = None
    max_rounds = 10  # ReAct 循环上限，防止无限 tool call

    for _ in range(max_rounds):
        ai_msg: AIMessage = llm_tools.invoke(lc_messages)
        lc_messages.append(ai_msg)
        # 无 tool_calls → LLM 给出最终回答，结束循环
        if not ai_msg.tool_calls:
            final_text = ai_msg.content or ""
            break

        for tc in ai_msg.tool_calls:
            tool_call_count += 1
            tool_name = tc["name"]
            args = tc.get("args") or {}
            tool_fn = tools_by_name.get(tool_name)
            if tool_fn:
                try:
                    out = tool_fn.invoke(args)
                except Exception as exc:
                    out = f"工具错误: {exc}"
            else:
                out = "未知工具"

            # 仅当 skill_view 成功加载磁盘上存在的技能时，才算「已匹配」（供 evaluator 统计/创建/patch）
            if tool_name == "skill_view":
                viewed = (args.get("name") or "").strip()
                if viewed:
                    print(f"  ▶ skill_view({viewed!r}, tier={args.get('tier', 2)})", flush=True)
                    if viewed in runtime.loader.list_skill_names():
                        loaded_skill_name = viewed
                    elif str(out).startswith(f"Skill '{viewed}' not found"):
                        print(f"  ⚠ skill_view 未命中（{viewed!r} 不在技能库），视为首次任务", flush=True)
            # call_refine_itinerary → 标记「经历行程完善」
            if tool_name == "call_refine_itinerary":
                had_error_recovery = True

            execution_steps.append(f"{tool_name}({json.dumps(args, ensure_ascii=False)[:80]})")
            lc_messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    else:
        # 达到 max_rounds 仍未结束
        final_text = lc_messages[-1].content if lc_messages else ""

    # 若 LLM 未调工具，后备提取步骤列表
    if not execution_steps:
        steps_resp = LLM_STRICT.invoke(
            f'从以下回答提取步骤 JSON：{{"steps": [...]}}\n{final_text}'
        )
        try:
            execution_steps = json.loads(steps_resp.content).get("steps", [])
        except json.JSONDecodeError:
            execution_steps = ["analyze", "respond"]

    return {
        "messages": [AIMessage(content=final_text)],
        "result": final_text,
        "execution_steps": execution_steps,
        "tool_call_count": tool_call_count,
        "had_error_recovery": had_error_recovery,
        "matched_skill_name": loaded_skill_name,
    }


def evaluator_node(state: AgentState) -> Dict[str, Any]:
    """
    评估节点：LLM 打分 + 决定后续分支。

    分支优先级（after_evaluator）：
        patch > create > publish > memory_update > END
    """
    prompt = f"""
    评估以下旅游行程规划任务完成质量（1-10），输出 JSON {{"score": 数字, "comment": "理由", "had_issues": true/false}}
    
    评分标准：逐日行程、预算/交通/住宿、注意事项是否完整；动线是否合理。
    
    任务：{state['task']}
    工具调用次数：{state['tool_call_count']}
    是否经历行程完善：{state['had_error_recovery']}
    用户纠正：{state.get('user_correction') or '无'}
    步骤：{json.dumps(state['execution_steps'], ensure_ascii=False)}
    结果：{state['result'][:3000]}
    """
    ev = json.loads(LLM_STRICT.invoke(prompt).content)
    score = float(ev.get("score", 5))
    comment = ev.get("comment", "")
    had_issues = bool(ev.get("had_issues", score < 6))

    skill_name = state.get("matched_skill_name")
    # skill_view 可能叫了不存在的 name（如误用 task_type），不计为已匹配
    if skill_name and runtime.loader.get(skill_name) is None:
        skill_name = None
    # 更新已有技能的使用统计（滑动平均）
    if skill_name:
        runtime.skill_manager.update_stats(skill_name, score)

    # --- Hermes 创建触发：5+ tools / 修错 / 用户纠正 ---
    hermes_create = (
        state["tool_call_count"] >= MIN_TOOL_CALLS_FOR_SKILL
        or state["had_error_recovery"]
        or bool(state.get("user_correction"))
    )
    # --- 兜底：高分 + 多步骤 + 未加载已有技能 ---
    fallback_create = (
        score >= MIN_SCORE_TO_CREATE
        and len(state["execution_steps"]) >= MIN_STEPS_TO_CREATE
        and not skill_name
    )
    should_create_skill = (
        (hermes_create or fallback_create)
        and not skill_name
        and runtime.loader.get(GENERIC_TRAVEL_SKILL_NAME) is None
    )

    # --- 自动改进：已加载技能但得分低或有问题 ---
    should_patch_skill = bool(skill_name) and (score <= MAX_SCORE_BEFORE_PATCH or had_issues)

    task_count = state["task_count"] + 1
    should_update_memory = task_count % 10 == 0  # 每 10 个任务整合 L1 记忆
    should_publish_skill = bool(skill_name) and score >= MIN_SCORE_TO_PUBLISH and not should_patch_skill

    return {
        "evaluation_score": score,
        "evaluation_comment": comment,
        "execution_had_issues": had_issues,
        "should_create_skill": should_create_skill,
        "should_patch_skill": should_patch_skill,
        "should_update_memory": should_update_memory,
        "should_publish_skill": should_publish_skill,
        "task_count": task_count,
    }


def skill_extractor_node(state: AgentState) -> Dict[str, Any]:
    """
    技能创建节点：LLM 从成功任务抽取 agentskills.io 标准 JSON → skill_manage(create)。

    产出：skills/{name}.md（YAML frontmatter + Markdown body）
    """
    # 已有通用旅游技能则不再创建
    if runtime.loader.get(GENERIC_TRAVEL_SKILL_NAME):
        print("  ▶ skill_extractor 跳过：已有 travel_planning 技能", flush=True)
        return {}

    prompt = f"""
        从以下成功执行的国内多日游规划任务中提取可复用技能，输出 JSON（agentskills.io，**全部中文**）：
        
        {{
          "name": "{GENERIC_TRAVEL_SKILL_NAME}",
          "description": "一行中文描述（通用，不含具体城市）",
          "version": "1.0.0",
          "platforms": ["macos", "linux", "windows"],
          "metadata": {{
            "hermes": {{
              "tags": ["{GENERIC_TRAVEL_TASK_TYPE}", "旅游", "行程规划"],
              "related_skills": [],
              "requires_toolsets": ["terminal"],
              "config": []
            }}
          }},
          "title": "{GENERIC_TRAVEL_SKILL_TITLE}",
          "trigger_conditions": ["用户需要规划国内 N 日游", "含预算/天数/偏好"],
          "steps": [
            "skill_view 加载本技能",
            "call_weather_agent 查天气",
            "call_attraction_agent 选 POI",
            "call_hotel_agent 定住宿区",
            "call_itinerary_agent 出逐日行程",
            "必要时 call_refine_itinerary 完善"
          ],
          "pitfalls": ["行程过满", "忽略交通耗时"],
          "verification": ["含逐日安排", "含预算与注意事项"]
        }}
        
        要求：name 必须为 {GENERIC_TRAVEL_SKILL_NAME!r}；steps 至少 5 条；不写死丽江/大理。
        
        任务：{state['task']}
        执行步骤记录：{json.dumps(state['execution_steps'], ensure_ascii=False)}
        评分：{state['evaluation_score']}
    """
    skill_data = json.loads(LLM_STRICT.invoke(prompt).content)
    skill_data["name"] = GENERIC_TRAVEL_SKILL_NAME
    skill_data["title"] = skill_data.get("title") or GENERIC_TRAVEL_SKILL_TITLE
    meta = skill_data.setdefault("metadata", {}).setdefault("hermes", {})
    tags = meta.setdefault("tags", [])
    for tag in (GENERIC_TRAVEL_TASK_TYPE, "旅游", "行程规划"):
        if tag not in tags:
            tags.append(tag)
    result = runtime.skill_manager.manage("create", skill_data=skill_data)
    if result.get("ok"):
        runtime.add_to_memory(f"学习了新技能：{result['skill']}")
    return {"messages": [AIMessage(content=f"[skill_manage create] {result}")]}


def skill_patcher_node(state: AgentState) -> Dict[str, Any]:
    """
    技能改进节点：LLM 生成 patch → Fuzzy Match 定位 → 原子替换。

    失败时 skill_manage(rollback) 回滚到 versions/ 备份。
    """
    skill_name = state["matched_skill_name"]
    if not skill_name:
        return {}

    skill = runtime.loader.get(skill_name)
    if not skill:
        return {}

    prompt = f"""
技能 `{skill_name}` 执行不佳，请生成 patch JSON（agentskills.io 章节名）：
{{
  "section": "trigger_conditions|steps|pitfalls|verification",
  "old_fragment": "要替换的原句（尽量精确）",
  "new_fragment": "改进后的内容"
}}

问题：{state['evaluation_comment']}
当前 steps：{json.dumps(skill.steps, ensure_ascii=False)}
当前 pitfalls：{json.dumps(skill.pitfalls, ensure_ascii=False)}
"""
    patch = json.loads(LLM_STRICT.invoke(prompt).content)
    result = runtime.skill_manager.manage("patch", skill_name=skill_name, patch=patch)
    if not result.get("ok"):
        runtime.skill_manager.manage("rollback", skill_name=skill_name)
    return {"messages": [AIMessage(content=f"[skill_manage patch] {result}")]}


def hub_publish_node(state: AgentState) -> Dict[str, Any]:
    """社区发布节点：将成熟技能导出到 hub_export/（模拟 agentskills.io）。"""
    name = state.get("matched_skill_name")
    if not name:
        return {}
    result = runtime.hub.publish(name, min_score=MIN_SCORE_TO_PUBLISH)
    return {"messages": [AIMessage(content=f"[skills_hub publish] {result}")]}


def memory_updater_node(state: AgentState) -> Dict[str, Any]:
    """记忆整合节点：LLM 压缩 L1 热记忆（每 10 个任务触发一次）。"""
    runtime.consolidate_memory()
    runtime.add_to_memory(f"完成 {state['task_count']} 个任务，已整合 L1 记忆")
    return {}


# ==============================
# 构建 LangGraph 图
# ==============================


def _make_checkpointer(storage_dir: str):
    """
    多轮对话 checkpoint。

    默认 MemorySaver（进程内）；
    设置 HERMES_CHECKPOINT=sqlite 且安装 langgraph-checkpoint-sqlite 可持久化到磁盘。
    """
    use_sqlite = os.getenv("HERMES_CHECKPOINT", "memory").strip().lower() == "sqlite"
    if use_sqlite:
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            db_path = os.path.join(storage_dir, "checkpoints.db")
            conn = sqlite3.connect(db_path, check_same_thread=False)
            return SqliteSaver(conn)
        except ImportError:
            try:
                import sqlite3

                from langgraph_checkpoint_sqlite import SqliteSaver

                db_path = os.path.join(storage_dir, "checkpoints.db")
                conn = sqlite3.connect(db_path, check_same_thread=False)
                return SqliteSaver(conn)
            except ImportError:
                print(
                    "⚠ 未安装 langgraph-checkpoint-sqlite，已回退 MemorySaver。"
                    "持久化 checkpoint 请: pip install langgraph-checkpoint-sqlite",
                    flush=True,
                )
    return MemorySaver()


def create_self_improving_agent(storage_dir: str = "./agent_memory"):
    """
    工厂函数：组装并编译 Hermes 自进化 Agent 图。

    返回 compiled graph，可 invoke / astream。
    """
    global runtime
    runtime = HermesRuntime(storage_dir)

    workflow = StateGraph(AgentState)

    # 注册节点
    workflow.add_node("router", router_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("skill_extractor", skill_extractor_node)
    workflow.add_node("skill_patcher", skill_patcher_node)
    workflow.add_node("rebuild_index", rebuild_index_node)
    workflow.add_node("hub_publish", hub_publish_node)
    workflow.add_node("memory_updater", memory_updater_node)

    # 固定边：router → executor → evaluator
    workflow.set_entry_point("router")
    workflow.add_edge("router", "executor")
    workflow.add_edge("executor", "evaluator")

    def after_evaluator(state: AgentState) -> str:
        """评估后条件路由：patch 优先于 create。"""
        if state["should_patch_skill"]:
            return "skill_patcher"
        if state["should_create_skill"]:
            return "skill_extractor"
        if state["should_publish_skill"]:
            return "hub_publish"
        if state["should_update_memory"]:
            return "memory_updater"
        return END

    workflow.add_conditional_edges(
        "evaluator",
        after_evaluator,
        {
            "skill_patcher": "skill_patcher",
            "skill_extractor": "skill_extractor",
            "hub_publish": "hub_publish",
            "memory_updater": "memory_updater",
            END: END,
        },
    )
    # create/patch 后必须重建索引，再结束
    workflow.add_edge("skill_patcher", "rebuild_index")
    workflow.add_edge("skill_extractor", "rebuild_index")
    workflow.add_edge("rebuild_index", END)
    workflow.add_edge("hub_publish", END)
    workflow.add_edge("memory_updater", END)

    return workflow.compile(checkpointer=_make_checkpointer(storage_dir))


# ==============================
# Demo：丽江 → 大理（Hermes + SubAgent 接法 A）
# ==============================

TASK1_LIJIANG = """
请为一对情侣规划丽江 3 日游行程，要求：
- 出发日期：2026-05-01，共 3 天 2 晚
- 预算：人均 2500 元（含住宿、餐饮、门票、市内交通）
- 偏好：古城漫步、轻度徒步、本地美食，节奏不要太赶
- 请给出逐日安排（上午/下午/晚上）、推荐住宿区域、大致费用拆分和注意事项
"""

TASK2_DALI = """
请为两位朋友规划大理 3 日游行程，要求：
- 出发日期：2026-06-10，共 3 天 2 晚
- 预算：人均 2200 元
- 偏好：洱海骑行、古镇闲逛、白族特色餐饮
- 请给出逐日安排、交通方式建议、住宿区域推荐和预算概览
"""

if __name__ == "__main__":
    storage_dir = "./my_agent_memory"

    if os.getenv("TRAVEL_DEMO_FRESH", "").strip().lower() in ("1", "true", "yes"):
        import shutil

        for sub in ("skills", "hub_export"):
            p = os.path.join(storage_dir, sub)
            if os.path.isdir(p):
                shutil.rmtree(p)
        snap = os.path.join(storage_dir, ".skills_prompt_snapshot.json")
        if os.path.isfile(snap):
            os.remove(snap)
        print("（已清空旧技能与索引快照）\n")

    agent = create_self_improving_agent(storage_dir=storage_dir)
    config = {"configurable": {"thread_id": "hermes-travel-subagent-1"}}

    print("=== 任务1：丽江 3 日游（SubAgent + skill 创建）===")
    r1 = agent.invoke(_empty_state(TASK1_LIJIANG.strip()), config)
    print(f"工具调用：{r1['tool_call_count']}  得分：{r1['evaluation_score']}")
    print(f"创建技能：{r1['should_create_skill']}  patch：{r1['should_patch_skill']}")
    print(f"skill_view：{r1.get('matched_skill_name') or '无（首次）'}")

    print("\n=== 任务2：大理 3 日游（skill_view 复用 SubAgent）===")
    r2 = agent.invoke(_empty_state(TASK2_DALI.strip(), task_count=r1["task_count"]), config)
    print(f"工具调用：{r2['tool_call_count']}  得分：{r2['evaluation_score']}")
    print(f"skill_view：{r2.get('matched_skill_name') or '无'}  Hub：{r2['should_publish_skill']}")

    print("\n=== 技能库（SKILL.md）===")
    for name in runtime.loader.list_skill_names():
        s = runtime.loader.get(name)
        if s:
            print(f"- {name} ({s.title}) v{s.version}  uses={s.use_count}  avg={s.avg_score:.1f}")

    published = runtime.hub.list_published()
    if published:
        print("\n=== Hub 已发布 ===")
        for p in published:
            print(f"- {p['name']}: {p.get('hub_url')}")
