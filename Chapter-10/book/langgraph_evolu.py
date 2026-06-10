"""
LangGraph 自我进化 Agent 书稿示例

闭环拓扑：

    router → executor → evaluator → [skill_extractor] → [memory_updater] → END

设计要点：
  - L1 热记忆（memory.md）：每次注入 prompt，有字符上限
  - L2 技能库（skills/*.json）：渐进披露，router 只看索引，executor 加载全文
  - 领域与图解耦：旅游只是 TRAVEL_DOMAIN 配置，换 DomainConfig 可问别的问题
  - Demo：丽江 3 日游 → 学技能 → 大理 3 日游 router 复用
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

import operator
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 环境变量：API Key / 模型 / 网关（与全书其他章节一致）
# ---------------------------------------------------------------------------
load_dotenv()
_api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY") or "").strip()
if _api_key:
    os.environ["OPENAI_API_KEY"] = _api_key


def _make_chat_openai(*, json_mode: bool = False) -> ChatOpenAI:
    """创建 ChatOpenAI；json_mode=True 时强制 JSON 输出（router/评估/抽技能用）。"""
    kw: dict = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": 0,
        "api_key": _api_key or "your-api-key-here",
    }
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if base_url:
        kw["base_url"] = base_url.rstrip("/")
    if json_mode:
        kw["model_kwargs"] = {"response_format": {"type": "json_object"}}
    # 内网自签名网关时可关闭 SSL 校验
    if os.getenv("OPENAI_SSL_VERIFY", "1").strip().lower() in ("0", "false", "no", "off"):
        import httpx

        kw["http_client"] = httpx.Client(verify=False)
    return ChatOpenAI(**kw)


# 普通对话 LLM（executor 生成行程正文）
LLM = _make_chat_openai()
# 结构化 JSON LLM（router / evaluator / skill_extractor / 抽步骤）
LLM_STRICT = _make_chat_openai(json_mode=True)


# ==============================
# JSON 解析（容错 markdown 围栏与前后说明文字）
# ==============================


def _parse_json(text: str) -> Optional[Any]:
    """
    从 LLM 回复中提取第一个 JSON 对象或数组。

    模型常返回 ```json ... ``` 或在 JSON 外加说明文字，直接 json.loads 会失败。
    """
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, candidate, re.DOTALL)
        if not match:
            continue
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    return None


# ==============================
# 领域配置（通用框架 + 可插拔领域 prompt）
# ==============================


@dataclass
class DomainConfig:
    """
    领域相关 prompt 片段；核心图节点不硬编码「旅游」等业务。

    各字段注入位置：
      executor_system_prompt  → _executor
      evaluation_rubric       → _evaluator（LLM 打分标准）
      skill_extraction_hint   → _skill_extractor
      task_type_examples      → _router
      router_extra_rules      → _router（如丽江/大理应复用同一技能）
    """

    name: str = "general"
    executor_system_prompt: str = (
        "你是通用 AI 助手。逐步思考，给出结构清晰、可执行的回答。"
    )
    evaluation_rubric: str = (
        "从正确性、完整性、效率三方面评估（1-10）。"
        "10=超出预期；8-9=良好；6-7=基本可用；4-5=有明显缺陷；1-3=不可用。"
    )
    skill_extraction_hint: str = (
        "提取可复用技能：name 应通用（不含具体人名/城市/文件名等实例），"
        "procedure 至少 3 步，全部字段使用中文。"
    )
    task_type_examples: str = "code_review, data_analysis, travel_planning, writing, research"
    router_extra_rules: str = ""


# 书稿 Demo 默认领域：旅游行程规划
TRAVEL_DOMAIN = DomainConfig(
    name="travel",
    executor_system_prompt=(
        "你是专业的旅游行程规划助手，熟悉国内热门目的地。"
        "请根据用户需求给出可执行的逐日行程，并说明交通、住宿区域、预算与注意事项。"
        "输出结构清晰，包含思考过程与具体步骤。"
    ),
    evaluation_rubric=(
        "评分标准（旅游行程规划）：\n"
        "- 10分：逐日行程合理、景点衔接顺畅、预算/交通/住宿建议完整，超出预期\n"
        "- 8-9分：行程可执行，覆盖主要需求，无明显逻辑问题\n"
        "- 6-7分：有行程框架但缺少预算、交通或注意事项等关键信息\n"
        "- 4-5分：景点堆砌、动线不合理或忽略用户约束（天数/预算/人群）\n"
        "- 1-3分：未给出可用行程或严重偏离需求"
    ),
    skill_extraction_hint=(
        "从成功执行的旅游规划任务中提取**通用**技能（全部中文）：\n"
        "- name 不要包含丽江、大理等具体城市，应描述流程（如「国内三日游行程规划」）\n"
        "- task_type 使用 travel_planning\n"
        "- procedure 至少 5 步，适用任意国内短途游（收集约束→住宿区→逐日安排→预算→注意事项）\n"
        "- 步骤中用占位说明（如「核心景点 A」「古城区域」），不要写死某一城市景点"
    ),
    task_type_examples="travel_planning, hotel_recommendation, flight_search, general",
    router_extra_rules=(
        "若任务为「国内 N 日游行程规划」（目的地可不同），且已有 task_type=travel_planning 的技能，"
        "应复用该技能而非返回 null；matched_skill 必须与列表中 name 完全一致。"
    ),
)


# ==============================
# 数据结构
# ==============================


class Skill(BaseModel):
    """技能：从成功任务中抽取的可复用程序性知识，持久化到 skills/{name}.json。"""

    name: str = Field(description="技能名称（通用，不含具体城市等实例）")
    description: str = Field(description="一行描述，供 router 索引匹配")
    task_type: str = Field(description="任务类型标签，如 travel_planning，用于同类任务复用")
    procedure: List[str] = Field(description="执行步骤，executor 匹配后注入 system prompt")
    pitfalls: List[str] = Field(default_factory=list, description="常见陷阱")
    verification: List[str] = Field(default_factory=list, description="结果验证标准")
    use_count: int = 0  # 被 router 匹配并执行的次数
    avg_score: float = 0.0  # 滑动平均得分（0.7 历史 + 0.3 本次）
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class AgentState(TypedDict):
    """LangGraph 图状态：各节点读/写部分字段，在节点间传递。"""

    messages: Annotated[Sequence[BaseMessage], operator.add]  # 对话消息（累加）
    task: str  # 用户任务原文
    task_type: str  # router 分类结果，如 travel_planning
    matched_skill: Optional[Skill]  # router 匹配到的完整技能（None=首次或无匹配）
    execution_steps: List[str]  # executor 记录的主要步骤
    result: str  # executor 最终文本（完整行程规划）
    evaluation_score: float  # evaluator LLM 打分 1-10
    evaluation_comment: str  # 评估理由
    should_create_skill: bool  # 是否进入 skill_extractor
    should_update_memory: bool  # 是否进入 memory_updater
    task_count: int  # 累计完成任务数（跨 invoke 传入以触发记忆整合）


class MemoryStore:
    """
    两层持久化记忆（热记忆 + 技能库，渐进披露）。

    目录结构（storage_dir）：
        memory.md           — L1 热记忆，超限时 LLM 压缩
        skills/*.json       — 技能库，router 只读索引，executor 读全文
    """

    def __init__(self, storage_dir: str = "./agent_memory", memory_char_limit: int = 2000):
        self.storage_dir = storage_dir
        self.memory_char_limit = memory_char_limit
        os.makedirs(storage_dir, exist_ok=True)
        self.memory_path = os.path.join(storage_dir, "memory.md")
        self.skills_dir = os.path.join(storage_dir, "skills")
        os.makedirs(self.skills_dir, exist_ok=True)
        self._load_memory()
        self._load_skills()

    def _load_memory(self) -> None:
        """从磁盘加载 L1 热记忆。"""
        if os.path.exists(self.memory_path):
            with open(self.memory_path, encoding="utf-8") as f:
                self.memory = f.read()
        else:
            self.memory = ""

    def _save_memory(self) -> None:
        with open(self.memory_path, "w", encoding="utf-8") as f:
            f.write(self.memory)

    def _load_skills(self) -> None:
        """扫描 skills/*.json 加载到内存字典。"""
        self.skills: Dict[str, Skill] = {}
        for filename in os.listdir(self.skills_dir):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(self.skills_dir, filename), encoding="utf-8") as f:
                skill = Skill(**json.load(f))
            self.skills[skill.name] = skill

    def reload_skills(self) -> None:
        """Demo 打印前刷新，确保磁盘上新写入的技能统计可见。"""
        self._load_skills()

    def get_memory_snapshot(self) -> str:
        """供 router / executor 注入的 L1 快照。"""
        return self.memory or "[尚无持久记忆]"

    def get_skill_index(self) -> List[Dict[str, Any]]:
        """
        渐进披露第一层：仅 name + description + task_type + 统计。

        router 只看索引，不加载 procedure 全文，节省 prompt token。
        """
        return [
            {
                "name": s.name,
                "description": s.description,
                "task_type": s.task_type,
                "use_count": s.use_count,
                "avg_score": round(s.avg_score, 1),
            }
            for s in sorted(self.skills.values(), key=lambda x: x.name)
        ]

    def get_skill(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def find_by_task_type(self, task_type: str) -> Optional[Skill]:
        """按 task_type 兜底匹配（丽江→大理复用 travel_planning 技能）。"""
        for skill in self.skills.values():
            if skill.task_type == task_type:
                return skill
        return None

    def resolve_skill_name(self, raw_name: Optional[str]) -> Optional[Skill]:
        """解析 LLM 返回的技能名：精确 → 去空格 → 过滤 null 字符串。"""
        if not raw_name:
            return None
        raw = str(raw_name).strip()
        if raw.lower() in ("null", "none", ""):
            return None
        if raw in self.skills:
            return self.skills[raw]
        for key, skill in self.skills.items():
            if key.replace(" ", "") == raw.replace(" ", ""):
                return skill
        return None

    def save_skill(self, skill: Skill) -> None:
        """持久化技能到 skills/{name}.json。"""
        skill.updated_at = datetime.now().isoformat()
        self.skills[skill.name] = skill
        path = os.path.join(self.skills_dir, f"{skill.name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(skill.model_dump(), f, indent=2, ensure_ascii=False)

    def update_skill_stats(self, name: str, score: float) -> None:
        """复用技能后更新 use_count 与滑动平均 avg_score。"""
        skill = self.skills.get(name)
        if not skill:
            return
        skill.use_count += 1
        skill.avg_score = 0.7 * skill.avg_score + 0.3 * score
        skill.updated_at = datetime.now().isoformat()
        self.save_skill(skill)

    def add_to_memory(self, entry: str) -> None:
        """追加 L1 记忆；超限则先 consolidate 再写入。"""
        new_memory = f"{self.memory}\n- {entry}".strip()
        if len(new_memory) > self.memory_char_limit:
            self.consolidate_memory()
            new_memory = f"{self.memory}\n- {entry}".strip()
        self.memory = new_memory
        self._save_memory()

    def consolidate_memory(self) -> None:
        """LLM 压缩热记忆，防止 prompt 膨胀。"""
        if not self.memory:
            return
        prompt = (
            f"请将以下记忆整合为不超过 {self.memory_char_limit // 2} 字符的核心要点，"
            f"每行以 '- ' 开头：\n{self.memory}"
        )
        self.memory = LLM.invoke(prompt).content.strip()
        self._save_memory()


# ==============================
# SelfImprovingPattern（自我进化 Agent 核心实现）
# ==============================


class SelfImprovingPattern:
    """
    自我进化 Agent：跨 invoke 持久化技能与热记忆。

    Parameters
    ----------
    storage_dir : 技能与记忆目录
    domain : 领域配置（Demo 默认 TRAVEL_DOMAIN）
    skill_threshold : 创建技能的最低 LLM 评估分（默认 7.0）
    min_steps_for_skill : 创建技能的最少 execution_steps 数
    nudge_interval : 每 N 个任务触发 memory_updater
    use_sqlite_checkpoint : 是否用 SqliteSaver 持久化 LangGraph checkpoint
    """

    def __init__(
        self,
        storage_dir: str = "./agent_memory",
        domain: DomainConfig = TRAVEL_DOMAIN,
        skill_threshold: float = 7.0,
        min_steps_for_skill: int = 3,
        nudge_interval: int = 10,
        use_sqlite_checkpoint: bool = True,
    ):
        self.store = MemoryStore(storage_dir)
        self.domain = domain
        self.skill_threshold = skill_threshold
        self.min_steps_for_skill = min_steps_for_skill
        self.nudge_interval = nudge_interval
        self.use_sqlite_checkpoint = use_sqlite_checkpoint
        self.storage_dir = storage_dir

    def _resolve_matched_skill(
        self, llm_result: Dict[str, Any], task_type: str
    ) -> Optional[Skill]:
        """
        技能匹配：LLM 返回 name 优先，否则按 task_type 兜底。

        例：任务2 大理规划，LLM 可能返回 null，但 task_type=travel_planning
        时仍复用任务1 学到的「国内三日游行程规划」技能。
        """
        skill = self.store.resolve_skill_name(llm_result.get("matched_skill"))
        if skill:
            return skill
        if task_type and task_type != "general":
            return self.store.find_by_task_type(task_type)
        return None

    def _router(self, state: AgentState) -> Dict[str, Any]:
        """
        节点1：任务路由。

        - 技能库为空 → 直接 general，无 matched_skill
        - 否则 LLM 看技能索引 + 热记忆，输出 task_type 与 matched_skill
        - 使用 LLM_STRICT + JSON
        """
        task = state["task"]
        skill_index = self.store.get_skill_index()
        memory_snapshot = self.store.get_memory_snapshot()

        if not skill_index:
            return {"matched_skill": None, "task_type": "general"}

        skill_list = json.dumps(skill_index, ensure_ascii=False, indent=2)
        extra = f"\n补充规则：{self.domain.router_extra_rules}" if self.domain.router_extra_rules else ""

        prompt = f"""
你是任务路由器。请分析用户任务并：
1. 分类 task_type（示例：{self.domain.task_type_examples}）
2. 若已有技能匹配，返回其 name（必须与列表完全一致）；否则返回 null

可用技能：
{skill_list}

持久记忆：
{memory_snapshot}
{extra}

用户任务：{task}

输出 JSON：{{"task_type": "...", "matched_skill": "技能名或null"}}
"""
        response = LLM_STRICT.invoke(prompt)
        parsed = _parse_json(response.content) or json.loads(response.content)
        task_type = parsed.get("task_type", "general")
        matched_skill = self._resolve_matched_skill(parsed, task_type)

        if matched_skill:
            print(f"  ▶ router 匹配技能：{matched_skill.name!r}", flush=True)
        else:
            print("  ▶ router 未匹配到技能", flush=True)

        return {"matched_skill": matched_skill, "task_type": task_type}

    def _executor(self, state: AgentState) -> Dict[str, Any]:
        """
        节点2：执行任务。

        - system = 领域 prompt + 热记忆 +（若有）技能 procedure/pitfalls/verification
        - LLM 生成完整回答 → result（Demo 中为逐日行程）
        - 第二次 LLM 调用从 result 抽取 execution_steps（供评估与抽技能）
        """
        task = state["task"]
        matched_skill = state.get("matched_skill")
        memory_snapshot = self.store.get_memory_snapshot()

        system_prompt = self.domain.executor_system_prompt
        if memory_snapshot and memory_snapshot != "[尚无持久记忆]":
            system_prompt += f"\n\n持久记忆：\n{memory_snapshot}"

        if matched_skill:
            system_prompt += (
                f"\n\n请按以下技能执行：\n"
                f"技能：{matched_skill.name}\n"
                f"步骤：\n" + "\n".join(f"- {s}" for s in matched_skill.procedure) + "\n"
                f"陷阱：\n" + "\n".join(f"- {p}" for p in matched_skill.pitfalls) + "\n"
                f"验证：\n" + "\n".join(f"- {v}" for v in matched_skill.verification)
            )

        response = LLM.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ])

        steps_resp = LLM_STRICT.invoke(
            f'从以下回答提取主要执行步骤，输出 JSON：{{"steps": ["步骤1", ...]}}\n{response.content}'
        )
        steps_parsed = _parse_json(steps_resp.content) or json.loads(steps_resp.content)
        steps = steps_parsed.get("steps", ["direct_response"]) if isinstance(steps_parsed, dict) else ["direct_response"]

        return {
            "messages": [AIMessage(content=response.content)],
            "result": response.content,
            "execution_steps": steps,
        }

    def _evaluator(self, state: AgentState) -> Dict[str, Any]:
        """
        节点3：评估。

        - score / comment：LLM 根据 evaluation_rubric 打分（非规则引擎）
        - should_create_skill：规则判定（高分 + 多步骤 + 未匹配技能 + 同 task_type 尚无技能）
        - 若本次复用了技能，更新 use_count / avg_score
        """
        prompt = f"""
评估任务完成质量（1-10），输出 JSON：{{"score": 数字, "comment": "理由"}}

{self.domain.evaluation_rubric}

任务：{state['task']}
步骤：{json.dumps(state['execution_steps'], ensure_ascii=False)}
结果：{state['result'][:4000]}
"""
        response = LLM_STRICT.invoke(prompt)
        parsed = _parse_json(response.content) or json.loads(response.content)
        score = float(parsed.get("score", 5))
        comment = parsed.get("comment", "")

        matched = state.get("matched_skill")
        if matched:
            self.store.update_skill_stats(matched.name, score)

        task_type = state.get("task_type") or "general"
        already_has_type_skill = (
            task_type != "general" and self.store.find_by_task_type(task_type) is not None
        )
        should_create_skill = (
            score >= self.skill_threshold
            and len(state.get("execution_steps", [])) >= self.min_steps_for_skill
            and matched is None
            and not already_has_type_skill
        )

        task_count = state.get("task_count", 0) + 1
        should_update_memory = task_count % self.nudge_interval == 0

        return {
            "evaluation_score": score,
            "evaluation_comment": comment,
            "should_create_skill": should_create_skill,
            "should_update_memory": should_update_memory,
            "task_count": task_count,
        }

    def _skill_extractor(self, state: AgentState) -> Dict[str, Any]:
        """
        节点4：从成功任务抽取技能 JSON 并写入 skills/。

        仅在 should_create_skill=True 时由条件边进入。
        """
        if not state.get("should_create_skill"):
            return {}

        task_type = state.get("task_type") or "general"
        if task_type != "general" and self.store.find_by_task_type(task_type):
            print(f"  ▶ skill_extractor 跳过：已有 task_type={task_type} 技能", flush=True)
            return {}

        prompt = f"""
从以下成功任务中提取可复用技能（JSON，中文字段）：

{self.domain.skill_extraction_hint}

任务：{state['task']}
任务类型：{task_type}
步骤：{json.dumps(state['execution_steps'], ensure_ascii=False)}
结果：{state['result'][:2000]}
评分：{state['evaluation_score']}

输出 JSON：
{{
  "name": "通用技能名",
  "description": "一行描述",
  "task_type": "{task_type}",
  "procedure": ["步骤1", "..."],
  "pitfalls": ["..."],
  "verification": ["..."]
}}
"""
        response = LLM_STRICT.invoke(prompt)
        skill_data = _parse_json(response.content) or json.loads(response.content)
        if not isinstance(skill_data, dict) or "name" not in skill_data:
            print("  ▶ skill_extractor 解析失败，跳过", flush=True)
            return {}

        skill_data.setdefault("task_type", task_type)
        new_skill = Skill(**skill_data)
        self.store.save_skill(new_skill)
        self.store.add_to_memory(f"学习了新技能：{new_skill.name} - {new_skill.description}")
        print(f"  ▶ skill_extractor 创建技能：{new_skill.name}", flush=True)
        return {}

    def _memory_updater(self, state: AgentState) -> Dict[str, Any]:
        """节点5：每 nudge_interval 个任务压缩 L1 热记忆。"""
        if not state.get("should_update_memory"):
            return {}
        self.store.consolidate_memory()
        self.store.add_to_memory(f"完成了 {state['task_count']} 个任务，已整合记忆")
        return {}

    def _after_evaluator(self, state: AgentState) -> str:
        """评估后路由：create 优先于 memory_update。"""
        if state.get("should_create_skill"):
            return "skill_extractor"
        if state.get("should_update_memory"):
            return "memory_updater"
        return "end"

    def _after_skill_extractor(self, state: AgentState) -> str:
        """抽技能后若仍需整合记忆，继续进入 memory_updater。"""
        if state.get("should_update_memory"):
            return "memory_updater"
        return "end"

    def build_graph(self):
        """组装并 compile LangGraph；可选 SqliteSaver checkpoint。"""
        graph = StateGraph(AgentState)
        graph.add_node("router", self._router)
        graph.add_node("executor", self._executor)
        graph.add_node("evaluator", self._evaluator)
        graph.add_node("skill_extractor", self._skill_extractor)
        graph.add_node("memory_updater", self._memory_updater)

        graph.set_entry_point("router")
        graph.add_edge("router", "executor")
        graph.add_edge("executor", "evaluator")
        graph.add_conditional_edges(
            "evaluator",
            self._after_evaluator,
            {
                "skill_extractor": "skill_extractor",
                "memory_updater": "memory_updater",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "skill_extractor",
            self._after_skill_extractor,
            {"memory_updater": "memory_updater", "end": END},
        )
        graph.add_edge("memory_updater", END)

        if self.use_sqlite_checkpoint:
            db_path = os.path.join(self.storage_dir, "checkpoints.db")
            conn = sqlite3.connect(db_path, check_same_thread=False)
            return graph.compile(checkpointer=SqliteSaver(conn))
        return graph.compile()

    def run(self, task: str, task_count: int = 0, thread_id: str = "self-improving-1") -> dict:
        """
        便捷入口：执行单条任务并返回最终 AgentState。

        task_count 需跨多次 run 手动递增，用于触发 memory_updater。
        """
        graph = self.build_graph()
        initial: AgentState = {
            "task": task,
            "messages": [],
            "task_type": "",
            "matched_skill": None,
            "execution_steps": [],
            "result": "",
            "evaluation_score": 0.0,
            "evaluation_comment": "",
            "should_create_skill": False,
            "should_update_memory": False,
            "task_count": task_count,
        }
        config = {"configurable": {"thread_id": thread_id}}
        return graph.invoke(initial, config)


def create_self_improving_agent(
    storage_dir: str = "./agent_memory",
    domain: DomainConfig = TRAVEL_DOMAIN,
) -> Any:
    """兼容旧 API：返回 compiled graph，供 agent.invoke(...) 调用。"""
    pattern = SelfImprovingPattern(storage_dir=storage_dir, domain=domain)
    return pattern.build_graph()


# 兼容旧全局变量（Demo / 测试脚本可能引用 memory_store.skills）
memory_store: MemoryStore


# ==============================
# Demo：旅游行程规划（默认 TRAVEL_DOMAIN）
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


def _print_task_outcome(result: dict, *, show_steps: bool = False) -> None:
    """打印 executor 的 result（完整行程）及评估摘要。"""
    body = (result.get("result") or "").strip()
    if body:
        print("\n--- 规划结果 ---")
        print(body)
    else:
        print("\n（无 result 文本）")

    if show_steps:
        steps = result.get("execution_steps") or []
        if steps:
            print("\n--- 执行步骤 ---")
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step}")

    print(f"\n评估得分：{result.get('evaluation_score')}")
    print(f"评估理由：{result.get('evaluation_comment')}")
    if result.get("should_create_skill"):
        print("创建技能：True")
    matched = result.get("matched_skill")
    label = "无"
    if matched:
        label = matched.name
    elif result.get("task_count", 0) <= 1 and not matched:
        label = "无（首次）"
    print(f"匹配技能：{label}")


def _empty_invoke_state(task: str, task_count: int = 0) -> dict:
    """构造一次 agent.invoke 的初始状态（LangGraph 必填字段）。"""
    return {
        "task": task.strip(),
        "messages": [],
        "execution_steps": [],
        "result": "",
        "evaluation_score": 0.0,
        "evaluation_comment": "",
        "should_create_skill": False,
        "should_update_memory": False,
        "task_count": task_count,
    }


if __name__ == "__main__":
    storage_dir = "./my_agent_memory"

    # TRAVEL_DEMO_FRESH=1 清空旧技能，重新演示「学习 → 复用」
    if os.getenv("TRAVEL_DEMO_FRESH", "").strip().lower() in ("1", "true", "yes"):
        import shutil

        skills_dir = os.path.join(storage_dir, "skills")
        if os.path.isdir(skills_dir):
            shutil.rmtree(skills_dir)
        print("（已清空旧技能目录，重新演示）\n")

    pattern = SelfImprovingPattern(storage_dir=storage_dir, domain=TRAVEL_DOMAIN)
    memory_store = pattern.store
    agent = pattern.build_graph()
    config = {"configurable": {"thread_id": "travel-demo-1"}}

    print("=== 任务1：丽江 3 日游（首次，将学习技能）===")
    r1 = agent.invoke(_empty_invoke_state(TASK1_LIJIANG), config)
    _print_task_outcome(r1)

    print("\n=== 任务2：大理 3 日游（复用已学技能）===")
    r2 = agent.invoke(_empty_invoke_state(TASK2_DALI, task_count=r1["task_count"]), config)
    _print_task_outcome(r2)

    memory_store.reload_skills()

    print("\n=== 技能库 ===")
    for name, skill in memory_store.skills.items():
        print(f"- {name}: {skill.description}")
        print(f"  task_type={skill.task_type}  uses={skill.use_count}  avg={skill.avg_score:.1f}")


#  程序运行结果
# == 任务1：丽江 3 日游（首次，将学习技能）===
#   ▶ skill_extractor 创建技能：国内短途旅游行程规划
#
# --- 规划结果 ---
# ## 1️⃣ 思考过程与制定原则
#
# | 需求 | 关键点 | 规划思路 |
# |------|--------|----------|
# | **出发日期** | 2026‑05‑01（春季，气温 12‑22℃） | 选取天气舒适、景区人流相对平稳的线路。 |
# | **时长** | 3 天 2 晚 | 以“古城 → 轻度徒步 → 本地美食”为主线，避免连日高强度爬山。 |
# | **预算** | 人均 2500 元（含住宿、餐饮、门票、市内交通） | 预算≈5000 元/对。<br>‑ 住宿 2 晚≈800 元<br>‑ 餐饮 3 天≈1000 元<br>‑ 门票/景区≈800 元<br>‑ 市内交通≈200 元<br>‑ 预留 200 元弹性（小费、纪念品） |
# | **偏好** | 古城漫步、轻度徒步、本地美食、节奏悠闲 | ① 第1天以古城为核心；<br>② 第2天安排玉龙雪山轻度徒步（蓝月谷+冰川公园，走走走走不爬到山顶）；<br>③ 第3天走访束河古镇、黑龙潭，兼顾美食。 |
# | **交通** | 市内公交、滴滴/出租、景区环保车 | 采用公共交通+少量打车，控制费用。 |
# | **住宿** | 推荐古城内或古城北侧（靠近束河） | 方便步行古城、夜间灯光、早起前往景区。 |
#
#...................
# 评估得分：10.0
# 评估理由：答案提供了完整的逐日行程（上午/下午/晚上），涵盖古城漫步、轻度徒步和本地美食，行程节奏舒适；明确推荐住宿区域并列出具体客栈选项；给出详细的费用拆分，严格控制在人均 2500 元预算内；提供了交通、门票预约、装备准备等实用注意事项；整体结构清晰、信息完整，超出预期要求。
# 创建技能：True
# 匹配技能：无（首次）
#
# === 任务2：大理 3 日游（复用已学技能）===
#   ▶ router 匹配技能：'国内短途旅游行程规划'
#
# --- 规划结果 ---
# **一、出行约束收集（思考过程）**
#
# | 项目 | 内容 |
# |------|------|
# | 出发日期 | 2026‑06‑10（周四） |
# | 天数/住宿 | 3 天 2 晚（6‑30 Jun） |
# | 人数 | 2 位朋友 |
# | 人均预算上限 | 2 200 元 |
# | 偏好 | ① 洱海骑行 ② 古镇闲逛 ③ 白族特色餐饮 |
# | 特殊需求 | 无（普通青年） |
#
# **二、核心景点划分**
#
# | 区块 | 主要景点（占位） | 预计游玩时长 |
# |------|----------------|--------------|
# | **古城文化区** | 大理古城（四方街、洋人街） | 半天 |
# | **洱海骑行区** | 环洱海自行车道（大理‑双廊‑喜洲‑大理） | 1‑1.5 天（分两段） |
# | **白族古镇区** | 喜洲古镇、周城古镇（可选） | 半天‑1 天 |
# | **自然轻度区** | 苍山索道+小苍山步道（可选） | 2 h（索道）+1 h步道 |
#
#...................
#
# **祝两位朋友大理之旅轻松愉快，骑行洱海、漫步古镇、品味白族美食，留下难忘的回忆！**
#
# 评估得分：10.0
# 评估理由：答案提供了完整的逐日行程、交通方式、住宿推荐及详细预算，涵盖了所有用户约束（出发日期、天数、预算、偏好），并列出执行步骤、注意事项和检查清单，信息齐全且逻辑顺畅，超出预期。
# 匹配技能：国内短途旅游行程规划
#
# === 技能库 ===
# - 国内短途旅游行程规划: 针对任意国内短途旅行，收集约束、确定住宿区域、制定逐日行程、预算拆分并给出注意事项的完整流程。
#   task_type=travel_planning  uses=1  avg=3.0

