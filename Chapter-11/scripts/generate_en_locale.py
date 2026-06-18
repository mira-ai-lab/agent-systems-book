"""Generate English locale files from zh.json structure (Phase 17)."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ROUTER_ZH = ROOT / "agent_framework" / "router" / "prompts" / "locales" / "zh.json"
ROUTER_EN = ROOT / "agent_framework" / "router" / "prompts" / "locales" / "en.json"
PLATFORM_ZH = ROOT / "agent_framework" / "prompts" / "locales" / "zh.json"
PLATFORM_EN = ROOT / "agent_framework" / "prompts" / "locales" / "en.json"

ROUTER_STRING_MAP = {
    "知识支持": "Knowledge Support",
    "包含的知识点有": "Knowledge points include",
    "、": ", ",
    "无": "None",
    "整体目标：": "Overall Goal:",
    "子任务：": "Subtasks:",
    "名称:{name}\n定义:{description}\n能力列表:{skills}\n": "Name:{name}\nDescription:{description}\nSkills:{skills}\n",
    "领域标识:{name}\n显示名称:{display_name}\nAgent能力:{agents}\n": "Domain:{name}\nDisplay Name:{display_name}\nAgent Capabilities:{agents}\n",
}

PLATFORM_STRING_MAP = {
    "你是多智能体编排中枢，负责理解用户目标、拆解子任务并协调子智能体完成请求。": (
        "You are a multi-agent orchestration hub. Understand user goals, decompose subtasks, "
        "and coordinate sub-agents to fulfill requests."
    ),
    "请根据用户原始请求的范围，综合子任务执行结果生成简洁、准确的最终回复。不要添加用户未询问的内容。": (
        "Based on the user's original request, synthesize subtask results into a concise, accurate reply. "
        "Do not add information the user did not ask for."
    ),
    "从对话中提取与任务相关的关键事实，以 JSON 列表返回。": (
        "Extract task-relevant key facts from the conversation and return them as a JSON list."
    ),
    "你是任务依赖分析专家。根据子任务列表与智能体能力，输出 JSON 格式的 depends_on 映射。": (
        "You are a task dependency analyst. Output a JSON depends_on map from subtasks and agent capabilities."
    ),
    "子任务列表：{subtasks}\n可用智能体：{agents}\n请输出 depends_on JSON。": (
        "Subtasks: {subtasks}\nAvailable agents: {agents}\nOutput depends_on JSON."
    ),
    "根据子任务描述选择最合适的子智能体。子任务：{task_description}": (
        "Select the best sub-agent for the subtask. Subtask: {task_description}"
    ),
    "你是 Supervisor，通过 handoff 将子任务分派给可用子智能体，汇总结果后回复用户。handoff 时给出完整、独立的子任务指令。": (
        "You are a Supervisor. Hand off subtasks to available sub-agents and summarize results for the user. "
        "Each handoff must include a complete, self-contained subtask instruction."
    ),
    "📋 最终规划": "Final Plan",
    "📋 最终回复": "Final Reply",
    "单任务查询，直接使用子智能体回复（跳过聚合 LLM）": (
        "Single-task query: use the sub-agent reply directly (skip aggregation LLM)."
    ),
}


def _translate_router_prompt(text: str, key_path: str) -> str:
    if key_path.endswith("classification.prompt_base"):
        return (
            "Role: You are a precise classification engine. Follow the rules strictly.\n\n"
            "Agent Classification: Build an empty list []. Classify the user input. "
            "Available agents: {agent_names}. If no match, output "
            '[{{"name": "other", "score": 1}}].\n\n'
            "Step 1: Match user input against each agent's capabilities.\n"
            "Step 2: Generalize by agent name/description if step 1 is empty.\n"
            "Step 3: Default to other if still empty.\n\n"
            "Notes:\n{note}\n\nAgent catalog:\n{agent_catalog}\n"
            "Return ONLY a JSON list of {{name, score}}.\n\nUser input:\n{query}\n"
        )
    if key_path.endswith("extraction.single"):
        return (
            "Role: Extract core event phrases from the user query.\n"
            "Return ONLY a JSON list of concise event strings.\n\nquery:{query}\nreturn:\n"
        )
    if key_path.endswith("extraction.multi"):
        return (
            "Role: Analyze history and extract events from the query.\n"
            "Return [turn_id, rewrite_query, [events...]].\n\n"
            "Notes:{note}\nhistory:{history}\nquery:{query}\nreturn:\n"
        )
    if key_path.endswith("history_prompt"):
        return (
            "Role: Judge whether conversation history is relevant to the current query.\n"
            "Output only 0 (not relevant) or 1 (relevant).\n\nHistory:\n{history}\nQuery:\n{query}\n"
        )
    if key_path.endswith("task_decomposition_prompt"):
        return (
            "Role: Decompose the user input into subtasks matching the agent team.\n"
            "Background:\n    {}\nAgent team:\n    {}\nUser input:\n    {}\n"
            "Output overall goal on one line, then subtasks each starting with '-'.\n"
        )
    if key_path.endswith("domain_classification.prompt_base"):
        return (
            "Role: Classify which business domain handles the user input.\n"
            "Available domains: {domain_names}. Output a JSON list of {{name, score}}.\n\n"
            "Notes:\n{note}\n\nDomains:\n{domain_catalog}\n\nUser input:\n{query}\n"
        )
    if "build_instruction" in key_path and key_path.endswith("system"):
        return (
            "You build executable instructions for sub-agents in a multi-agent system.\n"
            "Return only the rebuilt task instruction.\n"
        )
    if "build_instruction" in key_path and key_path.endswith("user_template"):
        return (
            "\ninit_task: {init_task}\ntarget_agent: {target_agent}\n"
            "agent_skill: {agent_skill}\nprevious_step_info: {previous_step_info}\n"
        )
    if "prompt_rewrite" in key_path and key_path.endswith("system"):
        return (
            "You rewrite multi-turn dialogue into a self-contained user request (new_query).\n"
            "Output new_query only.\n"
        )
    if "prompt_rewrite" in key_path and key_path.endswith("user_template"):
        return "\nHistory:\n{history}\n\nCurrent query:\n{query}\n"
    if key_path.endswith("task.summary.round_info"):
        return (
            "Round {idx}:\n- Instruction to {agent_name}: {agent_query}\n"
            "- Response from {agent_name}: {agent_response}\n"
        )
    if key_path.endswith("task.summary.prompt") or key_path.endswith("task.stage.prompt"):
        return text  # keep zh for long prompts if not mapped; use English stub below
    if key_path.endswith("task.stage.no_previous_summary"):
        return "No previous stage summary"
    if key_path.endswith("task.stage.all_steps_completed"):
        return "All steps completed; no next subtask description"
    if key_path.endswith("task.stage.step_level_context_desc"):
        return "Subtask: {step_desc}\n"
    if key_path.endswith("task.stage.step_level_context_detail"):
        return "- Instruction to {agent_name}: {query}\n- Response from {agent_name}: {response}\n"
    for zh, en in ROUTER_STRING_MAP.items():
        if text == zh:
            return en
    return text


def _walk_translate(obj, key_path: str = "") -> object:
    if isinstance(obj, dict):
        return {k: _walk_translate(v, f"{key_path}.{k}" if key_path else k) for k, v in obj.items()}
    if isinstance(obj, str):
        return _translate_router_prompt(obj, key_path)
    return obj


def _walk_platform_translate(obj) -> object:
    if isinstance(obj, dict):
        return {k: _walk_platform_translate(v) for k, v in obj.items()}
    if isinstance(obj, str):
        return PLATFORM_STRING_MAP.get(obj, obj)
    return obj


def main() -> None:
    router_zh = json.loads(ROUTER_ZH.read_text(encoding="utf-8"))
    router_en = _walk_translate(deepcopy(router_zh))
    ROUTER_EN.write_text(json.dumps(router_en, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    platform_zh = json.loads(PLATFORM_ZH.read_text(encoding="utf-8"))
    platform_en = _walk_platform_translate(deepcopy(platform_zh))
    # English decomposition prompt stub with same placeholders
    dp = platform_en["domain_prompts"]
    dp["decomposition_prompt"] = (
        "Decompose the user goal into subtasks for the agent team.\n"
        "Background: {background_info}\nAgent team: {agent_team}\nUser input: {user_input}\n"
        "Output # Goal and bullet subtasks."
    )
    dp["memory_aggregation_instruction"] = (
        "Synthesize subtask results to answer the user's original request only. "
        "Do not add unsolicited information."
    )
    PLATFORM_EN.write_text(json.dumps(platform_en, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", ROUTER_EN)
    print("wrote", PLATFORM_EN)


if __name__ == "__main__":
    main()
