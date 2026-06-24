"""system_prompt Variable 适配与读写。"""

from __future__ import annotations

from agent_framework.optimization.agents.runtime import extract_agent_system_prompt


AGENT_SYSTEM_PROMPT_ROLE = "travel sub-agent system prompt template"


def agent_system_prompt_variable(prompt_template: str, *, agent_name: str = "FlightAgent"):
    """将 Agent system_prompt 模板包装为 requires_grad=True 的 Variable。"""
    from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad

    _, Variable, _, _ = require_textgrad()
    cleaned = extract_agent_system_prompt(prompt_template, agent_name=agent_name)
    return Variable(
        cleaned,
        requires_grad=True,
        role_description=f"{agent_name} {AGENT_SYSTEM_PROMPT_ROLE}",
    )


def read_agent_system_prompt_value(prompt_var, *, agent_name: str = "FlightAgent") -> str:
    """从 Variable 读回并校验 prompt 模板。"""
    return extract_agent_system_prompt(str(getattr(prompt_var, "value", prompt_var)), agent_name=agent_name)
