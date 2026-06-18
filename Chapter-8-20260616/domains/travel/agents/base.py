"""子智能体公共构建逻辑（兼容层，实现已迁至 agent_framework.infra.agent_runtime）。"""

from __future__ import annotations

from agent_framework.infra.agent_runtime import build_agent, configure_agent_llm, reset_agent_llm

__all__ = ["build_agent", "configure_agent_llm", "reset_agent_llm"]
