"""子 Agent 运行时：共享 LLM 注入与 LangChain Agent 构建（领域无关）。"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from agent_framework.config import create_llm

_shared_llm: Any = None


def configure_agent_llm(llm: Any) -> None:
    """由编排器注入共享 LLM，子 Agent 创建时复用（避免重复 create_llm）。"""
    global _shared_llm
    _shared_llm = llm


def reset_agent_llm() -> None:
    """测试用：清除共享 LLM。"""
    global _shared_llm
    _shared_llm = None


def build_agent(
    tools: Sequence[Any],
    system_prompt: str,
    llm: Optional[Any] = None,
) -> Any:
    """创建 LangChain Agent；优先使用编排器注入的 LLM。"""
    resolved = llm or _shared_llm or create_llm()
    return create_agent(
        resolved,
        tools=list(tools),
        system_prompt=system_prompt,
        checkpointer=MemorySaver(),
    )
