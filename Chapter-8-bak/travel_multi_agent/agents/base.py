"""子智能体公共构建逻辑。"""

from __future__ import annotations

from typing import Any, Sequence

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from travel_multi_agent.config import create_llm


def build_agent(tools: Sequence[Any], system_prompt: str) -> Any:
    """用统一 LLM 配置创建 LangChain Agent。"""
    return create_agent(
        create_llm(),
        tools=list(tools),
        system_prompt=system_prompt,
        checkpointer=MemorySaver(),
    )
