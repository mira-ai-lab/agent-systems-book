"""LangChain ChatOpenAI → textgrad EngineLM adapter."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def create_textgrad_engine(llm: ChatOpenAI):
    """Return a textgrad ``EngineLM`` backed by LangChain ChatOpenAI."""
    from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad

    require_textgrad()
    from textgrad import EngineLM

    class Chapter11TextGradEngine(EngineLM):
        def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            response = llm.invoke(messages, **kwargs)
            content = response.content
            if isinstance(content, str):
                return content
            return str(content)

        def __call__(self, prompt: str, **kwargs) -> str:
            return self.generate(prompt, **kwargs)

    return Chapter11TextGradEngine()
