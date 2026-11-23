"""Adapter that turns a LangChain chat model into a textgrad ``EngineLM``.

:author: zhanghong, liuhui
:date: 2026-11-23
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def create_textgrad_engine(llm: ChatOpenAI):
    """Return a textgrad ``EngineLM`` backed by a LangChain chat model."""
    from textgrad import EngineLM

    class Chapter10TextGradEngine(EngineLM):
        def generate(
            self,
            prompt: str,
            system_prompt: str | None = None,
            **kwargs,
        ) -> str:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            response = llm.invoke(messages, **kwargs)
            content = response.content
            return content if isinstance(content, str) else str(content)

        def __call__(self, prompt: str, **kwargs) -> str:
            return self.generate(prompt, **kwargs)

    return Chapter10TextGradEngine()
