"""Model client for Chapter 5.

This module loads model-related environment variables and creates the chat
model used by the hotel recommendation agent.

:author: zhanghong, liuhui
:date: 2026-11-15
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_FILE)


def create_llm(
    *,
    temperature: float = 0,
    model: str | None = None,
):
    """Create a DashScope- or OpenAI-compatible chat model.

    The model is loaded from the project-level environment configuration.
    OpenAI is used when ``OPENAI_API_KEY`` is configured. When it is absent
    and ``DASHSCOPE_API_KEY`` is available, the function falls back to the
    OpenAI-compatible DashScope endpoint.

    Args:
        temperature: Model sampling temperature.
        model: Explicit model name. When omitted, the name is read from the
            environment.

    Returns:
        A configured LangChain ``ChatOpenAI`` instance.
    """
    from langchain_openai import ChatOpenAI

    openai_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or ""
    ).strip()
    dashscope_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    use_dashscope = not openai_key and bool(dashscope_key)

    kwargs: dict[str, Any] = {
        "model": (
            model
            or (
                os.getenv("DASHSCOPE_CHAT_MODEL", "qwen3.7-plus")
                if use_dashscope
                else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            )
        ),
        "temperature": temperature,
        "api_key": openai_key or dashscope_key or None,
    }
    base_url = (
        (os.getenv("DASHSCOPE_CHAT_BASE_URL") or "").strip()
        if use_dashscope
        else (os.getenv("OPENAI_BASE_URL") or "").strip()
    )
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    if os.getenv("OPENAI_SSL_VERIFY", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        import httpx

        kwargs["http_client"] = httpx.Client(verify=False)
    return ChatOpenAI(**kwargs)
