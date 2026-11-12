"""Chapter-5 模型客户端。

更新时间：2026-11-13

本模块负责加载模型相关环境变量并创建对话模型。
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
    """创建 DashScope 或 OpenAI 兼容的对话模型。

    Args:
        temperature: 模型采样温度。
        model: 显式指定的模型名称；未提供时从环境变量读取。
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
                os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus")
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
