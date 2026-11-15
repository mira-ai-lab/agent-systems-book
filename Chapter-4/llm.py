"""Model client and invocation helpers for Chapter 4.

This module centralizes environment loading, chat-model construction, response
text extraction, and JSON parsing. Task decomposition and dependency ordering
remain in the chapter notebook so the teaching flow stays visible.

:author: zhanghong, liuhui
:date: 2026-11-15
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)


def model_configured() -> bool:
    """Return whether an OpenAI-compatible model credential is configured."""
    return bool(
        (os.getenv("OPENAI_API_KEY") or "").strip()
        or (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    )


def create_llm(
    *,
    temperature: float = 0,
    model: str | None = None,
):
    """Create the chat model used by the Chapter 4 examples.

    OpenAI is preferred when ``OPENAI_API_KEY`` is configured. Otherwise, the
    function uses ``DASHSCOPE_API_KEY`` with the DashScope OpenAI-compatible
    endpoint. TLS verification is enabled by default.

    Args:
        temperature: Model sampling temperature.
        model: Explicit model name. When omitted, the model is read from the
            matching environment variable.

    Returns:
        A configured LangChain ``ChatOpenAI`` instance.

    Raises:
        RuntimeError: If neither supported API key is configured.
    """
    from langchain_openai import ChatOpenAI

    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    dashscope_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not openai_key and not dashscope_key:
        raise RuntimeError(
            "Configure OPENAI_API_KEY or DASHSCOPE_API_KEY in the repository "
            "root .env file."
        )

    use_dashscope = not openai_key
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
        "api_key": openai_key or dashscope_key,
    }
    base_url = (
        (os.getenv("DASHSCOPE_CHAT_BASE_URL") or "").strip()
        if use_dashscope
        else (os.getenv("OPENAI_BASE_URL") or "").strip()
    )
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    if use_dashscope:
        kwargs["extra_body"] = {
            "enable_thinking": os.getenv(
                "DASHSCOPE_ENABLE_THINKING",
                "0",
            ).strip().lower()
            in {"1", "true", "yes", "on"}
        }

    ssl_verify = os.getenv("OPENAI_SSL_VERIFY", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if not ssl_verify:
        import httpx

        kwargs["http_client"] = httpx.Client(verify=False)
        kwargs["http_async_client"] = httpx.AsyncClient(verify=False)
    return ChatOpenAI(**kwargs)


def extract_response_text(response: Any) -> str:
    """Extract plain text from a LangChain chat response."""
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()
    return str(content or "").strip()


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a JSON object from plain, fenced, or lightly wrapped model text."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("The model returned an empty response; JSON parsing failed.")

    candidates = [raw]
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    braced = re.search(r"\{[\s\S]*\}", raw)
    if braced:
        candidates.append(braced.group(0))

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not parse a JSON object from model output: {last_error}")


async def invoke_chat(
    llm: Any,
    prompt: str,
    *,
    system_prompt: str | None = None,
    json_mode: bool = False,
) -> str:
    """Invoke a chat model and return response text.

    Args:
        llm: A LangChain-compatible chat model.
        prompt: User message to send.
        system_prompt: Optional system message.
        json_mode: Whether to request a JSON object response from the provider.

    Returns:
        The response text with surrounding whitespace removed.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages: list[Any] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    kwargs: dict[str, Any] = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = await llm.ainvoke(messages, **kwargs)
    text = extract_response_text(response)
    if not text:
        raise ValueError("The model returned an empty response.")
    return text


async def invoke_json(
    llm: Any,
    prompt: str,
    *,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Invoke a chat model and parse its response as a JSON object."""
    text = await invoke_chat(
        llm,
        prompt,
        system_prompt=system_prompt,
        json_mode=True,
    )
    return parse_json_response(text)
