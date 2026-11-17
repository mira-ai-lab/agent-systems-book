"""Shared model client and invocation helpers for the book.

This module is the repository-level source of model configuration, synchronous
and asynchronous JSON calls, text extraction, JSON parsing, and structured
output invocation.

:author: liuhui
:date: 2026-11-17
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)


def load_project_env() -> Path:
    """Load the repository-level environment file and return its path."""
    if not ENV_FILE.is_file():
        raise FileNotFoundError(f"Project environment file not found: {ENV_FILE}")
    load_dotenv(ENV_FILE)
    return ENV_FILE


def model_configured() -> bool:
    """Return whether an OpenAI-compatible model credential is configured."""
    return bool(
        (os.getenv("OPENAI_API_KEY") or "").strip()
        or (os.getenv("OPENROUTER_API_KEY") or "").strip()
        or (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    )


def create_llm(
    *,
    temperature: float = 0,
    model: str | None = None,
):
    """Create a shared OpenAI- or DashScope-compatible chat model.

    Args:
        temperature: Model sampling temperature.
        model: Optional explicit model name.

    Returns:
        A configured LangChain ``ChatOpenAI`` instance.

    Raises:
        RuntimeError: If no supported model API key is configured.
    """
    from langchain_openai import ChatOpenAI

    openai_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or ""
    ).strip()
    dashscope_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not openai_key and not dashscope_key:
        raise RuntimeError(
            "Configure OPENAI_API_KEY, OPENROUTER_API_KEY, or "
            "DASHSCOPE_API_KEY in the repository root .env file."
        )

    use_dashscope = not openai_key
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

    if os.getenv("OPENAI_SSL_VERIFY", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        import httpx

        kwargs["http_client"] = httpx.Client(verify=False)
        kwargs["http_async_client"] = httpx.AsyncClient(verify=False)
    return ChatOpenAI(**kwargs)


def extract_response_text(response: Any) -> str:
    """Extract plain text from a LangChain response."""
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


def parse_json_response(text: str) -> Any:
    """Parse one JSON object or array from model output."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("The model returned an empty response; JSON parsing failed.")

    candidates = [raw]
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    decoder = json.JSONDecoder()
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            value, _ = decoder.raw_decode(candidate)
            if isinstance(value, (dict, list)):
                return value
        except json.JSONDecodeError as exc:
            last_error = exc

    for index in sorted(
        position
        for token in ("{", "[")
        if (position := raw.find(token)) >= 0
    ):
        try:
            value, _ = decoder.raw_decode(raw[index:])
            if isinstance(value, (dict, list)):
                return value
        except json.JSONDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not parse JSON from model output: {last_error}")


def invoke_json_sync(
    llm: Any,
    prompt: str,
    *,
    system_prompt: str | None = None,
    require_object: bool = False,
) -> Any:
    """Synchronously invoke the model and parse a JSON object or array.

    Args:
        llm: LangChain-compatible chat model.
        prompt: User prompt.
        system_prompt: Optional system instruction.
        require_object: Require a JSON object instead of accepting an array.

    Returns:
        The decoded JSON object or array.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages: list[Any] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))
    response = llm.invoke(
        messages,
        response_format={"type": "json_object"},
    )
    value = parse_json_response(extract_response_text(response))
    if require_object and not isinstance(value, dict):
        raise ValueError("Expected a JSON object response")
    return value


async def invoke_chat(
    llm: Any,
    prompt: str,
    *,
    system_prompt: str | None = None,
    json_mode: bool = False,
) -> str:
    """Invoke one asynchronous chat request and return response text."""
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
    require_object: bool = False,
) -> Any:
    """Invoke one asynchronous request and parse a JSON object or array."""
    text = await invoke_chat(
        llm,
        prompt,
        system_prompt=system_prompt,
        json_mode=True,
    )
    value = parse_json_response(text)
    if require_object and not isinstance(value, dict):
        raise ValueError("Expected a JSON object response")
    return value


async def invoke_structured(
    llm: Any,
    prompt: str,
    *,
    schema: Any,
) -> Any:
    """Invoke one asynchronous structured-output request."""
    structured_llm = llm.with_structured_output(schema)
    return await structured_llm.ainvoke(prompt)
