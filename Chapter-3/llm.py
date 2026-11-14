"""Model client and structured-output helpers.

This module centralizes model environment configuration and JSON response
parsing so that memory extraction, merging, and other business modules do not
duplicate model invocation details.

:author: zhanghong, liuhui
:date: 2026-11-14
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAPTER3 = PROJECT_ROOT / "Chapter-3"
ENV_FILE = PROJECT_ROOT / ".env"


def load_project_env() -> Path:
    """Load the project-level .env file and return its path."""
    if not ENV_FILE.is_file():
        raise FileNotFoundError(f"Project environment file not found: {ENV_FILE}")
    load_dotenv(ENV_FILE)
    return ENV_FILE


def create_llm():
    """Create the chat model used by the examples from the project environment.

    OpenAI is used when ``OPENAI_API_KEY`` is configured. When it is absent
    and ``DASHSCOPE_API_KEY`` is available, the function falls back to the
    OpenAI-compatible DashScope endpoint. Thinking mode is disabled by default
    for DashScope models.
    """
    from langchain_openai import ChatOpenAI
    import httpx

    load_project_env()
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    dashscope_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    use_dashscope = not openai_key and bool(dashscope_key)

    kwargs: dict[str, Any] = {
        "model": (
            os.getenv("DASHSCOPE_CHAT_MODEL", "qwen3.7-plus")
            if use_dashscope
            else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        ),
        "temperature": 0,
        "api_key": openai_key or dashscope_key or None,
    }
    base_url = (
        (os.getenv("DASHSCOPE_CHAT_BASE_URL") or "").strip()
        if use_dashscope
        else (os.getenv("OPENAI_BASE_URL") or "").strip()
    )
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    if use_dashscope:
        enable_thinking = os.getenv(
            "DASHSCOPE_ENABLE_THINKING",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        kwargs["extra_body"] = {"enable_thinking": enable_thinking}
    if os.getenv("OPENAI_SSL_VERIFY", "1").strip().lower() in ("0", "false", "no", "off"):
        kwargs["http_client"] = httpx.Client(verify=False)
    return ChatOpenAI(**kwargs)


def extract_response_text(response: Any) -> str:
    """Extract plain text from a LangChain response.

    String responses, empty content, and list responses composed of multiple
    text blocks are all supported.
    """
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
    """Parse a JSON object returned by the model.

    Pure JSON, fenced Markdown JSON blocks, and JSON mixed with explanatory
    text are supported.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("The model returned an empty response; JSON parsing failed")

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
    raise ValueError(f"Could not parse JSON from the model output: {last_error}")


def chat(llm: Any, prompt: str) -> dict[str, Any]:
    """Send one chat request and parse its JSON object response.

    The call explicitly sets ``response_format={"type": "json_object"}`` so
    that the model service constrains the response body to JSON. This function
    only extracts the text and parses the object; it does not rely on prompt
    retries.
    """
    response = llm.invoke(
        prompt,
        response_format={"type": "json_object"},
    )
    return parse_json_response(extract_response_text(response))
