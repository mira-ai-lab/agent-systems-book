"""模型客户端与结构化输出辅助函数。

更新时间：2026-11-12

本模块集中处理模型环境配置和 JSON 响应解析，避免记忆抽取、记忆合并等
业务模块重复实现模型调用细节。
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
    """加载项目根目录的 .env，并返回其路径。"""
    if not ENV_FILE.is_file():
        raise FileNotFoundError(f"未找到项目根目录环境文件: {ENV_FILE}")
    load_dotenv(ENV_FILE)
    return ENV_FILE


def default_llm():
    """加载项目根目录环境变量并创建课堂演示使用的对话模型。

    优先使用 OpenAI 或 OpenRouter 配置；当两者均未配置且存在
    ``DASHSCOPE_API_KEY`` 时，自动切换到阿里云百炼的 OpenAI 兼容接口。
    """
    from langchain_openai import ChatOpenAI
    import httpx

    load_project_env()
    openai_key = (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY") or "").strip()
    dashscope_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    use_dashscope = not openai_key and bool(dashscope_key)

    kwargs: dict[str, Any] = {
        "model": (
            os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus")
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
    if os.getenv("OPENAI_SSL_VERIFY", "1").strip().lower() in ("0", "false", "no", "off"):
        kwargs["http_client"] = httpx.Client(verify=False)
    return ChatOpenAI(**kwargs)


def response_text(response: Any) -> str:
    """提取 LangChain 响应中的纯文本内容。

    同时兼容字符串响应、空内容以及由多个文本块组成的列表响应。
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
    """解析模型返回的 JSON 对象。

    支持纯 JSON、Markdown JSON 代码块，以及夹带解释文字的响应。
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("模型返回为空，无法解析 JSON")

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
    raise ValueError(f"无法从模型输出解析 JSON: {last_error}")


def call_structured_llm(llm: Any, prompt: str) -> dict[str, Any]:
    """通过模型的 JSON 输出模式获取结构化响应。

    调用时显式设置 ``response_format={"type": "json_object"}``，由模型服务
    保证响应正文为 JSON；函数只负责提取文本和解析对象，不依赖提示词重试。
    """
    response = llm.invoke(
        prompt,
        response_format={"type": "json_object"},
    )
    return parse_json_response(response_text(response))
