"""项目配置：路径、环境变量、LLM 工厂。"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

# travel_multi_agent/config.py → Chapter-8/（书稿目录，非包名）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOK_ROOT = PROJECT_ROOT.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_memory"
GRAPH_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "orchestration" / "fixed_graph" / "output"
)
TRACES_DIR = PROJECT_ROOT / "traces"

_dotenv_loaded = False


def load_project_dotenv(*, override: bool = False) -> None:
    """依次尝试 Chapter-8/.env 与书仓库根 .env（幂等）。"""
    global _dotenv_loaded
    load_dotenv(PROJECT_ROOT / ".env", override=override)
    load_dotenv(BOOK_ROOT / ".env", override=override)
    _dotenv_loaded = True


def create_llm(*, temperature: float = 0):
    """创建统一的 LLM 客户端（全项目唯一入口）。"""
    from langchain_core.messages import BaseMessage
    from langchain_openai import ChatOpenAI

    if not _dotenv_loaded:
        load_project_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")

    base_url = os.getenv(
        "DASHSCOPE_CHAT_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).rstrip("/")
    model = os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus")
    ssl_verify = os.getenv("OPENAI_SSL_VERIFY", "false").lower() not in (
        "0",
        "false",
        "no",
    )

    class DashScopeSafeChatOpenAI(ChatOpenAI):
        """百炼兼容模式拒绝空 content；Agent 纯 tool_call 轮次需补占位符。"""

        @staticmethod
        def _sanitize_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
            sanitized: list[BaseMessage] = []
            for msg in messages:
                content = getattr(msg, "content", None)
                if content is None or content == "" or content == []:
                    sanitized.append(msg.model_copy(update={"content": " "}))
                else:
                    sanitized.append(msg)
            return sanitized

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return super()._generate(
                self._sanitize_messages(messages),
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            return await super()._agenerate(
                self._sanitize_messages(messages),
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

    return DashScopeSafeChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(verify=ssl_verify),
        http_async_client=httpx.AsyncClient(verify=ssl_verify),
    )
