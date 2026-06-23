"""项目配置：路径、环境变量、LLM 工厂。"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

# agent_framework/config.py → Chapter-8/（书稿目录，非包名）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOK_ROOT = PROJECT_ROOT.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_memory"
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
DYNAMIC_AGENTS_PATH = PROJECT_ROOT / "data" / "dynamic_agents.json"
THREAD_STAGE_CONTEXT_PATH = PROJECT_ROOT / "data" / "thread_stage_context.json"
GRAPH_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "orchestration" / "fixed_graph" / "output"
)
TRACES_DIR = PROJECT_ROOT / "traces"

# 单次编排请求默认超时（秒），可通过环境变量覆盖
REQUEST_TIMEOUT_SEC = float(os.getenv("REQUEST_TIMEOUT_SEC", "120"))

# HTTP API 未传 domain 时的可选回落（默认空 = 必须显式指定领域）
DEFAULT_DOMAIN = os.getenv("DEFAULT_DOMAIN", "").strip()

# /ready 探针可选预热领域（如 travel）；未设置则仅检查插件注册表
READY_DOMAIN = os.getenv("READY_DOMAIN", "").strip()

# OpenTelemetry 服务名（平台产品标识，与具体领域无关）
PLATFORM_SERVICE_NAME = (
    os.getenv("OTEL_SERVICE_NAME", "multi-agent-platform").strip() or "multi-agent-platform"
)

# LLM 重试
LLM_RETRY_MAX_ATTEMPTS = max(1, int(os.getenv("LLM_RETRY_MAX_ATTEMPTS", "3")))
LLM_RETRY_BASE_DELAY_SEC = float(os.getenv("LLM_RETRY_BASE_DELAY_SEC", "0.8"))

# HTTP 重试（领域 infra）
HTTP_RETRY_MAX_ATTEMPTS = max(1, int(os.getenv("HTTP_RETRY_MAX_ATTEMPTS", "3")))
HTTP_RETRY_BASE_DELAY_SEC = float(os.getenv("HTTP_RETRY_BASE_DELAY_SEC", "0.5"))

# 记忆向量 namespace 前缀
MEMORY_NAMESPACE_PREFIX = os.getenv("MEMORY_NAMESPACE_PREFIX", "chapter8_memories")

# 并发槽位等待（秒）；超时则 asyncio.TimeoutError，API 可映射为 429
REQUEST_SLOT_WAIT_SEC = float(os.getenv("REQUEST_SLOT_WAIT_SEC", "30"))

# 多租户编排器 LRU 缓存大小
TENANT_ORCHESTRATOR_CACHE_SIZE = max(1, int(os.getenv("TENANT_ORCHESTRATOR_CACHE_SIZE", "32")))

# 知识库多租户隔离（Phase 25）：非 default user_id 可写入 tenants/{user_id}/ overlay
KNOWLEDGE_TENANT_ISOLATION = os.getenv("KNOWLEDGE_TENANT_ISOLATION", "true").lower() not in (
    "0",
    "false",
    "no",
)

# Registry 联邦（Phase 25 P1）：逗号分隔远程 platform 基址
REGISTRY_FEDERATION_URLS = os.getenv("REGISTRY_FEDERATION_URLS", "").strip()
REGISTRY_FEDERATION_API_KEY = os.getenv("REGISTRY_FEDERATION_API_KEY", "").strip()

# 异步任务 SQLite 路径
JOB_DB_PATH = os.getenv("JOB_DB_PATH", str(PROJECT_ROOT / "data" / "jobs.db"))

_dotenv_loaded = False


def load_project_dotenv(*, override: bool = False) -> None:
    """依次尝试 Chapter-8/.env 与书仓库根 .env（幂等）。"""
    global _dotenv_loaded
    load_dotenv(PROJECT_ROOT / ".env", override=override)
    load_dotenv(BOOK_ROOT / ".env", override=override)
    _dotenv_loaded = True


def create_llm(*, temperature: float = 0, model: str | None = None):
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
    model = model or os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus")
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
