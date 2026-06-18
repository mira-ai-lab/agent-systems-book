"""异步重试与退避（LLM / 通用协程）。"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Tuple, Type

from agent_framework.config import LLM_RETRY_BASE_DELAY_SEC, LLM_RETRY_MAX_ATTEMPTS


def is_retryable_llm_error(exc: BaseException) -> bool:
    """判断是否值得对 LLM 调用重试。"""
    name = type(exc).__name__.lower()
    retry_hints = (
        "timeout",
        "ratelimit",
        "rate limit",
        "429",
        "503",
        "502",
        "connection",
        "temporarily",
        "overloaded",
    )
    text = f"{name} {exc}".lower()
    return any(h in text for h in retry_hints)


async def async_retry(
    factory: Callable[[], Awaitable[Any]],
    *,
    max_attempts: int | None = None,
    base_delay_sec: float | None = None,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
    should_retry: Callable[[BaseException], bool] | None = None,
) -> Any:
    """执行异步工厂函数，失败时指数退避重试。"""
    attempts = max_attempts if max_attempts is not None else LLM_RETRY_MAX_ATTEMPTS
    base_delay = base_delay_sec if base_delay_sec is not None else LLM_RETRY_BASE_DELAY_SEC
    last_exc: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return await factory()
        except retry_on as exc:
            last_exc = exc
            retryable = should_retry(exc) if should_retry else True
            if not retryable or attempt >= attempts:
                raise
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))

    if last_exc:
        raise last_exc
    raise RuntimeError("async_retry exhausted without result")
