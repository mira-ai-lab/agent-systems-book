"""全局限流：编排请求并发控制。"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional


class RequestSlotTimeoutError(TimeoutError):
    """并发槽位等待超时。"""


_semaphore: Optional[asyncio.Semaphore] = None


def max_concurrent_requests() -> int:
    return max(1, int(os.getenv("MAX_CONCURRENT_REQUESTS", "4")))


def get_request_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(max_concurrent_requests())
    return _semaphore


def reset_request_semaphore_for_tests() -> None:
    """测试用：按当前环境变量重建信号量。"""
    global _semaphore
    _semaphore = None


@asynccontextmanager
async def acquire_request_slot(
    *,
    wait_timeout_sec: Optional[float] = None,
) -> AsyncIterator[None]:
    """获取编排执行槽位；超时未获取则抛出 asyncio.TimeoutError。"""
    sem = get_request_semaphore()
    if wait_timeout_sec is None:
        await sem.acquire()
        try:
            yield
        finally:
            sem.release()
        return

    try:
        await asyncio.wait_for(sem.acquire(), timeout=wait_timeout_sec)
    except asyncio.TimeoutError as exc:
        raise RequestSlotTimeoutError("concurrent request slot wait timeout") from exc
    try:
        yield
    finally:
        sem.release()