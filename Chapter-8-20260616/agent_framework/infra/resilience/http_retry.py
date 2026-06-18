"""HTTP 请求重试（供 travel_api 等领域 infra 使用）。"""

from __future__ import annotations

from typing import Any

import httpx

from agent_framework.config import HTTP_RETRY_BASE_DELAY_SEC, HTTP_RETRY_MAX_ATTEMPTS
from agent_framework.infra.resilience.retry import async_retry


def _is_retryable_http_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (408, 429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError))


async def async_http_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs: Any,
) -> httpx.Response:
    """带重试的 HTTP 请求；client 由调用方创建并传入。"""

    async def _do() -> httpx.Response:
        response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    return await async_retry(
        _do,
        max_attempts=HTTP_RETRY_MAX_ATTEMPTS,
        base_delay_sec=HTTP_RETRY_BASE_DELAY_SEC,
        retry_on=(httpx.HTTPError,),
        should_retry=_is_retryable_http_error,
    )
