"""请求级子 Agent 话术 locale 上下文。"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

from agent_framework.i18n.locale import normalize_locale

_agent_locale: contextvars.ContextVar[str] = contextvars.ContextVar("agent_locale", default="zh")


def get_agent_locale() -> str:
    return _agent_locale.get()


@contextmanager
def agent_locale_context(locale: str) -> Iterator[None]:
    token = _agent_locale.set(normalize_locale(locale))
    try:
        yield
    finally:
        _agent_locale.reset(token)
