"""Optional ``textgrad`` import helpers."""

from __future__ import annotations

from typing import Any, Tuple


def require_textgrad() -> Tuple[Any, Any, Any, Any]:
    """Import textgrad core symbols or raise a helpful error."""
    try:
        import textgrad as tg
        from textgrad import Variable
        from textgrad.loss import TextLoss
        from textgrad.optimizer import TextualGradientDescent
    except ImportError as exc:
        raise ImportError(
            "textgrad 未安装。请运行: pip install -e \".[evolution]\" "
            "或 pip install textgrad>=0.1.5"
        ) from exc
    return tg, Variable, TextLoss, TextualGradientDescent
