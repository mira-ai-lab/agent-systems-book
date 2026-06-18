"""知识路由分数归一化：keyword / vector 映射到统一 [0, 1]。"""

from __future__ import annotations

from typing import Any, Dict

KEYWORD_SCORE_CEIL = 1.0


def normalize_keyword_score(raw: float, *, min_score: float = 0.65) -> float:
    if raw < min_score:
        return 0.0
    span = KEYWORD_SCORE_CEIL - min_score
    if span <= 0:
        return 1.0
    return min(1.0, max(0.0, (raw - min_score) / span))


def normalize_vector_score(raw: float, *, min_score: float = 0.15) -> float:
    if raw < min_score:
        return 0.0
    span = 1.0 - min_score
    if span <= 0:
        return 1.0
    return min(1.0, max(0.0, (raw - min_score) / span))


def attach_normalized_scores(
    meta: Dict[str, Any],
    *,
    source: str,
    vector_min_score: float,
    keyword_min_score: float,
) -> Dict[str, Any]:
    raw = float(meta.get("raw_score", meta.get("score", 0.0)))
    if source == "vector":
        normalized = normalize_vector_score(raw, min_score=vector_min_score)
    else:
        normalized = normalize_keyword_score(raw, min_score=keyword_min_score)
    return {
        **meta,
        "source": source,
        "raw_score": raw,
        "normalized_score": normalized,
        "score": normalized,
    }
