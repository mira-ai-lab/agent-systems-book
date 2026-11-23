"""知识关键词 + 向量知识库路由。"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.router.config import RouterConfig
from agent_framework.router.kb.loader import get_domain_knowledge_store
from agent_framework.router.kb.scoring import attach_normalized_scores
from agent_framework.router.plan import AgentCandidate


def _match_texts(*chunks: str) -> str:
    return " ".join(c.strip() for c in chunks if c and c.strip()).lower()


def match_knowledge_candidates(
    registry: SubAgentRegistry,
    *,
    query: str,
    events: Sequence[str] = (),
    min_score: float = 0.65,
) -> List[AgentCandidate]:
    haystack = _match_texts(query, *events)
    if not haystack:
        return []

    candidates: dict[str, float] = {}
    for agent_name in registry.get_agent_names():
        info = registry.agents.get(agent_name, {})
        score = 0.0
        for skill in info.get("skills") or []:
            if not isinstance(skill, dict):
                continue
            for tag in skill.get("tags") or []:
                text = str(tag or "").strip()
                if not text:
                    continue
                parts = text.split("-", 1)
                needles = [text.lower()]
                if len(parts) == 2:
                    needles.extend([parts[0].lower(), parts[1].lower()])
                if any(n and n in haystack for n in needles):
                    score = max(score, 0.75)
            for keyword in skill.get("keywords") or []:
                kw = str(keyword or "").strip().lower()
                if kw and kw in haystack:
                    score = max(score, 0.8)
        if score >= min_score:
            candidates[agent_name] = score

    return [
        AgentCandidate(name=name, score=score)
        for name, score in sorted(candidates.items(), key=lambda item: -item[1])
    ]


def match_vector_knowledge_candidates(
    domain: str,
    *,
    query: str,
    events: Sequence[str] = (),
    top_k: int = 5,
    min_score: float = 0.65,
    embedding_backend: str = "hashing",
    storage: str = "auto",
    vector_min_score: float = 0.15,
    keyword_min_score: float = 0.65,
    tenant_id: str = "default",
) -> Tuple[List[AgentCandidate], List[dict]]:
    store = get_domain_knowledge_store(
        domain,
        embedding_backend=embedding_backend,
        storage=storage,  # type: ignore[arg-type]
        tenant_id=tenant_id,
    )
    if store is None:
        return [], []
    hits = store.match_agents(query, events, top_k=top_k, min_score=min_score)
    meta: List[dict] = []
    candidates: List[AgentCandidate] = []
    for agent, raw_score, doc_id in hits:
        item = attach_normalized_scores(
            {
                "name": agent,
                "raw_score": raw_score,
                "doc_id": doc_id,
                "embedding_backend": store.embedding_backend_name,
            },
            source="vector",
            vector_min_score=vector_min_score,
            keyword_min_score=keyword_min_score,
        )
        item["source"] = "vector"
        meta.append(item)
        candidates.append(AgentCandidate(name=agent, score=item["normalized_score"]))
    return candidates, meta


def resolve_knowledge_candidates(
    registry: SubAgentRegistry,
    *,
    domain: str,
    query: str,
    events: Sequence[str] = (),
    config: RouterConfig,
    tenant_id: str = "default",
) -> Tuple[List[AgentCandidate], List[dict]]:
    """按 RouterConfig.knowledge_backend 合并 keyword / vector 命中。"""
    backend = (config.knowledge_backend or "hybrid").strip().lower()
    min_score = config.knowledge_min_score
    meta: List[dict] = []
    groups: List[List[AgentCandidate]] = []

    if backend in ("keyword", "hybrid"):
        keyword_hits = match_knowledge_candidates(
            registry,
            query=query,
            events=events,
            min_score=min_score,
        )
        groups.append(
            [
                AgentCandidate(
                    name=c.name,
                    score=attach_normalized_scores(
                        {"name": c.name, "raw_score": c.score},
                        source="keyword",
                        vector_min_score=config.knowledge_vector_min_score,
                        keyword_min_score=min_score,
                    )["normalized_score"],
                )
                for c in keyword_hits
            ]
        )
        meta.extend(
            attach_normalized_scores(
                {"name": c.name, "raw_score": c.score},
                source="keyword",
                vector_min_score=config.knowledge_vector_min_score,
                keyword_min_score=min_score,
            )
            for c in keyword_hits
        )

    if backend in ("vector", "hybrid") and (domain or "").strip():
        vector_hits, vector_meta = match_vector_knowledge_candidates(
            domain,
            query=query,
            events=events,
            top_k=config.knowledge_top_k,
            min_score=config.knowledge_vector_min_score,
            embedding_backend=config.knowledge_embedding_backend,
            storage=config.knowledge_storage,
            vector_min_score=config.knowledge_vector_min_score,
            keyword_min_score=min_score,
            tenant_id=tenant_id,
        )
        if vector_hits:
            groups.append(vector_hits)
        meta.extend(vector_meta)

    if not groups:
        return [], meta
    return merge_agent_candidates(*groups), meta


def merge_agent_candidates(
    *groups: Iterable[AgentCandidate],
) -> List[AgentCandidate]:
    merged: dict[str, float] = {}
    for group in groups:
        for item in group:
            if item.name == "other":
                continue
            merged[item.name] = max(merged.get(item.name, 0.0), item.score)
    return [
        AgentCandidate(name=name, score=score)
        for name, score in sorted(merged.items(), key=lambda item: -item[1])
    ]
