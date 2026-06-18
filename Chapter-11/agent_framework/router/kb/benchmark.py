"""知识库召回评测：hashing vs embedding hit@k。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agent_framework.config import PROJECT_ROOT
from agent_framework.router.kb.loader import reset_domain_knowledge_cache
from agent_framework.router.kb.repository import ingest_domain_knowledge
from agent_framework.router.stages.knowledge_routing import match_vector_knowledge_candidates

DEFAULT_FIXTURES_PATH = PROJECT_ROOT / "data" / "knowledge" / "benchmark" / "fixtures.json"
DEFAULT_TOP_K = (1, 3, 5)


@dataclass(frozen=True)
class RecallBenchmarkCase:
    case_id: str
    query: str
    expected_agent: str = ""
    expected_doc_id: str = ""


@dataclass
class RecallCaseResult:
    case_id: str
    query: str
    hit: bool
    rank: Optional[int]
    top_match: Dict[str, Any] = field(default_factory=dict)
    matches: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BackendRecallReport:
    backend: str
    domain: str
    case_count: int
    hit_at_k: Dict[str, float] = field(default_factory=dict)
    cases: List[RecallCaseResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "domain": self.domain,
            "case_count": self.case_count,
            "hit_at_k": dict(self.hit_at_k),
            "cases": [
                {
                    "case_id": item.case_id,
                    "query": item.query,
                    "hit": item.hit,
                    "rank": item.rank,
                    "top_match": dict(item.top_match),
                    "matches": list(item.matches),
                }
                for item in self.cases
            ],
        }


def default_fixtures_path() -> Path:
    return DEFAULT_FIXTURES_PATH


def load_benchmark_fixtures(path: Optional[Path] = None) -> tuple[str, List[RecallBenchmarkCase]]:
    fixture_path = path or default_fixtures_path()
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    domain = str(payload.get("domain") or "").strip()
    if not domain:
        raise ValueError("benchmark fixtures 缺少 domain")
    raw_cases = payload.get("cases") or []
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("benchmark fixtures 缺少 cases")

    cases: List[RecallBenchmarkCase] = []
    for idx, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            raise ValueError(f"benchmark case[{idx}] 缺少 query")
        case_id = str(item.get("id") or f"case-{idx + 1}").strip()
        cases.append(
            RecallBenchmarkCase(
                case_id=case_id,
                query=query,
                expected_agent=str(item.get("expected_agent") or "").strip(),
                expected_doc_id=str(item.get("expected_doc_id") or "").strip(),
            )
        )
    return domain, cases


def _case_hit(
    case: RecallBenchmarkCase,
    matches: Sequence[Dict[str, Any]],
    *,
    top_k: int,
) -> tuple[bool, Optional[int], Dict[str, Any]]:
    limited = list(matches[: max(1, top_k)])
    for rank, item in enumerate(limited, start=1):
        agent = str(item.get("name") or "")
        doc_id = str(item.get("doc_id") or "")
        agent_ok = not case.expected_agent or agent == case.expected_agent
        doc_ok = not case.expected_doc_id or doc_id == case.expected_doc_id
        if agent_ok and doc_ok and (case.expected_agent or case.expected_doc_id):
            return True, rank, dict(item)
    top_match = dict(limited[0]) if limited else {}
    return False, None, top_match


def evaluate_recall_case(
    domain: str,
    case: RecallBenchmarkCase,
    *,
    embedding_backend: str,
    top_k: int = 5,
    vector_min_score: float = 0.01,
    storage: str = "chroma",
) -> RecallCaseResult:
    _candidates, meta = match_vector_knowledge_candidates(
        domain,
        query=case.query,
        top_k=top_k,
        min_score=vector_min_score,
        embedding_backend=embedding_backend,
        storage=storage,
        vector_min_score=vector_min_score,
    )
    ordered = sorted(
        meta,
        key=lambda item: float(item.get("normalized_score") or item.get("score") or 0.0),
        reverse=True,
    )
    hit, rank, top_match = _case_hit(case, ordered, top_k=top_k)
    return RecallCaseResult(
        case_id=case.case_id,
        query=case.query,
        hit=hit,
        rank=rank,
        top_match=top_match,
        matches=ordered[:top_k],
    )


def evaluate_backend_recall(
    domain: str,
    cases: Sequence[RecallBenchmarkCase],
    *,
    embedding_backend: str,
    top_k_values: Sequence[int] = DEFAULT_TOP_K,
    vector_min_score: float = 0.01,
    storage: str = "chroma",
    ensure_ingested: bool = True,
) -> BackendRecallReport:
    if ensure_ingested:
        ingest_domain_knowledge(domain, embedding_backend=embedding_backend)
        reset_domain_knowledge_cache()

    max_k = max(top_k_values) if top_k_values else 5
    per_case_max: List[RecallCaseResult] = []
    for case in cases:
        per_case_max.append(
            evaluate_recall_case(
                domain,
                case,
                embedding_backend=embedding_backend,
                top_k=max_k,
                vector_min_score=vector_min_score,
                storage=storage,
            )
        )

    hit_at_k: Dict[str, float] = {}
    total = len(per_case_max) or 1
    for k in top_k_values:
        hits = 0
        for case, result in zip(cases, per_case_max):
            case_hit, _, _ = _case_hit(case, result.matches, top_k=k)
            if case_hit:
                hits += 1
        hit_at_k[f"hit@{k}"] = round(hits / total, 4)

    report_cases: List[RecallCaseResult] = []
    eval_k = top_k_values[0] if top_k_values else 1
    for case, result in zip(cases, per_case_max):
        hit, rank, top_match = _case_hit(case, result.matches, top_k=eval_k)
        report_cases.append(
            RecallCaseResult(
                case_id=case.case_id,
                query=case.query,
                hit=hit,
                rank=rank,
                top_match=top_match,
                matches=result.matches[:eval_k],
            )
        )

    return BackendRecallReport(
        backend=embedding_backend,
        domain=domain,
        case_count=len(cases),
        hit_at_k=hit_at_k,
        cases=report_cases,
    )


def compare_backend_recall(
    domain: str,
    cases: Sequence[RecallBenchmarkCase],
    *,
    backends: Sequence[str],
    top_k_values: Sequence[int] = DEFAULT_TOP_K,
    vector_min_score: float = 0.01,
    storage: str = "chroma",
) -> Dict[str, Any]:
    reports: Dict[str, BackendRecallReport] = {}
    for backend in backends:
        reports[backend] = evaluate_backend_recall(
            domain,
            cases,
            embedding_backend=backend,
            top_k_values=top_k_values,
            vector_min_score=vector_min_score,
            storage=storage,
        )
        reset_domain_knowledge_cache()

    primary_k = f"hit@{top_k_values[0]}" if top_k_values else "hit@1"
    comparison: Dict[str, Any] = {"primary_metric": primary_k, "winners": {}}
    if len(reports) >= 2:
        names = list(reports.keys())
        left, right = names[0], names[1]
        left_score = reports[left].hit_at_k.get(primary_k, 0.0)
        right_score = reports[right].hit_at_k.get(primary_k, 0.0)
        if left_score > right_score:
            comparison["winners"][primary_k] = left
        elif right_score > left_score:
            comparison["winners"][primary_k] = right
        else:
            comparison["winners"][primary_k] = "tie"

    return {
        "domain": domain,
        "case_count": len(cases),
        "top_k_values": list(top_k_values),
        "backends": {name: report.to_dict() for name, report in reports.items()},
        "comparison": comparison,
    }
