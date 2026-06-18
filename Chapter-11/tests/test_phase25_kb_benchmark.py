"""Phase 25 P2：KB Embedding 召回评测（25.9–25.10）。"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from agent_framework.router.kb.benchmark import (
    RecallBenchmarkCase,
    _case_hit,
    compare_backend_recall,
    evaluate_backend_recall,
    load_benchmark_fixtures,
)
from agent_framework.router.kb.loader import reset_domain_knowledge_cache


@pytest.fixture(autouse=True)
def _reset_kb_cache():
    reset_domain_knowledge_cache()
    yield
    reset_domain_knowledge_cache()


@pytest.fixture
def kb_tmp_path(tmp_path, monkeypatch):
    import agent_framework.router.kb.repository as repo

    monkeypatch.setattr(repo, "KNOWLEDGE_DIR", tmp_path)
    return tmp_path


def test_load_benchmark_fixtures():
    domain, cases = load_benchmark_fixtures()
    assert domain == "customer_service"
    assert len(cases) >= 5
    assert cases[0].query


def test_case_hit_by_doc_id():
    case = RecallBenchmarkCase(
        case_id="x",
        query="q",
        expected_agent="FAQAgent",
        expected_doc_id="cs-return-policy",
    )
    matches = [
        {"name": "TicketAgent", "doc_id": "other", "normalized_score": 0.9},
        {"name": "FAQAgent", "doc_id": "cs-return-policy", "normalized_score": 0.8},
    ]
    hit, rank, top = _case_hit(case, matches, top_k=3)
    assert hit is True
    assert rank == 2
    assert top["doc_id"] == "cs-return-policy"


def test_evaluate_backend_recall_hashing(kb_tmp_path):
    report = evaluate_backend_recall(
        "customer_service",
        [
            RecallBenchmarkCase(
                case_id="return",
                query="退换货政策",
                expected_agent="FAQAgent",
                expected_doc_id="cs-return-policy",
            ),
            RecallBenchmarkCase(
                case_id="ticket",
                query="提交工单投诉",
                expected_agent="TicketAgent",
                expected_doc_id="cs-ticket-escalation",
            ),
        ],
        embedding_backend="hashing",
        top_k_values=[1, 3],
    )
    assert report.backend == "hashing"
    assert report.case_count == 2
    assert "hit@1" in report.hit_at_k
    assert report.hit_at_k["hit@1"] >= 0.5
    assert report.cases[0].top_match.get("raw_score") is not None or report.cases[0].matches


def test_compare_backend_recall_report_shape(kb_tmp_path):
    report = compare_backend_recall(
        "customer_service",
        [
            RecallBenchmarkCase(
                case_id="return",
                query="7天无理由退货",
                expected_doc_id="cs-return-policy",
            )
        ],
        backends=["hashing"],
        top_k_values=[1, 3],
    )
    assert report["domain"] == "customer_service"
    assert "hashing" in report["backends"]
    assert "hit@1" in report["backends"]["hashing"]["hit_at_k"]
    case = report["backends"]["hashing"]["cases"][0]
    assert "matches" in case
    if case["matches"]:
        assert "raw_score" in case["matches"][0]
        assert "normalized_score" in case["matches"][0]


def test_benchmark_script_json_output(kb_tmp_path, tmp_path, monkeypatch):
    fixtures = tmp_path / "fixtures.json"
    fixtures.write_text(
        json.dumps(
            {
                "domain": "customer_service",
                "cases": [
                    {
                        "id": "return",
                        "query": "退换货政策",
                        "expected_doc_id": "cs-return-policy",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_knowledge_recall.py"
    spec = importlib.util.spec_from_file_location("benchmark_knowledge_recall", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    monkeypatch.setattr(sys, "argv", ["benchmark", "--fixtures", str(fixtures), "--json"])
    assert mod.main() == 0
