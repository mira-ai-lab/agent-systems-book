"""Tests for travel decomposition prompt optimizer (P1)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_framework.optimization.decomposition.evaluator import DecompositionBenchmarkReport
from agent_framework.optimization.decomposition.fixtures import load_decomposition_fixtures
from agent_framework.optimization.decomposition.optimizer_prompts import FAILURE_CASE_TEMPLATE
from agent_framework.optimization.decomposition.prompt_optimizer import (
    extract_decomposition_prompt,
    format_failure_feedback_from_cases,
    optimize_decomposition_prompt,
    propose_prompt_revision,
)
from agent_framework.optimization.decomposition.scorer import DecompositionScore
from domains.travel.prompt_bundle import TravelPrompts
from domains.travel.specs import create_travel_registry_stub


def _prompt_with_placeholders(tag: str = "") -> str:
    suffix = f"\n# marker {tag}" if tag else ""
    return (
        "背景: {background_info}\n"
        "团队: {agent_team}\n"
        "输入: {user_input}"
        f"{suffix}"
    )


def test_extract_decomposition_prompt_plain():
    text = _prompt_with_placeholders("plain")
    assert extract_decomposition_prompt(text) == text


def test_extract_decomposition_prompt_from_fence():
    inner = _prompt_with_placeholders("fenced")
    raw = f"```markdown\n{inner}\n```"
    assert extract_decomposition_prompt(raw) == inner


def test_extract_decomposition_prompt_requires_placeholders():
    with pytest.raises(ValueError, match="缺少占位符"):
        extract_decomposition_prompt("no placeholders here")


def test_format_failure_feedback_from_cases():
    fixtures = load_decomposition_fixtures()
    case = fixtures.cases[0]
    result = SimpleNamespace(
        case_id=case.case_id,
        query=case.query,
        raw_output="# 目标\nx\n# 任务拆解\n- y",
        score=DecompositionScore(
            total=0.5,
            format_ok=True,
            subtask_count_ok=False,
            slot_ok=True,
            keyword_ok=True,
            forbidden_ok=True,
            dependency_ok=True,
            routing_assignment_ok=False,
            agent_coverage_ok=False,
            subtask_count=1,
            details=["子任务数不对"],
        ),
    )
    text = format_failure_feedback_from_cases([(result, case)])
    assert case.case_id in text
    assert "WeatherAgent" in text
    assert FAILURE_CASE_TEMPLATE.split("{")[0].strip()[:10] in text or "Case" in text


def test_propose_prompt_revision_mock():
    fixtures = load_decomposition_fixtures()
    registry = create_travel_registry_stub()
    case = fixtures.cases[0]
    result = SimpleNamespace(
        case_id=case.case_id,
        query=case.query,
        raw_output="bad",
        score=DecompositionScore(
            total=0.4,
            format_ok=False,
            subtask_count_ok=False,
            slot_ok=False,
            keyword_ok=False,
            forbidden_ok=True,
            dependency_ok=False,
            routing_assignment_ok=False,
            agent_coverage_ok=False,
            subtask_count=0,
            details=["bad"],
        ),
    )
    optimizer = MagicMock()
    improved = _prompt_with_placeholders("improved")

    async def ainvoke(messages):
        msg = MagicMock()
        msg.content = improved
        return msg

    optimizer.ainvoke = AsyncMock(side_effect=ainvoke)

    async def _run():
        return await propose_prompt_revision(
            optimizer,
            current_prompt=_prompt_with_placeholders("base"),
            failures=[(result, case)],
            agent_team=registry.get_all_agents_text(),
        )

    assert asyncio.run(_run()) == improved


def test_optimize_accepts_improved_prompt(monkeypatch):
    fixtures = load_decomposition_fixtures()
    registry = create_travel_registry_stub()
    base_prompt = TravelPrompts.build("zh", use_optimized=False).decomposition_prompt
    improved_prompt = base_prompt + "\n## 优化标记 IMPROVED"

    call_state = {"eval": 0}

    async def fake_evaluate_benchmark(planner, *, registry, fixtures, split, lang=None):
        call_state["eval"] += 1
        prompt = planner.prompts.decomposition_prompt
        score = 0.9 if "IMPROVED" in prompt else 0.5
        return DecompositionBenchmarkReport(
            domain="travel",
            locale="zh",
            split=split,
            case_count=1,
            average_score=score,
        )

    async def fake_collect_failures(planner, cases, *, registry, lang, failure_threshold):
        if "IMPROVED" in planner.prompts.decomposition_prompt:
            return []
        case = cases[0]
        result = SimpleNamespace(
            case_id=case.case_id,
            query=case.query,
            raw_output="bad",
            score=DecompositionScore(
                total=0.4,
                format_ok=False,
                subtask_count_ok=False,
                slot_ok=False,
                keyword_ok=False,
                forbidden_ok=True,
                dependency_ok=False,
                routing_assignment_ok=False,
                agent_coverage_ok=False,
                subtask_count=0,
                details=["bad"],
            ),
        )
        return [(result, case)]

    async def fake_propose(optimizer_llm, *, current_prompt, failures, agent_team):
        return improved_prompt

    async def fake_evaluate_case(planner, case, *, registry, lang="zh"):
        from agent_framework.optimization.decomposition.evaluator import DecompositionCaseResult

        score_val = 0.9 if "IMPROVED" in planner.prompts.decomposition_prompt else 0.4
        return DecompositionCaseResult(
            case_id=case.case_id,
            query=case.query,
            score=DecompositionScore(
                total=score_val,
                format_ok=score_val >= 0.8,
                subtask_count_ok=score_val >= 0.8,
                slot_ok=score_val >= 0.8,
                keyword_ok=score_val >= 0.8,
                forbidden_ok=True,
                dependency_ok=score_val >= 0.8,
                routing_assignment_ok=score_val >= 0.8,
                agent_coverage_ok=score_val >= 0.8,
                subtask_count=1,
                details=[] if score_val >= 0.8 else ["bad"],
            ),
            raw_output="mock",
        )

    monkeypatch.setattr(
        "agent_framework.optimization.decomposition.prompt_optimizer.evaluate_case",
        fake_evaluate_case,
    )
    monkeypatch.setattr(
        "agent_framework.optimization.decomposition.prompt_optimizer.evaluate_decomposition_benchmark",
        fake_evaluate_benchmark,
    )
    monkeypatch.setattr(
        "agent_framework.optimization.decomposition.prompt_optimizer.collect_failures",
        fake_collect_failures,
    )
    monkeypatch.setattr(
        "agent_framework.optimization.decomposition.prompt_optimizer.propose_prompt_revision",
        fake_propose,
    )

    async def _run():
        return await optimize_decomposition_prompt(
            decomposition_prompt=base_prompt,
            registry=registry,
            executor_llm=MagicMock(),
            optimizer_llm=MagicMock(),
            fixtures=fixtures,
            max_steps=3,
            rollback=True,
        )

    result = asyncio.run(_run())
    assert "IMPROVED" in result.best_prompt
    assert result.best_dev_score == 0.9
    assert result.baseline_dev_score == 0.5
    assert any(step.accepted for step in result.steps)


def test_optimize_rollback_rejects_worse_candidate(monkeypatch):
    fixtures = load_decomposition_fixtures()
    registry = create_travel_registry_stub()
    base_prompt = _prompt_with_placeholders("base")
    worse_prompt = _prompt_with_placeholders("worse")

    async def fake_evaluate_benchmark(planner, *, registry, fixtures, split, lang=None):
        prompt = planner.prompts.decomposition_prompt
        if "worse" in prompt:
            score = 0.3
        elif "base" in prompt:
            score = 0.6
        else:
            score = 0.5
        return DecompositionBenchmarkReport(
            domain="travel",
            locale="zh",
            split=split,
            case_count=1,
            average_score=score,
        )

    async def fake_collect_failures(planner, cases, *, registry, lang, failure_threshold):
        case = cases[0]
        result = SimpleNamespace(
            case_id=case.case_id,
            query=case.query,
            raw_output="bad",
            score=DecompositionScore(
                total=0.4,
                format_ok=False,
                subtask_count_ok=False,
                slot_ok=False,
                keyword_ok=False,
                forbidden_ok=True,
                dependency_ok=False,
                routing_assignment_ok=False,
                agent_coverage_ok=False,
                subtask_count=0,
                details=["bad"],
            ),
        )
        return [(result, case)]

    async def fake_propose(optimizer_llm, *, current_prompt, failures, agent_team):
        return worse_prompt

    async def fake_evaluate_case(planner, case, *, registry, lang="zh"):
        from agent_framework.optimization.decomposition.evaluator import DecompositionCaseResult

        return DecompositionCaseResult(
            case_id=case.case_id,
            query=case.query,
            score=DecompositionScore(
                total=0.4,
                format_ok=False,
                subtask_count_ok=False,
                slot_ok=False,
                keyword_ok=False,
                forbidden_ok=True,
                dependency_ok=False,
                routing_assignment_ok=False,
                agent_coverage_ok=False,
                subtask_count=0,
                details=["bad"],
            ),
            raw_output="bad",
        )

    monkeypatch.setattr(
        "agent_framework.optimization.decomposition.prompt_optimizer.evaluate_case",
        fake_evaluate_case,
    )
    monkeypatch.setattr(
        "agent_framework.optimization.decomposition.prompt_optimizer.evaluate_decomposition_benchmark",
        fake_evaluate_benchmark,
    )
    monkeypatch.setattr(
        "agent_framework.optimization.decomposition.prompt_optimizer.collect_failures",
        fake_collect_failures,
    )
    monkeypatch.setattr(
        "agent_framework.optimization.decomposition.prompt_optimizer.propose_prompt_revision",
        fake_propose,
    )

    async def _run():
        return await optimize_decomposition_prompt(
            decomposition_prompt=base_prompt,
            registry=registry,
            executor_llm=MagicMock(),
            optimizer_llm=MagicMock(),
            fixtures=fixtures,
            max_steps=1,
            rollback=True,
        )

    result = asyncio.run(_run())
    assert "worse" not in result.best_prompt
    assert result.best_dev_score == 0.6
    assert result.steps[0].accepted is False
