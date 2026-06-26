"""Travel end-to-end benchmark tests (mock orchestrator, no live API)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from agent_framework.optimization.decomposition.fixtures import load_decomposition_fixtures
from agent_framework.optimization.e2e.evaluator import evaluate_e2e_benchmark
from agent_framework.optimization.e2e.expectations import resolve_e2e_expect
from agent_framework.optimization.e2e.rules import build_e2e_expectation_label, format_e2e_rule_checklist
from agent_framework.optimization.e2e.scorer import build_e2e_keyword_corpus, score_e2e_run


def test_resolve_e2e_expect_from_weather_case():
    case = next(item for item in load_decomposition_fixtures().cases if item.case_id == "case-01")
    expect = resolve_e2e_expect(case)
    assert "WeatherAgent" in expect.required_agents
    assert "北京" in expect.required_response_keywords
    assert "酒店" in expect.forbidden_response_keywords
    assert expect.min_completed_subtasks == 1
    assert len(expect.tool_checks) == 1
    assert expect.tool_checks[0].task_id == "T1"
    assert "北京" in expect.tool_checks[0].field_contains["city"]


def test_score_e2e_run_passes_for_complete_weather_flow():
    case = next(item for item in load_decomposition_fixtures().cases if item.case_id == "case-01")
    expect = resolve_e2e_expect(case)
    score = score_e2e_run(
        {
            "final_response": "北京明天天气晴，气温 25 度左右，适合出行。",
            "subtask_results": {
                "T1": {"agent": "WeatherAgent", "status": "completed"},
            },
        },
        expect,
    )
    assert score.total >= 0.8
    assert score.response_ok
    assert score.agents_ok
    assert score.completion_ok


def test_score_e2e_run_detects_missing_agent():
    case = next(item for item in load_decomposition_fixtures().cases if item.case_id == "case-08")
    expect = resolve_e2e_expect(case)
    score = score_e2e_run(
        {
            "final_response": "西安天气不错，也推荐了几家酒店。",
            "subtask_results": {
                "T1": {"agent": "WeatherAgent", "status": "completed"},
                "T2": {"agent": "HotelAgent", "status": "completed"},
            },
        },
        expect,
    )
    assert score.total < 0.8
    assert not score.agents_ok


def test_score_e2e_run_keywords_in_agent_summary_only():
    case = next(item for item in load_decomposition_fixtures().cases if item.case_id == "case-01")
    expect = resolve_e2e_expect(case)
    score = score_e2e_run(
        {
            "final_response": "查询完成，详见下方摘要。",
            "subtask_results": {
                "T1": {
                    "agent": "WeatherAgent",
                    "status": "completed",
                    "agent_summary": "北京明天天气晴，适合出行。",
                },
            },
        },
        expect,
    )
    assert score.keyword_ok
    assert score.total >= 0.8


def test_build_e2e_keyword_corpus_merges_final_and_summaries():
    corpus = build_e2e_keyword_corpus(
        {
            "final_response": "汇总如下",
            "subtask_results": {
                "T1": {"agent_summary": "上海航班"},
            },
        }
    )
    assert "汇总如下" in corpus
    assert "上海航班" in corpus


def test_build_e2e_expectation_label_includes_rule_failures():
    case = next(item for item in load_decomposition_fixtures().cases if item.case_id == "case-01")
    expect = resolve_e2e_expect(case)
    label = build_e2e_expectation_label(
        case,
        rule_failures=["回复/子任务摘要缺少关键词: ['天气']"],
    )
    assert "rule_scorer_checklist" in label or "Rule scorer checklist" in label
    assert "rule_scorer_failures_on_this_run" in label
    assert "天气" in label
    checklist = format_e2e_rule_checklist(expect)
    assert "0.35" in checklist
    assert "agent_summary" in checklist


def test_score_e2e_run_tool_data_city_check():
    case = next(item for item in load_decomposition_fixtures().cases if item.case_id == "case-01")
    expect = resolve_e2e_expect(case)
    good = score_e2e_run(
        {
            "final_response": "北京明天天气晴。",
            "subtask_results": {
                "T1": {
                    "agent": "WeatherAgent",
                    "status": "completed",
                    "tool_data": {"city": "北京", "date": "2026-06-23"},
                },
            },
        },
        expect,
    )
    bad = score_e2e_run(
        {
            "final_response": "北京明天天气晴。",
            "subtask_results": {
                "T1": {
                    "agent": "WeatherAgent",
                    "status": "completed",
                    "tool_data": {"city": "上海"},
                },
            },
        },
        expect,
    )
    assert good.tool_data_ok
    assert not bad.tool_data_ok
    assert bad.total < good.total


def test_evaluate_e2e_benchmark_with_mock_orchestrator():
    fixtures = load_decomposition_fixtures()
    dev_cases = fixtures.cases_for_split("dev")

    async def _fake_process_request(user_query: str, thread_id: str = "default", timeout_sec=None):
        if thread_id == dev_cases[0].case_id:
            return {
                "final_response": "已查询西安下周天气，并推荐酒店和西安特色美食餐厅。",
                "subtask_results": {
                    "T1": {"agent": "WeatherAgent", "status": "completed"},
                    "T2": {"agent": "HotelAgent", "status": "completed"},
                    "T3": {"agent": "RestaurantAgent", "status": "completed"},
                },
                "trace_id": "trace-dev-08",
                "orchestration_mode": "fixed_graph",
            }
        return {
            "final_response": "已查询北京到三亚航班，并推荐海棠湾附近酒店。",
            "subtask_results": {
                "T1": {"agent": "FlightAgent", "status": "completed"},
                "T2": {"agent": "HotelAgent", "status": "completed"},
            },
            "trace_id": "trace-dev-09",
            "orchestration_mode": "fixed_graph",
        }

    orchestrator = MagicMock()
    orchestrator.process_request = AsyncMock(side_effect=_fake_process_request)

    async def _run():
        return await evaluate_e2e_benchmark(
            orchestrator,
            fixtures=fixtures,
            split="dev",
            profile="workflow",
        )

    report = asyncio.run(_run())
    assert report.case_count == 2
    assert report.average_score >= 0.8
    assert report.cases[0].case_id == "case-08"
    assert report.cases[0].invoked_agents == ["HotelAgent", "RestaurantAgent", "WeatherAgent"]
