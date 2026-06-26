"""E2E train evaluation helpers."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from agent_framework.optimization.decomposition.fixtures import load_decomposition_fixtures
from agent_framework.optimization.e2e.collect import evaluate_e2e_train_cases


def test_evaluate_e2e_train_cases_single_pass_and_cap():
    fixtures = load_decomposition_fixtures()
    cases = fixtures.cases_for_split("dev")[:2]

    async def _fake_process_request(user_query: str, thread_id: str = "default", timeout_sec=None):
        if thread_id == cases[0].case_id:
            return {
                "final_response": "只有酒店",
                "subtask_results": {
                    "T1": {"agent": "HotelAgent", "status": "completed"},
                },
            }
        return {
            "final_response": "已查询北京到三亚航班，并推荐海棠湾附近酒店。",
            "subtask_results": {
                "T1": {"agent": "FlightAgent", "status": "completed"},
                "T2": {"agent": "HotelAgent", "status": "completed"},
            },
        }

    orchestrator = MagicMock()
    orchestrator.process_request = AsyncMock(side_effect=_fake_process_request)

    async def _run():
        return await evaluate_e2e_train_cases(
            orchestrator,
            cases,
            failure_threshold=0.8,
            max_failure_cases=1,
        )

    report = asyncio.run(_run())
    assert len(report.case_results) == 2
    assert len(report.train_scores) == 2
    assert orchestrator.process_request.await_count == 2
    assert len(report.failures) == 1
    assert report.failures[0][0].score.total < 0.8
