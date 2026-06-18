"""Phase 13：stage_summary。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage

from agent_framework.domain.pipeline import PipelineConfig
from agent_framework.orchestration.supervisor.stage_summary import (
    StageSummarizer,
    build_multi_step_context,
    run_stage_summary,
)


def test_build_multi_step_context():
    subtasks = {
        "T1": {"agent": "faq_agent", "agent_summary": "退货政策：7天"},
        "T2": {"agent": "ticket_agent", "agent_summary": "已创建工单"},
    }
    contexts = [
        {"agent_name": "faq_agent", "agent_query": "查退货政策", "agent_response": "7天"},
        {"agent_name": "ticket_agent", "agent_query": "创建投诉工单", "agent_response": "已创建"},
    ]
    text = build_multi_step_context(subtasks, contexts, locale="zh")
    assert "faq_agent" in text
    assert "ticket_agent" in text


def test_run_stage_summary_mock_llm():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="Stage Summary:\n1. **阶段目标与总体结果**：完成咨询与工单")
    )
    summary = asyncio.run(
        run_stage_summary(
            mock_llm,
            global_query="我要退货并投诉物流",
            multi_step_context="step1 完成政策咨询\nstep2 完成工单创建",
            locale="zh",
        )
    )
    assert "Stage Summary" in summary


def test_stage_summarizer_min_steps():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="stage"))
    summarizer = StageSummarizer(mock_llm, min_steps=2)
    result = asyncio.run(
        summarizer.summarize_stage(
            user_query="hello",
            messages=[],
            subtask_results={"T1": {"agent": "a", "agent_summary": "x"}},
            handoff_node_names=set(),
        )
    )
    assert result == ""
    mock_llm.ainvoke.assert_not_called()


def test_pipeline_stage_summary_flags():
    pipe = PipelineConfig(enable_stage_summary=True, stage_summary_min_steps=3)
    assert pipe.enable_stage_summary is True
    assert pipe.stage_summary_min_steps == 3
