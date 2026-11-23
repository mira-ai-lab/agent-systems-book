"""Phase 11D：Supervisor step_summary。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from agent_framework.domain.pipeline import PipelineConfig
from agent_framework.orchestration.supervisor.step_summary import (
    StepSummarizer,
    build_multi_round_context,
    extract_handoff_contexts,
    run_step_summary,
)


def test_build_multi_round_context():
    text = build_multi_round_context(
        [
            {
                "agent_name": "faq_agent",
                "agent_query": "查询退货政策",
                "agent_response": "支持 7 天无理由退货",
            }
        ],
        locale="zh",
    )
    assert "faq_agent" in text
    assert "退货政策" in text


def test_extract_handoff_contexts():
    messages = [
        HumanMessage(content="我要退货"),
        AIMessage(content="支持 7 天无理由退货", name="faq_agent"),
    ]
    contexts = extract_handoff_contexts(messages, {"faq_agent"})
    assert len(contexts) == 1
    assert contexts[0]["agent_name"] == "faq_agent"
    assert "退货" in contexts[0]["agent_query"]


def test_step_summarizer_skips_short_response():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="不应调用"))
    summarizer = StepSummarizer(mock_llm, min_chars=200)
    result = asyncio.run(
        summarizer.summarize_handoff(
            agent_name="faq_agent",
            agent_query="查询政策",
            agent_response="简短回复",
        )
    )
    assert result == "简短回复"
    mock_llm.ainvoke.assert_not_called()


def test_run_step_summary_mock_llm():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="退货政策：7 天内可退"))
    summary = asyncio.run(
        run_step_summary(
            mock_llm,
            step_desc="查询退货政策",
            agent_name="faq_agent",
            multi_round_context="用户咨询退货政策，已返回 7 天无理由规则",
            locale="zh",
        )
    )
    assert "退货" in summary


def test_pipeline_config_step_summary_flags():
    pipe = PipelineConfig(enable_step_summary=True, step_summary_min_chars=100)
    assert pipe.enable_step_summary is True
    assert pipe.step_summary_min_chars == 100
