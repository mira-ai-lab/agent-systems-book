"""Phase 16：travel 产品域 + 跨域推断 + task_decomposition + hybrid profile。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from agent_framework.bootstrap.platform import create_runtime
from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.orchestration.protocol import MODE_SUPERVISOR
from agent_framework.orchestration.supervisor.orchestrator import SupervisorOrchestrator
from agent_framework.router.engine import RouterEngine
from agent_framework.router.plan import AgentCandidate
from agent_framework.router.platform_domain_router import (
    DomainCandidate,
    parse_domain_classification_response,
    resolve_request_domain,
    select_domain,
)
from agent_framework.router.profile import PROFILE_HYBRID, normalize_profile, profile_to_mode
from agent_framework.router.stages.task_decomposition import (
    build_routing_steps,
    parse_decomposition_response,
)


def test_normalize_hybrid_profile():
    assert normalize_profile("hybrid") == PROFILE_HYBRID
    assert profile_to_mode(PROFILE_HYBRID) == MODE_SUPERVISOR


def test_parse_decomposition_response():
    raw = "整体目标：规划北京三日游\n子任务：\n- 查询天气\n- 推荐酒店"
    goal, steps = parse_decomposition_response(raw, locale="zh")
    assert "北京" in goal
    assert steps == ["查询天气", "推荐酒店"]


def test_build_routing_steps_assigns_agents():
    steps = build_routing_steps(
        ["查天气", "订酒店"],
        [AgentCandidate("WeatherAgent", 0.9), AgentCandidate("HotelAgent", 0.8)],
    )
    assert steps[0].agent == "WeatherAgent"
    assert steps[1].agent == "HotelAgent"


def test_select_domain_picks_best():
    assert select_domain([DomainCandidate("travel", 0.9)]) == "travel"
    assert select_domain([DomainCandidate("other", 1.0)]) is None


def test_parse_domain_classification_filters_unknown():
    parsed = parse_domain_classification_response(
        [{"name": "travel", "score": 0.9}, {"name": "unknown", "score": 0.8}],
        known_domains={"travel", "demo"},
    )
    assert len(parsed) == 1
    assert parsed[0].name == "travel"


def test_router_engine_task_decomposition_stage():
    registry = SubAgentRegistry()
    registry.register("A", lambda: MagicMock(), description="A")
    registry.register("B", lambda: MagicMock(), description="B")
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content='["复合任务"]'),
            AIMessage(content='[{"name": "A", "score": 0.9}, {"name": "B", "score": 0.85}]'),
            AIMessage(content="整体目标：完成复合任务\n子任务：\n- 步骤一\n- 步骤二"),
        ]
    )
    plan = asyncio.run(RouterEngine(mock_llm, registry).route("复合请求"))
    assert plan.profile == "workflow"
    assert len(plan.steps) == 2
    assert "task_decomposition" in plan.metadata["stages"]


def test_create_runtime_hybrid_returns_supervisor(monkeypatch):
    monkeypatch.setattr(
        "agent_framework.orchestration.supervisor.orchestrator.load_project_dotenv",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent_framework.orchestration.supervisor.orchestrator.setup_observability",
        lambda: None,
    )
    runtime = create_runtime("demo", profile="hybrid", enable_memory=False, llm=MagicMock())
    assert isinstance(runtime, SupervisorOrchestrator)
    assert runtime.transport == "mixed"


def test_resolve_request_domain_explicit():
    domain, candidates = asyncio.run(resolve_request_domain("hello", "demo"))
    assert domain == "demo"
    assert candidates is None


def test_resolve_request_domain_auto_classify():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content='[{"name": "travel", "score": 0.92}]')
    )
    with patch.dict("os.environ", {"DEFAULT_DOMAIN": ""}, clear=False):
        domain, candidates = asyncio.run(
            resolve_request_domain("规划杭州三日游", None, llm=mock_llm)
        )
    assert domain == "travel"
    assert candidates is not None
    assert candidates[0].name == "travel"
