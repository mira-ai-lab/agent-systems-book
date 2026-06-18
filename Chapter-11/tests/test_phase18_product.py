"""Phase 18：API locale + router pre_survey 联动 + 领域 prompt 英文化。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from agent_framework.domain.domain_config import DomainConfig
from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.i18n.locale import normalize_locale
from agent_framework.orchestration.fixed_graph.nodes import GraphContext, make_nodes
from agent_framework.orchestration.fixed_graph.state import CentralAgentState
from agent_framework.router.execution_plan_bridge import execution_plan_from_routing_plan
from agent_framework.router.plan import AgentCandidate, RoutingPlan, RoutingStep
from agent_framework.router.pre_survey_bridge import pre_survey_from_routing_plan
from domains.customer_service.prompt_bundle import CustomerServicePrompts
from domains.travel.prompt_bundle import TravelPrompts


def test_normalize_locale_en():
    assert normalize_locale("en") == "en"
    assert normalize_locale("EN") == "en"


def test_pre_survey_from_routing_plan_events():
    plan = RoutingPlan(
        rewritten_query="咨询退货并投诉物流",
        events=["咨询退货政策", "投诉物流"],
        candidates=[AgentCandidate("FAQAgent", 0.9), AgentCandidate("TicketAgent", 0.8)],
        profile="workflow",
        metadata={"stages": ["extraction", "classification"], "knowledge_matches": []},
    )
    pre = pre_survey_from_routing_plan(plan)
    assert pre["source"] == "router_engine"
    assert "咨询退货政策" in pre["given_facts"]
    assert any("FAQAgent" in item for item in pre["facts_to_lookup"])
    assert any("extraction" in item for item in pre["educated_guesses"])


def test_execution_plan_includes_router_pre_survey():
    plan = RoutingPlan(
        rewritten_query="plan trip",
        events=["plan itinerary"],
        candidates=[AgentCandidate("ItineraryAgent", 0.9)],
        steps=[RoutingStep("T1", "Plan itinerary", "ItineraryAgent")],
        profile="workflow",
        metadata={"decomposition_goal": "Plan a trip"},
    )
    ep = execution_plan_from_routing_plan(plan, user_query="plan trip")
    assert ep is not None
    assert ep["pre_survey"]["source"] == "router_engine"
    assert "plan itinerary" in ep["pre_survey"]["given_facts"]


def test_pre_survey_node_skips_llm_with_prefilled():
    registry = MagicMock()
    ctx = GraphContext(
        MagicMock(),
        None,
        registry=registry,
        prompts=DomainPrompts(
            central_agent_system="sys",
            aggregation="agg",
            facts_prompt="facts",
            decomposition_prompt="decomp",
            dependency_system="dep sys",
            dependency_user="dep user",
            agent_routing="route",
        ),
        domain_config=DomainConfig(),
    )
    ctx.planner = MagicMock()
    ctx.planner.run_pre_survey = AsyncMock()
    nodes = make_nodes(ctx)
    prefilled = pre_survey_from_routing_plan(
        RoutingPlan(rewritten_query="q", events=["event-a"], profile="workflow")
    )
    state: CentralAgentState = {
        "user_query": "q",
        "prefilled_pre_survey": prefilled,
        "logs": [],
    }
    result = asyncio.run(nodes["pre_survey"](state))
    ctx.planner.run_pre_survey.assert_not_called()
    assert result["pre_survey"]["source"] == "router_engine"


def test_customer_service_prompts_en():
    prompts = CustomerServicePrompts.build(locale="en")
    assert "customer-service orchestration hub" in prompts.central_agent_system.lower()
    assert prompts.multi_task_title == "Customer Service Summary"


def test_travel_prompts_en():
    prompts = TravelPrompts.build(locale="en")
    assert "travel planning central agent" in prompts.central_agent_system.lower()
    assert "{background_info}" in prompts.decomposition_prompt
