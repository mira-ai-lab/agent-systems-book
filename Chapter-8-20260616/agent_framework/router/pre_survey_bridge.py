"""Router events / metadata → FixedGraph pre_survey 结构。"""

from __future__ import annotations

from typing import Any, Dict, List

from agent_framework.router.plan import AgentCandidate, RoutingPlan


def pre_survey_from_routing_plan(plan: RoutingPlan) -> Dict[str, Any]:
    """将 RouterEngine 抽取结果写入 TaskPlanner 预调查四段式结构。"""
    given_facts: List[str] = list(plan.events or [])
    if plan.rewritten_query.strip():
        given_facts.append(f"rewritten_query: {plan.rewritten_query.strip()}")

    facts_to_lookup: List[str] = []
    strong_candidates = [
        c for c in plan.candidates if c.name.lower() != "other" and c.score >= 0.3
    ]
    if strong_candidates:
        facts_to_lookup.append(
            "matched_agents: "
            + ", ".join(f"{c.name}({c.score:.2f})" for c in strong_candidates)
        )
    knowledge = plan.metadata.get("knowledge_matches") or []
    if knowledge:
        facts_to_lookup.append(
            "knowledge_routing: "
            + ", ".join(
                f"{item.get('name')}({item.get('score')})"
                for item in knowledge
                if isinstance(item, dict)
            )
        )

    stages = plan.metadata.get("stages") or []
    educated: List[str] = []
    if stages:
        educated.append("router_stages: " + " → ".join(str(s) for s in stages))
    if plan.profile:
        educated.append(f"resolved_profile: {plan.profile}")
    if plan.primary_agent:
        educated.append(f"primary_agent: {plan.primary_agent}")

    return {
        "given_facts": given_facts,
        "facts_to_lookup": facts_to_lookup,
        "facts_to_derive": [],
        "educated_guesses": educated,
        "raw_text": "",
        "source": "router_engine",
    }


def merge_candidate_notes(candidates: List[AgentCandidate]) -> str:
    filtered = [c for c in candidates if c.name.lower() != "other"]
    if not filtered:
        return ""
    return ", ".join(f"{c.name}({c.score:.2f})" for c in filtered)
