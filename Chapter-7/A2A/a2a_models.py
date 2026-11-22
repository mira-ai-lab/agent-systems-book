"""Structured contracts exchanged across the Chapter-7 A2A boundary.

The module defines the Planner request/response models and converts a
discovered Agent Card into the capability metadata consumed by Chapter-6 task
planning.

Main capabilities:

- Define Planner request and task plan models.
- Convert a discovered Agent Card into planner capability metadata.

:author: zhanghong, liuhui
:date: 2026-11-22
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field


class AgentSkillSpec(BaseModel):
    """Describe one skill exposed by a remote A2A agent."""

    name: str
    description: str
    input_fields: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)


class AgentCapability(BaseModel):
    """Describe one A2A agent for planning and task routing."""

    name: str
    description: str
    parallel_safe: bool = True
    skills: list[AgentSkillSpec] = Field(default_factory=list)


class PlannerRequest(BaseModel):
    """Payload sent from the Supervisor to the Planner A2A service."""

    query: str
    available_agents: list[AgentCapability] = Field(default_factory=list)


class PlannedTask(BaseModel):
    """One dependency-aware task produced by the Planner."""

    task_id: str
    description: str
    depends_on: list[str] = Field(default_factory=list)


class TravelPlan(BaseModel):
    """Validated task plan returned by the Planner A2A service."""

    total_goal: str
    tasks: list[PlannedTask]
    execution_order: list[str]
    pre_survey: dict[str, Any] = Field(default_factory=dict)


def capability_from_card(card: Mapping[str, Any]) -> AgentCapability:
    """Derive planner metadata from a discovered Agent Card."""
    skills: list[AgentSkillSpec] = []
    parallel_safe = True
    for skill in card.get("skills", []):
        if not isinstance(skill, Mapping):
            continue
        tags = [
            str(tag)
            for tag in skill.get("tags", [])
        ]
        if "parallel-safe:false" in tags:
            parallel_safe = False
        skills.append(
            AgentSkillSpec(
                name=str(skill.get("id") or skill.get("name") or ""),
                description=str(skill.get("description") or ""),
                input_fields=[
                    tag.removeprefix("input:")
                    for tag in tags
                    if tag.startswith("input:")
                ],
                output_fields=[
                    tag.removeprefix("output:")
                    for tag in tags
                    if tag.startswith("output:")
                ],
            )
        )
    return AgentCapability(
        name=str(card["name"]),
        description=str(card.get("description") or ""),
        parallel_safe=parallel_safe,
        skills=skills,
    )
