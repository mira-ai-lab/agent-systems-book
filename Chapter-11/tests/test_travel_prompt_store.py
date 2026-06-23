"""Tests for optimized prompt store and runtime auto-load."""

from __future__ import annotations

import json
from pathlib import Path

from agent_framework.optimization.prompt_store import (
    apply_prompt_overrides,
    load_optimized_prompts,
    save_optimized_prompts,
)
from domains.travel.prompt_bundle import TravelPrompts


def test_save_and_load_optimized_prompts(tmp_path: Path, monkeypatch):
    path = tmp_path / "zh.json"
    save_optimized_prompts(
        path,
        updates={
            "decomposition_prompt": "DECOMP {background_info} {agent_team} {user_input}",
            "agent_routing": "ROUTE {agent_team} {subtasks_json}",
        },
        metadata={"best_dev_score": 0.91},
    )
    monkeypatch.setenv("TRAVEL_OPTIMIZED_PROMPTS_FILE", str(path))

    loaded = load_optimized_prompts("zh")
    assert "DECOMP" in loaded["decomposition_prompt"]
    assert "ROUTE" in loaded["agent_routing"]

    base = TravelPrompts.build("zh", use_optimized=False)
    merged = apply_prompt_overrides(base, locale="zh")
    assert merged.decomposition_prompt.startswith("DECOMP")
    assert merged.agent_routing.startswith("ROUTE")


def test_travel_prompts_build_auto_loads_optimized(tmp_path: Path, monkeypatch):
    path = tmp_path / "zh.json"
    path.write_text(
        json.dumps(
            {
                "decomposition_prompt": "AUTO {background_info} {agent_team} {user_input}",
                "agent_routing": "AUTO_ROUTE {agent_team} {subtasks_json}",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAVEL_OPTIMIZED_PROMPTS_FILE", str(path))
    monkeypatch.setenv("TRAVEL_OPTIMIZED_PROMPTS", "1")

    prompts = TravelPrompts.build("zh")
    assert prompts.decomposition_prompt.startswith("AUTO")
    assert prompts.agent_routing.startswith("AUTO_ROUTE")


def test_travel_prompts_build_can_disable_optimized(tmp_path: Path, monkeypatch):
    path = tmp_path / "zh.json"
    path.write_text(
        json.dumps({"decomposition_prompt": "SHOULD_NOT_LOAD {background_info} {agent_team} {user_input}"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAVEL_OPTIMIZED_PROMPTS_FILE", str(path))

    prompts = TravelPrompts.build("zh", use_optimized=False)
    assert not prompts.decomposition_prompt.startswith("SHOULD_NOT_LOAD")
