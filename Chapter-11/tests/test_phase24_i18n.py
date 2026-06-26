"""Phase 24 P1 i18n：子 Agent locales + 缺失告警 + API locale 端到端。"""

import warnings
from unittest.mock import patch

import pytest

from agent_framework.domain.locale_loader import (
    agent_system_prompt,
    domain_prompts_from_locale,
    load_domain_locale_payload,
    reset_locale_loader_cache,
)
from agent_framework.i18n.agent_locale_context import agent_locale_context, get_agent_locale
from domains.demo.prompt_bundle import DemoPrompts
from domains.travel.agents.prompt_loader import travel_agent_prompt
from domains.travel.prompt_bundle import TravelPrompts


@pytest.fixture(autouse=True)
def _clear_locale_cache():
    reset_locale_loader_cache()
    yield
    reset_locale_loader_cache()


@pytest.mark.parametrize("domain", ["travel", "demo"])
def test_domain_prompts_zh_en_differ(domain: str):
    zh = domain_prompts_from_locale(domain, "zh")
    en = domain_prompts_from_locale(domain, "en")
    assert zh.central_agent_system
    assert en.central_agent_system
    assert zh.central_agent_system != en.central_agent_system


def test_travel_agent_prompts_zh_en_differ():
    zh = travel_agent_prompt("WeatherAgent", locale="zh")
    en = travel_agent_prompt("WeatherAgent", locale="en")
    assert "天气" in zh or "查询" in zh
    assert "weather" in en.lower()


@pytest.mark.parametrize(
    "agent_name",
    ["WeatherAgent", "HotelAgent", "RestaurantAgent", "FlightAgent", "ItineraryAgent"],
)
def test_travel_five_agents_have_en_locale(agent_name: str):
    prompt = agent_system_prompt("travel", agent_name, "en")
    assert prompt.strip()
    assert "Agent" in prompt or "assistant" in prompt.lower()


def test_locale_missing_keys_logs_and_falls_back_to_zh():
    primary = {"central_agent_system": "English title only"}
    fallback = load_domain_locale_payload("demo", "zh")
    with patch("agent_framework.domain.locale_loader.log_info") as mock_log:
        from agent_framework.domain.locale_loader import _merge_locale_dict, _DOMAIN_PROMPT_FIELDS

        merged = _merge_locale_dict(
            primary,
            fallback,
            domain="demo",
            locale="en",
            kind="domain",
            fields=_DOMAIN_PROMPT_FIELDS,
        )
    assert merged["central_agent_system"] == "English title only"
    assert merged["aggregation"] == fallback["aggregation"]
    assert mock_log.called
    assert mock_log.call_args[0][1] == "locale.missing_keys"


def test_prompt_bundles_use_locale_json():
    tr_zh = TravelPrompts.build("zh")
    tr_en = TravelPrompts.build("en")
    assert tr_zh.multi_task_title != tr_en.multi_task_title

    demo_en = DemoPrompts.build("en")
    assert "Demo orchestrator" in demo_en.central_agent_system


def test_agent_locale_context():
    assert get_agent_locale() == "zh"
    with agent_locale_context("en"):
        assert get_agent_locale() == "en"
    assert get_agent_locale() == "zh"


def test_prompts_en_shim_deprecated():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import importlib

        import domains.travel.prompts_en as travel_en

        importlib.reload(travel_en)
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    assert travel_en.CENTRAL_AGENT_SYSTEM_PROMPT


def test_api_chat_locale_en_response():
    pytest.importorskip("fastapi")
    from importlib import import_module
    from unittest.mock import AsyncMock, MagicMock

    from fastapi.testclient import TestClient

    api_mod = import_module("services.api.app")
    mock_orch = MagicMock()
    mock_orch.process_request = AsyncMock(return_value={"final_response": "ok"})
    with patch.object(api_mod, "_get_orchestrator", AsyncMock(return_value=mock_orch)):
        client = TestClient(api_mod.app)
        resp = client.post(
            "/v1/chat",
            json={"query": "hello", "domain": "demo", "locale": "en"},
        )
    assert resp.status_code == 200
    assert resp.json()["locale"] == "en"
