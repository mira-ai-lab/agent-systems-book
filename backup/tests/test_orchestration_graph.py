"""编排层：StateGraph 构建、条件边、图可视化（无 LLM 执行）。"""

from unittest.mock import MagicMock

import pytest

from agent_framework.domain.domain_config import DomainConfig
from agent_framework.domain.pipeline import PipelineConfig
from agent_framework.orchestration.fixed_graph.graph import build_central_agent_graph, compile_graph
from agent_framework.orchestration.fixed_graph.nodes import has_more_layers
from agent_framework.orchestration.fixed_graph.stream_sink import StreamSink
from agent_framework.orchestration.fixed_graph.visualize import GraphVisualizer
from domains.travel.prompt_bundle import TravelPrompts
from domains.travel.specs import create_travel_registry_stub


@pytest.fixture
def mock_llm() -> MagicMock:
    return MagicMock()


@pytest.fixture
def travel_domain_bundle():
    return {
        "registry": create_travel_registry_stub(),
        "prompts": TravelPrompts.build(),
        "domain_config": DomainConfig(),
    }


def _node_names(graph_builder) -> set[str]:
    return set(graph_builder.nodes.keys())


def test_build_graph_full_pipeline(mock_llm, travel_domain_bundle):
    graph = build_central_agent_graph(mock_llm, pipeline=PipelineConfig(), **travel_domain_bundle)
    names = _node_names(graph)
    assert names == {
        "pre_survey",
        "retrieve_memory",
        "build_plan",
        "execute_layer",
        "aggregate",
        "save_memory",
    }


def test_build_graph_minimal_pipeline(mock_llm, travel_domain_bundle):
    pipe = PipelineConfig(enable_pre_survey=False, enable_memory=False)
    graph = build_central_agent_graph(mock_llm, pipeline=pipe, **travel_domain_bundle)
    names = _node_names(graph)
    assert names == {"build_plan", "execute_layer", "aggregate"}


def test_build_graph_memory_only(mock_llm, travel_domain_bundle):
    pipe = PipelineConfig(enable_pre_survey=False, enable_memory=True)
    graph = build_central_agent_graph(mock_llm, pipeline=pipe, **travel_domain_bundle)
    names = _node_names(graph)
    assert "retrieve_memory" in names
    assert "save_memory" in names
    assert "pre_survey" not in names


def test_compile_graph_returns_invokable_app(mock_llm, travel_domain_bundle):
    app = compile_graph(
        mock_llm,
        memory_system=None,
        pipeline=PipelineConfig(enable_memory=False, enable_pre_survey=False),
        **travel_domain_bundle,
    )
    assert hasattr(app, "ainvoke")
    assert hasattr(app, "get_graph")


@pytest.mark.parametrize(
    "state,expected",
    [
        ({"pending_layers": [["T1"], ["T2"]], "current_layer_index": 0}, "execute_layer"),
        ({"pending_layers": [["T1"]], "current_layer_index": 1}, "aggregate"),
        ({"pending_layers": [], "current_layer_index": 0}, "aggregate"),
    ],
)
def test_has_more_layers(state, expected):
    assert has_more_layers(state) == expected


def test_stream_sink_emit_and_reset():
    tokens: list[str] = []
    sink = StreamSink(enabled=True, on_token=tokens.append)
    sink.emit_token("hello")
    sink.emit_token("")
    assert tokens == ["hello"]
    sink.reset()
    assert not sink.enabled
    assert sink.on_token is None


def test_graph_visualizer_from_compiled(mock_llm, travel_domain_bundle):
    app = compile_graph(
        mock_llm,
        pipeline=PipelineConfig(enable_memory=False, enable_pre_survey=False),
        **travel_domain_bundle,
    )
    viz = GraphVisualizer.from_compiled(app)
    nodes = viz.get_nodes()
    assert "build_plan" in nodes
    mermaid = viz.get_mermaid()
    assert "build_plan" in mermaid
