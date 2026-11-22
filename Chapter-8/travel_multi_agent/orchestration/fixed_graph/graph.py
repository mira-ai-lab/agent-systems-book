"""LangGraph graph assembly driven by PipelineConfig.

:author: zhanghong, liuhui, xingrui
:date: 2026-11-22"""

from __future__ import annotations

from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from travel_multi_agent.agents.registry import DomainAgentRegistry
from travel_multi_agent.a2a.client import A2ARegistry
from travel_multi_agent.orchestration.pipeline import PipelineConfig
from travel_multi_agent.memory import LongTermMemory

from .nodes import GraphContext, has_more_layers, make_nodes
from .state import CentralAgentState
from .stream_sink import StreamSink


def build_central_agent_graph(
    llm: ChatOpenAI,
    memory_system: Optional[LongTermMemory] = None,
    stream_sink: Optional[StreamSink] = None,
    registry: Optional[DomainAgentRegistry] = None,
    a2a_registry: Optional[A2ARegistry] = None,
    pipeline: Optional[PipelineConfig] = None,
) -> StateGraph:
    """Build an uncompiled StateGraph; node functions are bound by make_nodes(ctx)."""
    pipe = pipeline or PipelineConfig()
    ctx = GraphContext(
        llm,
        memory_system,
        stream_sink=stream_sink,
        registry=registry,
        a2a_registry=a2a_registry,
        pipeline=pipe,
    )
    nodes = make_nodes(ctx)
    graph = StateGraph(CentralAgentState)

    # --- optional nodes controlled by PipelineConfig ---
    if pipe.enable_pre_survey:
        graph.add_node("pre_survey", nodes["pre_survey"])
    if pipe.enable_memory:
        graph.add_node("retrieve_memory", nodes["retrieve_memory"])
        graph.add_node("save_memory", nodes["save_memory"])

    # --- required nodes: plan -> route -> layered execution -> aggregate ---
    graph.add_node("build_plan", nodes["build_plan"])
    graph.add_node("route", nodes["route"])
    graph.add_node("execute_layer", nodes["execute_layer"])
    graph.add_node("aggregate", nodes["aggregate"])

    # --- entry and pre-plan edges ---
    if pipe.enable_pre_survey:
        graph.set_entry_point("pre_survey")
        if pipe.enable_memory:
            graph.add_edge("pre_survey", "retrieve_memory")
            graph.add_edge("retrieve_memory", "build_plan")
        else:
            graph.add_edge("pre_survey", "build_plan")
    else:
        if pipe.enable_memory:
            graph.set_entry_point("retrieve_memory")
            graph.add_edge("retrieve_memory", "build_plan")
        else:
            graph.set_entry_point("build_plan")

    graph.add_edge("build_plan", "route")
    graph.add_edge("route", "execute_layer")

    # execute_layer self-loop: advance current_layer_index until all layers are done
    graph.add_conditional_edges(
        "execute_layer",
        has_more_layers,
        {"execute_layer": "execute_layer", "aggregate": "aggregate"},
    )

    if pipe.enable_memory:
        graph.add_edge("aggregate", "save_memory")
        graph.add_edge("save_memory", END)
    else:
        graph.add_edge("aggregate", END)

    return graph


def compile_graph(
    llm: ChatOpenAI,
    memory_system: Optional[LongTermMemory] = None,
    checkpointer: Any = None,
    store: Any = None,
    stream_sink: Optional[StreamSink] = None,
    registry: Optional[DomainAgentRegistry] = None,
    a2a_registry: Optional[A2ARegistry] = None,
    pipeline: Optional[PipelineConfig] = None,
):
    """Compile the graph for ainvoke / astream; default checkpointer is MemorySaver."""
    graph = build_central_agent_graph(
        llm,
        memory_system,
        stream_sink=stream_sink,
        registry=registry,
        a2a_registry=a2a_registry,
        pipeline=pipeline,
    )
    if checkpointer is None:
        checkpointer = MemorySaver()
    compile_kwargs: dict = {"checkpointer": checkpointer}
    if store is not None:
        # LangGraph Store: shared with long-term memory for cross-thread retrieval
        compile_kwargs["store"] = store
    return graph.compile(**compile_kwargs)