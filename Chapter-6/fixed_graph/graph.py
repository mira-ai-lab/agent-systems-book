"""LangGraph 图定义：中心智能体 StateGraph"""

from __future__ import annotations

from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from . import bootstrap

bootstrap.setup()

from memory_system import LongTermMemory

from .nodes import GraphContext, has_more_layers, make_nodes
from .state import CentralAgentState


def build_central_agent_graph(
    llm: ChatOpenAI,
    memory_system: Optional[LongTermMemory] = None,
) -> StateGraph:
    """
    构建中心智能体 StateGraph

    流程:
        pre_survey (预调查阶段) → retrieve_memory (检索记忆) → build_plan (构建计划)
            → execute_layer (执行层 - 循环，按依赖层执行)
            → aggregate (聚合结果) → save_memory (保存记忆) → END (结束)
    """
    ctx = GraphContext(llm, memory_system)
    nodes = make_nodes(ctx)

    graph = StateGraph(CentralAgentState)

    graph.add_node("pre_survey", nodes["pre_survey"])
    graph.add_node("retrieve_memory", nodes["retrieve_memory"])
    graph.add_node("build_plan", nodes["build_plan"])
    graph.add_node("execute_layer", nodes["execute_layer"])
    graph.add_node("aggregate", nodes["aggregate"])
    graph.add_node("save_memory", nodes["save_memory"])

    graph.set_entry_point("pre_survey")
    graph.add_edge("pre_survey", "retrieve_memory")
    graph.add_edge("retrieve_memory", "build_plan")
    graph.add_edge("build_plan", "execute_layer")
    graph.add_conditional_edges(
        "execute_layer",
        has_more_layers,
        {
            "execute_layer": "execute_layer",
            "aggregate": "aggregate",
        },
    )
    graph.add_edge("aggregate", "save_memory")
    graph.add_edge("save_memory", END)

    return graph


def compile_graph(
    llm: ChatOpenAI,
    memory_system: Optional[LongTermMemory] = None,
    checkpointer: Any = None,
    store: Any = None,
):
    """编译可执行的 LangGraph 应用"""
    graph = build_central_agent_graph(llm, memory_system)
    if checkpointer is None:
        checkpointer = MemorySaver()
    compile_kwargs: dict = {"checkpointer": checkpointer}
    if store is not None:
        compile_kwargs["store"] = store
    return graph.compile(**compile_kwargs)
