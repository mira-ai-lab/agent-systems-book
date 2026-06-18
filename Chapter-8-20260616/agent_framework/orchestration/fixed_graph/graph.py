"""LangGraph 图定义：按 PipelineConfig 可组装流水线。

默认全流程：
    pre_survey → retrieve_memory → build_plan
        → execute_layer（按层循环）→ aggregate → save_memory → END

可通过 PipelineConfig 关闭 pre_survey 或 memory 节点，graph.py 会动态调整入口与边。
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agent_framework.domain.agent_registry import SubAgentRegistry
from agent_framework.domain.domain_config import DomainConfig
from agent_framework.domain.domain_prompts import DomainPrompts
from agent_framework.domain.pipeline import PipelineConfig
from agent_framework.infra.checkpoint_factory import resolve_checkpointer
from agent_framework.infra.memory.memory_system import LongTermMemory

from .nodes import GraphContext, has_more_layers, make_nodes
from .state import CentralAgentState
from .stream_sink import StreamSink


def build_central_agent_graph(
    llm: ChatOpenAI,
    memory_system: Optional[LongTermMemory] = None,
    stream_sink: Optional[StreamSink] = None,
    registry: Optional[SubAgentRegistry] = None,
    prompts: Optional[DomainPrompts] = None,
    domain_config: Optional[DomainConfig] = None,
    pipeline: Optional[PipelineConfig] = None,
) -> StateGraph:
    """构建未编译的 StateGraph；节点函数由 make_nodes(ctx) 按 GraphContext 绑定生成。"""
    pipe = pipeline or PipelineConfig()
    ctx = GraphContext(
        llm,
        memory_system,
        stream_sink=stream_sink,
        registry=registry,
        prompts=prompts,
        domain_config=domain_config,
        pipeline=pipe,
    )
    nodes = make_nodes(ctx)
    graph = StateGraph(CentralAgentState)

    # --- 注册节点（按 PipelineConfig 决定是否加入图中）---
    if pipe.runs_pre_survey_node:
        graph.add_node("pre_survey", nodes["pre_survey"])
    if pipe.enable_memory:
        graph.add_node("retrieve_memory", nodes["retrieve_memory"])
        graph.add_node("save_memory", nodes["save_memory"])

    # 规划 / 执行 / 聚合为必选节点
    graph.add_node("build_plan", nodes["build_plan"])
    graph.add_node("execute_layer", nodes["execute_layer"])
    graph.add_node("aggregate", nodes["aggregate"])

    # --- 入口与前置边：pre_survey 与 memory 可独立开关 ---
    if pipe.runs_pre_survey_node:
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

    graph.add_edge("build_plan", "execute_layer")

    # execute_layer 自循环：每层执行完后 current_layer_index++，直至层耗尽
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
    registry: Optional[SubAgentRegistry] = None,
    prompts: Optional[DomainPrompts] = None,
    domain_config: Optional[DomainConfig] = None,
    pipeline: Optional[PipelineConfig] = None,
):
    """编译图为可 ainvoke / astream 的应用；默认使用 MemorySaver 作为 checkpoint。"""
    graph = build_central_agent_graph(
        llm,
        memory_system,
        stream_sink=stream_sink,
        registry=registry,
        prompts=prompts,
        domain_config=domain_config,
        pipeline=pipeline,
    )
    if checkpointer is None:
        checkpointer = resolve_checkpointer()
    compile_kwargs: dict = {"checkpointer": checkpointer}
    if store is not None:
        # LangGraph Store：与长期记忆向量库配合，供跨线程检索
        compile_kwargs["store"] = store
    return graph.compile(**compile_kwargs)
