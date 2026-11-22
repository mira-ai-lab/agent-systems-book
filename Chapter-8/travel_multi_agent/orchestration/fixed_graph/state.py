"""LangGraph state schema for the fixed graph.

:author: zhanghong, liuhui, xingrui
:date: 2026-11-22"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class CentralAgentState(TypedDict, total=False):
    """Central-agent graph state (Ch2 -> Ch3 -> Ch4 -> route -> Ch5+ -> aggregate)."""

    # --- request context ---
    user_query: str          # original user input
    thread_id: str           # LangGraph checkpoint / A2A conversation isolation key

    # --- Ch2 fact survey ---
    pre_survey: Dict[str, Any]  # given_facts / facts_to_lookup / ...

    # --- Ch3 long-term memory ---
    retrieved_memories: List[Dict[str, Any]]  # retrieved + reranked memory entries
    enable_memory: bool       # whether memory read/write is enabled

    # --- Ch4 task decomposition ---
    execution_plan: Dict[str, Any]   # full plan from TaskPlanner (pre_survey snapshot)
    total_goal: str                  # decomposed overall goal
    tasks: List[Dict[str, Any]]      # task_id / description / depends_on (no agent yet)
    execution_order: List[str]       # dependency-safe task_id order (T1, T2, ...)

    # --- routing (Ch6 agent_routing) ---
    subtasks: List[Dict[str, Any]]   # routed tasks: agent / params / depends_on / routing_error

    # --- Ch5+ layered A2A execution ---
    subtask_results: Dict[str, Any]       # task_id -> {status, agent_summary, ...}
    pending_layers: List[List[str]]       # dependency layers; same layer runs in parallel
    current_layer_index: int              # execute_layer loop cursor

    # --- final output ---
    final_response: str     # user-facing response composed by the aggregate node
    logs: List[str]         # human-readable execution log

    # --- streaming ---
    enable_stream: bool     # stream final response token-by-token