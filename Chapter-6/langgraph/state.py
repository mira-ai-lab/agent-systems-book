"""LangGraph 中心智能体状态定义"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class CentralAgentState(TypedDict, total=False):
    """中心智能体图状态（Ch2→Ch3→Ch4→Ch5+→聚合）"""

    user_query: str
    thread_id: str

    # Ch2 预调查
    pre_survey: Dict[str, Any]

    # Ch3 记忆
    retrieved_memories: List[Dict[str, Any]]
    enable_memory: bool

    # Ch4 + 路由
    execution_plan: Dict[str, Any]
    total_goal: str
    subtasks: List[Dict[str, Any]]
    execution_order: List[str]

    # 子任务执行（按层循环）
    subtask_results: Dict[str, Any]
    pending_layers: List[List[str]]
    current_layer_index: int

    # 输出
    final_response: str
    logs: List[str]
