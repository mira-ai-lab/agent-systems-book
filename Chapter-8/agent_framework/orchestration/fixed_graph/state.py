"""LangGraph 中心智能体状态定义。

CentralAgentState 是整张图共享的「黑板」：各节点读取上游字段、写入本阶段产出。
LangGraph 会将节点返回的 dict 合并进 state（浅合并），未返回的字段保持不变。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class CentralAgentState(TypedDict, total=False):
    """中心智能体图状态（对应书稿 Ch2→Ch3→Ch4→Ch5+→聚合 各阶段）。"""

    # --- 请求上下文 ---
    user_query: str          # 用户原始输入
    thread_id: str           # LangGraph checkpoint / 子 Agent 会话隔离键

    # --- Ch2 思维链预调查 ---
    pre_survey: Dict[str, Any]  # given_facts / facts_to_lookup 等四段式结构

    # --- Ch3 长期记忆 ---
    retrieved_memories: List[Dict[str, Any]]  # 向量检索 + 重排后的记忆条目
    enable_memory: bool       # 运行时是否读写记忆（与 PipelineConfig.enable_memory 对齐）

    # --- Ch4 任务拆解 + Ch6 子智能体路由 ---
    execution_plan: Dict[str, Any]   # TaskPlanner 产出的完整计划（含 pre_survey 快照）
    total_goal: str                  # 拆解得到的整体目标
    subtasks: List[Dict[str, Any]]   # 带 agent / params / depends_on 的子任务列表
    execution_order: List[str]       # task_id 执行顺序（T1, T2, …）

    # --- Ch5+ 分层执行 ---
    subtask_results: Dict[str, Any]       # task_id → {status, tool_data, agent_summary, …}
    pending_layers: List[List[str]]       # 按依赖拆层后的 task_id 矩阵，同层可并行
    current_layer_index: int              # execute_layer 循环游标

    # --- 最终输出 ---
    final_response: str     # 聚合节点产出的用户可读回复
    logs: List[str]         # 人类可读的执行日志（CLI / 调试）

    # --- 流式展示 ---
    enable_stream: bool     # True 时聚合 LLM 逐 token 输出，并减少 tracing 日志刷屏
