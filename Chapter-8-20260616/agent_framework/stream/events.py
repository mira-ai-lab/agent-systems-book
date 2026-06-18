"""结构化流式事件 schema（FixedGraph / final）。"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

SubtaskSummaryFn = Callable[[Dict[str, Any]], Optional[str]]


def build_subtask_summary(
    result: Dict[str, Any],
    *,
    max_len: int = 500,
    summarizer: Optional[SubtaskSummaryFn] = None,
) -> str:
    """子任务完成摘要：优先 Agent 自然语言回答，其次领域自定义，最后通用 fallback。"""
    agent_summary = str(result.get("agent_summary") or "").strip()
    if agent_summary:
        text = agent_summary
    elif summarizer is not None:
        custom = summarizer(result)
        text = str(custom).strip() if custom else _generic_subtask_fallback(result)
    else:
        text = _generic_subtask_fallback(result)
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _generic_subtask_fallback(result: Dict[str, Any]) -> str:
    """领域无关 fallback：不解析 tool 业务字段，避免耦合某一垂直场景。"""
    agent = str(result.get("agent") or "子任务").strip()
    status = str(result.get("status") or "completed")
    tool_data = result.get("tool_data")
    tool_count = _resolve_tool_call_count(result, tool_data)

    if isinstance(tool_data, dict):
        err = tool_data.get("error") or tool_data.get("message")
        if err:
            prefix = f"{agent} 失败" if status == "failed" else f"{agent} 工具错误"
            return f"{prefix}: {err}"

    if tool_count > 0:
        return f"{agent} 已完成 {tool_count} 次工具调用"
    if tool_data is not None:
        return f"{agent} 已完成工具调用"
    return status


def _resolve_tool_call_count(result: Dict[str, Any], tool_data: Any) -> int:
    raw = result.get("tool_call_count")
    if raw is not None:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            pass
    if isinstance(tool_data, dict) and "calls" in tool_data:
        calls = tool_data.get("calls")
        if isinstance(calls, list):
            return len(calls)
    if tool_data is not None:
        return 1
    return 0


def graph_node_event(chunk: Dict[str, Any]) -> Dict[str, Any]:
    node = next(iter(chunk))
    update = chunk[node] if isinstance(chunk.get(node), dict) else {}
    data: Dict[str, Any] = {"node": node, "keys": list(update.keys())}
    if "final_response" in update:
        preview = str(update.get("final_response") or "")
        data["final_response_preview"] = preview[:240]
    execution_plan = update.get("execution_plan")
    if isinstance(execution_plan, dict):
        data["subtask_count"] = len(execution_plan.get("subtasks") or [])
    subtask_results = update.get("subtask_results")
    if isinstance(subtask_results, dict):
        data["completed_subtasks"] = list(subtask_results.keys())
    return {"type": "graph.node", "stage": node, "data": data}


def graph_progress_event(message: str, *, node: Optional[str] = None) -> Dict[str, Any]:
    text = (message or "").strip()
    stage = node or _infer_progress_stage(text)
    return {"type": "graph.progress", "stage": stage, "data": {"message": text}}


def graph_token_event(token: str, *, stage: str = "aggregate") -> Dict[str, Any]:
    return {"type": "graph.token", "stage": stage, "data": {"token": token}}


def graph_subtask_completed_event(
    result: Dict[str, Any],
    *,
    summarizer: Optional[SubtaskSummaryFn] = None,
) -> Dict[str, Any]:
    """子 Agent 完成时推送结构化摘要（execute_layer）。"""
    return {
        "type": "graph.subtask.completed",
        "stage": "execute_layer",
        "data": {
            "task_id": result.get("task_id"),
            "agent": result.get("agent"),
            "status": result.get("status"),
            "summary": build_subtask_summary(result, summarizer=summarizer),
        },
    }


def graph_subtask_token_event(task_id: str, agent: str, token: str) -> Dict[str, Any]:
    """子 Agent LLM 逐 token 输出（execute_layer）。"""
    return {
        "type": "graph.subtask.token",
        "stage": "execute_layer",
        "data": {
            "task_id": task_id,
            "agent": agent,
            "token": token,
        },
    }


def final_event(data: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "final", "stage": "done", "data": data}


def handoff_event(
    *,
    target: str,
    transport: str = "local",
    status: str = "completed",
    preview: str = "",
) -> Dict[str, Any]:
    """Supervisor handoff 进度事件（SSE / iter_request_stream）。"""
    return {
        "type": "handoff.completed",
        "stage": "handoff",
        "data": {
            "target": target,
            "transport": transport,
            "status": status,
            "response_preview": (preview or "")[:240],
        },
    }


def error_event(message: str, *, code: str = "internal_error") -> Dict[str, Any]:
    return {"type": "error", "stage": "error", "data": {"code": code, "message": message}}


def registry_updated_event(
    *,
    domain: str,
    action: str,
    agent_name: Optional[str] = None,
    scope: str = "domain",
    source: str = "dynamic",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Registry 变更事件（供 SSE / webhook / API 响应附带）。"""
    data: Dict[str, Any] = {
        "domain": domain,
        "action": action,
        "scope": scope,
        "source": source,
    }
    if agent_name:
        data["agent_name"] = agent_name
    if extra:
        data.update(extra)
    return {"type": "registry.updated", "stage": "registry", "data": data}


def public_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """剥离内部字段后再推送给前端 / SSE。"""
    return {key: value for key, value in event.items() if not key.startswith("_")}


def _infer_progress_stage(message: str) -> str:
    if "[Ch2]" in message:
        return "pre_survey"
    if "[Ch3]" in message:
        return "retrieve_memory"
    if "[Ch4]" in message:
        return "build_plan"
    if "[Ch5]" in message or "执行层" in message or "子任务" in message:
        return "execute_layer"
    if "[Ch6]" in message or "汇聚" in message:
        return "aggregate"
    return "graph"
