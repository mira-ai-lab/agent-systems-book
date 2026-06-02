"""聚合阶段辅助：单任务直达 vs 多任务 LLM 综合"""

from __future__ import annotations

import json
from typing import Any, Dict


def is_single_direct_response(results: Dict[str, Any]) -> bool:
    """仅一个子任务时，直接返回子智能体结果，避免被旅行规划模板扩写"""
    return len(results) == 1


def direct_response_from_results(results: Dict[str, Any]) -> str:
    """从单个子任务结果提取用户可读回复"""
    res = next(iter(results.values()))
    summary = (res.get("agent_summary") or "").strip()
    if summary:
        return summary
    tool_data = res.get("tool_data")
    if tool_data is not None:
        return json.dumps(tool_data, ensure_ascii=False, indent=2)
    return json.dumps(res, ensure_ascii=False, indent=2)


MEMORY_AGGREGATION_INSTRUCTION = (
    "请根据用户原始请求的范围，综合子任务执行结果生成回复。"
    "严格匹配用户问题，不要添加用户未询问的内容（例如用户只问天气，不要输出行程/酒店/美食攻略）。"
    "仅当用户明确要求旅行规划时，才提供完整的多日行程方案。"
)
