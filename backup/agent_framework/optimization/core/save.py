"""优化产物持久化：prompt JSON 与优化报告。

将 ``OptimizationResult`` 写入运行时加载的 optimized prompt 文件，
并生成独立的 JSON 报告供复盘与对比不同 Optimizer 效果。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from agent_framework.optimization.planner_runtime import merge_with_saved_prompts
from agent_framework.optimization.prompt_store import save_optimized_prompts

from .result import OptimizationResult


def save_decomposition_optimization_artifacts(
    result: OptimizationResult,
    *,
    locale: str,
    output_path: Path,
    report_path: Path,
    executor_model: str,
    optimizer_model: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """保存 decomposition_prompt 单 slot 优化结果。

    - ``output_path``：合并已有 optimized prompts 后写入（通常 zh.json）
    - ``report_path``：本 slot 逐步得分与 optimizer 信息的 JSON 报告
    """
    # 与磁盘上已有 routing 等字段合并，避免单 slot 优化覆盖其他 slot
    merged = merge_with_saved_prompts(
        locale=locale,
        decomposition_prompt=result.best_prompt,
    )
    metadata = {
        "slot": "decomposition_prompt",
        "optimizer": result.optimizer,
        "baseline_dev_score": result.baseline_dev_score,
        "best_dev_score": result.best_dev_score,
        "executor_model": executor_model,
        "optimizer_model": optimizer_model,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    save_optimized_prompts(output_path, updates=merged, metadata=metadata)

    report = {
        **result.to_dict(),
        "output_file": str(output_path),
        "executor_model": executor_model,
        "optimizer_model": optimizer_model,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_routing_optimization_artifacts(
    result: OptimizationResult,
    *,
    locale: str,
    output_path: Path,
    report_path: Path,
    executor_model: str,
    optimizer_model: str,
    decomposition_prompt: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """保存 agent_routing 单 slot 优化结果。

    Args:
        decomposition_prompt: 可选；routing 优化时若同时更新了拆解 prompt，一并写入合并结果。
    """
    merged = merge_with_saved_prompts(
        locale=locale,
        agent_routing=result.best_prompt,
        decomposition_prompt=decomposition_prompt,
    )
    metadata = {
        "slot": "agent_routing",
        "optimizer": result.optimizer,
        "baseline_dev_score": result.baseline_dev_score,
        "best_dev_score": result.best_dev_score,
        "executor_model": executor_model,
        "optimizer_model": optimizer_model,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    save_optimized_prompts(output_path, updates=merged, metadata=metadata)

    report = {
        **result.to_dict(),
        "output_file": str(output_path),
        "executor_model": executor_model,
        "optimizer_model": optimizer_model,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_planner_optimization_artifacts(
    *,
    locale: str,
    output_path: Path,
    report_path: Path,
    executor_model: str,
    optimizer_model: str,
    backend: str,
    decomposition_result: Optional[OptimizationResult] = None,
    routing_result: Optional[OptimizationResult] = None,
) -> None:
    """保存 planner 一键串联优化结果（decomposition + routing 可只含其一）。

    ``optimize_travel_planner.py`` 在 pipeline 结束后调用本函数，
    将两个 slot 的最优 prompt 与分 slot 得分写入同一套文件。
    """
    merged = merge_with_saved_prompts(
        locale=locale,
        decomposition_prompt=(
            decomposition_result.best_prompt if decomposition_result is not None else None
        ),
        agent_routing=routing_result.best_prompt if routing_result is not None else None,
    )
    metadata: Dict[str, Any] = {
        "slot": "planner",
        "optimizer_backend": backend,  # local 或 textgrad_lib
        "executor_model": executor_model,
        "optimizer_model": optimizer_model,
    }
    if decomposition_result is not None:
        metadata["decomposition"] = {
            "optimizer": decomposition_result.optimizer,
            "baseline_dev_score": decomposition_result.baseline_dev_score,
            "best_dev_score": decomposition_result.best_dev_score,
        }
    if routing_result is not None:
        metadata["routing"] = {
            "optimizer": routing_result.optimizer,
            "baseline_dev_score": routing_result.baseline_dev_score,
            "best_dev_score": routing_result.best_dev_score,
        }

    save_optimized_prompts(output_path, updates=merged, metadata=metadata)

    report = {
        "backend": backend,
        "output_file": str(output_path),
        "executor_model": executor_model,
        "optimizer_model": optimizer_model,
        "decomposition": decomposition_result.to_dict() if decomposition_result else None,
        "routing": routing_result.to_dict() if routing_result else None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_agent_optimization_artifacts(
    result: OptimizationResult,
    *,
    agent_name: str,
    locale: str,
    output_path: Path,
    report_path: Path,
    executor_model: str,
    optimizer_model: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """保存子 Agent system_prompt 优化结果（Agent-B1 等）。"""
    from agent_framework.optimization.agent_prompt_store import save_optimized_agent_prompts

    metadata = {
        "slot": "agent_system_prompt",
        "agent_name": agent_name,
        "optimizer": result.optimizer,
        "baseline_dev_score": result.baseline_dev_score,
        "best_dev_score": result.best_dev_score,
        "executor_model": executor_model,
        "optimizer_model": optimizer_model,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    save_optimized_agent_prompts(
        output_path,
        agent_name=agent_name,
        system_prompt_template=result.best_prompt,
        metadata=metadata,
    )

    report = {
        **result.to_dict(),
        "agent_name": agent_name,
        "output_file": str(output_path),
        "executor_model": executor_model,
        "optimizer_model": optimizer_model,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_multi_agent_optimization_artifacts(
    results: Dict[str, OptimizationResult],
    *,
    locale: str,
    output_path: Path,
    report_dir: Path,
    executor_model: str,
    optimizer_model: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Agent-B2：批量保存多个子 Agent 的优化结果。

    逐个 merge 写入同一 ``output_path``（agent_prompt_store 已支持合并），
    每个 Agent 另存独立报告 JSON 到 ``report_dir``。
    """
    from agent_framework.optimization.agent_prompt_store import save_optimized_agent_prompts
    from agent_framework.optimization.optimizers.textgrad_agent.optimize import default_agent_report_path

    for agent_name, result in results.items():
        metadata = {
            "slot": "agent_system_prompt",
            "agent_name": agent_name,
            "optimizer": result.optimizer,
            "baseline_dev_score": result.baseline_dev_score,
            "best_dev_score": result.best_dev_score,
            "executor_model": executor_model,
            "optimizer_model": optimizer_model,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        save_optimized_agent_prompts(
            output_path,
            agent_name=agent_name,
            system_prompt_template=result.best_prompt,
            metadata=metadata,
        )

        report_path = Path(default_agent_report_path(agent_name, locale))
        if report_dir:
            report_path = report_dir / report_path.name

        report = {
            **result.to_dict(),
            "agent_name": agent_name,
            "output_file": str(output_path),
            "executor_model": executor_model,
            "optimizer_model": optimizer_model,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
