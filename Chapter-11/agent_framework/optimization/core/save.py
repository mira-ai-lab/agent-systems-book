"""Persist optimized prompts and optimization reports."""

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
    merged = merge_with_saved_prompts(
        locale=locale,
        decomposition_prompt=(
            decomposition_result.best_prompt if decomposition_result is not None else None
        ),
        agent_routing=routing_result.best_prompt if routing_result is not None else None,
    )
    metadata: Dict[str, Any] = {
        "slot": "planner",
        "optimizer_backend": backend,
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
