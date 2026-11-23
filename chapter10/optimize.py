"""Planner prompt self-evolution: two TextGrad backends and a shared loop.

Backends:

- ``direct``: only the prompt template is a trainable ``Variable``; the loss is
  a natural-language critique of (prompt + failing cases).
- ``planner_graph``: the decompose -> dependency -> routing pipeline is built
  into a textgrad computation graph; the loss compares the pipeline output with
  the benchmark expectation.

:author: zhanghong, liuhui
:date: 2026-11-23
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple

from langchain_openai import ChatOpenAI

from evaluate import average_case_score, cases_for_split, collect_failures, evaluate_benchmark
from optimizable_prompts import extract_prompt_template
from runtime import Planner
from textgrad_engine import create_textgrad_engine

OptimizerBackend = Literal["direct", "planner_graph"]

ProposeCandidate = Callable[
    [str, str, str, List[Tuple[Any, Dict[str, Any]]]],
    Tuple[str, str],
]


@dataclass
class OptimizationStep:
    """One round: collect failures -> text gradient -> dev re-evaluation."""

    step: int
    train_average: float
    best_score: float
    candidate_score: float
    accepted: bool
    failure_count: int
    prompt_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "train_average": round(self.train_average, 4),
            "best_score": round(self.best_score, 4),
            "candidate_score": round(self.candidate_score, 4),
            "accepted": self.accepted,
            "failure_count": self.failure_count,
            "prompt_preview": self.prompt_preview,
        }


@dataclass
class OptimizationResult:
    """Final result of optimizing one prompt slot."""

    slot: str
    metric: str
    baseline_score: float
    best_score: float
    best_prompt: str
    optimizer: str
    steps: List[OptimizationStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot": self.slot,
            "metric": self.metric,
            "optimizer": self.optimizer,
            "baseline_score": round(self.baseline_score, 4),
            "best_score": round(self.best_score, 4),
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class SlotSpec:
    """One optimizable prompt slot and its evaluation metric."""

    slot: str
    placeholders: Tuple[str, ...]
    metric: str = "l1_l2"


@dataclass
class OptimizationConfig:
    """Loop settings shared by both TextGrad backends."""

    max_steps: int = 3
    failure_threshold: float = 0.8
    rollback: bool = True
    train_split: str = "train"
    dev_split: str = "dev"


# direct 后端的 loss 提示词：让 TextLoss 依据失败样例批评“当前模板”并给出改进方向。
# 改写自 Chapter-11 的 textgrad_lib/prompts.py，占位符与 optimizable_prompts.py 一致。
DECOMPOSITION_LOSS_PROMPT = """You are evaluating a travel task-decomposition prompt template.

The template must:
1. Keep placeholders {background_info}, {capability_catalog}, {user_input}
2. Decompose the user request into subtasks that keep every required slot and avoid the forbidden task types shown in the failures
3. Produce a total goal plus a minimal set of complete subtasks

Use the failure cases below as critique. Explain what should improve so the prompt produces better decompositions on similar queries.
"""

DECOMPOSITION_CONSTRAINTS = (
    "The optimized prompt must retain placeholders {background_info}, {capability_catalog}, {user_input}.",
    "Do not output analysis; return only the revised prompt template.",
)

ROUTING_LOSS_PROMPT = """You are evaluating a travel agent-routing prompt template.

The template must:
1. Keep placeholders {agent_team}, {pre_survey_json}, {subtasks_json}, {depends_on_json}
2. Assign each subtask to the correct specialist agent in the travel team
3. Fill tool params that match the chosen agent's inputSchema

Use the failure cases below as critique. Explain what should improve so routing matches the expected agents and params on similar subtask sets.
"""

ROUTING_CONSTRAINTS = (
    "The optimized prompt must retain placeholders {agent_team}, {pre_survey_json}, {subtasks_json}, {depends_on_json}.",
    "Do not output analysis; return only the revised routing prompt template.",
)

DIRECT_SLOT_CONFIG = {
    "decomposition_prompt": {
        "loss_prompt": DECOMPOSITION_LOSS_PROMPT,
        "constraints": DECOMPOSITION_CONSTRAINTS,
        "role_description": "travel decomposition prompt template",
    },
    "agent_routing": {
        "loss_prompt": ROUTING_LOSS_PROMPT,
        "constraints": ROUTING_CONSTRAINTS,
        "role_description": "travel agent_routing prompt template",
    },
}

FAILURE_CASE_TEMPLATE = """- 用例 {case_id}：query="{query}"，得分 {score:.3f}
  失败原因：{details}
"""


def _format_failures(failures: List[Tuple[Any, Dict[str, Any]]]) -> str:
    return "\n".join(
        FAILURE_CASE_TEMPLATE.format(
            case_id=result.case_id,
            query=case["query"],
            score=result.total,
            details="; ".join(result.details) or "得分低于阈值",
        )
        for result, case in failures
    )


def _run_textgrad_step(
    prompt_var: Any,
    engine: Any,
    failure_feedback: str,
    *,
    loss_prompt: str,
    constraints: Tuple[str, ...],
) -> None:
    """用失败反馈对 ``prompt_var`` 做一次 TextGrad 更新。

    构造 ``TextLoss`` → ``backward`` 得到「文本梯度」→
    ``TextualGradientDescent.step`` 更新 ``prompt_var``。``constraints`` 约束
    输出保留占位符。
    """
    from textgrad import Variable
    from textgrad.loss import TextLoss
    from textgrad.optimizer import TextualGradientDescent

    loss_instruction = Variable(
        f"{loss_prompt}\n\n# Failure cases\n{failure_feedback}",
        requires_grad=False,
        role_description="failure feedback for prompt optimization",
    )
    loss_fn = TextLoss(loss_instruction, engine)
    optimizer = TextualGradientDescent(
        parameters=[prompt_var],
        engine=engine,
        constraints=list(constraints),
    )

    critique_context = Variable(
        failure_feedback,
        requires_grad=False,
        role_description="benchmark failure details",
    )
    combined = Variable(
        f"Prompt template:\n{prompt_var.value}\n\nFailures:\n{critique_context.value}",
        requires_grad=False,
        role_description="prompt plus failures",
    )
    loss = loss_fn(combined)
    loss.backward(engine)
    optimizer.step()
    optimizer.zero_grad()


def make_direct_proposer(optimizer_llm: ChatOpenAI) -> ProposeCandidate:
    """构造 direct 后端的候选生成器（单 Variable + TextLoss + TGD）。

    返回闭包，签名与 ``planner_graph.propose_candidate_via_graph`` 一致：
    ``(decomposition_prompt, agent_routing, slot, failures) -> (新拆解, 新路由)``。
    """
    engine = create_textgrad_engine(optimizer_llm)

    def propose(
        decomposition_prompt: str,
        agent_routing: str,
        slot: str,
        failures: List[Tuple[Any, Dict[str, Any]]],
    ) -> Tuple[str, str]:
        from textgrad import Variable

        config = DIRECT_SLOT_CONFIG[slot]
        current_prompt = (
            decomposition_prompt if slot == "decomposition_prompt" else agent_routing
        )
        prompt_var = Variable(
            current_prompt,
            requires_grad=True,
            role_description=config["role_description"],
        )
        _run_textgrad_step(
            prompt_var,
            engine,
            _format_failures(failures),
            loss_prompt=config["loss_prompt"],
            constraints=config["constraints"],
        )
        new_prompt = str(prompt_var.value)
        if slot == "decomposition_prompt":
            return new_prompt, agent_routing
        return decomposition_prompt, new_prompt

    return propose


async def _optimize_slot(
    *,
    slot: str,
    metric: str,
    placeholders: Tuple[str, ...],
    decomposition_prompt: str,
    agent_routing: str,
    propose: ProposeCandidate,
    executor_llm: ChatOpenAI,
    registry: Any,
    fixtures: Dict[str, Any],
    config: OptimizationConfig,
    optimizer_name: str,
) -> OptimizationResult:
    """Optimize one slot: the loop is shared by both backends."""
    train_cases = cases_for_split(fixtures, config.train_split)

    def make_planner(decomposition: str, routing: str) -> Planner:
        return Planner(
            executor_llm,
            registry,
            decomposition_prompt=decomposition,
            agent_routing=routing,
        )

    best_decomposition = decomposition_prompt
    best_routing = agent_routing
    best_prompt = decomposition_prompt if slot == "decomposition_prompt" else agent_routing

    baseline = await evaluate_benchmark(
        make_planner(best_decomposition, best_routing),
        fixtures,
        config.dev_split,
        metric=metric,
    )
    best_score = baseline.average_score
    steps: List[OptimizationStep] = []

    for step in range(1, config.max_steps + 1):
        planner = make_planner(best_decomposition, best_routing)
        failures = await collect_failures(
            planner, train_cases, config.failure_threshold, metric
        )
        train_average = await average_case_score(planner, train_cases, metric)

        if not failures:
            steps.append(
                OptimizationStep(
                    step, train_average, best_score, best_score, False, 0, best_prompt[:160]
                )
            )
            break

        candidate_decomposition, candidate_routing = propose(
            best_decomposition, best_routing, slot, failures
        )
        candidate_prompt = (
            candidate_decomposition
            if slot == "decomposition_prompt"
            else candidate_routing
        )
        try:
            candidate_prompt = extract_prompt_template(candidate_prompt, placeholders)
        except ValueError:
            # 文本梯度可能生成含多余占位符/JSON 的非法模板，拒绝并回滚。
            steps.append(
                OptimizationStep(
                    step,
                    train_average,
                    best_score,
                    best_score,
                    False,
                    len(failures),
                    "(invalid candidate template)",
                )
            )
            continue
        if slot == "decomposition_prompt":
            candidate_decomposition = candidate_prompt
        else:
            candidate_routing = candidate_prompt

        candidate_report = await evaluate_benchmark(
            make_planner(candidate_decomposition, candidate_routing),
            fixtures,
            config.dev_split,
            metric=metric,
        )
        candidate_score = candidate_report.average_score
        accepted = candidate_score >= best_score if config.rollback else True
        if accepted:
            best_decomposition = candidate_decomposition
            best_routing = candidate_routing
            best_prompt = candidate_prompt
            best_score = candidate_score

        steps.append(
            OptimizationStep(
                step,
                train_average,
                best_score,
                candidate_score,
                accepted,
                len(failures),
                candidate_prompt[:160],
            )
        )
        if accepted and candidate_score >= 0.999:
            break

    return OptimizationResult(
        slot=slot,
        metric=metric,
        baseline_score=baseline.average_score,
        best_score=best_score,
        best_prompt=best_prompt,
        optimizer=optimizer_name,
        steps=steps,
    )


async def run_planner_optimization(
    *,
    backend: OptimizerBackend,
    slots: List[SlotSpec],
    decomposition_prompt: str,
    agent_routing: str,
    executor_llm: ChatOpenAI,
    optimizer_llm: ChatOpenAI,
    registry: Any,
    fixtures: Dict[str, Any],
    max_steps: int = 3,
    failure_threshold: float = 0.8,
    rollback: bool = True,
    train_split: str = "train",
    dev_split: str = "dev",
    objective: str = "l1_l2",
) -> Dict[str, Any]:
    """Optimize the requested slots in order with the chosen backend."""
    if backend == "planner_graph":
        from planner_graph import propose_candidate_via_graph

        def propose(
            decomposition: str,
            routing: str,
            slot: str,
            failures: List[Tuple[Any, Dict[str, Any]]],
        ) -> Tuple[str, str]:
            return propose_candidate_via_graph(
                decomposition,
                routing,
                slot,
                failures,
                executor_llm=executor_llm,
                optimizer_llm=optimizer_llm,
                registry=registry,
            )
    else:
        propose = make_direct_proposer(optimizer_llm)

    config = OptimizationConfig(
        max_steps=max_steps,
        failure_threshold=failure_threshold,
        rollback=rollback,
        train_split=train_split,
        dev_split=dev_split,
    )
    current_decomposition = decomposition_prompt
    current_routing = agent_routing
    results: List[OptimizationResult] = []

    for spec in slots:
        result = await _optimize_slot(
            slot=spec.slot,
            metric=spec.metric,
            placeholders=spec.placeholders,
            decomposition_prompt=current_decomposition,
            agent_routing=current_routing,
            propose=propose,
            executor_llm=executor_llm,
            registry=registry,
            fixtures=fixtures,
            config=config,
            optimizer_name=backend,
        )
        if spec.slot == "decomposition_prompt":
            current_decomposition = result.best_prompt
        else:
            current_routing = result.best_prompt
        results.append(result)

    return {
        "objective": objective,
        "train_split": train_split,
        "dev_split": dev_split,
        "prompts": {
            "decomposition_prompt": current_decomposition,
            "agent_routing": current_routing,
        },
        "results": results,
    }


def save_planner_optimization_artifacts(
    *,
    prompts: Dict[str, str],
    results: List[OptimizationResult],
    output_dir: Path,
    objective: str = "l1_l2",
    train_split: str = "train",
    dev_split: str = "dev",
) -> Tuple[Path, Path]:
    """Persist the retained prompts and the per-round optimization report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = output_dir / "optimized.json"
    prompts_path.write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report_path = output_dir / "optimization_report.json"
    report = {
        "objective": objective,
        "train_split": train_split,
        "dev_split": dev_split,
        "slots": [result.to_dict() for result in results],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return prompts_path, report_path
