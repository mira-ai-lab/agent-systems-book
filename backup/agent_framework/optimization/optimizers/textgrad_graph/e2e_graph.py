"""Planner prompts → 完整 E2E 编排的 textgrad 计算图（Phase B2）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Optional

from langchain_openai import ChatOpenAI

from agent_framework.optimization.decomposition.fixtures import DecompositionBenchmarkCase
from agent_framework.optimization.e2e.evaluator import E2eCaseResult
from agent_framework.optimization.decomposition.prompt_optimizer import extract_decomposition_prompt
from agent_framework.optimization.optimizers.textgrad_lib.adapter import (
    decomposition_prompt_variable,
    routing_prompt_variable,
)
from agent_framework.optimization.routing.prompt_optimizer import extract_agent_routing_prompt

from .e2e_bridge import E2eOrchestratorSyncBridge
from .e2e_loss import build_e2e_expectation_label, create_e2e_graph_loss_fn
from .graph import OptimizeSlot

TEXTGRAD_GRAPH_E2E_OPTIMIZER_NAME = "textgrad_graph_e2e"


@dataclass
class PlannerE2eGraphVariables:
    decomposition_prompt: Any
    agent_routing: Any


class PlannerPromptE2eGraph:
    """将 planner prompts 经完整 E2E 编排串成 textgrad 可反传图（LangGraph 不改动）。"""

    def __init__(
        self,
        *,
        bridge: E2eOrchestratorSyncBridge,
        decomposition_prompt: str,
        agent_routing: str,
        optimize_slot: OptimizeSlot,
        engine,
    ):
        from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad
        from textgrad.autograd import StringBasedFunction

        require_textgrad()
        self._bridge = bridge
        self._engine = engine
        self._optimize_slot = optimize_slot

        self.variables = PlannerE2eGraphVariables(
            decomposition_prompt=decomposition_prompt_variable(
                extract_decomposition_prompt(decomposition_prompt)
            ),
            agent_routing=routing_prompt_variable(extract_agent_routing_prompt(agent_routing)),
        )
        self.variables.decomposition_prompt.requires_grad = optimize_slot == "decomposition"
        self.variables.agent_routing.requires_grad = optimize_slot == "routing"

        self._e2e_fn = StringBasedFunction(
            self._e2e_call,
            "travel end-to-end orchestration with planner prompts",
        )
        self._loss_fn = create_e2e_graph_loss_fn(engine)

    def _e2e_call(
        self,
        decomposition_prompt,
        agent_routing,
        user_query,
        thread_id,
    ) -> str:
        result = self._bridge.process_request(
            decomposition_prompt=decomposition_prompt.value,
            agent_routing=agent_routing.value,
            user_query=user_query.value,
            thread_id=thread_id.value,
        )
        return self._bridge.format_e2e_output(result)

    def forward_case(
        self,
        case: DecompositionBenchmarkCase,
        *,
        rule_failures: Optional[List[str]] = None,
    ):
        from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad

        _, Variable, _, _ = require_textgrad()

        query_var = Variable(
            case.query,
            requires_grad=False,
            role_description="user travel query",
        )
        thread_var = Variable(
            case.case_id,
            requires_grad=False,
            role_description="e2e thread id",
        )

        e2e_out = self._e2e_fn(
            {
                "decomposition_prompt": self.variables.decomposition_prompt,
                "agent_routing": self.variables.agent_routing,
                "user_query": query_var,
                "thread_id": thread_var,
            }
        )
        label = Variable(
            build_e2e_expectation_label(case, rule_failures=rule_failures),
            requires_grad=False,
            role_description="e2e benchmark expectation specification",
        )
        loss = self._loss_fn([e2e_out, label])
        return e2e_out, loss

    def forward_failure(self, result: E2eCaseResult, case: DecompositionBenchmarkCase):
        """Forward a train failure using rule scorer details for loss alignment."""
        return self.forward_case(case, rule_failures=list(result.score.details))

    def trainable_parameters(self) -> List[Any]:
        params = []
        if self.variables.decomposition_prompt.requires_grad:
            params.append(self.variables.decomposition_prompt)
        if self.variables.agent_routing.requires_grad:
            params.append(self.variables.agent_routing)
        return params

    @classmethod
    def create(
        cls,
        *,
        executor_llm: ChatOpenAI,
        locale: str,
        decomposition_prompt: str,
        agent_routing: str,
        optimize_slot: OptimizeSlot,
        optimizer_llm: ChatOpenAI,
        e2e_profile: str = "workflow",
        e2e_timeout_sec: Optional[float] = None,
        enable_guess_agent: bool = True,
    ) -> "PlannerPromptE2eGraph":
        from agent_framework.optimization.optimizers.textgrad_lib.engine import create_textgrad_engine

        bridge = E2eOrchestratorSyncBridge(
            executor_llm=executor_llm,
            locale=locale,
            profile=e2e_profile,
            enable_memory=False,
            enable_guess_agent=enable_guess_agent,
            timeout_sec=e2e_timeout_sec,
        )
        engine = create_textgrad_engine(optimizer_llm)
        return cls(
            bridge=bridge,
            decomposition_prompt=decomposition_prompt,
            agent_routing=agent_routing,
            optimize_slot=optimize_slot,
            engine=engine,
        )

    def read_optimized_prompts(self) -> tuple[str, str]:
        from agent_framework.optimization.optimizers.textgrad_lib.adapter import (
            read_decomposition_prompt_value,
            read_routing_prompt_value,
        )

        return (
            read_decomposition_prompt_value(self.variables.decomposition_prompt),
            read_routing_prompt_value(self.variables.agent_routing),
        )
