"""TaskPlanner 三步 textgrad 计算图（StringBasedFunction 链）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

from langchain_openai import ChatOpenAI

from agent_framework.optimization.decomposition.fixtures import DecompositionBenchmarkCase
from agent_framework.optimization.decomposition.prompt_optimizer import extract_decomposition_prompt
from agent_framework.optimization.optimizers.textgrad_lib.adapter import (
    decomposition_prompt_variable,
    routing_prompt_variable,
)
from agent_framework.optimization.routing.prompt_optimizer import extract_agent_routing_prompt

from .bridge import TaskPlannerSyncBridge
from .loss import build_case_expectation_label, create_planner_graph_loss_fn

OptimizeSlot = Literal["decomposition", "routing"]


@dataclass
class PlannerGraphVariables:
    decomposition_prompt: Any
    agent_routing: Any


class PlannerTextGradGraph:
    """将 decomposition -> dependency -> routing 串成 textgrad 可反传计算图。"""

    def __init__(
        self,
        *,
        bridge: TaskPlannerSyncBridge,
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

        self.variables = PlannerGraphVariables(
            decomposition_prompt=decomposition_prompt_variable(
                extract_decomposition_prompt(decomposition_prompt)
            ),
            agent_routing=routing_prompt_variable(extract_agent_routing_prompt(agent_routing)),
        )
        self.variables.decomposition_prompt.requires_grad = optimize_slot == "decomposition"
        self.variables.agent_routing.requires_grad = optimize_slot == "routing"

        self._decomposition_fn = StringBasedFunction(
            self._decomposition_call,
            "travel task decomposition",
        )
        self._dependency_fn = StringBasedFunction(
            self._dependency_call,
            "travel dependency analysis",
        )
        self._routing_fn = StringBasedFunction(
            self._routing_call,
            "travel agent routing",
        )
        self._loss_fn = create_planner_graph_loss_fn(engine)

    def _active_substeps(self, parsed: dict) -> List[str]:
        return [
            step
            for step in parsed.get("subSteps") or []
            if step and str(step).upper() != "NULL"
        ]

    def _decomposition_call(
        self,
        decomposition_prompt,
        user_query,
        pre_survey_json,
    ) -> str:
        pre_survey = json.loads(pre_survey_json.value)
        parsed = self._bridge.run_decomposition(
            decomposition_prompt=decomposition_prompt.value,
            user_query=user_query.value,
            pre_survey=pre_survey,
        )
        return json.dumps(parsed, ensure_ascii=False)

    def _dependency_call(self, decomposition_json) -> str:
        parsed = json.loads(decomposition_json.value)
        sub_steps = self._active_substeps(parsed)
        execution_order, depends_map = self._bridge.run_dependency_analysis(sub_steps)
        payload = {
            "parsed": parsed,
            "sub_steps": sub_steps,
            "execution_order": execution_order,
            "depends_map": depends_map,
        }
        return json.dumps(payload, ensure_ascii=False)

    def _routing_call(self, agent_routing, dependency_json) -> str:
        state = json.loads(dependency_json.value)
        routed = self._bridge.route_to_agents(
            agent_routing=agent_routing.value,
            sub_steps=state["sub_steps"],
            execution_order=state["execution_order"],
            depends_map=state["depends_map"],
        )
        output = self._bridge.format_pipeline_output(
            parsed=state["parsed"],
            execution_order=state["execution_order"],
            depends_map=state["depends_map"],
            routed_subtasks=routed,
        )
        return output

    def forward_case(self, case: DecompositionBenchmarkCase):
        from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad

        _, Variable, _, _ = require_textgrad()

        query_var = Variable(
            case.query,
            requires_grad=False,
            role_description="user travel query",
        )
        pre_survey_var = Variable(
            json.dumps(case.pre_survey or {}, ensure_ascii=False),
            requires_grad=False,
            role_description="pre-survey context",
        )

        decomposition_out = self._decomposition_fn(
            {
                "decomposition_prompt": self.variables.decomposition_prompt,
                "user_query": query_var,
                "pre_survey_json": pre_survey_var,
            }
        )
        dependency_out = self._dependency_fn({"decomposition_json": decomposition_out})
        pipeline_out = self._routing_fn(
            {
                "agent_routing": self.variables.agent_routing,
                "dependency_json": dependency_out,
            }
        )
        label = Variable(
            build_case_expectation_label(case),
            requires_grad=False,
            role_description="benchmark expectation specification",
        )
        loss = self._loss_fn([pipeline_out, label])
        return pipeline_out, loss

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
        registry: Any,
        locale: str,
        decomposition_prompt: str,
        agent_routing: str,
        optimize_slot: OptimizeSlot,
        optimizer_llm: ChatOpenAI,
    ) -> "PlannerTextGradGraph":
        from agent_framework.optimization.optimizers.textgrad_lib.engine import create_textgrad_engine

        bridge = TaskPlannerSyncBridge.from_prompts(
            executor_llm=executor_llm,
            registry=registry,
            locale=locale,
            decomposition_prompt=extract_decomposition_prompt(decomposition_prompt),
            agent_routing=extract_agent_routing_prompt(agent_routing),
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
