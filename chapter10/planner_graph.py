"""Planner computation-graph TextGrad backend.

Builds the decompose -> dependency -> routing pipeline into a textgrad
``StringBasedFunction`` graph. The loss compares the pipeline output with the
benchmark expectation, so the text gradient flows through the real planning
steps back to the trainable prompt.

The synchronous bridge reuses the prompt-rendering and routing-normalization
helpers from :mod:`runtime`, so only the sync/async model invocation differs.

:author: zhanghong, liuhui
:date: 2026-11-23
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from langchain_openai import ChatOpenAI

from llm import invoke_chat_sync, invoke_json_sync
from runtime import (
    build_routing_subtasks,
    build_task_id_map,
    normalize_routing_output,
    render_decomposition_prompt,
    render_routing_prompt,
)
from textgrad_engine import create_textgrad_engine

# planner_graph 的 loss 用 MultiFieldEvaluation 比较「流水线输出 vs 期望」，
# 指令与角色描述改写自 Chapter-11 的 textgrad_graph/prompts.py；
# 约束模板的占位符与 optimizable_prompts.py 保持一致。
PLANNER_GRAPH_EVAL_INSTRUCTION = """You evaluate a travel planning pipeline output against benchmark expectations.

The pipeline runs: task decomposition -> dependency analysis -> agent routing.
Compare the actual pipeline output with the expected benchmark specification.
Identify concrete gaps in subtask structure, dependencies, and agent assignments.
Be specific so prompt improvements can fix the failures on similar queries.
"""

PLANNER_GRAPH_ROLE_DESCRIPTIONS = [
    "actual planner pipeline output",
    "benchmark expectation specification",
]

DECOMPOSITION_GRAPH_CONSTRAINTS = (
    "The optimized prompt must retain placeholders {background_info}, {capability_catalog}, {user_input}.",
    "Return only the revised decomposition prompt template.",
)

ROUTING_GRAPH_CONSTRAINTS = (
    "The optimized prompt must retain placeholders {agent_team}, {pre_survey_json}, {subtasks_json}, {depends_on_json}.",
    "Return only the revised agent_routing prompt template.",
)


class SyncPlannerBridge:
    """Run the three planning steps synchronously inside the textgrad graph."""

    def __init__(self, executor_llm: ChatOpenAI, registry: Any):
        self._executor_llm = executor_llm
        self._registry = registry

    def decompose_task(
        self,
        decomposition_prompt: str,
        user_query: str,
        pre_survey: Dict[str, Any],
    ) -> Dict[str, Any]:
        """同步跑任务拆解（``StringBasedFunction`` 要求同步前向）。"""
        from task_decomposition import parse_task_decomposition_response

        background = "【思维链预调查】\n" + json.dumps(
            {key: value for key, value in pre_survey.items() if key != "raw_text"},
            ensure_ascii=False,
        )
        prompt = render_decomposition_prompt(
            decomposition_prompt,
            background,
            self._registry.get_capability_catalog_text(),
            user_query,
        )
        response = invoke_chat_sync(self._executor_llm, prompt)
        return parse_task_decomposition_response(response)

    def analyze_task_dependencies(
        self,
        sub_steps: List[str],
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """同步跑依赖分析，返回执行顺序与 depends_on 映射。"""
        from dependency_analysis import (
            DEPENDENCY_ANALYSIS_SYSTEM_PROMPT,
            build_dependency_analysis_prompt,
            normalize_dependency_analysis,
        )

        id_to_task = build_task_id_map(sub_steps)
        if len(id_to_task) <= 1:
            return list(id_to_task), {task_id: [] for task_id in id_to_task}

        prompt = build_dependency_analysis_prompt(
            id_to_task,
            self._registry.get_agent_parameters_text(),
        )
        payload = invoke_json_sync(
            self._executor_llm,
            prompt,
            system_prompt=DEPENDENCY_ANALYSIS_SYSTEM_PROMPT,
        )
        return normalize_dependency_analysis(payload, list(id_to_task))

    def route_to_agents(
        self,
        agent_routing: str,
        sub_steps: List[str],
        execution_order: List[str],
        depends_map: Dict[str, List[str]],
        pre_survey: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """同步跑智能体路由，返回带 agent 与 params 的记录列表。"""
        subtasks_for_prompt = build_routing_subtasks(
            execution_order, sub_steps, depends_map
        )
        agent_team = (
            self._registry.get_all_agents_text()
            + "\n"
            + self._registry.get_agent_parameters_text()
        )
        prompt = render_routing_prompt(
            agent_routing,
            agent_team,
            pre_survey,
            subtasks_for_prompt,
            depends_map,
        )
        routed = invoke_json_sync(self._executor_llm, prompt)
        valid_agents = set(self._registry.get_agent_names())
        allowed_fields = {
            name: self._registry.get_agent_input_fields(name) for name in valid_agents
        }
        return normalize_routing_output(
            routed, execution_order, valid_agents, allowed_fields
        )

    @staticmethod
    def format_pipeline_output(
        parsed: Dict[str, Any],
        execution_order: List[str],
        depends_map: Dict[str, List[str]],
        routed_subtasks: List[Dict[str, Any]],
    ) -> str:
        """把 planner 三步结果序列化为 loss 输入文本。"""
        payload = {
            "total_goal": parsed.get("total_goal"),
            "sub_steps": parsed.get("sub_steps"),
            "execution_order": execution_order,
            "depends_map": depends_map,
            "routed_subtasks": routed_subtasks,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


def build_case_expectation_label(case: Dict[str, Any]) -> str:
    """Turn a benchmark case's expectations into loss-label text."""
    expect = case.get("expect") or {}
    lines = [
        f"case_id: {case.get('id')}",
        f"query: {case.get('query')}",
        f"min_subtasks: {expect.get('min_subtasks', 1)}",
        f"max_subtasks: {expect.get('max_subtasks', 99)}",
    ]
    if expect.get("required_keywords"):
        lines.append(f"required_keywords: {expect['required_keywords']}")
    if expect.get("forbidden_keywords"):
        lines.append(f"forbidden_keywords: {expect['forbidden_keywords']}")
    if expect.get("mappable_agents"):
        lines.append(f"mappable_agents: {expect['mappable_agents']}")
    if case.get("expect_dependency"):
        lines.append(
            "expected_dependency: "
            + json.dumps(case["expect_dependency"], ensure_ascii=False)
        )
    if case.get("expect_routing"):
        assignments = (case["expect_routing"] or {}).get("assignments") or []
        lines.append(
            "expected_routing: " + json.dumps(assignments, ensure_ascii=False)
        )
    return "\n".join(lines)


def create_graph_loss_fn(engine: Any):
    """Return a ``MultiFieldEvaluation`` comparing pipeline output and label."""
    from textgrad import Variable
    from textgrad.loss import MultiFieldEvaluation

    instruction = Variable(
        PLANNER_GRAPH_EVAL_INSTRUCTION,
        requires_grad=False,
        role_description="planner pipeline evaluation instruction",
    )
    return MultiFieldEvaluation(instruction, PLANNER_GRAPH_ROLE_DESCRIPTIONS, engine)


class PlannerGraph:
    """The decompose -> dependency -> routing chain as a textgrad graph."""

    def __init__(
        self,
        *,
        bridge: SyncPlannerBridge,
        decomposition_prompt: str,
        agent_routing: str,
        optimize_slot: str,
        engine: Any,
    ):
        from textgrad import Variable
        from textgrad.autograd import StringBasedFunction

        self._bridge = bridge
        self._engine = engine
        self._optimize_slot = optimize_slot
        self._decomposition_var = Variable(
            decomposition_prompt,
            requires_grad=optimize_slot == "decomposition_prompt",
            role_description="travel decomposition prompt template",
        )
        self._routing_var = Variable(
            agent_routing,
            requires_grad=optimize_slot == "agent_routing",
            role_description="travel agent_routing prompt template",
        )
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
        self._loss_fn = create_graph_loss_fn(engine)

    @staticmethod
    def _active_sub_steps(parsed: Dict[str, Any]) -> List[str]:
        return [
            step
            for step in (parsed.get("sub_steps") or [])
            if step and str(step).upper() != "NULL"
        ]

    def _decomposition_call(self, decomposition_prompt, user_query, pre_survey_json):
        pre_survey = json.loads(pre_survey_json.value)
        parsed = self._bridge.decompose_task(
            decomposition_prompt=decomposition_prompt.value,
            user_query=user_query.value,
            pre_survey=pre_survey,
        )
        return json.dumps({"parsed": parsed, "pre_survey": pre_survey}, ensure_ascii=False)

    def _dependency_call(self, decomposition_json):
        state = json.loads(decomposition_json.value)
        parsed = state["parsed"]
        sub_steps = self._active_sub_steps(parsed)
        execution_order, depends_map = self._bridge.analyze_task_dependencies(sub_steps)
        return json.dumps(
            {
                "parsed": parsed,
                "pre_survey": state["pre_survey"],
                "sub_steps": sub_steps,
                "execution_order": execution_order,
                "depends_map": depends_map,
            },
            ensure_ascii=False,
        )

    def _routing_call(self, agent_routing, dependency_json):
        state = json.loads(dependency_json.value)
        routed = self._bridge.route_to_agents(
            agent_routing=agent_routing.value,
            sub_steps=state["sub_steps"],
            execution_order=state["execution_order"],
            depends_map=state["depends_map"],
            pre_survey=state["pre_survey"],
        )
        return self._bridge.format_pipeline_output(
            parsed=state["parsed"],
            execution_order=state["execution_order"],
            depends_map=state["depends_map"],
            routed_subtasks=routed,
        )

    def forward_case(self, case: Dict[str, Any]):
        """Run one case through the graph and return ``(output, loss)``."""
        from textgrad import Variable

        query_var = Variable(
            case["query"],
            requires_grad=False,
            role_description="user travel query",
        )
        pre_survey_var = Variable(
            json.dumps(case.get("pre_survey") or {}, ensure_ascii=False),
            requires_grad=False,
            role_description="pre-survey context",
        )
        decomposition_out = self._decomposition_fn(
            {
                "decomposition_prompt": self._decomposition_var,
                "user_query": query_var,
                "pre_survey_json": pre_survey_var,
            }
        )
        dependency_out = self._dependency_fn({"decomposition_json": decomposition_out})
        pipeline_out = self._routing_fn(
            {
                "agent_routing": self._routing_var,
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
        parameters = []
        if self._decomposition_var.requires_grad:
            parameters.append(self._decomposition_var)
        if self._routing_var.requires_grad:
            parameters.append(self._routing_var)
        return parameters

    @classmethod
    def create(
        cls,
        *,
        executor_llm: ChatOpenAI,
        registry: Any,
        decomposition_prompt: str,
        agent_routing: str,
        optimize_slot: str,
        optimizer_llm: ChatOpenAI,
    ) -> "PlannerGraph":
        bridge = SyncPlannerBridge(executor_llm, registry)
        engine = create_textgrad_engine(optimizer_llm)
        return cls(
            bridge=bridge,
            decomposition_prompt=decomposition_prompt,
            agent_routing=agent_routing,
            optimize_slot=optimize_slot,
            engine=engine,
        )

    def read_optimized_prompts(self) -> Tuple[str, str]:
        return (
            str(self._decomposition_var.value),
            str(self._routing_var.value),
        )


def run_planner_graph_step(
    graph: PlannerGraph,
    failure_cases: List[Dict[str, Any]],
    *,
    constraints: Tuple[str, ...],
) -> None:
    """Forward all failing cases, sum losses, backward, and update prompts."""
    import textgrad as tg
    from textgrad.optimizer import TextualGradientDescent

    if not failure_cases:
        return

    losses = []
    for case in failure_cases:
        _, loss = graph.forward_case(case)
        losses.append(loss)

    total_loss = tg.sum(losses) if len(losses) > 1 else losses[0]
    total_loss.backward(graph._engine)

    parameters = graph.trainable_parameters()
    if not parameters:
        raise ValueError("graph 中没有 requires_grad=True 的 prompt Variable")

    optimizer = TextualGradientDescent(
        parameters=parameters,
        engine=graph._engine,
        constraints=list(constraints),
    )
    optimizer.step()
    optimizer.zero_grad()


def propose_candidate_via_graph(
    decomposition_prompt: str,
    agent_routing: str,
    slot: str,
    failures: List[Tuple[Any, Dict[str, Any]]],
    *,
    executor_llm: ChatOpenAI,
    optimizer_llm: ChatOpenAI,
    registry: Any,
) -> Tuple[str, str]:
    """Propose updated prompts by backpropagating through the planner graph."""
    constraints = (
        DECOMPOSITION_GRAPH_CONSTRAINTS
        if slot == "decomposition_prompt"
        else ROUTING_GRAPH_CONSTRAINTS
    )
    graph = PlannerGraph.create(
        executor_llm=executor_llm,
        registry=registry,
        decomposition_prompt=decomposition_prompt,
        agent_routing=agent_routing,
        optimize_slot=slot,
        optimizer_llm=optimizer_llm,
    )
    run_planner_graph_step(
        graph,
        [case for _, case in failures],
        constraints=constraints,
    )
    return graph.read_optimized_prompts()
