"""单 Agent 单节点 textgrad 计算图（StringBasedFunction）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from langchain_openai import ChatOpenAI

from agent_framework.optimization.agents.fixtures import SingleAgentCase
from agent_framework.optimization.agents.runtime import AgentSyncBridge, extract_agent_system_prompt

from .adapter import agent_system_prompt_variable, read_agent_system_prompt_value
from .loss import build_single_agent_expectation_label, create_agent_graph_loss_fn

TEXTGRAD_AGENT_GRAPH_OPTIMIZER_NAME = "textgrad_agent_graph"


@dataclass
class SingleAgentGraphVariables:
    """图中可训练变量集合（system_prompt）。"""

    system_prompt: Any


class SingleAgentTextGradGraph:
    """将 ``build_agent().invoke`` 包成单节点 StringBasedFunction 计算图。"""

    def __init__(
        self,
        *,
        bridge: AgentSyncBridge,
        system_prompt_template: str,
        agent_name: str,
        engine,
    ):
        from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad
        from textgrad.autograd import StringBasedFunction

        require_textgrad()
        self._bridge = bridge
        self._engine = engine
        self._agent_name = agent_name

        self.variables = SingleAgentGraphVariables(
            system_prompt=agent_system_prompt_variable(
                extract_agent_system_prompt(system_prompt_template, agent_name=agent_name),
                agent_name=agent_name,
            ),
        )

        self._agent_fn = StringBasedFunction(
            self._agent_call,
            f"travel {agent_name} react agent",
        )
        self._loss_fn = create_agent_graph_loss_fn(engine)

    def _agent_call(self, system_prompt, user_query, thread_id) -> str:
        """StringBasedFunction 回调：用当前 Variable 中的 prompt 执行 Agent。"""
        state = self._bridge.invoke(
            system_prompt_template=system_prompt.value,
            user_query=user_query.value,
            thread_id=thread_id.value,
        )
        return self._bridge.format_agent_output(state)

    def forward_case(self, case: SingleAgentCase):
        """单条 benchmark case 的 forward + loss。"""
        from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad

        _, Variable, _, _ = require_textgrad()

        query_var = Variable(
            case.user_query,
            requires_grad=False,
            role_description="user query to sub-agent",
        )
        thread_var = Variable(
            case.case_id,
            requires_grad=False,
            role_description="agent thread id",
        )

        agent_out = self._agent_fn(
            {
                "system_prompt": self.variables.system_prompt,
                "user_query": query_var,
                "thread_id": thread_var,
            }
        )
        label = Variable(
            build_single_agent_expectation_label(case),
            requires_grad=False,
            role_description="single-agent benchmark expectation",
        )
        loss = self._loss_fn([agent_out, label])
        return agent_out, loss

    def trainable_parameters(self) -> List[Any]:
        return [self.variables.system_prompt]

    @classmethod
    def create(
        cls,
        *,
        executor_llm: ChatOpenAI,
        locale: str,
        system_prompt_template: str,
        agent_name: str,
        optimizer_llm: ChatOpenAI,
    ) -> "SingleAgentTextGradGraph":
        from agent_framework.optimization.optimizers.textgrad_lib.engine import create_textgrad_engine

        bridge = AgentSyncBridge(llm=executor_llm, locale=locale, agent_name=agent_name)
        engine = create_textgrad_engine(optimizer_llm)
        return cls(
            bridge=bridge,
            system_prompt_template=system_prompt_template,
            agent_name=agent_name,
            engine=engine,
        )

    def read_optimized_prompt_template(self) -> str:
        return read_agent_system_prompt_value(
            self.variables.system_prompt,
            agent_name=self._agent_name,
        )
