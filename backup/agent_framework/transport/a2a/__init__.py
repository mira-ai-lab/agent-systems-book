"""Agent 传输层（本地 / A2A 远程）。"""

from agent_framework.transport.a2a.agent_graphs import build_a2a_agent_graph, build_a2a_agent_graphs
from agent_framework.transport.a2a.client import A2AClient

__all__ = ["A2AClient", "build_a2a_agent_graph", "build_a2a_agent_graphs"]
