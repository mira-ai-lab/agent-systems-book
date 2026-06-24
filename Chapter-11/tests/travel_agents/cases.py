"""Load travel single-agent test cases (兼容层，canonical 在 optimization.agents.fixtures)。"""

from agent_framework.optimization.agents.fixtures import (  # noqa: F401
    CASES_PATH,
    SingleAgentCase,
    SingleAgentCaseFixtures,
    default_cases_path,
    load_single_agent_cases,
)

__all__ = [
    "CASES_PATH",
    "SingleAgentCase",
    "SingleAgentCaseFixtures",
    "default_cases_path",
    "load_single_agent_cases",
]
