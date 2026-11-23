"""TextGrad graph 模式下的 loss 与 optimizer 约束模板。"""

PLANNER_GRAPH_EVAL_INSTRUCTION = """You evaluate a travel TaskPlanner pipeline output against benchmark expectations.

The pipeline runs: task decomposition -> dependency analysis -> agent routing.
Compare the actual pipeline output with the expected benchmark specification.
Identify concrete gaps in subtask structure, dependencies, and agent assignments.
Be specific so prompt improvements can fix the failures on similar queries.
"""

PLANNER_GRAPH_ROLE_DESCRIPTIONS = [
    "actual planner pipeline output",
    "benchmark expectation specification",
]

DECOMPOSITION_GRAPH_CONSTRAINTS = [
    "The optimized prompt must retain placeholders {background_info}, {agent_team}, {user_input}.",
    "Return only the revised decomposition prompt template.",
]

ROUTING_GRAPH_CONSTRAINTS = [
    "The optimized prompt must retain placeholders {agent_team}, {subtasks_json}, {today}, and {time_anchor}.",
    "The optimized prompt must forbid past years like 2024 in routing params dates.",
    "Return only the revised agent_routing prompt template.",
]
