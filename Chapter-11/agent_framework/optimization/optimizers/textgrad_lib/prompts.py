"""LLM templates for textgrad library decomposition optimization."""

DECOMPOSITION_TEXTGRAD_LOSS_PROMPT = """You are evaluating a travel task-decomposition prompt template.

The prompt must:
1. Keep placeholders {background_info}, {agent_team}, {user_input}
2. Decompose user requests into subtasks mappable to the travel agent team
3. Avoid over-expansion and forbidden task types shown in the failures

Use the failure cases below as critique. The variable you receive is the prompt template itself.
Explain what should improve so the prompt produces better decompositions on similar queries.
"""

DECOMPOSITION_TEXTGRAD_CONSTRAINTS = [
    "The optimized prompt must retain placeholders {background_info}, {agent_team}, {user_input}.",
    "Do not output analysis; return only the revised prompt template.",
]

# Backward-compatible alias
TEXTGRAD_OPTIMIZER_CONSTRAINTS = DECOMPOSITION_TEXTGRAD_CONSTRAINTS

ROUTING_TEXTGRAD_LOSS_PROMPT = """You are evaluating a travel agent_routing prompt template.

The prompt must:
1. Keep placeholders {agent_team}, {subtasks_json}, {today}, and {time_anchor}
2. Assign each subtask to the correct specialist agent in the travel team
3. Require YYYY-MM-DD dates aligned with the time anchor; never 2024 or other past years
4. Respect the expected routing patterns shown in the failure cases

Use the failure cases below as critique. The variable you receive is the routing prompt template itself.
Explain what should improve so routing matches expected agents on similar subtask sets.
"""

ROUTING_TEXTGRAD_CONSTRAINTS = [
    "The optimized prompt must retain placeholders {agent_team}, {subtasks_json}, {today}, and {time_anchor}.",
    "The optimized prompt must forbid past years like 2024 in routing params dates.",
    "Do not output analysis; return only the revised routing prompt template.",
]
