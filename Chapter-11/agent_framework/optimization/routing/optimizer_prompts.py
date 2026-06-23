"""LLM templates for TextGrad-style agent_routing prompt optimization."""

ROUTING_REVISION_TEMPLATE = """你是 travel 领域「子任务路由 prompt」优化器（TextGrad 风格）。

你的任务：根据失败样本反馈，改进 agent_routing prompt，使 TaskPlanner 为每个子任务分配到正确的 Agent 并给出 params。

【当前 agent_routing prompt】
{current_prompt}

【可用 Agent 团队】
{agent_team}

【失败样本】
{failure_feedback}

【硬性要求】
1. 只输出改进后的完整 agent_routing prompt，不要解释。
2. 必须保留占位符：{{agent_team}}、{{subtasks_json}}。
3. 输出仍要求模型返回 JSON 数组，每项含 task_id / description / agent / params / depends_on。
4. agent 必须是团队中的合法名称；params 字段名需匹配该 Agent 技能 inputSchema。
5. 单 Agent 意图不要路由到多个无关 Agent；多子任务应一一对应正确 Agent。
"""

ROUTING_FAILURE_CASE_TEMPLATE = """--- Case {case_id} ---
用户输入: {query}
子任务输入: {subtasks_input}
模型路由输出:
{raw_output}
得分: {score:.3f}
问题: {details}
期望分配: {expected_assignments}
"""
