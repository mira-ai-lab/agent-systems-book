"""LLM templates for TextGrad-style decomposition prompt optimization."""

PROMPT_REVISION_TEMPLATE = """你是 travel 领域「任务拆解 prompt」优化器（TextGrad 风格）。

你的任务：根据失败样本的文字反馈，改进 decomposition_prompt，使 TaskPlanner 产出更符合预期的子任务拆解。

【当前 decomposition_prompt】
{current_prompt}

【可用 Agent 团队（子任务必须可映射到这些 Agent）】
{agent_team}

【失败样本】
{failure_feedback}

【硬性要求】
1. 只输出改进后的完整 decomposition_prompt，不要解释。
2. 必须保留且仅使用这三个占位符：{{background_info}}、{{agent_team}}、{{user_input}}。
3. 保持输出格式要求：包含「# 目标」与「# 任务拆解」及「- 子任务」列表。
4. 强调：单意图查询（如只问天气）不要过度拆解；多意图查询应覆盖所需 Agent。
5. 子任务必须是信息完整的祈使句，包含城市/日期/预算等槽位。
"""

FAILURE_CASE_TEMPLATE = """--- Case {case_id} ---
用户输入: {query}
模型输出:
{raw_output}
得分: {score:.3f}
问题: {details}
期望 Agent: {expected_agents}
子任务数量范围: [{min_subtasks}, {max_subtasks}]
"""
