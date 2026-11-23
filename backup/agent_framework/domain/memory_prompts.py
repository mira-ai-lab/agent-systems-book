"""通用长期记忆 prompt（与具体领域无关）。"""

MEMORY_PROMPT_TEMPLATE = """你是一个有记忆的智能助手。请结合【最近对话】与【相关长期记忆】回答用户问题。

【最近对话】（来自 Checkpoint / 本线程短期缓冲，按时间顺序）
{recent_dialogue}

【相关长期记忆】（向量检索 + 重排后的 top-k）
{memory_context}

【当前问题】
{query}
"""

MEMORY_EXTRACT_PROMPT = """请从以下内容中提取核心记忆点，保留所有重要信息：
{content}

只输出一个 JSON 对象，字段如下：
{{
    "summary": "一句话总结",
    "key_points": ["要点1", "要点2"],
    "importance": 0.5
}}
"""
