"""记忆使用：构造短期上下文、长期记忆提示词并调用助手模型。

更新时间：2026-11-12

该模块负责把检索得到的画像事实和偏好记忆转换为模型可使用的上下文，
并维护按线程隔离的短期对话缓冲。
"""

from __future__ import annotations

from typing import Any


MEMORY_PROMPT_TEMPLATE = """你是一个具备长期记忆的智能助手。请结合【最近对话】与【相关长期记忆】回答用户问题。

【最近对话】（按时间顺序）
{recent_dialogue}

【相关长期记忆】（用户画像事实与偏好记忆）
{memory_context}

【当前问题】
{query}
"""


class ThreadShortTermMemory:
    """按线程隔离的短期对话缓冲。"""

    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self._threads: dict[str, list[dict[str, str]]] = {}

    def append(self, thread_id: str, role: str, content: str) -> None:
        """向指定线程追加一条消息，并保留最近 ``max_turns`` 条。"""
        self._threads.setdefault(thread_id, []).append(
            {"role": role, "content": content}
        )
        if len(self._threads[thread_id]) > self.max_turns:
            self._threads[thread_id] = self._threads[thread_id][-self.max_turns :]

    def format_recent(self, thread_id: str, last_n: int = 6) -> str:
        """格式化指定线程最近若干条消息。"""
        turns = self._threads.get(thread_id, [])[-last_n:]
        if not turns:
            return "（无历史对话）"
        return "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in turns
        )

    def clear(self) -> None:
        """清空全部线程的短期对话。"""
        self._threads.clear()


def build_memory_prompt(*, query: str, hits: list[dict[str, Any]]) -> str:
    """根据当前问题与检索结果构造只包含长期记忆的提示词。"""
    if hits:
        memory_context = "\n".join(
            f"- [{hit.get('memory_type', 'general')}] "
            f"{hit.get('summary') or hit.get('content')}"
            for hit in hits
        )
    else:
        memory_context = "（未检索到相关长期记忆）"
    return (
        "你是一个具备长期记忆的智能助手。请结合【相关长期记忆】回答用户问题。\n\n"
        f"【相关长期记忆】\n{memory_context}\n\n"
        f"【用户问题】\n{query}\n"
    )


class MemoryContextBuilder:
    """组合短期对话与长期记忆并生成最终提示词。"""

    def __init__(
        self,
        short_term: ThreadShortTermMemory | None = None,
        *,
        short_term_max_turns: int = 20,
    ) -> None:
        self.short_term = short_term or ThreadShortTermMemory(short_term_max_turns)

    def record_turn(
        self,
        thread_id: str,
        user_content: str,
        assistant_content: str | None = None,
    ) -> None:
        """记录一轮用户消息和可选的助手消息。"""
        self.short_term.append(thread_id, "user", user_content)
        if assistant_content:
            self.short_term.append(thread_id, "assistant", assistant_content)

    def build_prompt(
        self,
        thread_id: str,
        query: str,
        *,
        hits: list[dict[str, Any]],
        recent_n: int = 6,
    ) -> str:
        """构造包含短期对话和长期画像记忆的完整提示词。"""
        if hits:
            memory_context = "\n".join(
                f"- [{hit.get('memory_type', 'general')}] "
                f"{hit.get('summary') or hit.get('content')} "
                f"(final={hit.get('final_score', 0.0):.4f}, "
                f"vector_sim={hit.get('vector_sim', 0.0):.4f}, "
                f"rrf={hit.get('rrf_score', 0.0):.4f})"
                for hit in hits
            )
        else:
            memory_context = "（未检索到相关长期记忆）"
        return MEMORY_PROMPT_TEMPLATE.format(
            recent_dialogue=self.short_term.format_recent(thread_id, recent_n),
            memory_context=memory_context,
            query=query,
        )


class AssistantLLM:
    """将 LangChain 对话模型封装为统一的文本聊天接口。"""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def chat(self, messages: list[dict[str, str]]) -> str:
        """调用模型并返回纯文本答复。"""
        return str(self.llm.invoke(messages).content)
