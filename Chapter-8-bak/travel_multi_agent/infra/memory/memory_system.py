"""
Chapter-8: 长期记忆模块 — 基于 Chapter-3 的简化可运行实现

提供向量检索、短期对话缓冲、记忆写入与 build_prompt。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from travel_multi_agent.domain.prompts import MEMORY_EXTRACT_PROMPT, MEMORY_PROMPT_TEMPLATE


class ThreadShortTermMemory:
    """短期记忆"""

    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self._threads: Dict[str, List[Dict[str, str]]] = {}

    def append(self, thread_id: str, role: str, content: str) -> None:
        self._threads.setdefault(thread_id, []).append({"role": role, "content": content})
        if len(self._threads[thread_id]) > self.max_turns:
            self._threads[thread_id] = self._threads[thread_id][-self.max_turns :]

    def format_recent(self, thread_id: str, last_n: int = 6) -> str:
        turns = self._threads.get(thread_id, [])[-last_n:]
        if not turns:
            return "（无历史对话）"
        return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


class LongTermMemory:
    """
    长期记忆。
    优先使用 Chroma + DashScope 嵌入；不可用时回退到内存列表。
    """

    def __init__(
        self,
        user_id: str = "central_agent_user",
        persist_directory: str = "./chroma_memory",
        collection_name: str = "central_agent_memory",
        top_k: int = 5,
        llm: Any = None,
    ) -> None:
        self.user_id = user_id
        self.top_k = top_k
        self.llm = llm
        self.short_term = ThreadShortTermMemory()
        self._fallback_store: List[Dict[str, Any]] = []
        self._use_chroma = False

        try:
            import chromadb  # noqa: F401
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

            api_key = __import__("os").getenv("DASHSCOPE_API_KEY") or __import__("os").getenv("OPENAI_API_KEY")
            base_url = __import__("os").getenv(
                "DASHSCOPE_EMBEDDING_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            model = __import__("os").getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")

            ef = OpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=base_url,
                model_name=model,
            )
            client = chromadb.PersistentClient(path=persist_directory)
            self.collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=ef,
                metadata={"user_id": user_id},
            )
            self._use_chroma = True
        except Exception:
            self.collection = None

    def search_memories(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        if self._use_chroma and self.collection and self.collection.count() > 0:
            hits = self.collection.query(
                query_texts=[query],
                n_results=min(k, self.collection.count()),
                where={"user_id": self.user_id},
            )
            results = []
            for i, doc_id in enumerate(hits["ids"][0]):
                meta = hits["metadatas"][0][i]
                results.append({
                    "id": doc_id,
                    "summary": meta.get("summary", doc_id),
                    "key_points": json.loads(meta.get("key_points", "[]")),
                    "importance": float(meta.get("importance", 0.5)),
                    "content": hits["documents"][0][i],
                })
            return results

        scored = []
        q = query.lower()
        for item in self._fallback_store:
            text = (item.get("summary", "") + " " + " ".join(item.get("key_points", []))).lower()
            score = sum(1 for w in q.split() if w in text)
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]

    async def extract_memory(self, content: str) -> Dict[str, Any]:
        if not self.llm:
            return {
                "summary": content[:80],
                "key_points": [content[:120]],
                "importance": 0.6,
                "content": content,
            }
        from langchain_core.messages import HumanMessage

        prompt = MEMORY_EXTRACT_PROMPT.format(content=content)
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        try:
            data = json.loads(response.content or "{}")
        except json.JSONDecodeError:
            data = {"summary": content[:80], "key_points": [], "importance": 0.5}
        data["content"] = content
        data["id"] = str(uuid.uuid4())
        data["timestamp"] = datetime.now().isoformat()
        return data

    async def ingest(self, content: str, memory_type: str = "preference") -> str:
        extracted = await self.extract_memory(content)
        memory_id = extracted["id"]
        meta = {
            "user_id": self.user_id,
            "memory_type": memory_type,
            "summary": extracted.get("summary", ""),
            "key_points": json.dumps(extracted.get("key_points", []), ensure_ascii=False),
            "importance": float(extracted.get("importance", 0.5)),
            "timestamp": extracted.get("timestamp", datetime.now().isoformat()),
        }
        if self._use_chroma and self.collection is not None:
            self.collection.add(
                ids=[memory_id],
                documents=[extracted.get("content", content)],
                metadatas=[meta],
            )
        else:
            self._fallback_store.append({
                "id": memory_id,
                "summary": meta["summary"],
                "key_points": extracted.get("key_points", []),
                "importance": meta["importance"],
                "content": extracted.get("content", content),
            })
        return memory_id

    def record_turn(self, thread_id: str, user_msg: str, assistant_msg: str) -> None:
        self.short_term.append(thread_id, "user", user_msg)
        self.short_term.append(thread_id, "assistant", assistant_msg)

    def build_prompt(self, thread_id: str, query: str, hits: Optional[List[Dict[str, Any]]] = None) -> str:
        if hits is None:
            hits = self.search_memories(query)
        memory_lines = []
        for h in hits:
            kps = h.get("key_points") or []
            memory_lines.append(f"- {h.get('summary', '')} | 要点: {', '.join(kps)}")
        memory_context = "\n".join(memory_lines) if memory_lines else "（无相关长期记忆）"
        return MEMORY_PROMPT_TEMPLATE.format(
            recent_dialogue=self.short_term.format_recent(thread_id),
            memory_context=memory_context,
            query=query,
        )

    def format_memories_for_plan(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "summary": h.get("summary", ""),
                "key_points": h.get("key_points", []),
                "importance": h.get("importance", 0.5),
            }
            for h in hits
        ]
