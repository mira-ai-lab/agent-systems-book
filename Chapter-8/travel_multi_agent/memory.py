"""Long-term memory: reuse Chapter-6's MemoryAdapter plus a LangGraph Store backend.

:author: zhanghong, liuhui, xingrui
:date: 2026-11-22
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from travel_multi_agent.config import BOOK_ROOT

CHAPTER3_DIR = BOOK_ROOT / "Chapter-3"
CHAPTER6_DIR = BOOK_ROOT / "Chapter-6"
for _chapter_dir in (CHAPTER3_DIR, CHAPTER6_DIR):
    if str(_chapter_dir) not in sys.path:
        sys.path.insert(0, str(_chapter_dir))

from memory_adapter import MemoryAdapter
from memory_extract import MemoryExtractor
from memory_use import MEMORY_PROMPT_TEMPLATE, ThreadShortTermMemory

LongTermBackend = Literal["chroma", "store"]

# In-process shared store; the default InMemoryStore does not cross instances.
_SHARED_STORE: Any = None


def _get_or_create_shared_store() -> Any:
    """Return a process-wide LangGraph Store, creating it lazily."""
    global _SHARED_STORE
    if _SHARED_STORE is not None:
        return _SHARED_STORE

    from langgraph.store.base import IndexConfig
    from langgraph.store.memory import InMemoryStore

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv(
        "DASHSCOPE_EMBEDDING_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).rstrip("/")
    model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")
    dims = int(os.getenv("DASHSCOPE_EMBEDDING_DIMS", "1024"))

    try:
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=base_url,
            check_embedding_ctx_length=False,
        )
        _SHARED_STORE = InMemoryStore(
            index=IndexConfig(embed=embeddings, dims=dims, fields=["content"]),
        )
    except Exception:
        _SHARED_STORE = InMemoryStore()
    return _SHARED_STORE


class LongTermMemory(MemoryAdapter):
    """Reuse Chapter-6's MemoryAdapter (wrapping Chapter-3), plus build_prompt."""

    def __init__(
        self,
        user_id: str = "central_agent_user",
        persist_directory: str = "./chroma_memory",
        collection_name: str = "central_agent_memory",
        top_k: int = 5,
        llm: Any = None,
    ) -> None:
        super().__init__(
            user_id=user_id,
            persist_directory=persist_directory,
            collection_name=collection_name,
            top_k=top_k,
            llm=llm,
        )

    async def extract_memory(self, content: str) -> Dict[str, Any]:
        """Extract a structured memory via Chapter-3."""
        extracted = await self.extractor.extract_async(content)
        return {
            "id": extracted["id"],
            "content": extracted["content"],
            "summary": extracted["summary"],
            "key_points": extracted["key_points"],
            "importance": float(extracted.get("importance", 0.5)),
            "timestamp": extracted.get("timestamp", ""),
        }

    def build_prompt(
        self,
        thread_id: str,
        query: str,
        hits: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build a prompt combining recent dialogue and long-term memory."""
        if hits is None:
            hits = self.search_memories(query)
        memory_lines = []
        for hit in hits:
            key_points = hit.get("key_points") or []
            memory_lines.append(
                f"- {hit.get('summary', '')} | 要点: {', '.join(map(str, key_points))}"
            )
        memory_context = "\n".join(memory_lines) if memory_lines else "（无相关长期记忆）"
        return MEMORY_PROMPT_TEMPLATE.format(
            recent_dialogue=self.short_term.format_recent(thread_id),
            memory_context=memory_context,
            query=query,
        )

    def close(self) -> None:
        """Release Chroma handles so Windows can clean up the directory."""
        self.store.close()


class StoreLongTermMemory:
    """LangGraph Store backed long-term memory; same interface as LongTermMemory."""

    backend: LongTermBackend = "store"

    def __init__(
        self,
        user_id: str = "central_agent_user",
        llm: Any = None,
        store: Any = None,
        top_k: int = 5,
    ) -> None:
        self.user_id = user_id
        self.llm = llm
        self.top_k = top_k
        self.short_term = ThreadShortTermMemory()
        self.namespace = ("travel_multi_agent_memories", user_id)
        self.store = store or self._create_default_store()
        self._extractor = MemoryExtractor(llm)

    def _create_default_store(self) -> Any:
        return _get_or_create_shared_store()

    @property
    def backend_name(self) -> str:
        return "store"

    def search_memories(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search the LangGraph Store for relevant memories."""
        k = top_k or self.top_k
        try:
            items = self.store.search(self.namespace, query=query, limit=k)
        except Exception:
            items = self.store.search(self.namespace, limit=k)

        results = []
        for item in items:
            value = item.value or {}
            key_points = value.get("key_points", [])
            if isinstance(key_points, str):
                try:
                    key_points = json.loads(key_points)
                except json.JSONDecodeError:
                    key_points = [key_points]
            results.append({
                "id": item.key,
                "summary": value.get("summary", ""),
                "key_points": key_points,
                "importance": float(value.get("importance", 0.5)),
                "content": value.get("content", ""),
            })
        return results

    async def extract_memory(self, content: str) -> Dict[str, Any]:
        """Extract a structured memory via Chapter-3."""
        return await self._extractor.extract_async(content)

    async def ingest(self, content: str, memory_type: str = "preference") -> str:
        """Extract and persist one memory in the LangGraph Store."""
        extracted = await self.extract_memory(content)
        memory_id = extracted.get("id") or str(uuid.uuid4())
        value = {
            "user_id": self.user_id,
            "memory_type": memory_type,
            "summary": extracted.get("summary", ""),
            "key_points": extracted.get("key_points", []),
            "importance": float(extracted.get("importance", 0.5)),
            "content": extracted.get("content", content),
            "timestamp": extracted.get("timestamp", datetime.now().isoformat()),
        }
        self.store.put(self.namespace, memory_id, value)
        return memory_id

    def record_turn(self, thread_id: str, user_msg: str, assistant_msg: str) -> None:
        """Record one conversation turn in short-term memory."""
        self.short_term.append(thread_id, "user", user_msg)
        self.short_term.append(thread_id, "assistant", assistant_msg)

    def build_prompt(
        self,
        thread_id: str,
        query: str,
        hits: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build a prompt combining recent dialogue and long-term memory."""
        if hits is None:
            hits = self.search_memories(query)
        memory_lines = []
        for h in hits:
            kps = h.get("key_points") or []
            memory_lines.append(f"- {h.get('summary', '')} | 要点: {', '.join(map(str, kps))}")
        memory_context = "\n".join(memory_lines) if memory_lines else "（无相关长期记忆）"
        return MEMORY_PROMPT_TEMPLATE.format(
            recent_dialogue=self.short_term.format_recent(thread_id),
            memory_context=memory_context,
            query=query,
        )

    def format_memories_for_plan(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select compact memory fields for task-planning prompts."""
        return [
            {
                "summary": h.get("summary", ""),
                "key_points": h.get("key_points", []),
                "importance": h.get("importance", 0.5),
                "content": h.get("content", ""),
            }
            for h in hits
        ]


def resolve_memory_backend(explicit: Optional[str] = None) -> LongTermBackend:
    """Resolve the long-term memory backend: arg > MEMORY_BACKEND > chroma."""
    raw = (explicit or os.getenv("MEMORY_BACKEND", "chroma")).lower().strip()
    if raw in ("store", "langgraph_store", "fixed_graph", "langgraph_demo"):
        return "store"
    return "chroma"


def create_long_term_memory(
    backend: Optional[LongTermBackend | str] = None,
    *,
    user_id: str = "central_agent_user",
    llm: Any = None,
    persist_directory: str = "./chroma_memory",
    top_k: int = 5,
) -> Tuple[Any, Optional[Any]]:
    """Create the long-term memory instance and, for store, the LangGraph Store."""
    resolved = resolve_memory_backend(backend)

    if resolved == "store":
        memory = StoreLongTermMemory(user_id=user_id, llm=llm, top_k=top_k)
        return memory, memory.store

    memory = LongTermMemory(
        user_id=user_id,
        persist_directory=persist_directory,
        llm=llm,
        top_k=top_k,
    )
    return memory, None


__all__ = [
    "LongTermMemory",
    "ThreadShortTermMemory",
    "MemoryExtractor",
    "MEMORY_PROMPT_TEMPLATE",
    "create_long_term_memory",
    "resolve_memory_backend",
]