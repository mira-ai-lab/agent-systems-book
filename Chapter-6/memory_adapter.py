"""Chapter 6 adapter for the shared Chapter 3 memory implementation.

The adapter reuses memory extraction, merging, Chroma storage, hybrid retrieval,
and short-term context from Chapter 3 while preserving the interface expected by
the central controller.

:author: zhanghong, liuhui
:date: 2026-11-16
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import BOOK_ROOT


CHAPTER3_DIR = BOOK_ROOT / "Chapter-3"
if not CHAPTER3_DIR.is_dir():
    raise FileNotFoundError(f"Could not locate {CHAPTER3_DIR}")
if str(CHAPTER3_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER3_DIR))

from memory_extract import MemoryExtractor
from memory_merge import MemoryMerger
from memory_retrieve import MemoryRetriever
from memory_store import MemoryStore
from memory_store import reset_chroma_directory as _reset_ch3_chroma
from memory_use import MemoryContextBuilder, ThreadShortTermMemory


def _decode_key_points(value: Any) -> List[str]:
    """Decode Chroma's JSON-string key points into a list."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            return [value]
    return []


class MemoryAdapter:
    """Reuse Chapter 3 memory components through the Chapter 6 interface."""

    def __init__(
        self,
        user_id: str = "central_agent_user",
        persist_directory: str = "./chroma_memory",
        collection_name: str = "central_agent_memory",
        top_k: int = 5,
        llm: Any = None,
    ) -> None:
        """Initialize extraction, merging, storage, retrieval, and short-term context."""
        self.user_id = user_id
        self.top_k = top_k
        self.llm = llm
        self._persist_directory = persist_directory

        self.extractor = MemoryExtractor(llm)
        self.merger = MemoryMerger(llm)
        self.store = MemoryStore(
            user_id=user_id,
            persist_directory=persist_directory,
            collection_name=collection_name,
            extractor=self.extractor,
            merger=self.merger,
            embedding_model=(
                "dashscope"
                if (os.getenv("DASHSCOPE_API_KEY") or "").strip()
                else "openai"
            ),
        )
        self.retriever = MemoryRetriever(self.store, top_k=top_k)
        self.context_builder = MemoryContextBuilder()
        self.short_term: ThreadShortTermMemory = self.context_builder.short_term

    @staticmethod
    def _normalize_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **hit,
            "key_points": _decode_key_points(hit.get("key_points")),
            "importance": float(hit.get("importance", 0.5)),
        }

    def search_memories(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories through Chapter 3 hybrid retrieval."""
        hits = self.retriever.search(
            query,
            user_id=self.user_id,
            top_k=top_k,
        )
        return [self._normalize_hit(hit) for hit in hits]

    async def search_memories_async(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Asynchronously retrieve relevant memories through Chapter 3."""
        hits = await self.retriever.search_async(
            query,
            user_id=self.user_id,
            top_k=top_k,
        )
        return [self._normalize_hit(hit) for hit in hits]

    async def ingest(
        self,
        content: str,
        memory_type: str = "preference",
    ) -> str:
        """Extract, merge, and persist one memory through Chapter 3."""
        return await self.store.ingest_long_term_async(
            content,
            memory_type=memory_type,
        )

    def record_turn(
        self,
        thread_id: str,
        user_msg: str,
        assistant_msg: str,
    ) -> None:
        """Record one conversation turn in Chapter 3 short-term memory."""
        self.context_builder.record_turn(thread_id, user_msg, assistant_msg)

    def format_memories_for_plan(
        self,
        hits: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Select compact memory fields for task-planning prompts."""
        return [
            {
                "summary": hit.get("summary", ""),
                "key_points": _decode_key_points(hit.get("key_points")),
                "importance": float(hit.get("importance", 0.5)),
                "memory_type": hit.get("memory_type", "general"),
                "final_score": float(hit.get("final_score", 0.0)),
            }
            for hit in hits
        ]


def release_memory_handles(memory: Optional[MemoryAdapter]) -> None:
    """Clear and close Chapter 3 memory resources held by Chapter 6."""
    if memory is None:
        return
    memory.store.clear_all_memories()
    memory.store.close()
    memory.short_term._threads.clear()


def reset_chroma_directory(
    persist_directory: str | Path,
) -> bool:
    """Reset a Chapter 3 Chroma directory for clean notebook execution."""
    return _reset_ch3_chroma(Path(persist_directory))
