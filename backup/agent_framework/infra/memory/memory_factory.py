"""
长期记忆工厂：Chroma（Chapter-3）与 LangGraph Store 可切换

用法:
    memory, store = create_long_term_memory("store", user_id="u1", llm=llm, ...)
    memory, store = create_long_term_memory("chroma", persist_directory="...", ...)
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from agent_framework.config import MEMORY_NAMESPACE_PREFIX
from agent_framework.domain.memory_prompts import MEMORY_PROMPT_TEMPLATE
from agent_framework.infra.memory.memory_system import LongTermMemory, ThreadShortTermMemory

LongTermBackend = Literal["chroma", "store"]

# 进程内共享 Store（InMemoryStore 默认不跨实例；单例便于 demo / 同进程多轮对话）
_SHARED_STORE: Any = None


def _get_or_create_shared_store() -> Any:
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


def resolve_memory_backend(explicit: Optional[str] = None) -> LongTermBackend:
    """解析长期记忆后端：参数 > 环境变量 MEMORY_BACKEND > 默认 chroma"""
    raw = (explicit or os.getenv("MEMORY_BACKEND", "chroma")).lower().strip()
    if raw in ("store", "langgraph_store", "fixed_graph", "langgraph_demo"):
        return "store"
    return "chroma"


class StoreLongTermMemory:
    """
    基于 LangGraph Store 的长期记忆（语义检索走 Store 内置 index）

    与 LongTermMemory 保持相同对外接口，便于 orchestrator / nodes 无缝切换。
    """

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
        self.namespace = (MEMORY_NAMESPACE_PREFIX, user_id)
        self.store = store or self._create_default_store()
        # 复用 Chroma 版的 LLM 抽取逻辑（不写入 Chroma）
        self._extractor = LongTermMemory(user_id=user_id, llm=llm)

    def _create_default_store(self) -> Any:
        return _get_or_create_shared_store()

    @property
    def backend_name(self) -> str:
        return "store"

    def search_memories(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        try:
            items = self.store.search(
                self.namespace,
                query=query,
                limit=k,
            )
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
        return await self._extractor.extract_memory(content)

    async def ingest(self, content: str, memory_type: str = "preference") -> str:
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
        self.short_term.append(thread_id, "user", user_msg)
        self.short_term.append(thread_id, "assistant", assistant_msg)

    def build_prompt(
        self,
        thread_id: str,
        query: str,
        hits: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
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
        return [
            {
                "summary": h.get("summary", ""),
                "key_points": h.get("key_points", []),
                "importance": h.get("importance", 0.5),
                "content": h.get("content", ""),
            }
            for h in hits
        ]


def create_long_term_memory(
    backend: Optional[LongTermBackend | str] = None,
    *,
    user_id: str = "central_agent_user",
    llm: Any = None,
    persist_directory: str = "./chroma_memory",
    top_k: int = 5,
) -> Tuple[Any, Optional[Any]]:
    """
    创建长期记忆实例

    Returns:
        (memory_system, langgraph_store)
        - chroma 后端: store 为 None
        - store 后端: store 为 LangGraph Store，可传入 compile(store=...)
    """
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
