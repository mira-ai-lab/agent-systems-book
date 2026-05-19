"""
工业界常见「自建长期记忆」架构的改进版演示。

架构对照
─────────
                    ┌─────────────────────────────────┐
  对话结束 / 工具结果 ─►│ 记忆写入管道（可同步/异步）        │
                    │  抽取 → 去重 → 合并 → embed → 索引 │
                    └──────────────┬──────────────────┘
                                   ▼
              ┌────────────────────────────────────────────┐
              │  存储层：Chroma 向量 + BM25 + 元数据过滤      │
              └────────────────────────────────────────────┘
                                   ▲
  用户新问题 ──► 检索（混合）─► 重排（时间/重要性）─► 拼 prompt
       ▲
       └── ThreadShortTermMemory：本 thread 最近 messages（短期，不检索）
            对应 LangGraph Checkpoint 的职责，此处用内存模拟

本文件不修改 LangChainHybridMemory / SelfBuiltLongTermMemory 原实现。

嵌入模型（默认阿里云百炼，不下载本地 BGE）
  在 .env 配置：
    DASHSCOPE_API_KEY=你的百炼 API Key
    DASHSCOPE_EMBEDDING_MODEL=text-embedding-v3   # 或 text-embedding-v4
    DASHSCOPE_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  构造时也可：embedding_model="dashscope"（同义：qwen / aliyun）
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from EmbeddingFactory import HybridRetriever, MemoryCompressor

load_dotenv(Path(__file__).resolve().parent / ".env")

MEMORY_PROMPT_TEMPLATE = """你是一个有记忆的智能助手。请结合【最近对话】与【相关长期记忆】回答用户问题。

【最近对话】（来自 Checkpoint / 本线程短期缓冲，按时间顺序）
{recent_dialogue}

【相关长期记忆】（向量检索 + 重排后的 top-k）
{memory_context}

【当前问题】
{query}
"""


# ---------------------------------------------------------------------------
# 短期：模拟 LangGraph Checkpoint 中的 messages（只追加、不向量检索）
# ---------------------------------------------------------------------------
class ThreadShortTermMemory:
    """thread_id 隔离的短期对话缓冲，等价于 Checkpoint 里本会话的 messages 子集。"""

    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self._threads: Dict[str, List[Dict[str, str]]] = {}

    def append(self, thread_id: str, role: str, content: str) -> None:
        self._threads.setdefault(thread_id, []).append({"role": role, "content": content})
        # 只保留最近 max_turns 条
        if len(self._threads[thread_id]) > self.max_turns:
            self._threads[thread_id] = self._threads[thread_id][-self.max_turns :]

    def format_recent(self, thread_id: str, last_n: int = 6) -> str:
        turns = self._threads.get(thread_id, [])[-last_n:]
        if not turns:
            return "（无历史对话）"
        return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


# ---------------------------------------------------------------------------
# 写入管道：抽取 → 去重 → 合并 → 索引
# ---------------------------------------------------------------------------
@dataclass
class WritePipelineConfig:
    dedupe_similarity_threshold: float = 0.82  # Chroma 距离转相似度后的合并阈值
    merge_on_duplicate: bool = True


class MemoryWritePipeline:
    def __init__(
        self,
        collection,
        compressor: MemoryCompressor,
        hybrid_retriever: HybridRetriever,
        *,
        user_id: str,
        config: WritePipelineConfig | None = None,
        on_index_refreshed=None,
    ) -> None:
        self.collection = collection
        self.compressor = compressor
        self.hybrid_retriever = hybrid_retriever
        self.user_id = user_id
        self.config = config or WritePipelineConfig()
        self._on_index_refreshed = on_index_refreshed
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    def _distance_to_similarity(self, distance: float) -> float:
        return max(0.0, min(1.0, 1.0 - float(distance) / 2.0))

    def _sync_index_state(self, ltm: "SelfBuiltLongTermMemoryImproved") -> None:
        all_data = self.collection.get()
        ltm.documents = all_data.get("documents") or []
        ltm.metadatas = all_data.get("metadatas") or []
        ltm.ids = all_data.get("ids") or []
        if ltm.documents:
            ltm.hybrid_retriever.fit(ltm.documents)
        if self._on_index_refreshed:
            self._on_index_refreshed()

    def ingest_sync(
        self,
        ltm: "SelfBuiltLongTermMemoryImproved",
        content: str,
        *,
        memory_type: str = "general",
        importance: float | None = None,
        extra_metadata: Dict[str, Any] | None = None,
    ) -> str:
        """同步执行完整写入管道（课堂演示用）。"""
        extracted = self.compressor.extract_memory(content, extra_metadata)
        imp = float(importance if importance is not None else extracted.get("importance", 0.5))

        # 去重：用摘要/原文做一次向量近邻
        dup_id: Optional[str] = None
        if self.collection.count() > 0:
            hits = self.collection.query(
                query_texts=[extracted["summary"]],
                n_results=3,
                where={"user_id": self.user_id},
            )
            if hits["ids"] and hits["ids"][0]:
                best_dist = hits["distances"][0][0]
                if self._distance_to_similarity(best_dist) >= self.config.dedupe_similarity_threshold:
                    dup_id = hits["ids"][0][0]

        if dup_id and self.config.merge_on_duplicate:
            existing = self.collection.get(ids=[dup_id])
            old_meta = existing["metadatas"][0]
            old_mem = {
                "id": dup_id,
                "content": existing["documents"][0],
                "summary": old_meta.get("summary", ""),
                "key_points": json.loads(old_meta.get("key_points", "[]")),
                "importance": old_meta.get("importance", 0.5),
            }
            merged = self.compressor.merge_memories([old_mem, extracted])
            memory_id = dup_id
            extra_metadata = {**(extra_metadata or {}), "user_id": self.user_id}
            meta = self._build_metadata(merged, memory_type, float(merged.get("importance", imp)), extra_metadata)
            self.collection.update(
                ids=[memory_id],
                documents=[merged["content"]],
                metadatas=[meta],
            )
        else:
            memory_id = extracted["id"]
            extra_metadata = {**(extra_metadata or {}), "user_id": self.user_id}
            meta = self._build_metadata(extracted, memory_type, imp, extra_metadata)
            self.collection.add(
                ids=[memory_id],
                documents=[extracted["content"]],
                metadatas=[meta],
            )

        self._sync_index_state(ltm)
        return memory_id

    async def ingest_async(
        self,
        ltm: "SelfBuiltLongTermMemoryImproved",
        content: str,
        **kwargs: Any,
    ) -> str:
        """异步写入：在线程池中跑同步管道，避免阻塞事件循环。"""
        return await asyncio.to_thread(self.ingest_sync, ltm, content, **kwargs)

    async def enqueue(self, item: Dict[str, Any]) -> None:
        await self._queue.put(item)

    async def _worker(self, ltm: "SelfBuiltLongTermMemoryImproved") -> None:
        while True:
            item = await self._queue.get()
            if item.get("__stop__"):
                break
            try:
                await self.ingest_async(ltm, **item)
            finally:
                self._queue.task_done()

    def start_background_worker(self, ltm: "SelfBuiltLongTermMemoryImproved") -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker(ltm))

    async def stop_background_worker(self) -> None:
        if self._worker_task:
            await self._queue.put({"__stop__": True})
            await self._worker_task

    @staticmethod
    def _build_metadata(
        extracted: Dict[str, Any],
        memory_type: str,
        importance: float,
        extra: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        return {
            "user_id": (extra or {}).get("user_id") or extracted.get("metadata", {}).get("user_id") or "",
            "memory_type": memory_type,
            "importance": importance,
            "summary": extracted["summary"],
            "key_points": json.dumps(extracted["key_points"], ensure_ascii=False),
            "timestamp": extracted.get("timestamp", datetime.now().isoformat()),
            **(extra or {}),
        }


# ---------------------------------------------------------------------------
# 长期记忆 + 检索重排 + 与短期合并拼 prompt
# ---------------------------------------------------------------------------
class SelfBuiltLongTermMemoryImproved:
    def __init__(
        self,
        user_id: str,
        *,
        embedding_model: str = "dashscope",
        persist_directory: str = "./chroma_selfbuilt_improve",
        collection_name: str = "long_term_memory_v2",
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        time_decay_factor: float = 0.95,
        importance_weight: float = 0.3,
        short_term_max_turns: int = 20,
        llm=None,
    ) -> None:
        self.user_id = user_id
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.time_decay_factor = time_decay_factor
        self.importance_weight = importance_weight

        self.short_term = ThreadShortTermMemory(max_turns=short_term_max_turns)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = self._create_embedding_function(embedding_model)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"user_id": user_id},
        )

        self.hybrid_retriever = HybridRetriever()
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []

        self._llm = llm or self._default_llm()
        self.compressor = MemoryCompressor(self._llm)
        self.write_pipeline = MemoryWritePipeline(
            self.collection,
            self.compressor,
            self.hybrid_retriever,
            user_id=user_id,
            on_index_refreshed=None,
        )
        self.write_pipeline._sync_index_state(self)

    def _default_llm(self):
        from langchain_openai import ChatOpenAI
        import httpx

        kw: dict = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": 0,
            "api_key": (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY") or "").strip() or None,
        }
        base = (os.getenv("OPENAI_BASE_URL") or "").strip()
        if base:
            kw["base_url"] = base.rstrip("/")
        if os.getenv("OPENAI_SSL_VERIFY", "1").strip().lower() in ("0", "false", "no", "off"):
            kw["http_client"] = httpx.Client(verify=False)
        return ChatOpenAI(**kw)

    def _create_embedding_function(self, model_type: str):
        kind = (model_type or "dashscope").strip().lower()

        # 阿里云百炼：OpenAI 兼容 Embeddings（推荐，无需下载 BGE）
        if kind in ("dashscope", "qwen", "aliyun", "阿里云"):
            api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
            if not api_key:
                raise ValueError(
                    "使用阿里云嵌入请在 .env 设置 DASHSCOPE_API_KEY（百炼控制台 API Key）"
                )
            model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")
            base = os.getenv(
                "DASHSCOPE_EMBEDDING_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ).rstrip("/")
            # 写入环境变量，便于 Chroma OpenAIEmbeddingFunction 内部客户端读取
            os.environ["DASHSCOPE_API_KEY"] = api_key
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                api_key_env_var="DASHSCOPE_API_KEY",
                model_name=model,
                api_base=base,
            )

        if kind == "bge":
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3")
            )

        if kind == "openai":
            return embedding_functions.OpenAIEmbeddingFunction(
                model_name=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                api_key=os.getenv("OPENAI_API_KEY"),
                api_base=(os.getenv("OPENAI_BASE_URL") or "").strip() or None,
            )

        raise ValueError(
            f"不支持的 embedding_model={model_type!r}，请用 dashscope / openai / bge"
        )

    # ----- 对外：一轮对话结束（写短期 + 异步/同步长期）-----
    def record_turn(
        self,
        thread_id: str,
        user_content: str,
        assistant_content: str | None = None,
    ) -> None:
        """模拟 Checkpoint 追加 messages；assistant 可选（仅用户侧入库长期记忆）。"""
        self.short_term.append(thread_id, "user", user_content)
        if assistant_content:
            self.short_term.append(thread_id, "assistant", assistant_content)

    def close(self) -> None:
        """释放 Chroma 客户端（Windows/Jupyter 重复运行前建议调用）。"""
        self.collection = None
        self.client = None

    def clear_all_memories(self) -> None:
        """清空本 collection 内全部记忆（不删磁盘目录，适合 Notebook 重复运行）。"""
        data = self.collection.get()
        ids = data.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.hybrid_retriever = HybridRetriever()

    def ingest_long_term(
        self,
        content: str,
        *,
        memory_type: str = "general",
        importance: float | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        meta = {"user_id": self.user_id, **(metadata or {})}
        return self.write_pipeline.ingest_sync(
            self, content, memory_type=memory_type, importance=importance, extra_metadata=meta
        )

    async def ingest_long_term_async(self, content: str, **kwargs: Any) -> str:
        kwargs.setdefault("extra_metadata", {"user_id": self.user_id})
        return await self.write_pipeline.ingest_async(self, content, **kwargs)

    # ----- 检索 + 重排 -----
    def search_memories(
        self,
        query: str,
        *,
        memory_types: List[str] | None = None,
        top_k: int | None = None,
        use_time_decay: bool = True,
        use_importance_weight: bool = True,
    ) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        if not self.documents:
            return []

        where: Dict[str, Any] = {"user_id": self.user_id}
        if memory_types:
            where["memory_type"] = {"$in": memory_types}

        dense = self.collection.query(query_texts=[query], n_results=k * 2, where=where)
        dense_scores: List[tuple[int, float]] = []
        if dense["ids"] and dense["ids"][0]:
            for mem_id, dist in zip(dense["ids"][0], dense["distances"][0]):
                if mem_id in self.ids:
                    idx = self.ids.index(mem_id)
                    dense_scores.append((idx, self._distance_to_similarity(dist)))

        hybrid = self.hybrid_retriever.search(query, dense_scores, k * 2)

        results: List[Dict[str, Any]] = []
        for idx, base_score in hybrid:
            meta = self.metadatas[idx]
            score = base_score
            if use_time_decay:
                score *= self._time_decay(meta.get("timestamp", datetime.now().isoformat()))
            if use_importance_weight:
                imp = float(meta.get("importance", 0.5))
                score = score * (1 - self.importance_weight) + imp * self.importance_weight
            if score >= self.similarity_threshold:
                results.append(
                    {
                        "id": self.ids[idx],
                        "content": self.documents[idx],
                        "summary": meta.get("summary", ""),
                        "memory_type": meta.get("memory_type", "general"),
                        "importance": meta.get("importance", 0.5),
                        "timestamp": meta.get("timestamp", ""),
                        "score": score,
                    }
                )
        return sorted(results, key=lambda x: x["score"], reverse=True)[:k]

    def _time_decay(self, timestamp: str) -> float:
        try:
            days = (datetime.now() - datetime.fromisoformat(timestamp)).days
        except ValueError:
            days = 0
        return self.time_decay_factor ** max(days, 0)

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        return max(0.0, min(1.0, 1.0 - float(distance) / 2.0))

    # ----- 拼 prompt（短期 + 长期）-----
    def build_prompt(
        self,
        thread_id: str,
        query: str,
        *,
        long_term_hits: List[Dict[str, Any]] | None = None,
        recent_n: int = 6,
    ) -> str:
        hits = long_term_hits if long_term_hits is not None else self.search_memories(query)
        if hits:
            memory_context = "\n".join(
                f"- [{h['memory_type']}] {h['summary'] or h['content']} (score={h['score']:.4f})"
                for h in hits
            )
        else:
            memory_context = "（未检索到相关长期记忆）"
        recent = self.short_term.format_recent(thread_id, last_n=recent_n)
        return MEMORY_PROMPT_TEMPLATE.format(
            recent_dialogue=recent,
            memory_context=memory_context,
            query=query,
        )


# ---------------------------------------------------------------------------
# 演示：短期 Checkpoint 模拟 + 写入管道 + 检索拼 prompt
# ---------------------------------------------------------------------------
async def _demo_async_pipeline() -> None:
    ltm = SelfBuiltLongTermMemoryImproved(
        user_id="user_demo",
        embedding_model="dashscope",
        persist_directory="./chroma_selfbuilt_improve",
    )
    thread_id = "thread_001"

    # 1) 长期记忆：走写入管道（同步演示；生产可 start_background_worker + enqueue）
    print("=== 写入管道：抽取 → 去重 → 合并 → 索引 ===")
    ltm.ingest_long_term(
        "用户喜欢喝美式咖啡，不加糖不加奶",
        memory_type="preference",
        importance=0.8,
    )
    ltm.ingest_long_term(
        "用户住在北京朝阳区",
        memory_type="fact",
        importance=0.7,
    )
    # 近似重复，应触发合并而非新增一条
    ltm.ingest_long_term(
        "用户偏好美式咖啡，不要糖奶",
        memory_type="preference",
    )
    print(f"库内记忆条数: {len(ltm.documents)}")

    # 2) 短期：模拟 Checkpoint 多轮 messages
    print("\n=== 短期记忆（Checkpoint 模拟）===")
    ltm.record_turn(thread_id, "你好，我是小明")
    ltm.short_term.append(thread_id, "assistant", "你好小明，有什么可以帮你？")

    # 3) 检索 + 拼 prompt
    query = "你还记得我喜欢喝什么咖啡吗？"
    hits = ltm.search_memories(query)
    print("\n=== 长期记忆检索 ===")
    for h in hits:
        print(f"  - {h['summary']} ({h['score']:.4f})")

    prompt = ltm.build_prompt(thread_id, query, long_term_hits=hits)
    print("\n=== 拼好的完整 Prompt（节选）===\n")
    print(prompt[:1200] + ("…" if len(prompt) > 1200 else ""))


def main() -> None:
    asyncio.run(_demo_async_pipeline())


if __name__ == "__main__":
    main()
