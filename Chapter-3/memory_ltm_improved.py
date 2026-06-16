"""源码 B：长期记忆系统（SelfBuiltLongTermMemoryImproved 等）。"""

from __future__ import annotations

from memory_hybrid_compressor import HybridRetriever, MemoryCompressor

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions


def default_llm():
    """读取 .env，创建对话用大模型（课堂统一入口）。"""
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



_NOTEBOOK_STATE: dict = {}

def _bind_ltm(namespace: dict[str, Any] | None, ltm: "SelfBuiltLongTermMemoryImproved") -> None:
    if namespace is not None:
        namespace["ltm"] = ltm





def close_chroma_ltm(ltm_obj=None) -> None:
    """释放 Chroma 引用（尽量让 GC 回收文件句柄）。"""
    import gc
    import time

    target = ltm_obj if ltm_obj is not None else _NOTEBOOK_STATE.get("ltm")
    if target is not None:
        if hasattr(target, "close"):
            try:
                target.close()
            except Exception:
                pass
        try:
            target.collection = None
        except Exception:
            pass
        try:
            target.client = None
        except Exception:
            pass
    if ltm_obj is None:
        _NOTEBOOK_STATE.pop("ltm", None)
        _NOTEBOOK_STATE.pop("key", None)
    gc.collect()
    time.sleep(0.2)


def reset_chroma_directory(path: Path, ltm_obj=None) -> bool:
    """尝试删除整个 chroma 目录。Windows 上常会失败，返回 False 即可，不要抛错。"""
    import shutil
    import time

    close_chroma_ltm(ltm_obj)
    if not path.exists():
        return True
    for _ in range(6):
        try:
            shutil.rmtree(path)
            return True
        except PermissionError:
            time.sleep(0.5)
    print(
        f"⚠️ 未能删除文件夹 {path}（被系统占用）。"
        "将改为「复用 ltm + 只清空 collection」，不影响课堂演示。"
    )
    return False


def get_or_reset_ltm(
    *,
    user_id: str,
    persist_directory: str | Path,
    collection_name: str = "notebook_long_term_v1",
    force_rebuild_directory: bool = False,
    namespace: dict[str, Any] | None = None,
    **kwargs: Any,
) -> "SelfBuiltLongTermMemoryImproved":
    """Notebook 专用：第二次运行优先复用 ltm 并 clear_all_memories，避免 WinError 32。"""
    import gc
    import time

    path = Path(persist_directory)
    path.mkdir(parents=True, exist_ok=True)
    key = (str(path.resolve()), collection_name, user_id)

    if force_rebuild_directory:
        reset_chroma_directory(path, _NOTEBOOK_STATE.get("ltm"))

    existing = _NOTEBOOK_STATE.get("ltm")
    if (
        isinstance(existing, SelfBuiltLongTermMemoryImproved)
        and _NOTEBOOK_STATE.get("key") == key
        and not force_rebuild_directory
    ):
        existing.clear_all_memories()
        print("♻️ 已复用 ltm，并清空 collection（未删磁盘目录）")
        _bind_ltm(namespace, existing)
        return existing

    if isinstance(existing, SelfBuiltLongTermMemoryImproved):
        close_chroma_ltm(existing)
        gc.collect()
        time.sleep(0.3)

    ltm = SelfBuiltLongTermMemoryImproved(
        user_id=user_id,
        persist_directory=str(path),
        collection_name=collection_name,
        **kwargs,
    )
    _NOTEBOOK_STATE["ltm"] = ltm
    _NOTEBOOK_STATE["key"] = key
    _bind_ltm(namespace, ltm)
    return ltm

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

        self._llm = llm or default_llm()
        self.compressor = MemoryCompressor(self._llm)
        self.write_pipeline = MemoryWritePipeline(
            self.collection,
            self.compressor,
            self.hybrid_retriever,
            user_id=user_id,
            on_index_refreshed=None,
        )
        self.write_pipeline._sync_index_state(self)


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
        idx_to_vector_sim: Dict[int, float] = {}
        if dense["ids"] and dense["ids"][0]:
            for mem_id, dist in zip(dense["ids"][0], dense["distances"][0]):
                if mem_id in self.ids:
                    idx = self.ids.index(mem_id)
                    vector_sim = self._distance_to_similarity(dist)
                    idx_to_vector_sim[idx] = vector_sim
                    dense_scores.append((idx, vector_sim))

        hybrid = self.hybrid_retriever.search(query, dense_scores, k * 2)

        results: List[Dict[str, Any]] = []
        for idx, rrf_score in hybrid:
            meta = self.metadatas[idx]
            vector_sim = idx_to_vector_sim.get(idx, 0.0)
            # RRF 排名分 × 向量相似度幅度 → 随 query 变化的基础分
            score = rrf_score * vector_sim
            if use_time_decay:
                score *= self._time_decay(meta.get("timestamp", datetime.now().isoformat()))
            if use_importance_weight:
                imp = float(meta.get("importance", 0.5))
                score *= 0.5 + imp  # 乘性加权，避免常数项把分数拉平
            final_score = score
            if final_score >= self.similarity_threshold:
                results.append(
                    {
                        "id": self.ids[idx],
                        "content": self.documents[idx],
                        "summary": meta.get("summary", ""),
                        "memory_type": meta.get("memory_type", "general"),
                        "importance": meta.get("importance", 0.5),
                        "timestamp": meta.get("timestamp", ""),
                        "vector_sim": vector_sim,
                        "rrf_score": rrf_score,
                        "final_score": final_score,
                        "score": final_score,
                    }
                )
        return sorted(results, key=lambda x: x["final_score"], reverse=True)[:k]

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
                f"- [{h['memory_type']}] {h['summary'] or h['content']} "
                f"(final={h['final_score']:.4f}, vector_sim={h['vector_sim']:.4f}, rrf={h['rrf_score']:.4f})"
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