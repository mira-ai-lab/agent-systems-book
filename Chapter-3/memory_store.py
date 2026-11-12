"""记忆存储：管理 Chroma 持久化、去重和写入流程。

更新时间：2026-11-12

存储模块接收已经抽取的结构化记忆，根据相似度判断是否与已有记忆重复，
必要时调用合并模块，并同步维护内存中的文档、元数据和标识缓存。
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from memory_extract import MemoryExtractor
from memory_merge import MemoryMerger


@dataclass(frozen=True)
class MemoryStoreConfig:
    """控制重复检测和合并行为的配置。"""

    dedupe_similarity_threshold: float = 0.82
    merge_on_duplicate: bool = True


_STORE_STATE: dict[str, Any] = {}


class MemoryStore:
    """Chroma 长期记忆存储及其写入管道。"""

    def __init__(
        self,
        *,
        user_id: str,
        persist_directory: str | Path,
        collection_name: str,
        extractor: MemoryExtractor,
        merger: MemoryMerger,
        embedding_model: str = "dashscope",
        embedding_function: Any | None = None,
        config: MemoryStoreConfig | None = None,
    ) -> None:
        self.user_id = user_id
        self.extractor = extractor
        self.merger = merger
        self.config = config or MemoryStoreConfig()

        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.embedding_function = embedding_function or self._create_embedding_function(
            embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"user_id": user_id},
        )

        self.documents: list[str] = []
        self.metadatas: list[dict[str, Any]] = []
        self.ids: list[str] = []
        self._sync_index_state()

        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None

    def _create_embedding_function(self, model_type: str):
        """根据模型类型创建 Chroma 所需的在线嵌入函数。"""
        kind = (model_type or "dashscope").strip().lower()
        if kind in ("dashscope", "qwen", "aliyun", "阿里云"):
            api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
            if not api_key:
                raise ValueError(
                    "使用阿里云嵌入请在项目根目录 .env 中设置 DASHSCOPE_API_KEY"
                )
            model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")
            base_url = os.getenv(
                "DASHSCOPE_EMBEDDING_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ).rstrip("/")
            os.environ["DASHSCOPE_API_KEY"] = api_key
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                api_key_env_var="DASHSCOPE_API_KEY",
                model_name=model,
                api_base=base_url,
            )

        if kind == "openai":
            return embedding_functions.OpenAIEmbeddingFunction(
                model_name=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                api_key=os.getenv("OPENAI_API_KEY"),
                api_base=(os.getenv("OPENAI_BASE_URL") or "").strip() or None,
            )

        raise ValueError(
            f"不支持的 embedding_model={model_type!r}，请使用 dashscope 或 openai"
        )

    @staticmethod
    def distance_to_similarity(distance: float) -> float:
        """将 Chroma 距离转换为 [0, 1] 范围内的相似度。"""
        return max(0.0, min(1.0, 1.0 - float(distance) / 2.0))

    def _sync_index_state(self) -> None:
        """从 Chroma 重新读取文档和元数据缓存。"""
        data = self.collection.get()
        self.documents = data.get("documents") or []
        self.metadatas = data.get("metadatas") or []
        self.ids = data.get("ids") or []

    def count(self) -> int:
        """返回当前集合中的记忆数量。"""
        return self.collection.count()

    def get_user_records(
        self,
        user_id: str,
        memory_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """返回指定用户的记忆记录。

        Args:
            user_id: 用户标识。
            memory_types: 可选的记忆类型过滤器。
        """
        records: list[dict[str, Any]] = []
        allowed_types = set(memory_types or [])
        for memory_id, document, metadata in zip(
            self.ids,
            self.documents,
            self.metadatas,
        ):
            meta = metadata or {}
            if meta.get("user_id") != user_id:
                continue
            if allowed_types and meta.get("memory_type") not in allowed_types:
                continue
            records.append(
                {
                    "id": memory_id,
                    "content": document,
                    **meta,
                }
            )
        return records

    def _find_duplicate(self, summary: str) -> str | None:
        """查找当前用户下与摘要最相似且超过阈值的已有记忆。"""
        if self.collection.count() == 0:
            return None
        hits = self.collection.query(
            query_texts=[summary],
            n_results=3,
            where={"user_id": self.user_id},
        )
        ids = hits.get("ids") or []
        distances = hits.get("distances") or []
        if not ids or not ids[0] or not distances or not distances[0]:
            return None
        if (
            self.distance_to_similarity(distances[0][0])
            >= self.config.dedupe_similarity_threshold
        ):
            return ids[0][0]
        return None

    def add(
        self,
        extracted: dict[str, Any],
        *,
        memory_type: str = "general",
        importance: float | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """写入一条已抽取的记忆，并在需要时合并到已有记忆。

        Args:
            extracted: ``MemoryExtractor.extract`` 返回的结构化记忆。
            memory_type: 记忆类型，例如 ``preference`` 或 ``fact``。
            importance: 显式重要性；未提供时使用抽取结果中的值。
            extra_metadata: 需要写入 Chroma 的附加元数据。

        Returns:
            新增或更新后的记忆标识。
        """
        resolved_importance = float(
            importance
            if importance is not None
            else extracted.get("importance", 0.5)
        )
        metadata = {**(extra_metadata or {}), "user_id": self.user_id}
        duplicate_id = self._find_duplicate(extracted["summary"])

        if duplicate_id and self.config.merge_on_duplicate:
            existing = self.collection.get(ids=[duplicate_id])
            old_metadata = (existing.get("metadatas") or [{}])[0] or {}
            try:
                key_points = json.loads(old_metadata.get("key_points", "[]"))
            except (TypeError, json.JSONDecodeError):
                key_points = []
            old_memory = {
                "id": duplicate_id,
                "content": (existing.get("documents") or [""])[0],
                "summary": old_metadata.get("summary", ""),
                "key_points": key_points,
                "importance": old_metadata.get("importance", 0.5),
            }
            merged = self.merger.merge([old_memory, extracted])
            memory_id = duplicate_id
            self.collection.update(
                ids=[memory_id],
                documents=[merged["content"]],
                metadatas=[
                    self._build_metadata(
                        merged,
                        memory_type,
                        float(merged.get("importance", resolved_importance)),
                        metadata,
                    )
                ],
            )
        else:
            memory_id = extracted["id"]
            self.collection.add(
                ids=[memory_id],
                documents=[extracted["content"]],
                metadatas=[
                    self._build_metadata(
                        extracted,
                        memory_type,
                        resolved_importance,
                        metadata,
                    )
                ],
            )

        self._sync_index_state()
        return memory_id

    @staticmethod
    def _build_metadata(
        extracted: dict[str, Any],
        memory_type: str,
        importance: float,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """构造 Chroma 可持久化的扁平元数据。"""
        return {
            "user_id": extra.get("user_id", ""),
            "memory_type": memory_type,
            "importance": importance,
            "summary": extracted["summary"],
            "key_points": json.dumps(extracted["key_points"], ensure_ascii=False),
            "extraction_method": extracted.get("extraction_method", "unknown"),
            "timestamp": extracted.get("timestamp", datetime.now().isoformat()),
            **extra,
        }

    def ingest_long_term(
        self,
        content: str,
        *,
        memory_type: str = "general",
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """抽取并写入一条长期记忆。"""
        extra_metadata = {"user_id": self.user_id, **(metadata or {})}
        extracted = self.extractor.extract(content, metadata=extra_metadata)
        return self.add(
            extracted,
            memory_type=memory_type,
            importance=importance,
            extra_metadata=extra_metadata,
        )

    async def ingest_long_term_async(self, content: str, **kwargs: Any) -> str:
        """在线程池中执行同步写入，避免阻塞异步事件循环。"""
        return await asyncio.to_thread(self.ingest_long_term, content, **kwargs)

    async def enqueue(self, item: dict[str, Any]) -> None:
        """将一条待写入记忆放入后台队列。"""
        await self._queue.put(item)

    async def _worker(self) -> None:
        """持续消费后台写入队列，直到收到停止标记。"""
        while True:
            item = await self._queue.get()
            try:
                if item.get("__stop__"):
                    break
                await self.ingest_long_term_async(**item)
            finally:
                self._queue.task_done()

    def start_background_worker(self) -> None:
        """启动后台写入任务；已有运行中的任务时不会重复启动。"""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())

    async def stop_background_worker(self) -> None:
        """等待队列清空后停止后台写入任务。"""
        if self._worker_task:
            await self._queue.put({"__stop__": True})
            await self._worker_task
            self._queue.join()

    def clear_all_memories(self) -> None:
        """删除当前集合中的全部记忆并清空本地缓存。"""
        data = self.collection.get()
        ids = data.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
        self.documents = []
        self.metadatas = []
        self.ids = []

    def close(self) -> None:
        """释放 Chroma 客户端引用，便于 Windows 回收持久化目录。"""
        client = self.client
        self.collection = None
        self.client = None
        if client is not None and hasattr(client, "close"):
            client.close()
        gc.collect()


def close_memory_store(store: MemoryStore | None = None) -> None:
    """关闭记忆存储并清理 Notebook 会话中的全局引用。"""
    target = store if store is not None else _STORE_STATE.get("store")
    if target is not None:
        target.close()
    if store is None:
        _STORE_STATE.pop("store", None)
        _STORE_STATE.pop("key", None)
    gc.collect()
    time.sleep(0.2)


def reset_chroma_directory(path: Path, store: MemoryStore | None = None) -> bool:
    """尝试删除 Chroma 持久化目录。

    Windows 在文件句柄未释放时可能拒绝删除。该函数通过重试降低失败概率，
    返回 ``False`` 时调用方可以改用清空集合的方式继续演示。
    """
    close_memory_store(store)
    if not path.exists():
        return True
    for _ in range(6):
        try:
            shutil.rmtree(path)
            return True
        except PermissionError:
            time.sleep(0.5)
    print(
        f"未能删除目录 {path}，将复用存储实例并清空集合，不影响课堂演示。"
    )
    return False


def get_or_reset_store(
    *,
    user_id: str,
    persist_directory: str | Path,
    extractor: MemoryExtractor,
    merger: MemoryMerger,
    collection_name: str = "notebook_long_term_v1",
    force_rebuild_directory: bool = False,
    **kwargs: Any,
) -> MemoryStore:
    """获取可复用的 Notebook 记忆存储，并按需清空或重建。

    Notebook 重复执行时优先复用同一实例并清空集合，以降低 Windows 上
    Chroma 持久化目录被占用而无法删除的风险。
    """
    path = Path(persist_directory)
    key = (str(path.resolve()), collection_name, user_id)

    if force_rebuild_directory:
        reset_chroma_directory(path, _STORE_STATE.get("store"))

    existing = _STORE_STATE.get("store")
    if (
        isinstance(existing, MemoryStore)
        and _STORE_STATE.get("key") == key
        and not force_rebuild_directory
    ):
        existing.extractor = extractor
        existing.merger = merger
        existing.clear_all_memories()
        print("已复用记忆存储，并清空当前集合。")
        return existing

    if isinstance(existing, MemoryStore):
        close_memory_store(existing)
        time.sleep(0.3)

    path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(
        user_id=user_id,
        persist_directory=path,
        collection_name=collection_name,
        extractor=extractor,
        merger=merger,
        **kwargs,
    )
    _STORE_STATE["store"] = store
    _STORE_STATE["key"] = key
    return store
