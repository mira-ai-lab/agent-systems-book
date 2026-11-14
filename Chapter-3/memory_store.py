"""Long-term memory storage backed by the Chroma vector database.

Chroma ``PersistentClient`` stores documents, metadata, and identifiers in a
persistent collection. Embeddings use DashScope ``text-embedding-v3`` by
default, or ``text-embedding-3-small`` when OpenAI embeddings are selected.
The ingestion pipeline extracts structured memories, detects near-duplicates
by vector similarity, merges them when needed, and updates the local cache.

:author: zhanghong, liuhui
:date: 2026-11-14
"""

from __future__ import annotations

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
    """Configuration for duplicate detection and merging."""

    dedupe_similarity_threshold: float = 0.82
    merge_on_duplicate: bool = True


_STORE_STATE: dict[str, Any] = {}


class MemoryStore:
    """Chroma-backed long-term memory store and ingestion pipeline."""

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

    def _create_embedding_function(self, model_type: str):
        """Create the online embedding function required by Chroma."""
        kind = (model_type or "dashscope").strip().lower()
        if kind in ("dashscope", "qwen", "aliyun", "阿里云"):
            api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
            if not api_key:
                raise ValueError(
                    "Set DASHSCOPE_API_KEY in the project-level .env file to "
                    "use DashScope embeddings"
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
            f"Unsupported embedding_model={model_type!r}; "
            "use 'dashscope' or 'openai'"
        )

    @staticmethod
    def distance_to_similarity(distance: float) -> float:
        """Convert a Chroma distance to a similarity in the range [0, 1]."""
        return max(0.0, min(1.0, 1.0 - float(distance) / 2.0))

    def _sync_index_state(self) -> None:
        """Reload cached documents, metadata, and identifiers from Chroma."""
        data = self.collection.get()
        self.documents = data.get("documents") or []
        self.metadatas = data.get("metadatas") or []
        self.ids = data.get("ids") or []

    def get_user_records(
        self,
        user_id: str,
        memory_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return memory records for a specific user.

        Args:
            user_id: User identifier.
            memory_types: Optional memory-type filter.
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
        """Find the current user's most similar memory above the threshold."""
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
        """Write an extracted memory and merge it with an existing duplicate.

        Args:
            extracted: Structured memory returned by
                ``MemoryExtractor.extract``.
            memory_type: Memory type, such as ``preference`` or ``fact``.
            importance: Explicit importance. Defaults to the extraction value.
            extra_metadata: Additional metadata to persist in Chroma.

        Returns:
            Identifier of the newly added or updated memory.
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
        """Build flat, Chroma-compatible metadata for persistence."""
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
        """Extract and persist one long-term memory."""
        extra_metadata = {"user_id": self.user_id, **(metadata or {})}
        extracted = self.extractor.extract(content, metadata=extra_metadata)
        return self.add(
            extracted,
            memory_type=memory_type,
            importance=importance,
            extra_metadata=extra_metadata,
        )

    def clear_all_memories(self) -> None:
        """Delete every memory in the collection and clear local caches."""
        data = self.collection.get()
        ids = data.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
        self.documents = []
        self.metadatas = []
        self.ids = []

    def close(self) -> None:
        """Release Chroma references so Windows can release the data directory."""
        client = self.client
        self.collection = None
        self.client = None
        if client is not None and hasattr(client, "close"):
            client.close()
        gc.collect()


def close_memory_store(store: MemoryStore | None = None) -> None:
    """Close the memory store and clear its notebook-session references."""
    target = store if store is not None else _STORE_STATE.get("store")
    if target is not None:
        target.close()
    if store is None:
        _STORE_STATE.pop("store", None)
        _STORE_STATE.pop("key", None)
    gc.collect()
    time.sleep(0.2)


def reset_chroma_directory(path: Path, store: MemoryStore | None = None) -> bool:
    """Try to delete the Chroma persistence directory.

    Windows may reject deletion while file handles remain open. This function
    retries to reduce failures. When it returns ``False``, callers can clear
    the collection instead and continue.
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
        f"Could not delete {path}; reusing the store and clearing the "
        "collection instead."
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
    """Return a reusable notebook memory store, clearing or rebuilding as needed.

    Repeated notebook runs reuse the same instance and clear its collection
    when possible. This avoids Windows failures caused by a locked Chroma
    persistence directory.
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
        print("Reused the memory store and cleared the current collection.")
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
