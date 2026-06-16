"""chat_with_memories 三步封装：检索 → 回答 → 写回记忆。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from memory_hybrid_compressor import MemoryCompressor, _bm25_tokenize
from memory_ltm_improved import SelfBuiltLongTermMemoryImproved

DEMO_USER_ID = "student_demo"


class MemoryHybridRetriever:
    """混合检索器：dense + BM25 + RRF 融合。"""

    def __init__(self, ltm: SelfBuiltLongTermMemoryImproved):
        self.ltm = ltm

    def dense_search(self, query: str, user_id: str, topk: int) -> dict[str, list]:
        dense = self.ltm.collection.query(
            query_texts=[query],
            n_results=topk,
            where={"user_id": user_id},
        )
        return {
            "ids": dense["ids"][0] if dense.get("ids") else [],
            "distances": dense["distances"][0] if dense.get("distances") else [],
        }

    def bm25_search(self, query: str, records: list[dict[str, Any]], topk: int) -> list[tuple[int, float]]:
        if not records:
            return []
        docs = [r.get("content") or r.get("summary", "") for r in records]
        self.ltm.hybrid_retriever.fit(docs)
        if not self.ltm.hybrid_retriever.bm25:
            return []
        scores = self.ltm.hybrid_retriever.bm25.get_scores(_bm25_tokenize(query))
        ranked = np.argsort(scores)[::-1][:topk]
        return [(int(i), float(scores[i])) for i in ranked if scores[i] > 0]

    def fuse_search_results(
        self,
        *,
        query: str,
        records: list[dict[str, Any]],
        dense_ids: list[str],
        dense_distances: list[float],
        bm25_hits: list[tuple[int, float]],
        top_k: int = 3,
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        id_to_idx = {r["id"]: i for i, r in enumerate(records)}
        dense_scores: list[tuple[int, float]] = []
        for mem_id, dist in zip(dense_ids, dense_distances):
            idx = id_to_idx.get(mem_id)
            if idx is None:
                continue
            sim = SelfBuiltLongTermMemoryImproved._distance_to_similarity(dist)
            dense_scores.append((idx, sim))

        if bm25_hits:
            bm25_ranked = [idx for idx, _ in sorted(bm25_hits, key=lambda x: x[1], reverse=True)]
            rrf_k = self.ltm.hybrid_retriever.rrf_k
            rrf_scores: dict[int, float] = {}
            for rank, (idx, _) in enumerate(dense_scores):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
            for rank, idx in enumerate(bm25_ranked):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
            hybrid = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            hybrid = self.ltm.hybrid_retriever.search(query, dense_scores, top_k * 2)

        hits: list[dict[str, Any]] = []
        idx_to_sim = dict(dense_scores)
        for idx, rrf in hybrid[: top_k * 2]:
            rec = records[idx]
            vector_sim = idx_to_sim.get(idx, 0.0)
            final_score = rrf * vector_sim
            if final_score < threshold:
                continue
            hits.append({**rec, "rrf_score": rrf, "vector_sim": vector_sim, "final_score": final_score})
        return hits[:top_k]


class MemoryStore:
    def __init__(self, ltm: SelfBuiltLongTermMemoryImproved):
        self.ltm = ltm

    def get_user_records(self, user_id: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for i, mem_id in enumerate(self.ltm.ids):
            meta = self.ltm.metadatas[i]
            if meta.get("user_id") != user_id:
                continue
            records.append(
                {
                    "id": mem_id,
                    "content": self.ltm.documents[i],
                    **meta,
                }
            )
        return records

    def vectorize_texts(self, texts: list[str]) -> list[str]:
        """写入管道内部会向量化；此处返回文本供接口对齐。"""
        return list(texts)

    def add(
        self,
        extracted_memory: Any,
        *,
        user_id: str,
        embeddings: list[str] | None = None,
    ) -> dict[str, Any]:
        data = extracted_memory.data if hasattr(extracted_memory, "data") else extracted_memory
        key_points = data.get("key_points") or (embeddings or [])
        content = data.get("content") or data.get("summary") or "\n".join(key_points)
        memory_type = (data.get("metadata") or {}).get("memory_type", "preference")
        mid = self.ltm.ingest_long_term(
            content,
            memory_type=memory_type,
            importance=data.get("importance"),
            metadata={"user_id": user_id, **(data.get("metadata") or {})},
        )
        return {"ids": [mid], "key_points": key_points}


class MemoryExtractor:
    def __init__(self, compressor: MemoryCompressor):
        self.compressor = compressor

    def extract(self, interaction: list[dict[str, str]], metadata: dict[str, Any] | None = None) -> SimpleNamespace:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in interaction)
        data = self.compressor.extract_memory(text, metadata=metadata)
        meta = {**(metadata or {}), "key_points": data.get("key_points", [])}
        return SimpleNamespace(data=data, metadata=meta)


class AssistantLLM:
    def __init__(self, llm):
        self.llm = llm

    def chat(self, messages: list[dict[str, str]]) -> str:
        return str(self.llm.invoke(messages).content)


def build_memory_prompt(*, query: str, hits: list[dict[str, Any]]) -> str:
    if hits:
        memory_context = "\n".join(
            f"- [{h.get('memory_type', 'general')}] {h.get('summary') or h.get('content')}"
            for h in hits
        )
    else:
        memory_context = "（未检索到相关长期记忆）"
    return (
        "你是一个有记忆的智能助手。请结合【相关长期记忆】回答用户问题。\n\n"
        f"【相关长期记忆】\n{memory_context}\n\n"
        f"【用户问题】\n{query}\n"
    )


def chat_with_memories(
    message: str,
    *,
    retriever: MemoryHybridRetriever,
    memory_store: MemoryStore,
    extractor: MemoryExtractor,
    assistant_llm: AssistantLLM,
    user_id: str = DEMO_USER_ID,
    topk: int = 6,
    top_k: int = 3,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """三步：混合检索 → 结合记忆回答 → 抽取并写入新记忆。"""
    dense_result = retriever.dense_search(message, user_id, topk)
    user_records = memory_store.get_user_records(user_id)
    bm25_hits = retriever.bm25_search(message, user_records, topk)

    relevant_memories = retriever.fuse_search_results(
        query=message,
        records=user_records,
        dense_ids=dense_result["ids"],
        dense_distances=dense_result["distances"],
        bm25_hits=bm25_hits,
        top_k=top_k,
        threshold=threshold,
    )

    response_prompt = build_memory_prompt(query=message, hits=relevant_memories)
    assistant_response = assistant_llm.chat([{"role": "user", "content": response_prompt}])

    interaction = [
        {"role": "user", "content": message},
        {"role": "assistant", "content": assistant_response},
    ]
    extracted_memory = extractor.extract(interaction, metadata={"user_id": user_id})

    memory_texts = extracted_memory.metadata.get("key_points", []) if extracted_memory else []
    memory_vectors = memory_store.vectorize_texts(memory_texts)
    memory_results = memory_store.add(
        extracted_memory,
        user_id=user_id,
        embeddings=memory_vectors,
    )

    return {
        "relevant_memories": relevant_memories,
        "assistant_response": assistant_response,
        "extracted_key_points": memory_texts,
        "memory_write_result": memory_results,
    }
