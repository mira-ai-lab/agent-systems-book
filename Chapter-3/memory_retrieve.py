"""记忆检索：融合向量召回、BM25 与重排分数。

更新时间：2026-11-12

检索模块从 ``MemoryStore`` 读取指定用户的画像事实和偏好记忆，通过向量
相似度与关键词检索完成候选召回，再结合时间衰减和重要性生成最终排名。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from memory_store import MemoryStore

try:
    import jieba

    def tokenize_for_bm25(text: str) -> list[str]:
        """使用 jieba 对中英文文本进行 BM25 分词。"""
        return [token.strip() for token in jieba.cut(text.lower()) if token.strip()]

except ImportError:

    def tokenize_for_bm25(text: str) -> list[str]:
        """jieba 不可用时按空白切分文本。"""
        return text.lower().split()


class HybridRetriever:
    """向量排名与 BM25 排名的 RRF 融合器。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75, rrf_k: int = 60) -> None:
        self.bm25: BM25Okapi | None = None
        self.documents: list[str] = []
        self.k1 = k1
        self.b = b
        self.rrf_k = rrf_k

    def fit(self, documents: list[str]) -> None:
        """构建 BM25 索引。

        Args:
            documents: 按本地索引顺序排列的文档列表。
        """
        self.documents = list(documents)
        tokenized_documents = [tokenize_for_bm25(document) for document in documents]
        self.bm25 = BM25Okapi(tokenized_documents, k1=self.k1, b=self.b)

    def search(
        self,
        query: str,
        dense_scores: list[tuple[int, float]],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """融合向量排名和 BM25 排名，返回文档索引及 RRF 分数。

        ``dense_scores`` 的顺序表示向量召回排名，其中的相似度只用于确定
        名次；最终分数量级由检索器结合向量相似度计算。
        """
        if not self.bm25 or not self.documents:
            return [
                (index, 1.0 / (self.rrf_k + rank))
                for rank, (index, _) in enumerate(dense_scores[:top_k])
            ]

        bm25_scores = self.bm25.get_scores(tokenize_for_bm25(query))
        bm25_ranked = np.argsort(bm25_scores)[::-1]

        rrf_scores: dict[int, float] = {}
        for rank, (index, _) in enumerate(dense_scores):
            rrf_scores[index] = rrf_scores.get(index, 0.0) + 1.0 / (
                self.rrf_k + rank
            )
        for rank, index in enumerate(bm25_ranked):
            if bm25_scores[index] > 0:
                rrf_scores[int(index)] = rrf_scores.get(int(index), 0.0) + 1.0 / (
                    self.rrf_k + rank
                )

        return sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]


class MemoryRetriever:
    """面向用户画像和偏好记忆的混合检索器。"""

    def __init__(
        self,
        store: MemoryStore,
        *,
        hybrid_retriever: HybridRetriever | None = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        time_decay_factor: float = 0.95,
    ) -> None:
        self.store = store
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.time_decay_factor = time_decay_factor

    def dense_search(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """执行纯向量检索，并返回带相似度和元数据的候选记录。"""
        if not self.store.documents:
            return []
        dense = self.store.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"user_id": user_id},
        )
        ids = (dense.get("ids") or [[]])[0]
        distances = (dense.get("distances") or [[]])[0]
        documents = (dense.get("documents") or [[]])[0]
        metadatas = (dense.get("metadatas") or [[]])[0]
        return [
            {
                "id": memory_id,
                "distance": distance,
                "vector_sim": self.distance_to_similarity(distance),
                "content": document,
                "metadata": metadata or {},
            }
            for memory_id, distance, document, metadata in zip(
                ids,
                distances,
                documents,
                metadatas,
            )
        ]

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        memory_types: list[str] | None = None,
        top_k: int | None = None,
        use_time_decay: bool = True,
        use_importance_weight: bool = True,
    ) -> list[dict[str, Any]]:
        """检索与查询相关的用户画像事实和偏好记忆。

        Args:
            query: 当前用户问题。
            user_id: 用户标识；未提供时使用存储实例的用户标识。
            memory_types: 可选的记忆类型过滤条件。
            top_k: 返回数量；未提供时使用检索器默认值。
            use_time_decay: 是否应用时间衰减。
            use_importance_weight: 是否应用重要性加权。

        Returns:
            按 ``final_score`` 降序排列的记忆记录。
        """
        resolved_user_id = user_id or self.store.user_id
        limit = top_k or self.top_k
        records = self.store.get_user_records(resolved_user_id, memory_types)
        if not records:
            return []

        documents = [
            record.get("content") or record.get("summary", "")
            for record in records
        ]
        self.hybrid_retriever.fit(documents)
        dense_records = self.dense_search(query, resolved_user_id, limit * 2)
        id_to_local_index = {
            record["id"]: index for index, record in enumerate(records)
        }
        dense_scores: list[tuple[int, float]] = []
        vector_scores: dict[int, float] = {}
        for dense_record in dense_records:
            index = id_to_local_index.get(dense_record["id"])
            if index is None:
                continue
            vector_similarity = dense_record["vector_sim"]
            dense_scores.append((index, vector_similarity))
            vector_scores[index] = vector_similarity

        hybrid_results = self.hybrid_retriever.search(
            query,
            dense_scores,
            limit * 2,
        )
        results: list[dict[str, Any]] = []
        for index, rrf_score in hybrid_results:
            record = records[index]
            vector_similarity = vector_scores.get(index, 0.0)
            score = rrf_score * vector_similarity
            if use_time_decay:
                score *= self._time_decay(
                    str(record.get("timestamp") or datetime.now().isoformat())
                )
            if use_importance_weight:
                score *= 0.5 + float(record.get("importance", 0.5))
            if score < self.similarity_threshold:
                continue
            results.append(
                {
                    **record,
                    "vector_sim": vector_similarity,
                    "rrf_score": rrf_score,
                    "final_score": score,
                    "score": score,
                }
            )
        return sorted(
            results,
            key=lambda item: item["final_score"],
            reverse=True,
        )[:limit]

    def _time_decay(self, timestamp: str) -> float:
        """根据记忆时间计算指数衰减权重。"""
        try:
            days = (datetime.now() - datetime.fromisoformat(timestamp)).days
        except ValueError:
            days = 0
        return self.time_decay_factor ** max(days, 0)

    @staticmethod
    def distance_to_similarity(distance: float) -> float:
        """将 Chroma 距离转换为 [0, 1] 范围内的相似度。"""
        return MemoryStore.distance_to_similarity(distance)
