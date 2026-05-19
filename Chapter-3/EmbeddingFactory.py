import os
import json
import re
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from sentence_transformers import SentenceTransformer

load_dotenv()


# ==============================
# 1. 统一嵌入模型工厂
# ==============================
class EmbeddingFactory:
    """支持BGE/OpenAI/Qwen三种嵌入模型的工厂类"""

    @staticmethod
    def create_embedding(model_type: str, **kwargs) -> Embeddings:
        """
        创建嵌入模型实例
        :param model_type: bge/openai/qwen
        """
        if model_type.lower() == "bge":
            return BGEEmbeddings(model_name=kwargs.get("model_name", "BAAI/bge-m3"))
        elif model_type.lower() == "openai":
            return OpenAIEmbeddings(
                model=kwargs.get("model_name", "text-embedding-3-small"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif model_type.lower() == "qwen":
            return DashScopeEmbeddings(
                model=kwargs.get("model_name", "text-embedding-v2"),
                dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
            )
        else:
            raise ValueError(f"不支持的嵌入模型类型: {model_type}")


class BGEEmbeddings(Embeddings):
    """BGE嵌入模型包装器，兼容LangChain接口"""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


# ==============================
# 2. 混合检索核心算法（BM25 + 余弦相似度 + RRF融合）
# ==============================
class HybridRetriever:
    """混合检索器：BM25稀疏检索 + 余弦相似度密集检索 + RRF结果融合"""

    def __init__(self, k1: float = 1.5, b: float = 0.75, rrf_k: int = 60):
        self.bm25 = None
        self.documents = []
        self.k1 = k1
        self.b = b
        self.rrf_k = rrf_k

    def fit(self, documents: List[str]):
        """用文档列表构建 BM25 索引（统计词频/IDF 等，非神经网络训练）"""
        self.documents = documents
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)

    def search(self, query: str, dense_scores: List[Tuple[int, float]], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        执行混合检索
        :param query: 查询文本
        :param dense_scores: 密集检索结果，格式为[(文档索引, 相似度得分)]
        :param top_k: 返回结果数量
        :return: 融合后的结果，格式为[(文档索引, 融合得分)]
        """
        if not self.bm25 or not self.documents:
            return dense_scores[:top_k]

        # BM25检索
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked = np.argsort(bm25_scores)[::-1]

        # 构建RRF排名字典
        rrf_scores = {}

        # 密集检索RRF得分
        for rank, (idx, _) in enumerate(dense_scores):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (self.rrf_k + rank)

        # BM25检索RRF得分
        for rank, idx in enumerate(bm25_ranked):
            if bm25_scores[idx] > 0:
                rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (self.rrf_k + rank)

        # 排序并返回top_k
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


# ==============================
# 3. 记忆抽取与合并器
# ==============================
def _llm_text(response: Any) -> str:
    """从 LangChain AIMessage 取出文本（兼容空 content / 多段 content）。"""
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()
    return str(content or "").strip()


def _parse_json_from_llm(text: str) -> Dict[str, Any]:
    """解析 LLM 返回的 JSON（支持 ```json 代码块与夹杂说明文字）。"""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("LLM 返回为空，无法解析 JSON")

    candidates = [raw]
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    brace = re.search(r"\{[\s\S]*\}", raw)
    if brace:
        candidates.append(brace.group(0))

    last_err: Optional[Exception] = None
    for piece in candidates:
        try:
            obj = json.loads(piece)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError as e:
            last_err = e
            continue
    raise ValueError(f"无法从 LLM 输出解析 JSON: {last_err}")


class MemoryCompressor:
    """记忆抽取与合并器，自动将相似记忆合并为核心要点"""

    def __init__(self, llm):
        self.llm = llm

    def _invoke_json(self, prompt: str, *, retries: int = 2) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            p = prompt if attempt == 0 else (
                prompt.rstrip()
                + "\n\n请只输出一个合法 JSON 对象，不要 markdown、不要解释。"
            )
            try:
                return _parse_json_from_llm(_llm_text(self.llm.invoke(p)))
            except Exception as e:
                last_err = e
        raise ValueError(str(last_err))

    def _fallback_extract(self, content: str, metadata: Dict[str, Any] | None) -> Dict[str, Any]:
        snippet = content.strip().replace("\n", " ")
        summary = snippet[:120] + ("…" if len(snippet) > 120 else "")
        return {
            "id": str(uuid.uuid4()),
            "content": content,
            "summary": summary,
            "key_points": [snippet] if snippet else [],
            "importance": float((metadata or {}).get("importance", 0.5)),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

    def extract_memory(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """从原始内容中抽取结构化记忆"""
        prompt = f"""请从以下内容中提取核心记忆点，保留所有重要信息：
        {content}
        
        只输出一个 JSON 对象，字段如下：
        {{
            "summary": "一句话总结",
            "key_points": ["要点1", "要点2"],
            "importance": 0.5
        }}
        importance 为 0.0 到 1.0 之间的数字。"""

        try:
            extracted = self._invoke_json(prompt)
        except Exception:
            return self._fallback_extract(content, metadata)

        imp = extracted.get("importance", 0.5)
        try:
            imp = float(imp)
        except (TypeError, ValueError):
            imp = 0.5

        return {
            "id": str(uuid.uuid4()),
            "content": content,
            "summary": str(extracted.get("summary", content)),
            "key_points": list(extracted.get("key_points") or []),
            "importance": imp,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

    def merge_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并多个相似记忆为一个综合记忆"""
        memories_text = "\n\n".join([f"记忆{i + 1}:\n{mem['content']}" for i, mem in enumerate(memories)])

        prompt = f"""请将以下多个相似记忆合并为一个综合记忆，保留所有独特的重要信息：
        {memories_text}
        
        只输出一个 JSON 对象：
        {{
            "summary": "合并后的一句话总结",
            "key_points": ["合并后的要点1", "合并后的要点2"],
            "importance": 0.5
        }}"""

        try:
            merged = self._invoke_json(prompt)
        except Exception:
            base = memories[0]
            return {
                **base,
                "id": str(uuid.uuid4()),
                "metadata": {"merged_from": [mem["id"] for mem in memories], **(base.get("metadata") or {})},
                "is_merged": True,
            }

        imp = merged.get("importance", 0.5)
        try:
            imp = float(imp)
        except (TypeError, ValueError):
            imp = 0.5
        key_points = list(merged.get("key_points") or [])

        return {
            "id": str(uuid.uuid4()),
            "content": "\n".join(key_points) if key_points else memories_text,
            "summary": str(merged.get("summary", memories[0].get("summary", ""))),
            "key_points": key_points,
            "importance": imp,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"merged_from": [mem["id"] for mem in memories]},
            "is_merged": True,
        }