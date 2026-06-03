"""
长期记忆与混合检索示例 - Memory Text Vector Detail

完整链路（课堂按步骤对照输出）：
  ① 抽取  对话/原文 → LLM 结构化（summary、key_points、importance）
  ② 合并  与库中相似记忆去重 → 相似则 LLM 合并后 update，否则新增
  ③ 存储  Chroma 向量索引 + BM25 倒排 + metadata
  ④ 检索  向量 + BM25 → RRF 融合 → final_score 重排
  ⑤ 使用  检索结果拼 Prompt → 大模型结合记忆回答

运行前请先安装依赖（建议 conda 环境 ）：
    pip install chromadb rank-bm25 jieba python-dotenv numpy langchain-openai langchain-core httpx

.env 配置：
    DASHSCOPE_API_KEY=你的密钥
    RUN_LLM=1          # 0：跳过 LLM 抽取/合并/回答，仅用种子数据
    RUN_LLM_ANSWER=1   # 0：步骤 1-4 照常，仅跳过步骤 5 调用大模型（API 失效时用）

运行方式：
    python memory_text_vector_detail.py    # 完整链路（抽取→合并→存储→检索→使用）
"""

import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError as e:
    print("导入 chromadb 失败。请使用 conda 环境 enterprise_bench_agents_test 运行，例如：")
    print(r'  conda activate enterprise_bench_agents_test')
    print(f"  当前解释器: {sys.executable}")
    raise SystemExit(1) from e

load_dotenv(Path(__file__).parent / ".env")

RUN_LLM = os.getenv("RUN_LLM", "1").strip().lower() not in ("0", "false", "no", "off")
RUN_LLM_ANSWER = os.getenv("RUN_LLM_ANSWER", "1").strip().lower() not in ("0", "false", "no", "off")
MERGE_SIMILARITY_THRESHOLD = 0.82  # 向量相似度 ≥ 此值则触发合并


def _user_where(user_id: str) -> Dict[str, Any]:
    """Chroma 过滤：单条件用 $eq；多条件才用 $and。"""
    return {"user_id": {"$eq": user_id}}


# ---------------------------------------------------------------------------
# 记忆抽取与合并（内嵌，避免 import EmbeddingFactory 拉取 sentence_transformers）
# ---------------------------------------------------------------------------
def _llm_text(response: Any) -> str:
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
    raise ValueError(f"无法从 LLM 输出解析 JSON: {last_err}")


class MemoryCompressor:
    """记忆抽取与合并器。"""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def _invoke_json(self, prompt: str, *, retries: int = 2) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            p = (
                prompt
                if attempt == 0
                else prompt.rstrip() + "\n\n请只输出一个合法 JSON 对象，不要 markdown、不要解释。"
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

    def _is_trivial(self, content: str) -> bool:
        """第一层：规则过滤。识别明显的闲聊或无意义内容。"""
        trivial_keywords = ["你好", "再见", "谢谢", "嗯", "哦", "好的", "哈哈"]
        return any(kw in content for kw in trivial_keywords) and len(content) < 20

    def extract_memory(self, content: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        # 规则过滤：如果是闲聊，直接给低分并快速返回
        if self._is_trivial(content):
            return {
                "id": str(uuid.uuid4()),
                "content": content,
                "summary": content,
                "key_points": [],
                "importance": 0.1,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }

        prompt = f"""请从以下内容中提取核心记忆点，并评估其重要性：
{content}

只输出一个 JSON 对象：
{{
  "summary": "一句话总结",
  "key_points": ["要点1"],
  "importance": 0.8,
  "reason": "简短说明为什么给这个分数"
}}

重要性评分标准 (0.0-1.0)：
- 1.0：硬性约束（过敏、证件号、密码）、核心身份。
- 0.8：强烈偏好（必须住海景房）、长期计划。
- 0.5：一般事实（今天吃了什么）、临时兴趣。
- 0.1：纯闲聊、问候。
"""
        try:
            extracted = self._invoke_json(prompt)
        except Exception:
            return self._fallback_extract(content, metadata)
        
        imp = float(extracted.get("importance", 0.5))
        return {
            "id": str(uuid.uuid4()),
            "content": content,
            "summary": str(extracted.get("summary", content)),
            "key_points": list(extracted.get("key_points") or []),
            "importance": max(0.0, min(1.0, imp)),  # 确保在 0-1 之间
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

    def merge_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        memories_text = "\n\n".join(
            f"记忆{i + 1}:\n{mem['content']}" for i, mem in enumerate(memories)
        )
        prompt = f"""请将以下多个相似记忆合并为一个综合记忆，保留所有独特的重要信息：
{memories_text}

只输出一个 JSON 对象：
{{"summary": "合并后总结", "key_points": ["要点1"], "importance": 0.5}}"""
        try:
            merged = self._invoke_json(prompt)
        except Exception:
            base = memories[0]
            return {**base, "metadata": {"merged_from": [m["id"] for m in memories]}}
        imp = float(merged.get("importance", 0.5) or 0.5)
        key_points = list(merged.get("key_points") or [])
        return {
            "id": str(uuid.uuid4()),
            "content": "\n".join(key_points) if key_points else memories_text,
            "summary": str(merged.get("summary", memories[0].get("summary", ""))),
            "key_points": key_points,
            "importance": imp,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"merged_from": [m["id"] for m in memories]},
            "is_merged": True,
        }

# ---------------------------------------------------------------------------
# 工具：中文分词（BM25 需要把句子切成词）
# ---------------------------------------------------------------------------
try:
    import jieba

    def tokenize(text: str) -> List[str]:
        return [w.strip() for w in jieba.cut(text.lower()) if w.strip()]
except ImportError:

    def tokenize(text: str) -> List[str]:
        return text.lower().split()


def default_llm():
    """
    对话 LLM 入口：固定走阿里云百炼，不用 .env 里的 OPENAI_BASE_URL。

    说明：若用 OPENAI_BASE_URL（如 modelfactory.lenovo.com）+ DASHSCOPE_API_KEY，
    网关会返回 401「token invalid」—— 密钥与地址必须同属一家服务。
    """
    from langchain_core.messages import HumanMessage  # noqa: F401
    from langchain_openai import ChatOpenAI
    import httpx

    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("请在 .env 中设置 DASHSCOPE_API_KEY")

    chat_base = (
        os.getenv("DASHSCOPE_CHAT_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).rstrip("/")
    chat_model = os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus")

    return ChatOpenAI(
        model=chat_model,
        temperature=0,
        base_url=chat_base,
        api_key=api_key,
        http_client=httpx.Client(verify=False),
    )


# ---------------------------------------------------------------------------
# 第 1 部分：混合检索器（向量排名 + BM25 → RRF）
# ---------------------------------------------------------------------------
class HybridRetriever:
    """混合检索：语义向量 + BM25 关键词，RRF 融合排名。"""

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.bm25 = None
        self.documents: List[str] = []

    def fit(self, documents: List[str]) -> None:
        self.documents = documents
        if documents:
            self.bm25 = BM25Okapi([tokenize(doc) for doc in documents])
        else:
            self.bm25 = None

    def search(
        self, query: str, dense_scores: List[Tuple[int, float]], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        if not self.bm25:
            return [
                (idx, 1.0 / (self.rrf_k + rank))
                for rank, (idx, _) in enumerate(dense_scores[:top_k])
            ]

        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_ranked = np.argsort(bm25_scores)[::-1]

        rrf_scores: Dict[int, float] = {}
        for rank, (idx, _) in enumerate(dense_scores):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, idx in enumerate(bm25_ranked):
            if bm25_scores[idx] > 0:
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# 第 2 部分：长期记忆库（抽取 → 合并 → 存储 → 检索 → 拼 Prompt）
# ---------------------------------------------------------------------------
MEMORY_PROMPT = """你是一个有记忆的智能助手。请结合【最近对话】与【相关长期记忆】回答用户问题。

【最近对话】
{recent_dialogue}

【相关长期记忆】
{memory_context}

【当前问题】
{query}
"""


class SimpleLongTermMemory:
    """长期记忆：写入管道（抽取+合并）+ Chroma + 混合检索。"""

    def __init__(
        self,
        user_id: str,
        persist_directory: str = "./chroma_student_demo",
        *,
        llm=None,
        merge_threshold: float = MERGE_SIMILARITY_THRESHOLD,
    ) -> None:
        self.user_id = user_id
        self.merge_threshold = merge_threshold
        api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("请在 .env 中设置 DASHSCOPE_API_KEY（阿里云百炼 API Key）")

        model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")
        base_url = os.getenv(
            "DASHSCOPE_EMBEDDING_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).rstrip("/")
        os.environ["DASHSCOPE_API_KEY"] = api_key

        self.client = chromadb.PersistentClient(path=persist_directory)
        embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            api_key_env_var="DASHSCOPE_API_KEY",
            model_name=model,
            api_base=base_url,
        )
        self.collection = self.client.get_or_create_collection(
            name="student_long_term_memory",
            embedding_function=embed_fn,
        )

        self._llm = llm
        self.compressor: Optional[MemoryCompressor] = (
            MemoryCompressor(llm) if llm is not None else None
        )

        self.hybrid = HybridRetriever()
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self._sync_from_chroma()

    def _sync_from_chroma(self) -> None:
        data = self.collection.get(where=_user_where(self.user_id))
        self.ids = list(data.get("ids") or [])
        self.documents = list(data.get("documents") or [])
        self.metadatas = list(data.get("metadatas") or [])
        self.hybrid.fit(self.documents)

    def clear_all(self) -> None:
        if self.ids:
            self.collection.delete(ids=self.ids)
        self.documents, self.metadatas, self.ids = [], [], []
        self.hybrid = HybridRetriever()

    @staticmethod
    def distance_to_similarity(distance: float) -> float:
        return max(0.0, min(1.0, 1.0 - float(distance) / 2.0))

    @staticmethod
    def _document_text(extracted: Dict[str, Any]) -> str:
        """写入向量库的文本：优先用 summary，便于检索。"""
        return str(extracted.get("summary") or extracted.get("content") or "").strip()

    def _build_metadata(
        self,
        extracted: Dict[str, Any],
        memory_type: str,
        importance: float,
        *,
        merged_from: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "user_id": self.user_id,
            "memory_type": memory_type,
            "importance": importance,
            "summary": extracted.get("summary", ""),
            "key_points": json.dumps(extracted.get("key_points") or [], ensure_ascii=False),
            "timestamp": extracted.get("timestamp", datetime.now().isoformat()),
        }
        if merged_from:
            meta["merged_from"] = json.dumps(merged_from, ensure_ascii=False)
        return meta

    def _extract(self, raw_content: str, memory_type: str) -> Dict[str, Any]:
        if self.compressor:
            extracted = self.compressor.extract_memory(
                raw_content, metadata={"memory_type": memory_type, "user_id": self.user_id}
            )
            return extracted
        snippet = raw_content.strip()
        return {
            "id": str(uuid.uuid4()),
            "content": raw_content,
            "summary": snippet[:120] + ("…" if len(snippet) > 120 else ""),
            "key_points": [snippet] if snippet else [],
            "importance": 0.5,
            "timestamp": datetime.now().isoformat(),
            "metadata": {},
        }

    def _find_duplicate(self, query_text: str, memory_type: str) -> Optional[str]:
        """在同类记忆中找近邻；类型不同不合并（避免「三亚行程」并到「海景房」）。"""
        if not self.documents:
            return None
        hits = self.collection.query(
            query_texts=[query_text],
            n_results=8,
            where={
                "$and": [
                    {"user_id": {"$eq": self.user_id}},
                    {"memory_type": {"$eq": memory_type}},
                ]
            },
        )
        if not hits["ids"] or not hits["ids"][0]:
            return None
        for mem_id, dist in zip(hits["ids"][0], hits["distances"][0]):
            if self.distance_to_similarity(dist) >= self.merge_threshold:
                return mem_id
        return None

    def ingest(
        self,
        raw_content: str,
        *,
        memory_type: str = "general",
        importance: float | None = None,
    ) -> Dict[str, Any]:
        """
        完整写入管道：抽取 → 去重/合并 → 存储。
        返回 action=add|merge 及摘要，便于课堂打印。
        """
        extracted = self._extract(raw_content, memory_type)
        imp = float(importance if importance is not None else extracted.get("importance", 0.5))
        store_text = self._document_text(extracted)
        dup_id = self._find_duplicate(raw_content, memory_type)

        if dup_id and self.compressor:
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
            imp = float(merged.get("importance", imp))
            store_text = self._document_text(merged)
            meta = self._build_metadata(
                merged,
                memory_type,
                imp,
                merged_from=[dup_id, extracted["id"]],
            )
            self.collection.update(ids=[dup_id], documents=[store_text], metadatas=[meta])
            idx = self.ids.index(dup_id)
            self.documents[idx] = store_text
            self.metadatas[idx] = meta
            self.hybrid.fit(self.documents)
            return {
                "action": "merge",
                "memory_id": dup_id,
                "summary": meta["summary"],
                "importance": imp,
                "memory_type": memory_type,
            }

        mem_id = extracted["id"]
        meta = self._build_metadata(extracted, memory_type, imp)
        self.collection.add(ids=[mem_id], documents=[store_text], metadatas=[meta])
        self.ids.append(mem_id)
        self.documents.append(store_text)
        self.metadatas.append(meta)
        self.hybrid.fit(self.documents)
        return {
            "action": "add",
            "memory_id": mem_id,
            "summary": meta["summary"],
            "importance": imp,
            "memory_type": memory_type,
        }

    def add_memory(
        self,
        text: str,
        *,
        memory_type: str = "general",
        importance: float = 0.5,
    ) -> str:
        """跳过抽取，直接存储（离线演示 / 批量种子数据）。"""
        mem_id = str(uuid.uuid4())
        meta = self._build_metadata(
            {"summary": text, "key_points": [text], "timestamp": datetime.now().isoformat()},
            memory_type,
            importance,
        )
        self.collection.add(ids=[mem_id], documents=[text], metadatas=[meta])
        self.ids.append(mem_id)
        self.documents.append(text)
        self.metadatas.append(meta)
        self.hybrid.fit(self.documents)
        return mem_id

    def search_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        混合检索，返回每条记忆的详细分数（含时间衰减）。
        """
        if not self.documents:
            return []

        # 第 1 步：向量检索
        dense = self.collection.query(
            query_texts=[query],
            n_results=top_k * 3,
            where=_user_where(self.user_id),
        )

        dense_scores: List[Tuple[int, float]] = []
        idx_to_vector_sim: Dict[int, float] = {}
        for mem_id, dist in zip(dense["ids"][0], dense["distances"][0]):
            if mem_id not in self.ids:
                continue
            idx = self.ids.index(mem_id)
            vector_sim = self.distance_to_similarity(dist)
            idx_to_vector_sim[idx] = vector_sim
            dense_scores.append((idx, vector_sim))

        # 第 2 步：混合检索 (RRF)
        hybrid_hits = self.hybrid.search(query, dense_scores, top_k * 3)

        # 第 3 步：关键词匹配与重排
        query_tokens = set(tokenize(query))
       

        now = datetime.now()
        results: List[Dict[str, Any]] = []
        for idx, rrf_score in hybrid_hits:
            meta = self.metadatas[idx]
            vector_sim = idx_to_vector_sim.get(idx, 0.0)
            base_importance = float(meta.get("importance", 0.5))

            # --- 第三层：时间衰减计算 ---
            ts_str = meta.get("timestamp", now.isoformat())
            try:
                ts = datetime.fromisoformat(ts_str)
                months_diff = (now - ts).days / 30.0
                # 衰减公式：每月衰减 10% (0.9 ^ 月数)
                time_decay = 0.9 ** months_diff
            except Exception:
                time_decay = 1.0
            
            decayed_importance = base_importance * time_decay

            # 关键词加权
            doc_text = (meta.get("summary", "") + " " + self.documents[idx]).lower()
            doc_tokens = set(tokenize(doc_text))
            overlap = len(query_tokens & doc_tokens)
            keyword_boost = 1.0 + (overlap * 0.1)

            # 类型惩罚
            type_penalty = 1.0
         

            # 最终公式：加入时间衰减后的 importance
            final_score = rrf_score * vector_sim * (0.5 + decayed_importance) * keyword_boost * type_penalty

            results.append(
                {
                    "summary": meta.get("summary", self.documents[idx]),
                    "memory_type": meta.get("memory_type", "general"),
                    "importance": base_importance,      # 原始重要性
                    "decayed_importance": decayed_importance, # 衰减后的重要性
                    "vector_sim": vector_sim,
                    "rrf_score": rrf_score,
                    "final_score": final_score,
                    "keyword_boost": keyword_boost,
                    "age_months": round(months_diff, 1),
                }
            )

        return sorted(results, key=lambda x: x["final_score"], reverse=True)[:top_k]

    def build_prompt(
        self,
        query: str,
        hits: List[Dict[str, Any]] | None = None,
        *,
        recent_dialogue: str | None = None,
    ) -> str:
        hits = hits if hits is not None else self.search_memories(query)
        if hits:
            memory_context = "\n".join(
                f"- [{h['memory_type']}] {h['summary']} "
                f"(importance={h.get('importance', 0.5):.2f}, "
                f"final={h['final_score']:.4f}, vector_sim={h['vector_sim']:.4f})"
                for h in hits
            )
        else:
            memory_context = "（未找到相关记忆）"

        dialogue = recent_dialogue or "（无最近对话）"
        return MEMORY_PROMPT.format(
            recent_dialogue=dialogue,
            memory_context=memory_context,
            query=query,
        )


# ---------------------------------------------------------------------------
# 第 3 部分：演示主流程
# ---------------------------------------------------------------------------
def print_pipeline_overview() -> None:
    print("【记忆全流程】抽取 → 合并 → 存储 → 检索 → 使用")
    print("-" * 70)
    print("  ① 抽取  原始对话 → MemoryCompressor.extract_memory (LLM JSON)")
    print("  ② 合并  与库中相似记忆(向量≥阈值) → merge_memories → update")
    print("  ③ 存储  Chroma 向量 + 本地 BM25 索引 + metadata")
    print("  ④ 检索  向量 + BM25 → RRF → final_score 排序")
    print("  ⑤ 使用  build_prompt → ChatOpenAI 结合记忆回答")
    print(f"  当前 RUN_LLM={'开启' if RUN_LLM else '关闭'}  RUN_LLM_ANSWER={'开启' if RUN_LLM_ANSWER else '关闭'}")
    print(f"  Python: {sys.executable}")
    print("-" * 70)
    print()


def print_ingest_result(raw: str, result: Dict[str, Any]) -> None:
    action = "合并更新" if result["action"] == "merge" else "新增"
    print(f"  [{action}] [{result['memory_type']}] importance={result['importance']:.2f}")
    print(f"           原文片段: {raw[:50]}{'…' if len(raw) > 50 else ''}")
    print(f"           入库摘要: {result['summary']}")
    print()


def print_search_result(query: str, hits: List[Dict[str, Any]], *, teaching: bool = True) -> None:
    print("=" * 70)
    print(f"问题：{query}")
    print("-" * 70)
    if not hits:
        print("  （没有找到相关记忆）")
        return
    for i, h in enumerate(hits, 1):
        imp = float(h.get("importance", 0.5))
        decayed_imp = float(h.get("decayed_importance", imp))
        vs, rrf, final = h["vector_sim"], h["rrf_score"], h["final_score"]
        kb = h.get("keyword_boost", 1.0)
        age = h.get("age_months", 0)
        
        print(f"  [{i}] {h['summary']}")
        print(f"      [入库标注] 类型={h['memory_type']}  importance={imp:.2f}  (距今 {age} 个月)")
        print(f"      [检索计算] vector_sim={vs:.4f}  rrf_score={rrf:.4f}  keyword_boost={kb:.2f}")
        if teaching:
            weight = 0.5 + decayed_imp
            print(
                f"      [排序公式] final = rrf × vector_sim × (0.5+decay_imp) × keyword_boost"
                f"\n                = {rrf:.4f} × {vs:.4f} × {weight:.2f} × {kb:.2f} = {final:.4f}"
            )


# 模拟多轮对话原文（步骤 ② 抽取的输入）
DIALOGUE_SEGMENTS: List[Tuple[str, str]] = [
    # memory_type 分得越细，合并时越不容易把「行程」和「预算」并成一条
    ("用户：我计划下个月3月15日到20日去三亚旅游，想玩5天4夜。", "travel_plan"),
    (
        "用户：住宿一定要海景房，每晚预算不超过800元，酒店必须有WiFi和空调，不要青旅。。",
        "lodging_preference",
    ),
    ("用户：我是素食主义者，不吃海鲜，也不能吃辣，对辣椒过敏。", "diet"),
    ("用户：总旅游预算5000元（不含机票），住宿可以多花钱，吃饭要省一点。", "budget"),
]

# 故意与「海景房」相近，用于演示 ② 合并
MERGE_DEMO_SEGMENT = (
    "用户再次强调：一定要住海景酒店，每晚800块以内，必须有WiFi。",
    "lodging_preference",
)

# RUN_LLM=0 时的离线种子（跳过 LLM 抽取）
OFFLINE_SEED_MEMORIES: List[Tuple[str, str, float]] = [
    ("用户喜欢住海景房，预算每晚不超过800元。", "preference", 0.9),
    ("用户计划下个月（3月15日-20日）去三亚旅游。", "fact", 0.8),
    ("用户是素食主义者，不吃海鲜。", "fact", 0.9),
    ("用户不能吃辣，对辣椒过敏。", "fact", 0.8),
    ("用户总预算为5000元（不含往返机票）。", "fact", 0.9),
]

RECENT_DIALOGUE = (
    "user: 你好，我想规划一次三亚旅行\n"
    "assistant: 好的，请告诉我时间和偏好。\n"
    "user: 大概3月中下旬，5天4夜，要海景房。\n"
    "assistant: 已记录，还会帮您记住饮食和预算偏好。"
)


def main() -> None:
    """
        【记忆全流程】抽取 → 合并 → 存储 → 检索 → 使用
    ----------------------------------------------------------------------
      ① 抽取  原始对话 → MemoryCompressor.extract_memory (LLM JSON)
      ② 合并  与库中相似记忆(向量≥阈值) → merge_memories → update
      ③ 存储  Chroma 向量 + 本地 BM25 索引 + metadata
      ④ 检索  向量 + BM25 → RRF → final_score 排序
      ⑤ 使用  build_prompt → ChatOpenAI 结合记忆回答

    """

    print("【步骤 1】创建长期记忆库\n")
    llm = default_llm() if RUN_LLM else None
    memory = SimpleLongTermMemory(user_id="student_demo", llm=llm)
    memory.clear_all()

    # --- 步骤 2：抽取 + 合并 + 存储 ---
    if RUN_LLM:
        print("【步骤 2】抽取 → 合并 → 存储（ingest 管道）\n")
        for raw, mtype in DIALOGUE_SEGMENTS:
            result = memory.ingest(raw, memory_type=mtype)
            print_ingest_result(raw, result)

        print("【步骤 2b】演示合并：写入与「海景房」相近的一条\n")
        raw, mtype = MERGE_DEMO_SEGMENT
        result = memory.ingest(raw, memory_type=mtype)
        print_ingest_result(raw, result)
        if result["action"] == "merge":
            print("  → 已触发 merge_memories，更新原记忆而非新增一条。\n")
        else:
            print("  → 未命中相似阈值，作为新记忆写入（可调 MERGE_SIMILARITY_THRESHOLD）。\n")
    else:
        print("【步骤 2】RUN_LLM=0：跳过 LLM 抽取，直接写入种子记忆\n")
        for text, mtype, imp in OFFLINE_SEED_MEMORIES:
            memory.add_memory(text, memory_type=mtype, importance=imp)
            print(f"  [直接存储] [{mtype}] {text}")

    print(f"\n  库中共有 {len(memory.metadatas)} 条记忆")

    # --- 步骤 3：检索 ---
    print("\n【步骤 3】混合检索（不同问题 → 分数变化）\n")
    test_queries = [
        "我喜欢住什么样的酒店吗？",
        "我对什么过敏？",
        "我的旅行预算是多少？",
        "我想去哪里旅游？",
    ]
    for query in test_queries:
        print_search_result(query, memory.search_memories(query, top_k=3))
        print()

    # --- 步骤 4：使用（拼 Prompt）---
    print("\n【步骤 4】使用：检索结果 + 最近对话 → Prompt\n")
    example_query = "你还记得我喜欢住什么样的酒店吗？我喜欢吃什么？"
    example_hits = memory.search_memories(example_query)
    prompt = memory.build_prompt(
        example_query, example_hits, recent_dialogue=RECENT_DIALOGUE
    )
    print(prompt)

    from langchain_core.messages import HumanMessage

    print("\n【步骤 5】使用：调用大模型结合记忆回答\n")
    answer = llm.invoke([HumanMessage(content=prompt)])
    print(answer.content)


if __name__ == "__main__":
    main()


# 【步骤 1】创建长期记忆库

# 【步骤 2】抽取 → 合并 → 存储（ingest 管道）

#   [新增] [travel_plan] importance=0.80
#            原文片段: 用户：我计划下个月3月15日到20日去三亚旅游，想玩5天4夜。
#            入库摘要: 用户计划于3月15日至20日（5天4夜）赴三亚旅游。

#   [新增] [lodging_preference] importance=0.80
#            原文片段: 用户：住宿一定要海景房，每晚预算不超过800元，酒店必须有WiFi和空调，不要青旅。。
#            入库摘要: 用户坚持入住海景房，每晚预算上限800元，且酒店须配备WiFi和空调，明确排除青旅。

#   [新增] [diet] importance=0.80
#            原文片段: 用户：我是素食主义者，不吃海鲜，也不能吃辣，对辣椒过敏。
#            入库摘要: 用户是素食主义者，不吃海鲜且对辣椒过敏。

#   [新增] [budget] importance=0.80
#            原文片段: 用户：总旅游预算5000元（不含机票），住宿可以多花钱，吃饭要省一点。
#            入库摘要: 用户设定总旅游预算5000元（不含机票），倾向住宿多投入、餐饮严格节俭。

# 【步骤 2b】演示合并：写入与「海景房」相近的一条

#   [合并更新] [lodging_preference] importance=0.50
#            原文片段: 用户再次强调：一定要住海景酒店，每晚800块以内，必须有WiFi。
#            入库摘要: 用户坚持入住海景房（明确要求‘海景酒店’，排除青旅），每晚预算严格控制在800元以内，酒店必须配备WiFi和空调。

#   → 已触发 merge_memories，更新原记忆而非新增一条。


#   库中共有 4 条记忆

# 【步骤 3】混合检索（不同问题 → 分数变化）

# ======================================================================
# 问题：我喜欢住什么样的酒店吗？
# ----------------------------------------------------------------------
#   [1] 用户设定总旅游预算5000元（不含机票），倾向住宿多投入、餐饮严格节俭。
#       [入库标注] 类型=budget  importance=0.80  (距今 0.0 个月)
#       [检索计算] vector_sim=0.7938  rrf_score=0.0164  keyword_boost=1.00
#       [排序公式] final = rrf × vector_sim × (0.5+decay_imp) × keyword_boost
#                 = 0.0164 × 0.7938 × 1.30 × 1.00 = 0.0169
#   [2] 用户计划于3月15日至20日（5天4夜）赴三亚旅游。
#       [入库标注] 类型=travel_plan  importance=0.80  (距今 0.0 个月)
#       [检索计算] vector_sim=0.7398  rrf_score=0.0161  keyword_boost=1.00
#       [排序公式] final = rrf × vector_sim × (0.5+decay_imp) × keyword_boost
#                 = 0.0161 × 0.7398 × 1.30 × 1.00 = 0.0155
#   [3] 用户是素食主义者，不吃海鲜且对辣椒过敏。
#       [入库标注] 类型=diet  importance=0.80  (距今 0.0 个月)
#       [检索计算] vector_sim=0.7013  rrf_score=0.0159  keyword_boost=1.00
#       [排序公式] final = rrf × vector_sim × (0.5+decay_imp) × keyword_boost
#                 = 0.0159 × 0.7013 × 1.30 × 1.00 = 0.0145

# ======================================================================
# 问题：我对什么过敏？
# ----------------------------------------------------------------------
#   [1] 用户是素食主义者，不吃海鲜且对辣椒过敏。
#       [入库标注] 类型=diet  importance=0.80  (距今 0.0 个月)
#       [检索计算] vector_sim=0.8340  rrf_score=0.0167  keyword_boost=1.00
#       [排序公式] final = rrf × vector_sim × (0.5+decay_imp) × keyword_boost
#                 = 0.0167 × 0.8340 × 1.30 × 1.00 = 0.0181
#   [2] 用户计划于3月15日至20日（5天4夜）赴三亚旅游。
#       [入库标注] 类型=travel_plan  importance=0.80  (距今 0.0 个月)
#       [检索计算] vector_sim=0.7048  rrf_score=0.0164  keyword_boost=1.00
#       [排序公式] final = rrf × vector_sim × (0.5+decay_imp) × keyword_boost
#                 = 0.0164 × 0.7048 × 1.30 × 1.00 = 0.0150
#   [3] 用户设定总旅游预算5000元（不含机票），倾向住宿多投入、餐饮严格节俭。
#       [入库标注] 类型=budget  importance=0.80  (距今 0.0 个月)
#       [检索计算] vector_sim=0.6902  rrf_score=0.0159  keyword_boost=1.00
#       [排序公式] final = rrf × vector_sim × (0.5+decay_imp) × keyword_boost
#                 = 0.0159 × 0.6902 × 1.30 × 1.00 = 0.0142

# ======================================================================
# 问题：我的旅行预算是多少？
# ----------------------------------------------------------------------
#   [1] 用户设定总旅游预算5000元（不含机票），倾向住宿多投入、餐饮严格节俭。
#       [入库标注] 类型=budget  importance=0.80  (距今 0.0 个月)
#       [检索计算] vector_sim=0.8520  rrf_score=0.0167  keyword_boost=1.00
#       [排序公式] final = rrf × vector_sim × (0.5+decay_imp) × keyword_boost
#                 = 0.0167 × 0.8520 × 1.30 × 1.00 = 0.0185
#   [2] 用户计划于3月15日至20日（5天4夜）赴三亚旅游。
#       [入库标注] 类型=travel_plan  importance=0.80  (距今 0.0 个月)
#       [检索计算] vector_sim=0.7868  rrf_score=0.0161  keyword_boost=1.00
#       [排序公式] final = rrf × vector_sim × (0.5+decay_imp) × keyword_boost
#                 = 0.0161 × 0.7868 × 1.30 × 1.00 = 0.0165
#   [3] 用户是素食主义者，不吃海鲜且对辣椒过敏。
#       [入库标注] 类型=diet  importance=0.80  (距今 0.0 个月)
#       [检索计算] vector_sim=0.7171  rrf_score=0.0159  keyword_boost=1.00
#       [排序公式] final = rrf × vector_sim × (0.5+decay_imp) × keyword_boost
#                 = 0.0159 × 0.7171 × 1.30 × 1.00 = 0.0148

# ======================================================================
# 问题：我想去哪里旅游？
# ----------------------------------------------------------------------
#   [1] 用户计划于3月15日至20日（5天4夜）赴三亚旅游。
#       [入库标注] 类型=travel_plan  importance=0.80  (距今 0.0 个月)
#       [检索计算] vector_sim=0.7669  rrf_score=0.0167  keyword_boost=1.00
#       [排序公式] final = rrf × vector_sim × (0.5+decay_imp) × keyword_boost
#                 = 0.0167 × 0.7669 × 1.30 × 1.00 = 0.0166
#   [2] 用户设定总旅游预算5000元（不含机票），倾向住宿多投入、餐饮严格节俭。
#       [入库标注] 类型=budget  importance=0.80  (距今 0.0 个月)
#       [检索计算] vector_sim=0.7609  rrf_score=0.0164  keyword_boost=1.00
#       [排序公式] final = rrf × vector_sim × (0.5+decay_imp) × keyword_boost
#                 = 0.0164 × 0.7609 × 1.30 × 1.00 = 0.0162
#   [3] 用户是素食主义者，不吃海鲜且对辣椒过敏。
#       [入库标注] 类型=diet  importance=0.80  (距今 0.0 个月)
#       [检索计算] vector_sim=0.7260  rrf_score=0.0159  keyword_boost=1.00
#       [排序公式] final = rrf × vector_sim × (0.5+decay_imp) × keyword_boost
#                 = 0.0159 × 0.7260 × 1.30 × 1.00 = 0.0150


# 【步骤 4】使用：检索结果 + 最近对话 → Prompt

# 你是一个有记忆的智能助手。请结合【最近对话】与【相关长期记忆】回答用户问题。

# 【最近对话】
# user: 你好，我想规划一次三亚旅行
# assistant: 好的，请告诉我时间和偏好。
# user: 大概3月中下旬，5天4夜，要海景房。
# assistant: 已记录，还会帮您记住饮食和预算偏好。

# 【相关长期记忆】
# - [budget] 用户设定总旅游预算5000元（不含机票），倾向住宿多投入、餐饮严格节俭。 (importance=0.80, final=0.0170, vector_sim=0.7842)
# - [travel_plan] 用户计划于3月15日至20日（5天4夜）赴三亚旅游。 (importance=0.80, final=0.0153, vector_sim=0.7319)
# - [diet] 用户是素食主义者，不吃海鲜且对辣椒过敏。 (importance=0.80, final=0.0145, vector_sim=0.7032)
# - [lodging_preference] 用户坚持入住海景房（明确要求‘海景酒店’，排除青旅），每晚预算严格控制在800元以内，酒店必须配备WiFi和空调。 (importance=0.50, final=0.0126, vector_sim=0.7710)

# 【当前问题】
# 你还记得我喜欢住什么样的酒店吗？我喜欢吃什么？


# 【步骤 5】使用：调用大模型结合记忆回答

# 当然记得！根据我们的对话和您的长期偏好：

# **酒店偏好**：  
# - 必须是**海景房**（明确要求“海景酒店”，不接受青旅或无海景房型）；  
# - 每晚预算≤800元，4晚总住宿预算控制在3200元以内（您总预算5000元不含机票，住宿可适当多投入）；  
# - 酒店需配备**WiFi和空调**（必备设施）；  
# - 已锁定行程为**3月15日—20日（5天4夜）**。

# **饮食偏好**：  
# - **纯素食主义者**，完全不吃肉类、蛋奶类（含奶酪、酸奶等）、海鲜及任何动物副产品；  
# - **严格忌海鲜**（包括鱼露、虾皮、蚝油等隐形海鲜成分）；  
# - **对辣椒过敏**，需完全避免辣椒、辣油、花椒、辣酱及含辣椒提取物的调味品；  
# - 餐饮方面倾向**严格节俭**，优先选择干净、有素餐标识的本地餐馆或酒店自助早餐（可提前确认素食选项）。

# 需要我为您推荐几处符合海景+素食友好+预算可控的三亚酒店，并附上周边素食餐厅建议吗？
