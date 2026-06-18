"""OpenAI 兼容 Embedding 后端（DashScope / OpenAI）。"""

from __future__ import annotations

import os

import numpy as np

from agent_framework.config import load_project_dotenv


class OpenAIEmbeddingBackend:
    name = "embedding"

    def __init__(self) -> None:
        load_project_dotenv()
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("embedding 后端需要 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
        base_url = os.getenv(
            "DASHSCOPE_EMBEDDING_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).rstrip("/")
        model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")
        from langchain_openai import OpenAIEmbeddings

        self._client = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=base_url,
            check_embedding_ctx_length=False,
        )

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            return vec
        return vec / norm

    def embed(self, text: str) -> np.ndarray:
        vector = self._client.embed_query(text or " ")
        return self._normalize(np.asarray(vector, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        vectors = self._client.embed_documents([t or " " for t in texts])
        return [self._normalize(np.asarray(v, dtype=np.float32)) for v in vectors]
