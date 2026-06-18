"""Router Engine 阶段开关。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouterConfig:
    enable_classification: bool = True
    enable_history_gate: bool = True
    enable_interaction_rewrite: bool = True
    enable_extraction: bool = True
    enable_knowledge_routing: bool = True
    enable_instruction_build: bool = True
    enable_task_decomposition: bool = True
    semantic_task_routing: bool = True
    locale: str = "zh"
    classification_note: str = ""
    extraction_note: str = ""
    task_info: str = ""
    knowledge_backend: str = "hybrid"  # keyword | vector | hybrid
    knowledge_top_k: int = 5
    knowledge_min_score: float = 0.65
    knowledge_vector_min_score: float = 0.15
    knowledge_embedding_backend: str = "hashing"  # hashing | embedding
    knowledge_storage: str = "auto"  # auto | memory | chroma
