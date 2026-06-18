"""知识库文档模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class KnowledgeDocument:
    doc_id: str
    agent: str
    text: str
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.doc_id,
            "agent": self.agent,
            "text": self.text,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeDocument":
        doc_id = str(data.get("id") or data.get("doc_id") or "").strip()
        agent = str(data.get("agent") or "").strip()
        text = str(data.get("text") or "").strip()
        if not doc_id or not agent or not text:
            raise ValueError("知识文档需要 id、agent、text")
        tags = [str(t).strip() for t in (data.get("tags") or []) if str(t).strip()]
        return cls(doc_id=doc_id, agent=agent, text=text, tags=tags)
