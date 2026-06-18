"""异步任务模型。"""

from __future__ import annotations



from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class JobRecord:
    job_id: str
    user_id: str
    query: str
    thread_id: str
    status: JobStatus
    domain: str = "travel"
    mode: str = "fixed_graph"
    transport: str = "local"
    locale: str = "zh"
    result_json: Optional[str] = None
    error: Optional[str] = None
    trace_id: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "job_id": self.job_id,
            "domain": self.domain,
            "mode": self.mode,
            "transport": self.transport,
            "locale": self.locale,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.trace_id:
            out["trace_id"] = self.trace_id
        if self.error:
            out["error"] = self.error
        if self.result_json:
            import json

            try:
                out["result"] = json.loads(self.result_json)
            except json.JSONDecodeError:
                out["result_raw"] = self.result_json
        return out
