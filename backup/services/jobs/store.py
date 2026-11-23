"""SQLite 异步任务存储。"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent_framework.config import JOB_DB_PATH
from services.jobs.models import JobRecord, JobStatus


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or JOB_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    domain TEXT NOT NULL DEFAULT 'travel',
                    query TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    trace_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
            if "domain" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN domain TEXT NOT NULL DEFAULT 'travel'")
            if "mode" not in cols:
                conn.execute(
                    "ALTER TABLE jobs ADD COLUMN mode TEXT NOT NULL DEFAULT 'fixed_graph'"
                )
            if "transport" not in cols:
                conn.execute(
                    "ALTER TABLE jobs ADD COLUMN transport TEXT NOT NULL DEFAULT 'local'"
                )
            if "locale" not in cols:
                conn.execute(
                    "ALTER TABLE jobs ADD COLUMN locale TEXT NOT NULL DEFAULT 'zh'"
                )
            conn.commit()

    def create_job(
        self,
        *,
        user_id: str,
        query: str,
        thread_id: str,
        domain: str,
        mode: str = "fixed_graph",
        transport: str = "local",
        locale: str = "zh",
    ) -> JobRecord:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        now = _utc_now()
        resolved_domain = (domain or "travel").strip() or "travel"
        resolved_mode = (mode or "fixed_graph").strip() or "fixed_graph"
        resolved_transport = (transport or "local").strip() or "local"
        resolved_locale = (locale or "zh").strip() or "zh"
        record = JobRecord(
            job_id=job_id,
            user_id=user_id,
            query=query,
            thread_id=thread_id,
            domain=resolved_domain,
            mode=resolved_mode,
            transport=resolved_transport,
            locale=resolved_locale,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, user_id, domain, mode, transport, locale, query, thread_id, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.job_id,
                    record.user_id,
                    record.domain,
                    record.mode,
                    record.transport,
                    record.locale,
                    record.query,
                    record.thread_id,
                    record.status.value,
                    record.created_at,
                    record.updated_at,
                ),
            )
            conn.commit()
        return record

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def claim_next_pending(self) -> Optional[JobRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs WHERE status = ?
                ORDER BY created_at ASC LIMIT 1
                """,
                (JobStatus.PENDING.value,),
            ).fetchone()
            if not row:
                return None
            now = _utc_now()
            updated = conn.execute(
                """
                UPDATE jobs SET status = ?, updated_at = ?
                WHERE job_id = ? AND status = ?
                """,
                (JobStatus.RUNNING.value, now, row["job_id"], JobStatus.PENDING.value),
            )
            conn.commit()
            if updated.rowcount != 1:
                return None
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
        return self._row_to_record(row) if row else None

    def mark_succeeded(self, job_id: str, result: dict, trace_id: Optional[str] = None) -> None:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, result_json = ?, trace_id = ?, updated_at = ?, error = NULL
                WHERE job_id = ?
                """,
                (
                    JobStatus.SUCCEEDED.value,
                    json.dumps(result, ensure_ascii=False, default=str),
                    trace_id,
                    now,
                    job_id,
                ),
            )
            conn.commit()

    def mark_failed(self, job_id: str, error: str) -> None:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE job_id = ?
                """,
                (JobStatus.FAILED.value, error[:2000], now, job_id),
            )
            conn.commit()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        keys = set(row.keys())
        domain = row["domain"] if "domain" in keys else "travel"
        mode = row["mode"] if "mode" in keys else "fixed_graph"
        transport = row["transport"] if "transport" in keys else "local"
        locale = row["locale"] if "locale" in keys else "zh"
        return JobRecord(
            job_id=row["job_id"],
            user_id=row["user_id"],
            query=row["query"],
            thread_id=row["thread_id"],
            domain=domain,
            mode=mode,
            transport=transport,
            locale=locale,
            status=JobStatus(row["status"]),
            result_json=row["result_json"],
            error=row["error"],
            trace_id=row["trace_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
