"""后台任务 worker：轮询 JobStore 并执行编排。"""

from __future__ import annotations

import asyncio
from typing import Optional

from agent_framework.bootstrap.tenant_pool import get_tenant_pool
from agent_framework.infra.concurrency import RequestSlotTimeoutError
from agent_framework.observability.metrics import record_job_outcome
from services.jobs.store import JobStore


class JobWorker:
    def __init__(
        self,
        store: Optional[JobStore] = None,
        poll_interval_sec: float = 1.0,
    ) -> None:
        self.store = store or JobStore()
        self.poll_interval_sec = poll_interval_sec
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    async def run_forever(self) -> None:
        pool = get_tenant_pool()
        while not self._stop:
            job = self.store.claim_next_pending()
            if not job:
                await asyncio.sleep(self.poll_interval_sec)
                continue
            try:
                orch = await pool.get(
                    job.user_id,
                    domain=job.domain,
                    mode=job.mode,
                    transport=job.transport,
                    locale=job.locale,
                )
                result = await orch.process_request(job.query, thread_id=job.thread_id)
                self.store.mark_succeeded(
                    job.job_id,
                    {
                        "final_response": result.get("final_response", ""),
                        "execution_plan": result.get("execution_plan"),
                        "subtask_results": result.get("subtask_results"),
                    },
                    trace_id=result.get("trace_id"),
                )
                record_job_outcome(
                    job.domain,
                    "succeeded",
                    mode=job.mode,
                    transport=job.transport,
                )
            except RequestSlotTimeoutError as exc:
                self.store.mark_failed(job.job_id, f"slot timeout: {exc}")
                record_job_outcome(
                    job.domain,
                    "slot_timeout",
                    mode=job.mode,
                    transport=job.transport,
                )
            except asyncio.TimeoutError:
                self.store.mark_failed(job.job_id, "request timeout")
                record_job_outcome(
                    job.domain,
                    "timeout",
                    mode=job.mode,
                    transport=job.transport,
                )
            except Exception as exc:
                self.store.mark_failed(job.job_id, str(exc))
                record_job_outcome(
                    job.domain,
                    "failed",
                    mode=job.mode,
                    transport=job.transport,
                )
