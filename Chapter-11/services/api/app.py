"""多领域多智能体 HTTP API（agent-platform v0.6）。"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from agent_framework.bootstrap.tenant_pool import get_tenant_pool
from agent_framework.config import DEFAULT_DOMAIN, READY_DOMAIN, load_project_dotenv
from agent_framework.i18n.locale import normalize_locale
from agent_framework.router.observability import enrich_routing_observability
from agent_framework.domain.dynamic_registry import DynamicAgentRecord, get_dynamic_agent_store
from agent_framework.domain.plugin_registry import get_domain_plugin, list_domains
from agent_framework.infra.concurrency import RequestSlotTimeoutError
from agent_framework.stream.events import error_event
from agent_framework.stream.sse import format_sse
from agent_framework.tracing import setup_observability
from services.api.auth import require_api_key
from services.api.metrics import metrics_enabled, record_chat, record_job, render_metrics
from services.jobs.store import JobStore
from services.jobs.worker import JobWorker

load_project_dotenv()
setup_observability()

_job_store = JobStore()
_worker = JobWorker(store=_job_store)
_worker_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _worker_task
    _worker_task = asyncio.create_task(_worker.run_forever())
    yield
    _worker.stop()
    if _worker_task:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Multi-Agent Platform API",
    version="0.20.0",
    description=(
        "企业通用多智能体路由平台 HTTP 服务。\n\n"
        "**推荐调用**：`POST /v1/chat` 仅传 `query` 即可（`profile` 默认 `auto`）；"
        "`domain` 可省略，由 `DEFAULT_DOMAIN` 或 LLM 跨域推断。"
        "详见 `GET /v1/domains`。"
    ),
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "退货政策是什么？",
                    "profile": "auto",
                },
                {
                    "query": "订单号是 12345，我想咨询退货政策",
                    "domain": "customer_service",
                    "profile": "auto",
                },
            ]
        }
    )

    query: str = Field(..., min_length=1, max_length=8000, description="用户问题（唯一必填）")
    domain: Optional[str] = Field(
        default=None,
        max_length=64,
        description="已注册领域名；省略时按 DEFAULT_DOMAIN 或 LLM 跨域推断",
    )
    profile: str = Field(
        default="auto",
        description="执行 Profile：auto（推荐）| workflow | adaptive | hybrid",
    )
    mode: str = Field(
        default="fixed_graph",
        description="遗留字段；profile 未指定时映射为 workflow/adaptive",
    )
    transport: str = Field(
        default="local",
        description="Supervisor 子 Agent 传输：local | a2a | mixed（仅 mode=supervisor 生效）",
    )
    user_id: Optional[str] = Field(default="default", max_length=128)
    thread_id: Optional[str] = Field(default=None, max_length=128)
    timeout_sec: Optional[float] = Field(default=None, gt=0, le=600)
    conversation_history: Optional[str] = Field(
        default=None,
        max_length=32000,
        description="可选多轮对话历史（profile=auto 时供 history_gate / interaction_rewrite）",
    )
    locale: str = Field(
        default="zh",
        description="话术 locale：zh | en",
    )


class ChatResponse(BaseModel):
    domain: str
    resolved_domain: str
    domain_candidates: Optional[List[Dict[str, Any]]] = None
    profile: str
    mode: str
    resolved_profile: Optional[str] = None
    user_id: str
    thread_id: str
    final_response: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    routing_plan: Optional[Dict[str, Any]] = None
    knowledge_matches: Optional[List[Dict[str, Any]]] = None
    stage_summary: Optional[str] = None
    last_stage_summary: Optional[str] = None
    locale: str = "zh"


class JobSubmitRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    domain: Optional[str] = Field(default=None, max_length=64)
    profile: str = Field(default="auto", description="执行 Profile，默认 auto")
    mode: str = Field(default="fixed_graph")
    transport: str = Field(default="local")
    user_id: Optional[str] = Field(default="default", max_length=128)
    thread_id: Optional[str] = Field(default=None, max_length=128)
    locale: str = Field(default="zh", description="话术 locale：zh | en")


class JobSubmitResponse(BaseModel):
    job_id: str
    domain: str
    mode: str
    user_id: str
    thread_id: str
    status: str


class DynamicAgentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(default="", max_length=512)
    source: str = Field(default="metadata", description="metadata | a2a")
    scope: str = Field(default="domain", description="domain | shared（跨域共享）")
    skills: Optional[List[Dict[str, Any]]] = Field(default=None, description="knowledge_routing 用 skill 元数据")
    a2a_url: str = Field(default="", max_length=2048)
    a2a_node_name: str = Field(default="", max_length=64)
    registry_agent: Optional[str] = Field(default=None, max_length=64)
    alias_of: Optional[str] = Field(
        default=None,
        max_length=64,
        description="引用静态 Agent 能力（如 FAQAgent），与 registry_agent 类似但用于 metadata shared 别名",
    )


class DynamicAgentResponse(BaseModel):
    domain: str
    agent: Dict[str, Any]
    invalidated_runtimes: int
    registry_event: Optional[Dict[str, Any]] = None


class KnowledgeDocumentPayload(BaseModel):
    id: str = Field(..., min_length=1, max_length=128)
    agent: str = Field(..., min_length=1, max_length=64)
    text: str = Field(..., min_length=1, max_length=16000)
    tags: Optional[List[str]] = Field(default=None)


class KnowledgeUpsertRequest(BaseModel):
    documents: List[KnowledgeDocumentPayload] = Field(..., min_length=1)
    replace: bool = Field(
        default=False,
        description="true=全量替换；false=按 id upsert",
    )
    embedding_backend: str = Field(default="hashing", description="hashing | embedding")
    user_id: Optional[str] = Field(
        default=None,
        description="租户 KB overlay scope；省略=shared domain KB",
    )


class KnowledgeListResponse(BaseModel):
    domain: str
    tenant_id: str = "default"
    storage: str
    embedding_backend: str
    document_count: int
    documents: List[Dict[str, Any]]


class KnowledgeUpsertResponse(BaseModel):
    domain: str
    tenant_id: str = "default"
    storage: str = "chroma"
    document_count: int
    embedding_backend: str
    invalidated_runtimes: int


def _ensure_registered_domain(domain: str) -> str:
    resolved = _resolve_domain(domain)
    known = {item["name"] for item in list_domains()}
    if resolved not in known:
        raise HTTPException(status_code=404, detail=f"domain '{resolved}' 未注册")
    return resolved


def _invalidate_domain_runtimes(domain: str, *, scope: str = "domain") -> int:
    pool = get_tenant_pool()
    if scope == "shared":
        total = 0
        for item in list_domains():
            total += pool.invalidate(item["name"])
        return total
    return pool.invalidate(domain)


def _resolve_domain(domain: Optional[str]) -> str:
    resolved = (domain or DEFAULT_DOMAIN or "").strip()
    if not resolved:
        raise HTTPException(
            status_code=400,
            detail=(
                "缺少 domain。可省略 domain 由平台自动推断，"
                "或设置 DEFAULT_DOMAIN，或显式指定如 customer_service、travel。"
            ),
        )
    return resolved


async def _resolve_chat_domain(
    query: str,
    domain: Optional[str],
    *,
    locale: str = "zh",
) -> tuple[str, Optional[List[Dict[str, Any]]]]:
    from agent_framework.router.platform_domain_router import resolve_request_domain

    explicit = (domain or "").strip()
    if explicit or (DEFAULT_DOMAIN or "").strip():
        resolved = _resolve_domain(domain)
        _ensure_registered_domain(resolved)
        return resolved, None
    try:
        selected, candidates = await resolve_request_domain(query, None, locale=locale)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = None
    if candidates:
        payload = [{"name": c.name, "score": c.score} for c in candidates]
    return selected, payload


async def _get_orchestrator(
    *,
    domain: str,
    user_id: Optional[str] = None,
    profile: str = "workflow",
    mode: str = "fixed_graph",
    transport: str = "local",
    locale: str = "zh",
):
    pool = get_tenant_pool()
    resolved_profile = (profile or "").strip()
    if not resolved_profile or resolved_profile == "workflow" and mode == "supervisor":
        if not profile or profile == "workflow":
            resolved_profile = "adaptive" if mode == "supervisor" else "workflow"
    return await pool.get(
        user_id,
        domain=domain,
        profile=resolved_profile,
        mode=mode,
        transport=transport,
        locale=locale,
    )


def _resolve_api_profile(body_profile: str, body_mode: str) -> str:
    profile = (body_profile or "").strip()
    if profile and profile != "workflow":
        return profile
    mode = (body_mode or "fixed_graph").strip() or "fixed_graph"
    if profile == "workflow" and mode == "supervisor":
        return "adaptive"
    if not profile:
        return "adaptive" if mode == "supervisor" else "workflow"
    return profile


async def _prepare_chat(body: ChatRequest) -> Dict[str, Any]:
    """解析 chat / chat/stream 共享上下文。"""
    try:
        locale = normalize_locale(body.locale)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    domain, domain_candidates = await _resolve_chat_domain(body.query, body.domain, locale=locale)
    profile = _resolve_api_profile(body.profile, body.mode)
    mode = (body.mode or "fixed_graph").strip() or "fixed_graph"
    transport = (body.transport or "local").strip() or "local"
    if profile != "auto" and profile != "adaptive":
        transport = "local"
    if profile == "workflow":
        transport = "local"
    metrics_mode = mode if profile not in ("auto",) else "router"
    metrics_transport = transport if profile in ("auto", "adaptive") else "local"
    user_id = (body.user_id or "default").strip() or "default"
    try:
        runtime = await _get_orchestrator(
            domain=domain,
            user_id=user_id,
            profile=profile,
            mode=mode,
            transport=transport,
            locale=locale,
        )
    except KeyError as exc:
        record_chat(domain, "bad_domain", mode=metrics_mode, transport=metrics_transport)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        record_chat(domain, "bad_request", mode=metrics_mode, transport=metrics_transport)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    thread_id = body.thread_id or f"api-{uuid.uuid4().hex[:12]}"
    request_kwargs: Dict[str, Any] = {"timeout_sec": body.timeout_sec}
    if profile == "auto" and body.conversation_history:
        request_kwargs["conversation_history"] = body.conversation_history
    return {
        "domain": domain,
        "domain_candidates": domain_candidates,
        "profile": profile,
        "mode": mode,
        "locale": locale,
        "user_id": user_id,
        "thread_id": thread_id,
        "runtime": runtime,
        "request_kwargs": request_kwargs,
        "metrics_mode": metrics_mode,
        "metrics_transport": metrics_transport,
    }


def _build_chat_response(body: ChatRequest, ctx: Dict[str, Any], result: Dict[str, Any]) -> ChatResponse:
    enriched = enrich_routing_observability(dict(result), domain=ctx["domain"])
    return ChatResponse(
        domain=ctx["domain"],
        resolved_domain=enriched["resolved_domain"],
        domain_candidates=ctx["domain_candidates"],
        profile=ctx["profile"],
        mode=enriched.get("orchestration_mode") or ctx["mode"],
        resolved_profile=enriched.get("resolved_profile"),
        user_id=ctx["user_id"],
        thread_id=ctx["thread_id"],
        final_response=enriched.get("final_response") or "",
        trace_id=enriched.get("trace_id"),
        span_id=enriched.get("span_id"),
        routing_plan=enriched.get("routing_plan"),
        knowledge_matches=enriched.get("knowledge_matches"),
        stage_summary=enriched.get("stage_summary") or None,
        last_stage_summary=enriched.get("last_stage_summary") or None,
        locale=ctx["locale"],
    )


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "product": "agent-platform", "version": "0.20.0"}


@app.get("/ready")
async def ready() -> Dict[str, Any]:
    registered = list_domains()
    if not registered:
        raise HTTPException(status_code=503, detail="no domains registered")
    payload: Dict[str, Any] = {
        "status": "ready",
        "product": "agent-platform",
        "domains": [d["name"] for d in registered],
    }
    warmup = (READY_DOMAIN or "").strip()
    if warmup:
        try:
            await _get_orchestrator(domain=warmup, user_id="default")
            payload["warmed_domain"] = warmup
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return payload


@app.get("/metrics")
async def metrics() -> Response:
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


@app.get("/v1/domains")
async def domains(_: None = Depends(require_api_key)) -> Dict[str, Any]:
    return {
        "recommended_profile": "auto",
        "domains": list_domains(),
    }


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, _: None = Depends(require_api_key)) -> ChatResponse:
    ctx = await _prepare_chat(body)
    domain = ctx["domain"]
    try:
        result = await ctx["runtime"].process_request(
            body.query,
            thread_id=ctx["thread_id"],
            **ctx["request_kwargs"],
        )
    except RequestSlotTimeoutError as exc:
        record_chat(domain, "429", mode=ctx["metrics_mode"], transport=ctx["metrics_transport"])
        raise HTTPException(status_code=429, detail="too many concurrent requests") from exc
    except asyncio.TimeoutError as exc:
        record_chat(domain, "504", mode=ctx["metrics_mode"], transport=ctx["metrics_transport"])
        raise HTTPException(status_code=504, detail="request timeout") from exc
    except Exception as exc:
        record_chat(domain, "500", mode=ctx["metrics_mode"], transport=ctx["metrics_transport"])
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    record_chat(domain, "200", mode=ctx["metrics_mode"], transport=ctx["metrics_transport"])
    return _build_chat_response(body, ctx, result)


@app.post("/v1/chat/stream")
async def chat_stream(body: ChatRequest, _: None = Depends(require_api_key)) -> StreamingResponse:
    ctx = await _prepare_chat(body)
    domain = ctx["domain"]

    async def event_generator():
        seq = 0
        try:
            async for event in ctx["runtime"].iter_request_stream(
                body.query,
                thread_id=ctx["thread_id"],
                **ctx["request_kwargs"],
            ):
                seq += 1
                yield format_sse(event, event_id=str(seq))
            record_chat(domain, "200", mode=ctx["metrics_mode"], transport=ctx["metrics_transport"])
        except RequestSlotTimeoutError:
            record_chat(domain, "429", mode=ctx["metrics_mode"], transport=ctx["metrics_transport"])
            yield format_sse(error_event("too many concurrent requests", code="429"))
        except asyncio.TimeoutError:
            record_chat(domain, "504", mode=ctx["metrics_mode"], transport=ctx["metrics_transport"])
            yield format_sse(error_event("request timeout", code="504"))
        except Exception as exc:
            record_chat(domain, "500", mode=ctx["metrics_mode"], transport=ctx["metrics_transport"])
            yield format_sse(error_event(str(exc), code="500"))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/v1/agents")
async def list_all_agents(
    domain: Optional[str] = None,
    scope: Optional[str] = None,
    source: Optional[str] = None,
    federated: bool = False,
    health: bool = False,
    _: None = Depends(require_api_key),
) -> Dict[str, Any]:
    if federated:
        from agent_framework.domain.registry_federation import list_federated_agents

        return await list_federated_agents(
            domain=domain,
            scope=scope,
            source=source,
            include_health=health,
        )

    from agent_framework.domain.agent_catalog import list_platform_agent_entries

    agents = list_platform_agent_entries(
        domain=domain,
        scope=scope,
        source=source,
    )
    return {
        "agents": agents,
        "count": len(agents),
        "filters": {
            "domain": domain,
            "scope": scope,
            "source": source,
            "federated": False,
            "health": False,
        },
    }


@app.get("/v1/registry/federation")
async def get_registry_federation_status(
    health: bool = True,
    _: None = Depends(require_api_key),
) -> Dict[str, Any]:
    from agent_framework.domain.registry_federation import (
        list_federation_clusters,
        parse_federation_urls,
    )

    urls = parse_federation_urls()
    clusters = await list_federation_clusters(include_health=health, federation_urls=urls)
    return {
        "federation_urls": urls,
        "clusters": clusters,
        "cluster_count": len(clusters),
    }


@app.get("/v1/domains/{domain}/agents")
async def list_domain_agents(domain: str, _: None = Depends(require_api_key)) -> Dict[str, Any]:
    from agent_framework.domain.agent_catalog import list_dynamic_agent_records

    resolved = _ensure_registered_domain(domain)
    get_domain_plugin(resolved)
    static = get_domain_plugin(resolved).create_registry().list_agent_metadata()
    dynamic = list_dynamic_agent_records(resolved)
    return {"domain": resolved, "static_agents": static, "dynamic_agents": dynamic}


@app.post("/v1/domains/{domain}/agents", response_model=DynamicAgentResponse)
async def register_domain_agent(
    domain: str,
    body: DynamicAgentRequest,
    _: None = Depends(require_api_key),
) -> DynamicAgentResponse:
    resolved = _ensure_registered_domain(domain)
    get_domain_plugin(resolved)
    source = (body.source or "metadata").strip().lower()
    if source not in ("metadata", "a2a"):
        raise HTTPException(status_code=400, detail="source 可选: metadata, a2a")
    scope = (body.scope or "domain").strip().lower()
    if scope not in ("domain", "shared"):
        raise HTTPException(status_code=400, detail="scope 可选: domain, shared")
    try:
        record = get_dynamic_agent_store().register(
            resolved,
            DynamicAgentRecord(
                name=body.name.strip(),
                description=(body.description or "").strip(),
                skills=list(body.skills or []),
                source=source,
                scope=scope,
                a2a_url=(body.a2a_url or "").strip(),
                a2a_node_name=(body.a2a_node_name or "").strip(),
                registry_agent=(body.registry_agent or "").strip() or None,
                alias_of=(body.alias_of or "").strip() or None,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    invalidated = _invalidate_domain_runtimes(resolved, scope=scope)
    from agent_framework.domain.registry_events import notify_registry_updated

    registry_event = await notify_registry_updated(
        domain=resolved,
        action="register",
        agent_name=record.name,
        scope=record.scope,
        source="dynamic",
    )
    return DynamicAgentResponse(
        domain=resolved,
        agent=record.to_dict(),
        invalidated_runtimes=invalidated,
        registry_event=registry_event,
    )


@app.delete("/v1/domains/{domain}/agents/{agent_name}", response_model=DynamicAgentResponse)
async def unregister_domain_agent(
    domain: str,
    agent_name: str,
    _: None = Depends(require_api_key),
) -> DynamicAgentResponse:
    resolved = _ensure_registered_domain(domain)
    get_domain_plugin(resolved)
    removed = get_dynamic_agent_store().unregister(resolved, agent_name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"dynamic agent '{agent_name}' 不存在")
    invalidated = _invalidate_domain_runtimes(resolved)
    from agent_framework.domain.registry_events import notify_registry_updated

    registry_event = await notify_registry_updated(
        domain=resolved,
        action="unregister",
        agent_name=agent_name,
        scope="domain",
        source="dynamic",
    )
    return DynamicAgentResponse(
        domain=resolved,
        agent={"name": agent_name},
        invalidated_runtimes=invalidated,
        registry_event=registry_event,
    )


@app.get("/v1/domains/{domain}/knowledge", response_model=KnowledgeListResponse)
async def list_domain_knowledge(
    domain: str,
    embedding_backend: str = "hashing",
    user_id: Optional[str] = None,
    _: None = Depends(require_api_key),
) -> KnowledgeListResponse:
    from agent_framework.router.kb.repository import list_domain_knowledge as repo_list
    from agent_framework.router.kb.tenant import normalize_kb_tenant_id

    resolved = _ensure_registered_domain(domain)
    get_domain_plugin(resolved)
    tid = normalize_kb_tenant_id(user_id)
    payload = repo_list(resolved, embedding_backend=embedding_backend.strip() or "hashing", tenant_id=tid)
    return KnowledgeListResponse(**payload)


@app.post("/v1/domains/{domain}/knowledge", response_model=KnowledgeUpsertResponse)
async def upsert_domain_knowledge(
    domain: str,
    body: KnowledgeUpsertRequest,
    user_id: Optional[str] = None,
    _: None = Depends(require_api_key),
) -> KnowledgeUpsertResponse:
    from agent_framework.router.kb.models import KnowledgeDocument
    from agent_framework.router.kb.repository import upsert_domain_knowledge as repo_upsert
    from agent_framework.router.kb.tenant import normalize_kb_tenant_id

    resolved = _ensure_registered_domain(domain)
    get_domain_plugin(resolved)
    backend = (body.embedding_backend or "hashing").strip() or "hashing"
    tid = normalize_kb_tenant_id(user_id or body.user_id)
    try:
        documents = [
            KnowledgeDocument.from_dict(
                {
                    "id": item.id,
                    "agent": item.agent,
                    "text": item.text,
                    "tags": item.tags or [],
                }
            )
            for item in body.documents
        ]
        count = repo_upsert(
            resolved,
            documents,
            embedding_backend=backend,
            replace=body.replace,
            tenant_id=tid,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    invalidated = _invalidate_domain_runtimes(resolved)
    return KnowledgeUpsertResponse(
        domain=resolved,
        tenant_id=tid,
        document_count=count,
        embedding_backend=backend,
        invalidated_runtimes=invalidated,
    )


@app.post("/v1/jobs", response_model=JobSubmitResponse)
async def submit_job(body: JobSubmitRequest, _: None = Depends(require_api_key)) -> JobSubmitResponse:
    locale = normalize_locale(body.locale)
    domain, _ = await _resolve_chat_domain(body.query, body.domain, locale=locale)
    profile = _resolve_api_profile(body.profile, body.mode)
    mode = (body.mode or "fixed_graph").strip() or "fixed_graph"
    transport = (body.transport or "local").strip() or "local"
    if profile == "workflow":
        transport = "local"
    user_id = (body.user_id or "default").strip() or "default"
    thread_id = body.thread_id or f"job-{uuid.uuid4().hex[:12]}"
    record = _job_store.create_job(
        user_id=user_id,
        query=body.query,
        thread_id=thread_id,
        domain=domain,
        mode=mode,
        transport=transport,
        locale=locale,
    )
    record_job(domain, mode=mode, transport=transport)
    return JobSubmitResponse(
        job_id=record.job_id,
        domain=domain,
        mode=mode,
        user_id=record.user_id,
        thread_id=record.thread_id,
        status=record.status.value,
    )


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str, _: None = Depends(require_api_key)) -> Dict[str, Any]:
    record = _job_store.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="job not found")
    return record.to_dict()
