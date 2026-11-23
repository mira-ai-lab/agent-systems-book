export type ExecutionProfile = "auto" | "workflow" | "adaptive" | "hybrid";

export interface FetchPolicy {
  /** Client-side HTTP timeout in milliseconds. Default 120_000. */
  timeoutMs?: number;
  /** Extra retry attempts after the first request. Default 2. */
  retries?: number;
  /** Base delay between retries; doubles each attempt. Default 500ms. */
  retryDelayMs?: number;
}

export interface RouterClientConfig {
  baseUrl: string;
  apiKey?: string;
  fetch?: typeof fetch;
  fetchPolicy?: FetchPolicy;
}

export interface RouteOptions {
  domain?: string;
  profile?: ExecutionProfile;
  mode?: string;
  transport?: string;
  userId?: string;
  threadId?: string;
  locale?: "zh" | "en";
  conversationHistory?: string;
  /** Server-side orchestration timeout in seconds. */
  timeoutSec?: number;
  /** Per-request client HTTP policy override. */
  fetchPolicy?: FetchPolicy;
}

export interface ChatRequestBody {
  query: string;
  domain?: string;
  profile?: string;
  mode?: string;
  transport?: string;
  user_id?: string;
  thread_id?: string;
  locale?: string;
  conversation_history?: string;
  timeout_sec?: number;
}

export interface ChatResponse {
  domain: string;
  resolved_domain: string;
  domain_candidates?: Array<Record<string, unknown>> | null;
  profile: string;
  mode: string;
  resolved_profile?: string | null;
  user_id: string;
  thread_id: string;
  final_response: string;
  trace_id?: string | null;
  span_id?: string | null;
  routing_plan?: Record<string, unknown> | null;
  knowledge_matches?: Array<Record<string, unknown>> | null;
  stage_summary?: string | null;
  last_stage_summary?: string | null;
  locale: string;
}

export interface StreamEvent {
  type: string;
  stage?: string;
  data?: Record<string, unknown>;
}

export interface JobSubmitOptions {
  domain?: string;
  profile?: ExecutionProfile;
  mode?: string;
  transport?: string;
  userId?: string;
  threadId?: string;
  locale?: "zh" | "en";
  fetchPolicy?: FetchPolicy;
}

export interface GetJobOptions {
  apiKey?: string;
  fetchPolicy?: FetchPolicy;
}

export interface JobSubmitRequestBody {
  query: string;
  domain?: string;
  profile?: string;
  mode?: string;
  transport?: string;
  user_id?: string;
  thread_id?: string;
  locale?: string;
}

export interface JobSubmitResponse {
  job_id: string;
  domain: string;
  mode: string;
  user_id: string;
  thread_id: string;
  status: string;
}

export type JobStatus = "pending" | "running" | "succeeded" | "failed";

export interface JobRecord {
  job_id: string;
  domain: string;
  mode: string;
  transport: string;
  locale: string;
  user_id: string;
  thread_id: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  trace_id?: string;
  error?: string;
  result?: Record<string, unknown>;
  result_raw?: string;
}

export class RouterClientError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(`Router API ${status}: ${detail}`);
    this.name = "RouterClientError";
    this.status = status;
    this.detail = detail;
  }
}

export class RouterClientTimeoutError extends RouterClientError {
  constructor(detail = "request timeout") {
    super(408, detail);
    this.name = "RouterClientTimeoutError";
  }
}

export class RouterClientNetworkError extends RouterClientError {
  readonly cause?: unknown;

  constructor(detail: string, cause?: unknown) {
    super(0, detail);
    this.name = "RouterClientNetworkError";
    this.message = `Router network error: ${detail}`;
    this.cause = cause;
  }
}
