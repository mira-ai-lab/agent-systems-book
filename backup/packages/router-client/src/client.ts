import { fetchWithPolicy, mergeFetchPolicy } from "./http.js";
import { iterateSseStream } from "./sse.js";
import type {
  ChatRequestBody,
  ChatResponse,
  FetchPolicy,
  GetJobOptions,
  JobRecord,
  JobSubmitOptions,
  JobSubmitRequestBody,
  JobSubmitResponse,
  RouteOptions,
  RouterClientConfig,
  StreamEvent,
} from "./types.js";
import { RouterClientError } from "./types.js";

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
}

function buildHeaders(apiKey?: string): HeadersInit {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
  };
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  return headers;
}

export function buildChatRequestBody(
  query: string,
  options: RouteOptions = {},
): ChatRequestBody {
  const body: ChatRequestBody = {
    query,
    profile: options.profile ?? "auto",
    locale: options.locale ?? "zh",
  };
  if (options.domain) body.domain = options.domain;
  if (options.mode) body.mode = options.mode;
  if (options.transport) body.transport = options.transport;
  if (options.userId) body.user_id = options.userId;
  if (options.threadId) body.thread_id = options.threadId;
  if (options.conversationHistory) {
    body.conversation_history = options.conversationHistory;
  }
  if (options.timeoutSec !== undefined) {
    body.timeout_sec = options.timeoutSec;
  }
  return body;
}

export function buildJobRequestBody(
  query: string,
  options: JobSubmitOptions = {},
): JobSubmitRequestBody {
  const body: JobSubmitRequestBody = {
    query,
    profile: options.profile ?? "auto",
    locale: options.locale ?? "zh",
  };
  if (options.domain) body.domain = options.domain;
  if (options.mode) body.mode = options.mode;
  if (options.transport) body.transport = options.transport;
  if (options.userId) body.user_id = options.userId;
  if (options.threadId) body.thread_id = options.threadId;
  return body;
}

export class RouterClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly fetchImpl: typeof fetch;
  private readonly fetchPolicy: FetchPolicy;

  constructor(config: RouterClientConfig) {
    this.baseUrl = normalizeBaseUrl(config.baseUrl);
    this.apiKey = config.apiKey;
    this.fetchImpl = config.fetch ?? fetch;
    this.fetchPolicy = config.fetchPolicy ?? {};
  }

  private resolvePolicy(override?: FetchPolicy) {
    return mergeFetchPolicy(this.fetchPolicy, override);
  }

  private request(
    url: string,
    init: RequestInit,
    policy?: FetchPolicy,
  ): Promise<Response> {
    return fetchWithPolicy(this.fetchImpl, url, init, this.resolvePolicy(policy));
  }

  async route(query: string, options: RouteOptions = {}): Promise<ChatResponse> {
    const { fetchPolicy, ...routeOptions } = options;
    const response = await this.request(
      `${this.baseUrl}/v1/chat`,
      {
        method: "POST",
        headers: buildHeaders(this.apiKey),
        body: JSON.stringify(buildChatRequestBody(query, routeOptions)),
      },
      fetchPolicy,
    );
    return (await response.json()) as ChatResponse;
  }

  async *routeStream(
    query: string,
    options: RouteOptions = {},
  ): AsyncGenerator<StreamEvent> {
    const { fetchPolicy, ...routeOptions } = options;
    const response = await this.request(
      `${this.baseUrl}/v1/chat/stream`,
      {
        method: "POST",
        headers: {
          ...buildHeaders(this.apiKey),
          Accept: "text/event-stream",
        },
        body: JSON.stringify(buildChatRequestBody(query, routeOptions)),
      },
      fetchPolicy,
    );
    if (!response.body) {
      throw new RouterClientError(500, "empty stream body");
    }
    yield* iterateSseStream(response.body);
  }

  async submitJob(
    query: string,
    options: JobSubmitOptions = {},
  ): Promise<JobSubmitResponse> {
    const { fetchPolicy, ...jobOptions } = options;
    const response = await this.request(
      `${this.baseUrl}/v1/jobs`,
      {
        method: "POST",
        headers: buildHeaders(this.apiKey),
        body: JSON.stringify(buildJobRequestBody(query, jobOptions)),
      },
      fetchPolicy,
    );
    return (await response.json()) as JobSubmitResponse;
  }

  async getJob(jobId: string, options: GetJobOptions = {}): Promise<JobRecord> {
    const response = await this.request(
      `${this.baseUrl}/v1/jobs/${encodeURIComponent(jobId)}`,
      {
        method: "GET",
        headers: buildHeaders(options.apiKey ?? this.apiKey),
      },
      options.fetchPolicy,
    );
    return (await response.json()) as JobRecord;
  }
}

export function createRouterClient(config: RouterClientConfig): RouterClient {
  return new RouterClient(config);
}

export async function route(
  baseUrl: string,
  query: string,
  options: RouteOptions & { apiKey?: string } = {},
): Promise<ChatResponse> {
  const { apiKey, ...routeOptions } = options;
  const client = createRouterClient({ baseUrl, apiKey });
  return client.route(query, routeOptions);
}

export async function submitJob(
  baseUrl: string,
  query: string,
  options: JobSubmitOptions & { apiKey?: string } = {},
): Promise<JobSubmitResponse> {
  const { apiKey, ...jobOptions } = options;
  const client = createRouterClient({ baseUrl, apiKey });
  return client.submitJob(query, jobOptions);
}

export async function getJob(
  baseUrl: string,
  jobId: string,
  options: GetJobOptions = {},
): Promise<JobRecord> {
  const client = createRouterClient({ baseUrl, apiKey: options.apiKey });
  return client.getJob(jobId, options);
}
