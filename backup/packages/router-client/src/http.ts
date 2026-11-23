import type { FetchPolicy } from "./types.js";
import {
  RouterClientError,
  RouterClientNetworkError,
  RouterClientTimeoutError,
} from "./types.js";

export const DEFAULT_FETCH_POLICY: Required<FetchPolicy> = {
  timeoutMs: 120_000,
  retries: 2,
  retryDelayMs: 500,
};

export function mergeFetchPolicy(
  base?: FetchPolicy,
  override?: FetchPolicy,
): Required<FetchPolicy> {
  return {
    timeoutMs: override?.timeoutMs ?? base?.timeoutMs ?? DEFAULT_FETCH_POLICY.timeoutMs,
    retries: override?.retries ?? base?.retries ?? DEFAULT_FETCH_POLICY.retries,
    retryDelayMs:
      override?.retryDelayMs ?? base?.retryDelayMs ?? DEFAULT_FETCH_POLICY.retryDelayMs,
  };
}

export function isRetryableStatus(status: number): boolean {
  return status === 429 || status === 502 || status === 503 || status === 504;
}

export function isRouterClientError(error: unknown): error is RouterClientError {
  return error instanceof RouterClientError;
}

export function isRouterClientTimeoutError(
  error: unknown,
): error is RouterClientTimeoutError {
  return error instanceof RouterClientTimeoutError;
}

export function isRouterClientNetworkError(
  error: unknown,
): error is RouterClientNetworkError {
  return error instanceof RouterClientNetworkError;
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    return JSON.stringify(payload.detail ?? payload);
  } catch {
    return await response.text();
  }
}

function toClientError(error: unknown): RouterClientError {
  if (error instanceof RouterClientError) {
    return error;
  }
  if (error instanceof DOMException && error.name === "TimeoutError") {
    return new RouterClientTimeoutError();
  }
  if (error instanceof Error && error.name === "AbortError") {
    return new RouterClientTimeoutError();
  }
  const message = error instanceof Error ? error.message : String(error);
  return new RouterClientNetworkError(message, error);
}

export async function sleep(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

export async function fetchWithPolicy(
  fetchImpl: typeof fetch,
  url: string,
  init: RequestInit,
  policy: Required<FetchPolicy>,
): Promise<Response> {
  let lastError: RouterClientError = new RouterClientNetworkError("request failed");

  for (let attempt = 0; attempt <= policy.retries; attempt += 1) {
    try {
      const response = await fetchImpl(url, {
        ...init,
        signal: init.signal ?? AbortSignal.timeout(policy.timeoutMs),
      });

      if (response.ok) {
        return response;
      }

      if (!isRetryableStatus(response.status) || attempt === policy.retries) {
        throw new RouterClientError(response.status, await readErrorDetail(response));
      }
    } catch (error) {
      lastError = toClientError(error);
      if (lastError instanceof RouterClientError && lastError.status > 0) {
        if (!isRetryableStatus(lastError.status) || attempt === policy.retries) {
          throw lastError;
        }
      } else if (attempt === policy.retries) {
        throw lastError;
      }
    }

    if (attempt < policy.retries) {
      await sleep(policy.retryDelayMs * 2 ** attempt);
    }
  }

  throw lastError;
}
