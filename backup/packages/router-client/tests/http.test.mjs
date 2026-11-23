import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";

const root = dirname(fileURLToPath(import.meta.url));
const http = await import(pathToFileURL(join(root, "../dist/http.js")));
const types = await import(pathToFileURL(join(root, "../dist/types.js")));

const {
  DEFAULT_FETCH_POLICY,
  fetchWithPolicy,
  isRetryableStatus,
  isRouterClientError,
  isRouterClientNetworkError,
  isRouterClientTimeoutError,
  mergeFetchPolicy,
} = http;

const {
  RouterClientError,
  RouterClientNetworkError,
  RouterClientTimeoutError,
} = types;

test("mergeFetchPolicy applies overrides", () => {
  const merged = mergeFetchPolicy({ timeoutMs: 1000 }, { retries: 1 });
  assert.equal(merged.timeoutMs, 1000);
  assert.equal(merged.retries, 1);
  assert.equal(merged.retryDelayMs, DEFAULT_FETCH_POLICY.retryDelayMs);
});

test("isRetryableStatus matches transient HTTP codes", () => {
  assert.equal(isRetryableStatus(429), true);
  assert.equal(isRetryableStatus(503), true);
  assert.equal(isRetryableStatus(400), false);
});

test("fetchWithPolicy retries retryable HTTP errors", async () => {
  let attempts = 0;
  const fetchImpl = async () => {
    attempts += 1;
    if (attempts < 3) {
      return new Response(JSON.stringify({ detail: "busy" }), { status: 503 });
    }
    return new Response(JSON.stringify({ ok: true }), { status: 200 });
  };

  const response = await fetchWithPolicy(fetchImpl, "http://test/v1/chat", {}, {
    timeoutMs: 1000,
    retries: 2,
    retryDelayMs: 1,
  });

  assert.equal(response.status, 200);
  assert.equal(attempts, 3);
});

test("fetchWithPolicy does not retry client errors", async () => {
  let attempts = 0;
  const fetchImpl = async () => {
    attempts += 1;
    return new Response(JSON.stringify({ detail: "bad request" }), { status: 400 });
  };

  await assert.rejects(
    () =>
      fetchWithPolicy(fetchImpl, "http://test/v1/chat", {}, {
        timeoutMs: 1000,
        retries: 2,
        retryDelayMs: 1,
      }),
    (error) => {
      assert.ok(error instanceof RouterClientError);
      assert.equal(error.status, 400);
      return true;
    },
  );
  assert.equal(attempts, 1);
});

test("fetchWithPolicy throws RouterClientTimeoutError", async () => {
  const fetchImpl = async () => {
    throw new DOMException("The operation timed out.", "TimeoutError");
  };

  await assert.rejects(
    () =>
      fetchWithPolicy(fetchImpl, "http://test/v1/chat", {}, {
        timeoutMs: 1000,
        retries: 0,
        retryDelayMs: 1,
      }),
    (error) => isRouterClientTimeoutError(error),
  );
});

test("fetchWithPolicy throws RouterClientNetworkError after retries", async () => {
  let attempts = 0;
  const fetchImpl = async () => {
    attempts += 1;
    throw new TypeError("fetch failed");
  };

  await assert.rejects(
    () =>
      fetchWithPolicy(fetchImpl, "http://test/v1/chat", {}, {
        timeoutMs: 1000,
        retries: 1,
        retryDelayMs: 1,
      }),
    (error) => isRouterClientNetworkError(error),
  );
  assert.equal(attempts, 2);
});

test("error type guards", () => {
  const httpError = new RouterClientError(404, "not found");
  const timeoutError = new RouterClientTimeoutError();
  const networkError = new RouterClientNetworkError("offline");

  assert.equal(isRouterClientError(httpError), true);
  assert.equal(isRouterClientTimeoutError(timeoutError), true);
  assert.equal(isRouterClientNetworkError(networkError), true);
  assert.equal(isRouterClientNetworkError(httpError), false);
});
