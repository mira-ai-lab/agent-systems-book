import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";

const root = dirname(fileURLToPath(import.meta.url));
const { createRouterClient } = await import(pathToFileURL(join(root, "../dist/index.js")));

const baseUrl = process.env.ROUTER_CLIENT_BASE_URL?.replace(/\/+$/, "");
const apiKey = process.env.ROUTER_CLIENT_API_KEY;

function requireBaseUrl(t) {
  if (!baseUrl) {
    t.skip("ROUTER_CLIENT_BASE_URL not set");
    return false;
  }
  return true;
}

test("route() sync chat against live API", async (t) => {
  if (!requireBaseUrl(t)) return;

  const client = createRouterClient({ baseUrl, apiKey });
  const result = await client.route("北京明天天气怎么样？", {
    domain: "travel",
    userId: "sdk-integration",
    profile: "auto",
  });

  assert.equal(result.user_id, "sdk-integration");
  assert.equal(result.domain, "travel");
  assert.match(result.final_response, /晴/);
  assert.ok(result.trace_id);
});

test("routeStream() receives router stages and final SSE event", async (t) => {
  if (!requireBaseUrl(t)) return;

  const client = createRouterClient({ baseUrl, apiKey });
  const types = [];

  for await (const event of client.routeStream("北京明天天气怎么样？", {
    domain: "travel",
    profile: "auto",
  })) {
    types.push(event.type);
    if (event.type === "final") {
      assert.match(String(event.data?.final_response ?? ""), /晴/);
    }
  }

  assert.ok(types.includes("final"), `expected final event, got: ${types.join(", ")}`);
});

test("submitJob() and getJob() against live API", async (t) => {
  if (!requireBaseUrl(t)) return;

  const client = createRouterClient({ baseUrl, apiKey });
  const submitted = await client.submitJob("杭州三日游", {
    domain: "travel",
    userId: "sdk-jobs",
    profile: "auto",
  });

  assert.ok(submitted.job_id);
  assert.equal(submitted.status, "pending");
  assert.equal(submitted.user_id, "sdk-jobs");
  assert.equal(submitted.domain, "travel");

  const detail = await client.getJob(submitted.job_id);
  assert.equal(detail.job_id, submitted.job_id);
  assert.equal(detail.status, "pending");
  assert.equal(detail.user_id, "sdk-jobs");
});
