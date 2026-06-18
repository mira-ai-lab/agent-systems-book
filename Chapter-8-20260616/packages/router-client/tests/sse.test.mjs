import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import test from "node:test";

const root = dirname(fileURLToPath(import.meta.url));
const { parseSseBlock } = await import(pathToFileURL(join(root, "../dist/sse.js")));

test("parseSseBlock parses router final event", () => {
  const block = readFileSync(join(root, "fixtures/final.sse.txt"), "utf8");
  const event = parseSseBlock(block);
  assert.equal(event?.type, "final");
  assert.equal(event?.data?.final_response, "ok");
});

test("parseSseBlock parses router stage event", () => {
  const block =
    'event: router.classification\ndata: {"type":"router.classification","stage":"classification","data":{"candidates":[]}}\n';
  const event = parseSseBlock(block);
  assert.equal(event?.type, "router.classification");
});
