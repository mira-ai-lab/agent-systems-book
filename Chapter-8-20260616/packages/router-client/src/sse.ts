import type { StreamEvent } from "./types.js";

export function parseSseBlock(block: string): StreamEvent | null {
  const lines = block.split(/\r?\n/);
  let eventType = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("event:")) {
      eventType = line.slice(6).trim() || "message";
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (dataLines.length === 0) {
    return null;
  }

  const raw = dataLines.join("\n");
  try {
    const payload = JSON.parse(raw) as StreamEvent;
    if (payload && typeof payload === "object" && payload.type) {
      return payload;
    }
    return { type: eventType, data: payload as unknown as Record<string, unknown> };
  } catch {
    return { type: eventType, data: { raw } };
  }
}

export async function* iterateSseStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<StreamEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");

      let splitAt = buffer.indexOf("\n\n");
      while (splitAt >= 0) {
        const block = buffer.slice(0, splitAt);
        buffer = buffer.slice(splitAt + 2);
        const event = parseSseBlock(block);
        if (event) {
          yield event;
        }
        splitAt = buffer.indexOf("\n\n");
      }
    }

    if (buffer.trim()) {
      const event = parseSseBlock(buffer);
      if (event) {
        yield event;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
