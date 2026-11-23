import { createRouterClient } from "@agent-platform/router-client";
import type { StreamEvent } from "@agent-platform/router-client";
import "./style.css";

const DEFAULT_BASE =
  (import.meta as ImportMeta & { env?: { VITE_API_BASE_URL?: string } }).env
    ?.VITE_API_BASE_URL ?? "";

interface TimelineEntry {
  id: number;
  type: string;
  stage?: string;
  summary: string;
  raw?: unknown;
}

let timelineSeq = 0;

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function summarizeEvent(event: StreamEvent): string {
  if (event.type === "final") {
    const text = String(event.data?.final_response ?? "");
    return text.length > 120 ? `${text.slice(0, 120)}…` : text;
  }
  if (event.type.startsWith("router.")) {
    const data = event.data ?? {};
    if (Array.isArray(data.events)) {
      return `events: ${(data.events as string[]).join(", ")}`;
    }
    if (Array.isArray(data.candidates)) {
      return `candidates: ${(data.candidates as Array<{ name?: string }>)
        .map((c) => c.name ?? "?")
        .join(", ")}`;
    }
    return JSON.stringify(data).slice(0, 100);
  }
  if (event.type.startsWith("handoff.")) {
    return JSON.stringify(event.data ?? {}).slice(0, 100);
  }
  return JSON.stringify(event.data ?? {}).slice(0, 100);
}

function renderTimeline(container: HTMLElement, entries: TimelineEntry[]): void {
  container.replaceChildren();
  if (!entries.length) {
    container.append(el("p", "muted", "发送消息后，Router SSE 阶段会显示在这里。"));
    return;
  }
  for (const entry of entries) {
    const row = el("article", "timeline-item");
    row.append(el("div", "timeline-type", entry.type));
    if (entry.stage) {
      row.append(el("div", "timeline-stage", entry.stage));
    }
    row.append(el("pre", "timeline-summary", entry.summary));
    container.append(row);
  }
}

function renderResponse(container: HTMLElement, text: string, meta?: Record<string, string>): void {
  container.replaceChildren();
  container.append(el("div", "response-text", text || "（无回复）"));
  if (meta) {
    const metaRow = el("div", "response-meta");
    for (const [key, value] of Object.entries(meta)) {
      const chip = el("span", "chip", `${key}: ${value}`);
      metaRow.append(chip);
    }
    container.append(metaRow);
  }
}

function mount(): void {
  const root = document.querySelector("#app");
  if (!root) return;

  const shell = el("div", "shell");
  shell.append(
    el("header", "header", "Agent Platform Router Demo"),
    el("p", "subtitle", "Vite + @agent-platform/router-client · 同步 / SSE 流式"),
  );

  const form = el("form", "panel");
  const baseInput = el("input", "input") as HTMLInputElement;
  baseInput.type = "url";
  baseInput.placeholder = "API Base URL（留空=当前站点，dev 走 Vite proxy）";
  baseInput.value = DEFAULT_BASE;

  const queryInput = el("textarea", "textarea") as HTMLTextAreaElement;
  queryInput.placeholder = "输入 query，例如：规划杭州三日游";
  queryInput.rows = 3;
  queryInput.value = "帮我查北京明天天气，并推荐一家安静的酒店";

  const domainInput = el("input", "input") as HTMLInputElement;
  domainInput.placeholder = "domain（可选，如 travel）";
  domainInput.value = "travel";

  const apiKeyInput = el("input", "input") as HTMLInputElement;
  apiKeyInput.type = "password";
  apiKeyInput.placeholder = "X-API-Key（可选）";

  const streamToggle = el("label", "toggle") as HTMLLabelElement;
  const streamCheckbox = el("input") as HTMLInputElement;
  streamCheckbox.type = "checkbox";
  streamCheckbox.checked = true;
  streamToggle.append(streamCheckbox, document.createTextNode(" SSE 流式（routeStream）"));

  const submitBtn = el("button", "btn primary", "发送") as HTMLButtonElement;
  submitBtn.type = "submit";

  const status = el("div", "status muted", "就绪");

  form.append(
    el("label", "field-label", "API Base URL"),
    baseInput,
    el("label", "field-label", "Query"),
    queryInput,
    el("label", "field-label", "Domain"),
    domainInput,
    el("label", "field-label", "API Key"),
    apiKeyInput,
    streamToggle,
    submitBtn,
  );

  const layout = el("div", "layout");
  const timelinePanel = el("section", "panel");
  timelinePanel.append(el("h2", "", "Router SSE 时间线"));
  const timeline = el("div", "timeline");
  timelinePanel.append(timeline);

  const responsePanel = el("section", "panel");
  responsePanel.append(el("h2", "", "最终回复"));
  const responseBox = el("div", "response");
  responsePanel.append(responseBox);

  layout.append(timelinePanel, responsePanel);
  shell.append(form, status, layout);
  root.append(shell);

  renderTimeline(timeline, []);
  renderResponse(responseBox, "等待请求…");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    submitBtn.disabled = true;
    status.className = "status";
    status.textContent = "请求中…";
    renderTimeline(timeline, []);
    renderResponse(responseBox, "…");

    const baseUrl = baseInput.value.trim() || window.location.origin;
    const client = createRouterClient({
      baseUrl,
      apiKey: apiKeyInput.value.trim() || undefined,
    });
    const options = {
      domain: domainInput.value.trim() || undefined,
      profile: "auto" as const,
      userId: "demo-web",
    };
    const query = queryInput.value.trim();
    if (!query) {
      status.className = "status error";
      status.textContent = "请输入 query";
      submitBtn.disabled = false;
      return;
    }

    try {
      if (streamCheckbox.checked) {
        const entries: TimelineEntry[] = [];
        let finalText = "";
        const meta: Record<string, string> = {};

        for await (const streamEvent of client.routeStream(query, options)) {
          entries.push({
            id: ++timelineSeq,
            type: streamEvent.type,
            stage: streamEvent.stage,
            summary: summarizeEvent(streamEvent),
            raw: streamEvent.data,
          });
          renderTimeline(timeline, entries);

          if (streamEvent.type === "final") {
            finalText = String(streamEvent.data?.final_response ?? "");
            if (streamEvent.data?.trace_id) {
              meta.trace_id = String(streamEvent.data.trace_id);
            }
            if (streamEvent.data?.domain) {
              meta.domain = String(streamEvent.data.domain);
            }
          }
        }

        renderResponse(responseBox, finalText, Object.keys(meta).length ? meta : undefined);
        status.className = "status ok";
        status.textContent = `流式完成 · ${entries.length} 个事件`;
      } else {
        const result = await client.route(query, options);
        renderResponse(responseBox, result.final_response, {
          domain: result.domain,
          profile: result.profile,
          trace_id: result.trace_id ?? "—",
        });
        status.className = "status ok";
        status.textContent = "同步 route() 完成";
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      status.className = "status error";
      status.textContent = message;
      renderResponse(responseBox, `错误：${message}`);
    } finally {
      submitBtn.disabled = false;
    }
  });
}

mount();
