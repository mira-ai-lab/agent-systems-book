"""
Convert OpenTelemetry span JSONL traces to Who&When JSON format
for Automated_FA/inference.py failure attribution.

Usage:
    python testdata/convert_spans_to_whowhen.py testdata/spans_20260615_180713.jsonl
    python testdata/convert_spans_to_whowhen.py testdata/spans_*.jsonl -o testdata/converted
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# Planner wrapper spans carry little unique content; leaf planner steps are kept.
PLANNER_WRAPPER_SUFFIXES = (".build_plan",)


def load_spans(jsonl_path: Path) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                spans.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {jsonl_path}") from exc
    if not spans:
        raise ValueError(f"No spans found in {jsonl_path}")
    return spans


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value and str(value).strip():
            return str(value).strip()
    return ""


def _span_name(span: dict[str, Any]) -> str:
    return str(span.get("name", ""))


def is_planner_span(span: dict[str, Any]) -> bool:
    name = _span_name(span)
    if ".planner." not in name:
        return False
    return not any(name.endswith(suffix) for suffix in PLANNER_WRAPPER_SUFFIXES)


def is_agent_invoke_span(span: dict[str, Any]) -> bool:
    return _span_name(span).endswith(".agent.invoke")


def is_aggregate_span(span: dict[str, Any]) -> bool:
    return _span_name(span).endswith(".orchestration.aggregate")


def get_user_query(spans: list[dict[str, Any]]) -> str:
    for span in spans:
        attrs = span.get("attributes", {})
        query = _first_non_empty(attrs.get("user.query"), attrs.get("state.user_query"))
        if query:
            return query
        for event in span.get("events", []):
            event_attrs = event.get("attributes", {})
            query = _first_non_empty(event_attrs.get("user_query"))
            if query:
                return query
            state_raw = event_attrs.get("state")
            if isinstance(state_raw, str):
                try:
                    state = json.loads(state_raw)
                    query = _first_non_empty(state.get("user_query"))
                    if query:
                        return query
                except json.JSONDecodeError:
                    pass
    return ""


def _parse_preview_json(preview: str) -> dict[str, Any]:
    if not preview:
        return {}
    try:
        parsed = json.loads(preview)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def get_final_response(spans: list[dict[str, Any]]) -> str:
    candidates: list[tuple[str, str]] = []
    for span in spans:
        for event in span.get("events", []):
            if event.get("name") != "result":
                continue
            preview = event.get("attributes", {}).get("preview", "")
            data = _parse_preview_json(preview)
            final_response = data.get("final_response")
            if isinstance(final_response, str) and final_response.strip():
                candidates.append((span.get("start_time", ""), final_response.strip()))

    if not candidates:
        return ""

    # Prefer the latest result event that contains final_response (usually root request span).
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def build_default_ground_truth(question: str) -> str:
    return (
        f"针对用户问题「{question}」，系统应给出准确、完整、可验证的回答，"
        "且关键事实（时间范围、地点、数量、结论等）应与用户意图一致。"
    )


def _task_sort_key(task_id: str) -> tuple[int | str, ...]:
    task_id = task_id.strip()
    match = re.match(r"^T(\d+)$", task_id, re.IGNORECASE)
    if match:
        return (0, int(match.group(1)))
    match = re.match(r"^(\d+)$", task_id)
    if match:
        return (1, int(match.group(1)))
    return (2, task_id)


def _parse_task_id_list(raw: str) -> list[str]:
    return [task_id.strip() for task_id in raw.split(",") if task_id.strip()]


def collect_agent_invoke_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [span for span in spans if is_agent_invoke_span(span)]


def _collect_task_ids_from_agent_spans(spans: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    task_ids: list[str] = []
    for span in collect_agent_invoke_spans(spans):
        task_id = span.get("attributes", {}).get("task.id")
        if task_id and task_id not in seen:
            seen.add(str(task_id))
            task_ids.append(str(task_id))
    return sorted(task_ids, key=_task_sort_key)


def get_execution_order(spans: list[dict[str, Any]]) -> list[str]:
    for span in spans:
        for event in span.get("events", []):
            if event.get("name") != "plan.built":
                continue
            order = event.get("attributes", {}).get("execution.order", "")
            if order:
                return _parse_task_id_list(str(order))

    for span in spans:
        layer_tasks = span.get("attributes", {}).get("layer.tasks")
        if layer_tasks:
            return _parse_task_id_list(str(layer_tasks))

    inferred = _collect_task_ids_from_agent_spans(spans)
    if inferred:
        return inferred

    return []


def span_agent_name(span: dict[str, Any]) -> str:
    span_name = _span_name(span)
    attrs = span.get("attributes", {})

    if is_agent_invoke_span(span):
        return str(attrs.get("task.agent") or attrs.get("agent.name") or "UnknownAgent")

    if is_planner_span(span):
        return "Planner." + span_name.rsplit(".", 1)[-1]

    if is_aggregate_span(span):
        return "Orchestrator"

    return span_name.rsplit(".", 1)[-1] or "UnknownStep"


def _format_tool_events(events: list[dict[str, Any]]) -> list[str]:
    parts: list[str] = []
    for event in events:
        if event.get("name") != "tool.completed":
            continue
        attrs = event.get("attributes", {})
        parts.append(
            "[Tool] {tool} (task={task}, error={error})\n{output}".format(
                tool=attrs.get("tool.name", "unknown"),
                task=attrs.get("task.id", "?"),
                error=attrs.get("tool.has_error", False),
                output=attrs.get("tool.output_preview", ""),
            )
        )
    return parts


def span_content(span: dict[str, Any]) -> str:
    attrs = span.get("attributes", {})
    events = span.get("events", [])
    parts: list[str] = [
        f"Span: {_span_name(span)}",
        f"Status: {span.get('status_code', 'OK')}",
    ]

    task_id = attrs.get("task.id")
    if task_id:
        parts.append(f"Task ID: {task_id}")

    if attrs.get("task.description"):
        parts.append(f"[Task Description]\n{attrs['task.description']}")

    for event in events:
        event_name = event.get("name", "")
        event_attrs = event.get("attributes", {})

        if event_name == "sub_agent_conversation":
            parts.append(f"[Query]\n{event_attrs.get('query', '')}")
            parts.append(f"[Response]\n{event_attrs.get('response', '')}")
            parts.append(
                "[Meta] status={status}, tool_calls={tool_calls}, agent={agent}".format(
                    status=event_attrs.get("status", ""),
                    tool_calls=event_attrs.get("tool_call_count", 0),
                    agent=event_attrs.get("agent", ""),
                )
            )
        elif event_name == "result":
            preview = event_attrs.get("preview", "")
            if preview:
                parts.append(f"[Result]\n{preview}")
            else:
                parts.append(f"[Result] status={event_attrs.get('status', '')}")
        elif event_name == "request" and event_attrs.get("sub_steps"):
            parts.append(f"[Sub Steps]\n{event_attrs.get('sub_steps', '')}")
        elif event_name == "plan.built":
            parts.append(
                "[Plan Built] subtasks={count}, layers={layers}, order={order}".format(
                    count=event_attrs.get("subtask.count", "?"),
                    layers=event_attrs.get("layer.count", "?"),
                    order=event_attrs.get("execution.order", "?"),
                )
            )

    tool_parts = _format_tool_events(events)
    if tool_parts:
        parts.append("[Tool Outputs]")
        parts.extend(tool_parts)

    if len(parts) <= 2 and attrs:
        parts.append("[Attributes]\n" + json.dumps(attrs, ensure_ascii=False, indent=2))

    return "\n\n".join(parts)


def _history_entry(span: dict[str, Any]) -> dict[str, str]:
    return {
        "name": span_agent_name(span),
        "role": "user",
        "content": span_content(span),
    }


def _sort_agent_spans(spans: list[dict[str, Any]], execution_order: list[str]) -> list[dict[str, Any]]:
    order_index = {task_id: idx for idx, task_id in enumerate(execution_order)}

    def sort_key(span: dict[str, Any]) -> tuple[int, str, str]:
        task_id = str(span.get("attributes", {}).get("task.id", ""))
        if execution_order:
            return (order_index.get(task_id, 999), span.get("start_time", ""), task_id)
        return (0, span.get("start_time", ""), task_id)

    return sorted(spans, key=sort_key)


def build_history(spans: list[dict[str, Any]]) -> list[dict[str, str]]:
    execution_order = get_execution_order(spans)
    history: list[dict[str, str]] = []

    planner_spans = [span for span in spans if is_planner_span(span)]
    planner_spans.sort(key=lambda span: span.get("start_time", ""))
    history.extend(_history_entry(span) for span in planner_spans)

    agent_spans = _sort_agent_spans(collect_agent_invoke_spans(spans), execution_order)
    history.extend(_history_entry(span) for span in agent_spans)

    aggregate_spans = [span for span in spans if is_aggregate_span(span)]
    if aggregate_spans:
        aggregate_span = sorted(aggregate_spans, key=lambda span: span.get("start_time", ""))[0]
        content = span_content(aggregate_span)
        final_response = get_final_response(spans)
        if final_response:
            content += f"\n\n[Final Response]\n{final_response}"
        history.append(
            {
                "name": "Orchestrator",
                "role": "user",
                "content": content,
            }
        )

    if not history:
        raise ValueError(
            "Could not build history: no planner, agent.invoke, or aggregate spans found. "
            "Check span naming conventions in the input JSONL."
        )

    return history


def infer_output_name(jsonl_path: Path) -> str:
    return f"{jsonl_path.stem}.json"


def convert_jsonl_to_whowwhen(
    jsonl_path: Path,
    output_dir: Path,
    output_name: str | None = None,
    ground_truth: str | None = None,
    is_correct: bool = False,
    mistake_agent: str | None = None,
    mistake_step: str | None = None,
    mistake_reason: str | None = None,
    include_final_response_in_ground_truth: bool = False,
) -> Path:
    spans = load_spans(jsonl_path)
    history = build_history(spans)
    question = get_user_query(spans)
    final_response = get_final_response(spans)

    if not question:
        raise ValueError(f"Could not extract user query from spans: {jsonl_path}")

    if ground_truth is None:
        ground_truth = build_default_ground_truth(question)
        if include_final_response_in_ground_truth and final_response:
            ground_truth += f"\n\n[系统实际输出摘要]\n{final_response[:1500]}"

    record: dict[str, Any] = {
        "is_correct": is_correct,
        "question": question,
        "ground_truth": ground_truth,
        "history": history,
        "source_trace_id": spans[0].get("trace_id"),
        "source_file": jsonl_path.name,
        "execution_order": get_execution_order(spans),
    }

    if final_response:
        record["system_final_response"] = final_response

    if mistake_agent is not None:
        record["mistake_agent"] = mistake_agent
    if mistake_step is not None:
        record["mistake_step"] = str(mistake_step)
    if mistake_reason is not None:
        record["mistake_reason"] = mistake_reason

    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = output_name or infer_output_name(jsonl_path)
    out_path = output_dir / out_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return out_path


def _print_summary(out_path: Path) -> None:
    with out_path.open("r", encoding="utf-8") as f:
        record = json.load(f)

    print(f"Wrote {out_path}")
    print(f"trace_id: {record.get('source_trace_id')}")
    print(f"question: {record.get('question', '')[:80]}...")
    print(f"execution_order: {record.get('execution_order', [])}")
    print(f"history steps: {len(record['history'])}")
    for idx, step in enumerate(record["history"]):
        print(f"  Step {idx}: {step['name']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert span JSONL traces to Who&When JSON for Automated_FA."
    )
    parser.add_argument(
        "jsonl_paths",
        nargs="+",
        type=Path,
        help="One or more input spans JSONL files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("testdata/converted"),
        help="Output directory for converted JSON (default: testdata/converted)",
    )
    parser.add_argument("--output-name", type=str, default=None, help="Output JSON filename (single input only)")
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Expected correct answer/result description for attribution prompts",
    )
    parser.add_argument(
        "--is-correct",
        action="store_true",
        help="Mark the task as successful (default: false)",
    )
    parser.add_argument("--mistake-agent", type=str, default=None, help="Optional label for evaluate.py")
    parser.add_argument("--mistake-step", type=str, default=None, help="Optional label for evaluate.py")
    parser.add_argument("--mistake-reason", type=str, default=None, help="Optional label for evaluate.py")
    parser.add_argument(
        "--include-final-response",
        action="store_true",
        help="Append system final_response snippet to ground_truth",
    )
    args = parser.parse_args()

    if len(args.jsonl_paths) > 1 and args.output_name:
        parser.error("--output-name can only be used when converting a single JSONL file.")

    for jsonl_path in args.jsonl_paths:
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Input file not found: {jsonl_path}")

        out_path = convert_jsonl_to_whowwhen(
            jsonl_path=jsonl_path,
            output_dir=args.output_dir,
            output_name=args.output_name if len(args.jsonl_paths) == 1 else None,
            ground_truth=args.ground_truth,
            is_correct=args.is_correct,
            mistake_agent=args.mistake_agent,
            mistake_step=args.mistake_step,
            mistake_reason=args.mistake_reason,
            include_final_response_in_ground_truth=args.include_final_response,
        )
        _print_summary(out_path)
        print()


if __name__ == "__main__":
    main()
