from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from query_planner import QueryPlan
from query_planner_llm import plan_message
from query_scope import set_thread_plan


FILTER_RENDER_ORDER = (
    "region",
    "province",
    "status",
    "category",
    "contractor",
    "infra_year",
    "program_name",
)


def _render_filters(filters: dict[str, str]) -> str:
    ordered_items: list[tuple[str, str]] = []
    seen_keys: set[str] = set()
    for key in FILTER_RENDER_ORDER:
        value = filters.get(key)
        if value:
            ordered_items.append((key, value))
            seen_keys.add(key)
    for key, value in filters.items():
        if key in seen_keys or not value:
            continue
        ordered_items.append((key, value))
    clauses = [f"{key}={value}" for key, value in ordered_items]
    return " AND ".join(clauses)


def render_plan(plan: QueryPlan) -> str:
    if plan.intent == "lookup":
        return f"Lookup contract {plan.lookup_value}".strip()
    if plan.intent == "clarify":
        return f"Ask clarifying question: {plan.subject}".strip()
    if plan.intent == "availability":
        clause_text = _render_filters(plan.filters) or "all=true"
        return f"Check availability where {clause_text}".strip()
    if plan.intent == "browse":
        clause_text = _render_filters(plan.filters)
        rendered = f"Filter contracts where {clause_text}".strip()
        if plan.limit:
            rendered += f" LIMIT {plan.limit}"
        return rendered
    if plan.intent == "stats":
        clause_text = _render_filters(plan.filters) or "all=true"
        return f"Calculate metrics where {clause_text}".strip()
    if plan.intent == "search":
        if plan.filters:
            return (
                f"Find all contracts about {plan.subject or 'contracts'} "
                f"where {_render_filters(plan.filters)}"
            ).strip()
        return f"Find all contracts about {plan.subject or 'contracts'}".strip()
    if plan.intent == "compare":
        rendered = f"Compare contracts {plan.lookup_value}".strip()
        if plan.subject:
            rendered += f": {plan.subject}"
        return rendered
    if plan.intent == "anomaly":
        parts = [plan.analysis_type or "anomaly scan"]
        if plan.lookup_value:
            parts.append(plan.lookup_value)
        if plan.filters:
            parts.append(_render_filters(plan.filters))
        return "Analyze " + " | ".join(parts)
    return plan.subject or ""


def _detect_intent(expanded_query: str) -> str:
    lowered = expanded_query.lower().strip()
    if lowered.startswith("lookup contract"):
        return "lookup"
    if lowered.startswith("check availability where"):
        return "availability"
    if lowered.startswith("filter contracts where"):
        return "browse"
    if lowered.startswith("calculate metrics where"):
        return "stats"
    if lowered.startswith("find all contracts about"):
        return "search"
    if lowered.startswith("compare contracts"):
        return "compare"
    if lowered.startswith("ask clarifying question:"):
        return "clarify"
    if lowered.startswith("analyze "):
        return "anomaly"
    return "chat"


def log_query_expansion(
    raw_input: str, expanded_output: str, thread_id: str | None = None
) -> None:
    log_path = Path(
        os.environ.get(
            "QUERY_EXPAND_LOG_PATH",
            Path(__file__).parent / "logs" / "query_expand.jsonl",
        )
    )
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "raw_input": raw_input,
        "expanded_output": expanded_output,
        "intent": _detect_intent(expanded_output),
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"Query expansion log error: {exc}")


def query_expand(query: str, thread_id: str | None = None) -> str:
    plan = plan_message(query, thread_id)
    set_thread_plan(thread_id, plan.to_dict())
    return render_plan(plan) if plan.intent != "chat" else query
