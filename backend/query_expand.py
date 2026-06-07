import json
import os
from datetime import datetime, timezone
from pathlib import Path

from query_planner import (
    QueryPlan,
    RESULT_REFERENCE_TERMS,
    detect_intent_from_expanded_query,
    plan_query,
    render_plan,
)
from query_scope import get_thread_plan, get_thread_result, set_thread_plan

RESULT_REFERENCE_LIMIT_CAP = 10


def _plan_from_memory(thread_id: str | None) -> QueryPlan | None:
    payload = get_thread_plan(thread_id)
    if not payload:
        return None

    return QueryPlan(
        intent=payload.get("intent", "chat"),
        filters=dict(payload.get("filters", {})),
        subject=str(payload.get("subject", "") or ""),
        lookup_value=str(payload.get("lookup_value", "") or ""),
        limit=payload.get("limit"),
        has_location_phrase=bool(payload.get("has_location_phrase", False)),
        has_unresolved_location_hint=bool(
            payload.get("has_unresolved_location_hint", False)
        ),
        is_follow_up=bool(payload.get("is_follow_up", False)),
    )


def _merge_result_filters(base_filters: dict[str, str], new_filters: dict[str, str]) -> dict[str, str]:
    merged = dict(base_filters)
    for key, value in new_filters.items():
        merged[key] = value

    if "province" in new_filters:
        merged.pop("region", None)
    if "region" in new_filters:
        merged.pop("province", None)
    return merged


def _resolve_result_reference(
    query: str,
    previous_plan: QueryPlan | None,
    thread_id: str | None,
) -> QueryPlan | None:
    if not RESULT_REFERENCE_TERMS.search(query):
        return None

    result_state = get_thread_result(thread_id)
    result_filters = result_state.get("filters")
    if result_state.get("result_kind") != "contract_set" or not isinstance(result_filters, dict) or not result_filters:
        return None

    plan = plan_query(query, previous_plan=previous_plan)
    count = int(result_state.get("count") or 0)
    limit = min(count, RESULT_REFERENCE_LIMIT_CAP) if count > 0 else None

    return QueryPlan(
        intent="browse",
        filters=_merge_result_filters(
            {key: str(value) for key, value in result_filters.items() if isinstance(value, str)},
            plan.filters,
        ),
        subject="",
        lookup_value="",
        limit=plan.limit or limit,
        has_location_phrase=plan.has_location_phrase,
        has_unresolved_location_hint=plan.has_unresolved_location_hint,
        is_follow_up=True,
    )


def _detect_intent(expanded_query: str) -> str:
    return detect_intent_from_expanded_query(expanded_query)


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
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Query expansion log error: {e}")


def query_expand(query: str, thread_id: str | None = None) -> str:
    previous_plan = _plan_from_memory(thread_id)
    plan = _resolve_result_reference(query, previous_plan, thread_id)
    if plan is None:
        plan = plan_query(query, previous_plan=previous_plan)

    if plan.intent == "chat":
        expanded = query
    else:
        expanded = render_plan(plan)

    set_thread_plan(
        thread_id,
        {
            "intent": plan.intent,
            "filters": plan.filters,
            "subject": plan.subject,
            "lookup_value": plan.lookup_value,
            "limit": plan.limit,
            "has_location_phrase": plan.has_location_phrase,
            "has_unresolved_location_hint": plan.has_unresolved_location_hint,
            "is_follow_up": plan.is_follow_up,
        },
    )
    return expanded
