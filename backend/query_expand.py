import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from chat_memory import find_relevant_messages
from query_planner import (
    DOMAIN_TERMS,
    CONTRACTOR_REFERENCE_TERMS,
    FOLLOW_UP_TERMS,
    QueryPlan,
    RESULT_REFERENCE_TERMS,
    detect_intent_from_expanded_query,
    plan_query,
    render_plan,
)
from query_scope import get_thread_plan, get_thread_result, set_thread_plan, set_thread_result

RESULT_REFERENCE_LIMIT_CAP = 10
OTHER_PROJECTS_TERMS = re.compile(
    r"\b(other projects?|other contracts?|more projects?|another project|another contract|remaining projects?)\b",
    re.IGNORECASE,
)
ORDINAL_REFERENCE_PATTERNS = [
    (re.compile(r"(?:\b1st\b|\b1\s*(?:st)?\s+(?:one|contract|project|result)\b|#1\b|\bnumber\s+1\b)", re.IGNORECASE), 0),
    (re.compile(r"(?:\b2nd\b|\b2\s*(?:nd)?\s+(?:one|contract|project|result)\b|#2\b|\bnumber\s+2\b)", re.IGNORECASE), 1),
    (re.compile(r"(?:\b3rd\b|\b3\s*(?:rd)?\s+(?:one|contract|project|result)\b|#3\b|\bnumber\s+3\b)", re.IGNORECASE), 2),
    (re.compile(r"(?:\b4th\b|\b4\s*(?:th)?\s+(?:one|contract|project|result)\b|#4\b|\bnumber\s+4\b)", re.IGNORECASE), 3),
    (re.compile(r"(?:\b5th\b|\b5\s*(?:th)?\s+(?:one|contract|project|result)\b|#5\b|\bnumber\s+5\b)", re.IGNORECASE), 4),
    (re.compile(r"\bfirst\s+(?:one|contract|project|result)\b", re.IGNORECASE), 0),
    (re.compile(r"\bsecond\s+(?:one|contract|project|result)\b", re.IGNORECASE), 1),
    (re.compile(r"\bthird\s+(?:one|contract|project|result)\b", re.IGNORECASE), 2),
    (re.compile(r"\bfourth\s+(?:one|contract|project|result)\b", re.IGNORECASE), 3),
    (re.compile(r"\bfifth\s+(?:one|contract|project|result)\b", re.IGNORECASE), 4),
    (re.compile(r"\blast\s+(?:one|contract|project|result)\b", re.IGNORECASE), -1),
]
HISTORY_REFERENCE_TERMS = re.compile(
    r"\b(again|earlier|previous|before|same|that|those|these|them|it|compare)\b",
    re.IGNORECASE,
)


def _plan_from_payload(payload: dict[str, object] | None) -> QueryPlan | None:
    if not payload:
        return None

    intent = str(payload.get("intent", "chat"))
    if intent == "chat" and not payload.get("filters") and not payload.get("subject"):
        return None

    return QueryPlan(
        intent=intent,
        filters=dict(payload.get("filters", {})),
        subject=str(payload.get("subject", "") or ""),
        lookup_value=str(payload.get("lookup_value", "") or ""),
        limit=payload.get("limit"),
        exclude_selected_contract=bool(payload.get("exclude_selected_contract", False)),
        has_location_phrase=bool(payload.get("has_location_phrase", False)),
        has_unresolved_location_hint=bool(
            payload.get("has_unresolved_location_hint", False)
        ),
        is_follow_up=bool(payload.get("is_follow_up", False)),
    )


def _plan_from_memory(thread_id: str | None) -> QueryPlan | None:
    return _plan_from_payload(get_thread_plan(thread_id))


def _merge_result_filters(base_filters: dict[str, str], new_filters: dict[str, str]) -> dict[str, str]:
    merged = dict(base_filters)
    for key, value in new_filters.items():
        merged[key] = value

    if "province" in new_filters:
        merged.pop("region", None)
    if "region" in new_filters:
        merged.pop("province", None)
    return merged


def _resolve_ordinal_lookup(thread_id: str | None, query: str) -> QueryPlan | None:
    result_state = get_thread_result(thread_id)
    if result_state.get("result_kind") != "contract_set":
        return None

    displayed_ids = result_state.get("displayed_contract_ids") or result_state.get("contract_ids") or []
    if not isinstance(displayed_ids, list) or not displayed_ids:
        return None

    for pattern, index in ORDINAL_REFERENCE_PATTERNS:
        if not pattern.search(query):
            continue
        try:
            lookup_value = displayed_ids[index]
        except IndexError:
            return None
        return QueryPlan(intent="lookup", lookup_value=str(lookup_value))
    return None


def _resolve_result_reference(
    query: str,
    previous_plan: QueryPlan | None,
    thread_id: str | None,
) -> QueryPlan | None:
    result_state = get_thread_result(thread_id)
    result_filters = result_state.get("filters")
    if result_state.get("result_kind") != "contract_set" or not isinstance(result_filters, dict) or not result_filters:
        return None

    plan = plan_query(query, previous_plan=previous_plan)
    has_direct_reference = bool(RESULT_REFERENCE_TERMS.search(query))
    is_follow_up_modifier = bool(FOLLOW_UP_TERMS.search(query.strip()))
    if not has_direct_reference and not (is_follow_up_modifier and plan.filters and not plan.lookup_value):
        return None

    count = int(result_state.get("count") or 0)
    limit = min(count, RESULT_REFERENCE_LIMIT_CAP) if count > 0 else None
    resolved_limit = plan.limit or (limit if has_direct_reference else None)

    return QueryPlan(
        intent="browse",
        filters=_merge_result_filters(
            {key: str(value) for key, value in result_filters.items() if isinstance(value, str)},
            plan.filters,
        ),
        subject="",
        lookup_value="",
        limit=resolved_limit,
        exclude_selected_contract=plan.exclude_selected_contract,
        has_location_phrase=plan.has_location_phrase,
        has_unresolved_location_hint=plan.has_unresolved_location_hint,
        is_follow_up=True,
    )


def _extract_selected_contract_source(
    result_state: dict[str, object],
) -> dict[str, object] | None:
    if not isinstance(result_state, dict):
        return None

    displayed_sources = result_state.get("displayed_sources")
    if not isinstance(displayed_sources, list) or not displayed_sources:
        return None

    selected_id = str(result_state.get("selected_contract_id") or "").strip()
    if selected_id:
        for source in displayed_sources:
            if not isinstance(source, dict):
                continue
            contract_id = str(source.get("contractId") or "").strip()
            if contract_id and contract_id == selected_id:
                return source

    if len(displayed_sources) == 1 and isinstance(displayed_sources[0], dict):
        return displayed_sources[0]

    if result_state.get("result_kind") == "contract_detail" and isinstance(displayed_sources[0], dict):
        return displayed_sources[0]

    return None


def _resolve_same_contractor_reference(
    query: str,
    thread_id: str | None,
) -> QueryPlan | None:
    if not CONTRACTOR_REFERENCE_TERMS.search(query):
        return None

    result_state = get_thread_result(thread_id)
    source = _extract_selected_contract_source(result_state)
    if not source:
        return QueryPlan(
            intent="clarify",
            subject="Which contractor are you referring to?",
            filters={},
            lookup_value="",
            limit=None,
            exclude_selected_contract=False,
            has_location_phrase=False,
            has_unresolved_location_hint=False,
            is_follow_up=True,
        )

    contractor = str(source.get("contractor") or "").strip()
    if not contractor:
        return QueryPlan(
            intent="clarify",
            subject="Which contractor are you referring to?",
            filters={},
            lookup_value="",
            limit=None,
            exclude_selected_contract=False,
            has_location_phrase=False,
            has_unresolved_location_hint=False,
            is_follow_up=True,
        )

    exclude_selected_contract = bool(
        OTHER_PROJECTS_TERMS.search(query)
        or re.search(r"\bother\b", query, re.IGNORECASE)
        or re.search(r"\bmore\b", query, re.IGNORECASE)
    )

    return QueryPlan(
        intent="browse",
        filters={"contractor": contractor},
        subject="",
        lookup_value="",
        limit=None,
        exclude_selected_contract=exclude_selected_contract,
        has_location_phrase=False,
        has_unresolved_location_hint=False,
        is_follow_up=True,
    )


def _needs_older_context(
    query: str,
    previous_plan: QueryPlan | None,
    result_state: dict[str, object],
) -> bool:
    stripped = query.strip()
    if not stripped:
        return False
    if any(pattern.search(query) for pattern, _ in ORDINAL_REFERENCE_PATTERNS):
        return not bool(result_state)
    if RESULT_REFERENCE_TERMS.search(query):
        return not bool(result_state)
    if FOLLOW_UP_TERMS.search(stripped):
        return previous_plan is None or not previous_plan.filters
    if HISTORY_REFERENCE_TERMS.search(query) and not DOMAIN_TERMS.search(query):
        return previous_plan is None or not previous_plan.filters
    return False


def _load_older_context(
    thread_id: str | None,
    query: str,
) -> tuple[QueryPlan | None, dict[str, object]]:
    if not thread_id:
        return None, {}

    messages = find_relevant_messages(thread_id, query, limit=8)
    recovered_plan = None
    recovered_result: dict[str, object] = {}

    for message in messages:
        metadata = message.get("message_metadata")
        if not isinstance(metadata, dict):
            continue

        if recovered_plan is None:
            recovered_plan = _plan_from_payload(metadata.get("plan"))

        if not recovered_result:
            result_state = metadata.get("result_state")
            if isinstance(result_state, dict) and result_state.get("result_kind") == "contract_set":
                recovered_result = result_state

        if recovered_plan and recovered_result:
            break

    return recovered_plan, recovered_result


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
    current_result_state = get_thread_result(thread_id)

    if _needs_older_context(query, previous_plan, current_result_state):
        recovered_plan, recovered_result = _load_older_context(thread_id, query)
        if previous_plan is None and recovered_plan is not None:
            previous_plan = recovered_plan
        if not current_result_state and recovered_result:
            set_thread_result(thread_id, recovered_result)
            current_result_state = recovered_result

    plan = _resolve_ordinal_lookup(thread_id, query)
    if plan is None:
        plan = _resolve_result_reference(query, previous_plan, thread_id)
    if plan is None:
        plan = _resolve_same_contractor_reference(query, thread_id)
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
            "exclude_selected_contract": plan.exclude_selected_contract,
            "has_location_phrase": plan.has_location_phrase,
            "has_unresolved_location_hint": plan.has_unresolved_location_hint,
            "is_follow_up": plan.is_follow_up,
        },
    )
    return expanded
