from __future__ import annotations

import json
import os
import re

from features.chat.memory import find_relevant_messages
from features.chat.agent.query_planner import (
    PROXIMITY_PATTERN,
    QueryPlan,
    build_anchor_plan,
    extract_anchor_filters,
    find_lookup_contract_id,
    has_domain_terms,
    has_awarded_to_contractor,
    is_greeting,
)
from features.chat.agent.query_scope import (
    compact_thread_context,
    get_thread_plan,
    get_thread_result,
    set_thread_plan,
    set_thread_result,
)

PLANNER_SYSTEM_PROMPT = """
You are a query planner for the DPWH Watchdog system, a database of Philippines public works contracts.

Your only job is to read the user message and the CONTEXT block, then output a JSON plan. You do not reason about contract data. You do not explain. You output only valid JSON.

──────────────────────────────────────────
CONTEXT BLOCK FORMAT (you will receive this before the user message):
result_kind: contract_set | contract_detail | contract_compare | none
result_count: integer
active_filters: {region, province, status, category, contractor, infra_year}
active_filters may also include infra_year_start and infra_year_end for year windows
displayed_contracts: [{index, id, description_snippet}] (up to 5)
selected_contract: {id, description_snippet} or null
last_intent: string
──────────────────────────────────────────
OUTPUT SCHEMA (always valid JSON, no markdown fences):
{
  "intent": "browse|lookup|stats|availability|compare|anomaly|clarify|chat",
  "source_scope": "database|displayed_results|selected_contract",
  "selection": {"type": "all|ordinal|named|top_n|single", "indices": [], "ids": [],"limit": null},
  "filters": {"region": null, "province": null, "status": null, "category": null, "contractor": null, "infra_year": null, "infra_year_start": null, "infra_year_end": null, "program_name": null},
  "subject": "",
  "lookup_value": "",
  "analysis_type": null,
  "question": "",
  "needs_clarification": false,
  "clarification_question": null,
  "exclude_selected": false
}
──────────────────────────────────────────
INTENT RULES:
browse — user wants a list of contracts filtered by attributes
browse — ALSO use when the user asks "are there any X" AND simultaneously requests
specific per-contract data fields (budget, completion date, contractor, progress, etc.).
lookup — user wants full detail of one specific contract (name or ID)
stats — user wants counts, totals, averages, breakdowns
availability — ONLY when the user asks purely whether something exists, with no follow-up
request for individual contract data.
compare — user wants a side-by-side comparison of 2+ specific contracts
anomaly — user asks about suspicious patterns, outliers, red flags, corruption
clarify — request is too vague to act on; set clarification_question
chat — greeting, thanks, off-topic, general conversation

ANALYSIS_TYPE VALUES (set when intent=anomaly):
contractor_concentration, budget_outlier, award_anomaly,
timeline_anomaly, bidding_anomaly, document_gap, scope_similarity

REFERENCE RESOLUTION RULES:
- "first", "1st", "#1", "the first one" → indices=[0]
- "second", "2nd" → indices=[1]; "third" → indices=[2]
- "these three", "all three", "the three" → indices=[0,1,2]
- Resolve indices to ids using displayed_contracts from CONTEXT
- "same contractor" / "this contractor" → set filters.contractor from selected_contract or displayed_contracts[0]
- "other projects", "other contracts" → set exclude_selected=true
- "those", "them", "these results" with no new subject → source_scope=displayed_results, carry forward active_filters
- Explicit contract ID (pattern like 14K00302, 22AC0086) → intent=lookup, lookup_value=<id>
- Region numbers: 1=I, 2=II, 3=III, 4=IV, 5=V, 6=VI, 7=VII, 8=VIII, 9=IX, 10=X, 11=XI, 12=XII, 13=XIII, 15=XV, 16=XVI
- "NCR", "metro manila" → region="National Capital Region"

FILTER CARRY-FORWARD:
If user does not specify a filter that was active in active_filters, and the query is a follow-up (source_scope=displayed_results), carry the active filter forward into output filters.

CLARIFICATION TRIGGERS:
- intent would be browse/search but filters=null and subject="" and no active context → needs_clarification=true
- compare intent but fewer than 2 contracts can be resolved → needs_clarification=true
- lookup intent but no ID or name can be extracted → needs_clarification=true

ANOMALY DETECTION KEYWORDS:
suspicious, anomaly, red flag, irregular, corruption, overpriced, low-balled,
outlier, concentration, monopoly, single bidder, no competition, stalled,
overdue, missing documents, duplicate, similar scope, same contractor winning
""".strip()

COMPARE_TERMS = re.compile(
    r"\b(compare|comparison|differences?\s+between|versus|vs\.?|more expensive than)\b",
    re.IGNORECASE,
)
ANOMALY_TERMS = re.compile(
    r"\b(anomal(?:y|ies)|red flags?|suspicious|irregular|concentration|budget ratio|timeline|delayed|bidding|bidder|document gaps?|missing documents?|similar scope)\b",
    re.IGNORECASE,
)
AVAILABILITY_TERMS = re.compile(
    r"\b(are there|is there|do you have|available|any|exist)\b",
    re.IGNORECASE,
)
DETAIL_FIELD_TERMS = re.compile(
    r"\b(budget|budgets|completion date|completion dates|contractor|progress|how much|dates|each project|individual|list them|show them|details)\b",
    re.IGNORECASE,
)
STATS_TERMS = re.compile(
    r"\b(how many|count|counts|total|sum|average|avg|statistics|metrics|breakdown|top|highest|lowest)\b",
    re.IGNORECASE,
)
BROWSE_TERMS = re.compile(
    r"\b(show|list|give me|which|browse|filter)\b|\bwhat\s+(?:contracts?|projects?)\b|\b(?:contracts?|projects?)\s+are\s+there\b",
    re.IGNORECASE,
)
FOLLOW_UP_TERMS = re.compile(
    r"^(what about|how about|what if|and what about|show them|show those|show these|them|those|these|what about this|what about that|compare these|compare those|compare them)\b",
    re.IGNORECASE,
)
RESULT_REFERENCE_TERMS = re.compile(
    r"\b(show|list)\s+(them|those|these|results|projects|contracts)\b|\bwhat\s+are\s+(those|these)\b",
    re.IGNORECASE,
)
CONTRACTOR_REFERENCE_TERMS = re.compile(
    r"\b(the contractor|the same contractor|same contractor|this contractor|that contractor|this one|that one|same one)\b",
    re.IGNORECASE,
)
GENERIC_SUBJECTS = {
    "contract",
    "contracts",
    "project",
    "projects",
    "detail",
    "details",
    "the contractor",
    "same contractor",
    "this contractor",
    "that contractor",
}
ORDINAL_REFERENCE_PATTERNS = [
    (re.compile(r"(?:\b1st\b|\b1\s*(?:st)?\s+(?:one|contract|project|result)\b|#1\b|\bnumber\s+1\b|\bfirst\s+(?:one|contract|project|result)\b)", re.IGNORECASE), 0),
    (re.compile(r"(?:\b2nd\b|\b2\s*(?:nd)?\s+(?:one|contract|project|result)\b|#2\b|\bnumber\s+2\b|\bsecond\s+(?:one|contract|project|result)\b)", re.IGNORECASE), 1),
    (re.compile(r"(?:\b3rd\b|\b3\s*(?:rd)?\s+(?:one|contract|project|result)\b|#3\b|\bnumber\s+3\b|\bthird\s+(?:one|contract|project|result)\b)", re.IGNORECASE), 2),
    (re.compile(r"\blast\s+(?:one|contract|project|result)\b", re.IGNORECASE), -1),
]
HISTORY_REFERENCE_TERMS = re.compile(
    r"\b(again|earlier|previous|before|same|that|those|these|them|it|compare)\b",
    re.IGNORECASE,
)
CATEGORY_ALIASES = {
    "flood control": "flood control",
    "drainage": "flood control",
    "river control": "flood control",
    "covered court": "building",
    "multi-purpose building": "building",
    "multi purpose building": "building",
    "school building": "school",
    "school buildings": "school",
    "school": "school",
    "bridge": "bridge",
    "bridges": "bridge",
    "road": "road",
    "roads": "road",
    "road widening": "road",
    "water system": "water supply",
    "water supply": "water supply",
    "water": "water supply",
    "building": "building",
    "buildings": "building",
}

ANALYSIS_TYPE_ALIASES = {
    "budget_outlier": "budget_outlier",
    "budget_anomalies": "budget_outlier",
    "award_anomaly": "award_anomaly",
    "timeline_anomaly": "timeline_anomaly",
    "timeline_anomalies": "timeline_anomaly",
    "bidding_anomaly": "bidding_anomaly",
    "bidding_anomalies": "bidding_anomaly",
    "document_gap": "document_gap",
    "document_gaps": "document_gap",
    "scope_similarity": "scope_similarity",
    "similar_scope": "scope_similarity",
    "contractor_concentration": "contractor_concentration",
}


def _normalize_text(value: object) -> str:
    lowered = str(value or "").lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _match_category(query: str) -> str | None:
    normalized = _normalize_text(query)
    for alias in sorted(CATEGORY_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return CATEGORY_ALIASES[alias]
    return None


def _matched_category_span(query: str) -> str:
    normalized = _normalize_text(query)
    for alias in sorted(CATEGORY_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return alias
    return ""


def _description_matches_query(description: str, query_normalized: str) -> bool:
    description_normalized = _normalize_text(description)
    if not description_normalized:
        return False
    if description_normalized in query_normalized:
        return True
    words = description_normalized.split()
    for span_size in range(min(5, len(words)), 2, -1):
        for start in range(0, len(words) - span_size + 1):
            phrase = " ".join(words[start : start + span_size])
            if len(phrase) < 15:
                continue
            if phrase in query_normalized:
                return True
    return False


def _strip_subject_text(query: str, spans: list[str]) -> str:
    cleaned = query
    for span in spans:
        if span:
            cleaned = re.sub(re.escape(span), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(what about|how about|show me|list|give me|which|what|are there|is there|do you have|does .+ have|find|search for|search|contracts? about|projects? about|compare|details? about|tell me about)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b(in|from|near|around|within|at|across|for)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:contracts?|projects?|project|contract|there|anything|any|please|now|currently)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[?!.:,]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    return cleaned


def _is_generic_subject(subject: str) -> bool:
    return _normalize_text(subject) in GENERIC_SUBJECTS or not _normalize_text(subject)


def _is_scope_only_follow_up_subject(subject: str, filters: dict[str, str]) -> bool:
    normalized_subject = _normalize_text(subject)
    if not normalized_subject:
        return True
    if re.fullmatch(r"region\s+[ivxlcdm0-9]+(?:\s*[ab])?", normalized_subject):
        return True
    for value in filters.values():
        if normalized_subject == _normalize_text(str(value)):
            return True
    return False


def _extract_json(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError("No JSON object found", stripped, 0)
    return json.loads(stripped[start : end + 1])


def _normalize_analysis_type(value: object) -> str:
    normalized = _normalize_text(str(value or ""))
    return ANALYSIS_TYPE_ALIASES.get(normalized, str(value or "").strip())


def _normalize_awarded_to_filters(plan: QueryPlan, user_message: str) -> QueryPlan:
    if has_awarded_to_contractor(user_message) and plan.filters.get("status") == "Awarded":
        plan.filters.pop("status", None)
    return plan


def _plan_from_payload(payload: dict[str, object], anchors: dict[str, str] | None = None) -> QueryPlan:
    filters = payload.get("filters", {})
    if not isinstance(filters, dict):
        filters = {}
    merged_filters = {
        key: str(value).strip()
        for key, value in filters.items()
        if isinstance(value, str) and str(value).strip()
    }
    for key, value in (anchors or {}).items():
        merged_filters[key] = value

    selection = payload.get("selection", {})
    if not isinstance(selection, dict):
        selection = {}
    selection_ids = selection.get("ids", [])
    if not isinstance(selection_ids, list):
        selection_ids = []
    selection_indices = selection.get("indices", [])
    if not isinstance(selection_indices, list):
        selection_indices = []

    intent = str(payload.get("intent") or "chat")
    needs_clarification = bool(payload.get("needs_clarification", False))
    clarification_question = str(payload.get("clarification_question") or "").strip()
    lookup_value = str(payload.get("lookup_value") or "")
    if not lookup_value and selection_ids:
        lookup_value = ",".join(str(item).strip() for item in selection_ids if str(item).strip())

    if needs_clarification and clarification_question:
        intent = "clarify"

    return QueryPlan(
        intent=intent,
        filters=merged_filters,
        subject=clarification_question or str(payload.get("subject") or ""),
        lookup_value=lookup_value,
        limit=(
            payload.get("limit")
            if isinstance(payload.get("limit"), int)
            else selection.get("limit")
            if isinstance(selection.get("limit"), int)
            else None
        ),
        exclude_selected_contract=bool(
            payload.get("exclude_selected_contract", payload.get("exclude_selected", False))
        ),
        has_location_phrase=bool(payload.get("has_location_phrase", False)),
        has_unresolved_location_hint=bool(
            payload.get("has_unresolved_location_hint", False)
        ),
        is_follow_up=bool(payload.get("is_follow_up", False)),
        analysis_type=_normalize_analysis_type(payload.get("analysis_type")),
    )


def _merge_with_previous(plan: QueryPlan, raw_query: str, thread_id: str | None) -> QueryPlan:
    previous_payload = get_thread_plan(thread_id)
    previous_filters = previous_payload.get("filters", {})
    if not isinstance(previous_filters, dict):
        previous_filters = {}
    if not previous_filters:
        previous_result = get_thread_result(thread_id)
        result_filters = previous_result.get("filters", {})
        if isinstance(result_filters, dict):
            previous_filters = result_filters
    previous_subject = str(previous_payload.get("subject") or "")
    previous_intent = str(previous_payload.get("intent") or "")
    if not previous_intent:
        previous_result = get_thread_result(thread_id)
        previous_intent = str(previous_result.get("intent") or "")

    is_follow_up = bool(FOLLOW_UP_TERMS.search(raw_query.strip()))
    if not is_follow_up:
        return plan

    merged = QueryPlan(
        intent=plan.intent if plan.intent != "chat" else previous_intent or "chat",
        filters={
            key: str(value)
            for key, value in previous_filters.items()
            if isinstance(value, str)
        },
        subject=plan.subject or previous_subject,
        lookup_value=plan.lookup_value,
        limit=plan.limit,
        exclude_selected_contract=plan.exclude_selected_contract,
        has_location_phrase=plan.has_location_phrase,
        has_unresolved_location_hint=plan.has_unresolved_location_hint,
        is_follow_up=True,
        analysis_type=plan.analysis_type,
    )
    for key, value in plan.filters.items():
        merged.filters[key] = value
    if "province" in plan.filters:
        merged.filters.pop("region", None)
    if "region" in plan.filters:
        merged.filters.pop("province", None)
    return merged


def _selected_source(thread_id: str | None) -> dict[str, object] | None:
    result = get_thread_result(thread_id)
    displayed = result.get("displayed_sources")
    if not isinstance(displayed, list) or not displayed:
        return None
    if len(displayed) == 1 and isinstance(displayed[0], dict):
        return displayed[0]
    selected_id = str(result.get("selected_contract_id") or "").strip()
    if selected_id:
        for item in displayed:
            if isinstance(item, dict) and str(item.get("contractId") or "").strip() == selected_id:
                return item
    return displayed[0] if isinstance(displayed[0], dict) else None


def _resolve_compare_from_context(user_message: str, thread_id: str | None) -> QueryPlan:
    result = get_thread_result(thread_id)
    displayed_sources = result.get("displayed_sources")
    displayed_ids = result.get("displayed_contract_ids") or result.get("contract_ids") or []
    if not isinstance(displayed_ids, list) or not displayed_ids:
        return QueryPlan(intent="clarify", subject="Which contracts should I compare?")

    if re.search(r"\b(these three|the three|three projects|three contracts)\b", user_message, re.IGNORECASE):
        return QueryPlan(
            intent="compare",
            lookup_value=",".join(str(contract_id) for contract_id in displayed_ids[:3]),
            subject=user_message.strip(),
            is_follow_up=True,
        )

    matched_ids: list[str] = []
    normalized_query = _normalize_text(user_message)
    if isinstance(displayed_sources, list):
        for source in displayed_sources:
            if not isinstance(source, dict):
                continue
            contract_id = str(source.get("contractId") or "").strip()
            description = _normalize_text(source.get("description"))
            if not contract_id:
                continue
            raw_description = str(source.get("description") or "")
            if contract_id.lower() in normalized_query or _description_matches_query(raw_description, normalized_query):
                matched_ids.append(contract_id)
    matched_ids = list(dict.fromkeys(matched_ids))
    if len(matched_ids) >= 2:
        return QueryPlan(
            intent="compare",
            lookup_value=",".join(matched_ids),
            subject=user_message.strip(),
            is_follow_up=True,
        )
    if len(displayed_ids) <= 3 and re.search(r"\b(compare|these|those|them)\b", user_message, re.IGNORECASE):
        return QueryPlan(
            intent="compare",
            lookup_value=",".join(str(contract_id) for contract_id in displayed_ids),
            subject=user_message.strip(),
            is_follow_up=True,
        )
    return QueryPlan(intent="clarify", subject="Which contracts should I compare?")


def _resolve_ordinal_lookup(user_message: str, thread_id: str | None) -> QueryPlan | None:
    result = get_thread_result(thread_id)
    if result.get("result_kind") not in {"contract_set", "contract_compare"}:
        return None
    displayed_ids = result.get("displayed_contract_ids") or result.get("contract_ids") or []
    if not isinstance(displayed_ids, list) or not displayed_ids:
        return None
    for pattern, index in ORDINAL_REFERENCE_PATTERNS:
        if pattern.search(user_message):
            try:
                contract_id = displayed_ids[index]
            except IndexError:
                return None
            return QueryPlan(intent="lookup", lookup_value=str(contract_id), is_follow_up=True)
    return None


def _plan_from_history_payload(payload: dict[str, object] | None) -> QueryPlan | None:
    if not payload or not isinstance(payload, dict):
        return None
    intent = str(payload.get("intent") or "chat")
    filters = payload.get("filters", {})
    if not isinstance(filters, dict):
        filters = {}
    return QueryPlan(
        intent=intent,
        filters={key: str(value) for key, value in filters.items() if isinstance(value, str)},
        subject=str(payload.get("subject") or ""),
        lookup_value=str(payload.get("lookup_value") or ""),
        limit=payload.get("limit") if isinstance(payload.get("limit"), int) else None,
        exclude_selected_contract=bool(payload.get("exclude_selected_contract", False)),
        has_location_phrase=bool(payload.get("has_location_phrase", False)),
        has_unresolved_location_hint=bool(payload.get("has_unresolved_location_hint", False)),
        is_follow_up=bool(payload.get("is_follow_up", False)),
        analysis_type=_normalize_analysis_type(payload.get("analysis_type")),
    )


def _needs_history_recovery(
    user_message: str,
    thread_id: str | None,
) -> bool:
    if not thread_id:
        return False
    try:
        if get_thread_plan(thread_id) or get_thread_result(thread_id):
            return False
    except Exception:
        return False
    stripped = user_message.strip()
    if not stripped:
        return False
    if any(pattern.search(user_message) for pattern, _ in ORDINAL_REFERENCE_PATTERNS):
        return True
    if RESULT_REFERENCE_TERMS.search(user_message):
        return True
    if COMPARE_TERMS.search(user_message):
        return True
    if FOLLOW_UP_TERMS.search(stripped):
        return True
    if HISTORY_REFERENCE_TERMS.search(user_message) and not has_domain_terms(user_message):
        return True
    return False


def _recover_history_context(thread_id: str | None, user_message: str) -> None:
    if not thread_id:
        return
    try:
        messages = find_relevant_messages(thread_id, user_message, limit=8)
    except Exception:
        return
    recovered_plan: QueryPlan | None = None
    recovered_result: dict[str, object] = {}
    for message in messages:
        metadata = message.get("message_metadata")
        if not isinstance(metadata, dict):
            continue
        if recovered_plan is None:
            recovered_plan = _plan_from_history_payload(metadata.get("plan"))
        if not recovered_result:
            result_state = metadata.get("result_state")
            if isinstance(result_state, dict) and result_state.get("result_kind") in {
                "contract_set",
                "contract_compare",
                "contract_detail",
            }:
                recovered_result = result_state
        if recovered_plan and recovered_result:
            break

    try:
        if recovered_plan is not None and not get_thread_plan(thread_id):
            set_thread_plan(thread_id, recovered_plan.to_dict())
        if recovered_result and not get_thread_result(thread_id):
            set_thread_result(thread_id, recovered_result)
    except Exception:
        return


def _fallback_plan(user_message: str, thread_id: str | None) -> QueryPlan:
    anchor_plan = build_anchor_plan(user_message)
    filters = dict(anchor_plan.filters)
    category = _match_category(user_message)
    category_span = _matched_category_span(user_message)
    if category:
        filters["category"] = category

    stripped = user_message.strip()
    lowered = stripped.lower()

    ordinal_lookup = _resolve_ordinal_lookup(user_message, thread_id)
    if ordinal_lookup is not None:
        return ordinal_lookup

    if COMPARE_TERMS.search(user_message):
        return _resolve_compare_from_context(user_message, thread_id)

    if ANOMALY_TERMS.search(user_message):
        analysis_type = ""
        if "concentration" in lowered:
            analysis_type = "contractor_concentration"
        elif "budget" in lowered:
            analysis_type = "budget_outlier"
        elif "award" in lowered:
            analysis_type = "award_anomaly"
        elif "timeline" in lowered or "delayed" in lowered:
            analysis_type = "timeline_anomaly"
        elif "bidding" in lowered or "bidder" in lowered:
            analysis_type = "bidding_anomaly"
        elif "document" in lowered or "missing" in lowered:
            analysis_type = "document_gap"
        elif "similar scope" in lowered:
            analysis_type = "scope_similarity"
        lookup_value = find_lookup_contract_id(user_message) or ""
        if not lookup_value and analysis_type == "scope_similarity":
            selected = _selected_source(thread_id)
            if isinstance(selected, dict):
                lookup_value = str(selected.get("contractId") or "")
        return _merge_with_previous(
            QueryPlan(
                intent="anomaly",
                filters=filters,
                subject=stripped,
                lookup_value=lookup_value,
                is_follow_up=bool(FOLLOW_UP_TERMS.search(stripped)),
                analysis_type=analysis_type,
            ),
            user_message,
            thread_id,
        )

    if RESULT_REFERENCE_TERMS.search(user_message):
        result = get_thread_result(thread_id)
        result_filters = result.get("filters", {})
        if isinstance(result_filters, dict) and result_filters:
            limit = int(result.get("count") or 0) or None
            return QueryPlan(
                intent="browse",
                filters={key: str(value) for key, value in result_filters.items() if isinstance(value, str)},
                limit=min(limit, 10) if limit else None,
                is_follow_up=True,
            )

    if CONTRACTOR_REFERENCE_TERMS.search(lowered):
        selected = _selected_source(thread_id)
        contractor = str(selected.get("contractor") or "").strip() if isinstance(selected, dict) else ""
        previous = get_thread_result(thread_id)
        previous_filters = previous.get("filters", {})
        merged_filters = {
            key: str(value)
            for key, value in previous_filters.items()
            if isinstance(value, str)
        }
        if contractor:
            merged_filters["contractor"] = contractor
        if contractor:
            intent = (
                "browse"
                if DETAIL_FIELD_TERMS.search(user_message)
                else "availability"
                if AVAILABILITY_TERMS.search(user_message)
                else "browse"
            )
            return QueryPlan(
                intent=intent,
                filters=merged_filters,
                exclude_selected_contract=True,
                is_follow_up=True,
            )
        return QueryPlan(intent="clarify", subject="Which contractor are you referring to?")

    subject = _strip_subject_text(
        user_message,
        [
            filters.get("region", ""),
            filters.get("province", ""),
            filters.get("status", ""),
            filters.get("contractor", ""),
            filters.get("infra_year", ""),
            filters.get("infra_year_start", ""),
            filters.get("infra_year_end", ""),
            find_lookup_contract_id(user_message) or "",
        ],
    )
    subject_for_search = _strip_subject_text(
        user_message,
        [
            filters.get("region", ""),
            filters.get("province", ""),
            filters.get("status", ""),
            filters.get("contractor", ""),
            filters.get("infra_year", ""),
            filters.get("infra_year_start", ""),
            filters.get("infra_year_end", ""),
            find_lookup_contract_id(user_message) or "",
        ],
    )

    if anchor_plan.lookup_value:
        return QueryPlan(intent="lookup", lookup_value=anchor_plan.lookup_value, filters=filters)
    if BROWSE_TERMS.search(user_message) and not filters and _is_generic_subject(subject):
        return QueryPlan(
            intent="clarify",
            subject="Which region, contractor, category, or status should I narrow this to?",
        )
    if STATS_TERMS.search(user_message):
        if not filters and _is_generic_subject(subject):
            return QueryPlan(
                intent="clarify",
                subject="Which region, contractor, category, or status should I use?",
            )
        return _merge_with_previous(QueryPlan(intent="stats", filters=filters, subject=subject), user_message, thread_id)
    if AVAILABILITY_TERMS.search(user_message) and DETAIL_FIELD_TERMS.search(user_message):
        return _merge_with_previous(
            QueryPlan(intent="browse", filters=filters, subject=subject),
            user_message,
            thread_id,
        )
    if AVAILABILITY_TERMS.search(user_message):
        if not filters and _is_generic_subject(subject):
            return QueryPlan(
                intent="clarify",
                subject="Which region, contractor, category, or status should I narrow this to?",
            )
        return _merge_with_previous(QueryPlan(intent="availability", filters=filters, subject=subject), user_message, thread_id)
    if BROWSE_TERMS.search(user_message):
        return _merge_with_previous(QueryPlan(intent="browse", filters=filters), user_message, thread_id)
    if filters:
        previous_payload = get_thread_plan(thread_id)
        previous_intent = str(previous_payload.get("intent") or "")
        if not previous_intent:
            previous_result = get_thread_result(thread_id)
            previous_intent = str(previous_result.get("intent") or "")
        if (
            FOLLOW_UP_TERMS.search(stripped)
            and previous_intent in {"browse", "availability", "stats"}
            and _is_scope_only_follow_up_subject(subject_for_search, filters)
        ):
            return _merge_with_previous(
                QueryPlan(intent=previous_intent, filters=filters),
                user_message,
                thread_id,
            )
        if not _is_generic_subject(subject_for_search):
            return _merge_with_previous(
                QueryPlan(intent="search", filters=filters, subject=subject_for_search),
                user_message,
                thread_id,
            )
        return _merge_with_previous(QueryPlan(intent="browse", filters=filters), user_message, thread_id)
    if has_domain_terms(user_message):
        if _is_generic_subject(subject):
            return QueryPlan(
                intent="clarify",
                subject="Which region, contractor, category, or status should I narrow this to?",
            )
        return _merge_with_previous(QueryPlan(intent="search", filters=filters, subject=subject or stripped), user_message, thread_id)
    return QueryPlan(intent="chat", subject=stripped)


def plan_with_llm(user_message: str, thread_id: str | None) -> QueryPlan:
    anchors = extract_anchor_filters(user_message)
    try:
        context = compact_thread_context(thread_id)
    except Exception:
        context = (
            "result_kind: none\n"
            "result_count: 0\n"
            "active_filters: {}\n"
            "displayed_contracts: []\n"
            "selected_contract: null\n"
            "last_intent: none"
        )
    planner_model = os.environ.get("GROQ_PLANNER_MODEL") or os.environ.get("GROQ_MODEL")
    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=planner_model,
            temperature=0.0,
            max_tokens=int(os.environ.get("GROQ_PLANNER_MAX_TOKENS", "350")),
            top_p=1.0,
            streaming=False,
            max_retries=2,
            timeout=30,
        )
        response = llm.invoke(
            [
                ("system", PLANNER_SYSTEM_PROMPT),
                (
                    "user",
                    f"{context}\n\nUSER_MESSAGE:\n{user_message}\n\nReturn JSON only.",
                ),
            ]
        )
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            )
        payload = _extract_json(str(content))
        plan = _plan_from_payload(payload, anchors)
        if not plan.lookup_value:
            plan.lookup_value = find_lookup_contract_id(user_message) or plan.lookup_value
        normalized_plan = _normalize_awarded_to_filters(plan, user_message)
        if normalized_plan.intent == "availability" and DETAIL_FIELD_TERMS.search(user_message):
            normalized_plan.intent = "browse"
        return _merge_with_previous(normalized_plan, user_message, thread_id)
    except Exception:
        return _normalize_awarded_to_filters(_fallback_plan(user_message, thread_id), user_message)


def plan_message(user_message: str, thread_id: str | None) -> QueryPlan:
    if _needs_history_recovery(user_message, thread_id):
        _recover_history_context(thread_id, user_message)
    anchor_plan = build_anchor_plan(user_message)
    if anchor_plan.lookup_value:
        return QueryPlan(
            intent="lookup",
            lookup_value=anchor_plan.lookup_value,
            filters=anchor_plan.filters,
            has_location_phrase=anchor_plan.has_location_phrase,
            has_unresolved_location_hint=anchor_plan.has_unresolved_location_hint,
        )
    if is_greeting(user_message) and not has_domain_terms(user_message):
        return QueryPlan(intent="chat", subject=user_message.strip())
    if PROXIMITY_PATTERN.search(user_message):
        return QueryPlan(intent="proximity", subject=user_message.strip())
    return plan_with_llm(user_message, thread_id)
