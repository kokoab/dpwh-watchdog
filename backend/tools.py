import json
import importlib
import os
import re
import re as _re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from embeddings import LocalAPIEmbeddings
from filter_parser import FUZZY_FIELDS, parse_filter_string
from hybrid_search import hybrid_search, structured_match_count, structured_match_ids
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from lookup_parser import parse_lookup_string
from query_planner import QueryPlan
from query_scope import (
    get_current_thread_id,
    get_thread_plan,
    get_thread_result,
    set_thread_result,
)
from reranker import rerank
from stats_parser import parse_stats_filters

web_search = DuckDuckGoSearchRun()
embedding = LocalAPIEmbeddings()

PG_DSN: str = os.environ.get("PG_DSN") or (
    f"host={os.environ.get('POSTGRES_HOST')} "
    f"port={os.environ.get('POSTGRES_PORT')} "
    f"dbname={os.environ.get('POSTGRES_DB')} "
    f"user={os.environ.get('POSTGRES_USER')} "
    f"password={os.environ.get('POSTGRES_PASSWORD')}"
)

FILTER_MATCH_LIMIT = 10
RESULT_STATE_ID_CAP = 100


def _psycopg2():
    return importlib.import_module("psycopg2")


def _psycopg2_extras():
    return importlib.import_module("psycopg2.extras")


def _legacy_filter_query(filters: dict[str, str], *, limit: int | None = None) -> str:
    clauses = [f"{key}={value}" for key, value in filters.items() if value]
    query = "Filter contracts where " + " AND ".join(clauses)
    if limit:
        query += f" LIMIT {limit}"
    return query


def _legacy_stats_query(filters: dict[str, str], *, availability: bool = False) -> str:
    clauses = [f"{key}={value}" for key, value in filters.items() if value]
    prefix = "Check availability where" if availability else "Calculate metrics where"
    return f"{prefix} {' AND '.join(clauses) if clauses else 'all=true'}"


def _legacy_search_query(plan: QueryPlan) -> str:
    if plan.filters:
        clauses = [f"{key}={value}" for key, value in plan.filters.items() if value]
        return (
            f"Find all contracts about {plan.subject or 'contracts'} "
            f"where {' AND '.join(clauses)}"
        ).strip()
    return f"Find all contracts about {plan.subject or 'contracts'}".strip()


def _format_date(value) -> str:
    if value in (None, ""):
        return "N/A"
    if isinstance(value, (date, datetime)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    return str(value)[:10]


def _coerce_float(value) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


def _coerce_date(value):
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return None


def _contract_duration(start_value, completion_value) -> str:
    if not start_value or not completion_value:
        return "N/A"

    start = _coerce_date(start_value)
    completion = _coerce_date(completion_value)

    if not isinstance(start, date) or not isinstance(completion, date):
        return "N/A"

    delta_days = (completion - start).days
    if delta_days < 0:
        return "N/A"
    return f"{delta_days} day(s)"


def _build_stats_scope(
    region: Optional[str],
    province: Optional[str],
    infra_year: Optional[str],
    infra_year_start: Optional[str],
    infra_year_end: Optional[str],
    status: Optional[str],
    category_keyword: Optional[str],
    contractor: Optional[str],
) -> str:
    scope_parts = []
    if region:
        scope_parts.append(f"Region: {region}")
    if province:
        scope_parts.append(f"Province: {province}")
    if infra_year:
        scope_parts.append(f"Year: {infra_year}")
    elif infra_year_start and infra_year_end:
        scope_parts.append(f"Years: {infra_year_start}-{infra_year_end}")
    if status:
        scope_parts.append(f"Status: {status}")
    if category_keyword:
        scope_parts.append(f"Category: {category_keyword}")
    if contractor:
        scope_parts.append(f"Contractor: {contractor}")
    return f"[{' | '.join(scope_parts)}]" if scope_parts else "[Global Scope]"


def _truncate_text(value, limit: int = 220) -> str:
    text = " ".join(str(value or "N/A").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _row_get(row, key: str):
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, TypeError, IndexError):
        return None


def _extract_document_links(raw_json: dict | None) -> dict[str, str]:
    if not isinstance(raw_json, dict):
        return {}

    links = raw_json.get("links")
    if not isinstance(links, dict):
        return {}

    return {
        key: str(value).strip()
        for key, value in links.items()
        if isinstance(value, str) and value.strip()
    }


def _current_thread_id() -> str | None:
    return get_current_thread_id()


def _record_result_state(payload: dict[str, object]) -> None:
    thread_id = _current_thread_id()
    if not thread_id:
        return
    set_thread_result(thread_id, payload)


def _record_empty_contract_detail_state() -> None:
    _record_result_state(
        {
            "result_kind": "contract_detail",
            "intent": "lookup",
            "count": 0,
            "contract_ids": [],
            "displayed_contract_ids": [],
            "displayed_sources": [],
            "is_complete_result_set": True,
        }
    )


def _normalize_result_filters(filters: dict[str, object]) -> dict[str, str]:
    return {
        key: str(value).strip()
        for key, value in filters.items()
        if isinstance(value, str) and value.strip()
    }


def _format_filter_phrase(filters: dict[str, str]) -> str:
    category = filters.get("category")
    province = filters.get("province")
    region = filters.get("region")
    status = filters.get("status")
    contractor = filters.get("contractor")
    infra_year = filters.get("infra_year")
    infra_year_start = filters.get("infra_year_start")
    infra_year_end = filters.get("infra_year_end")
    program = filters.get("program_name")

    subject = f"{category} projects" if category else "contracts"
    if status:
        subject = f"{status} {subject}"

    parts = [subject]
    if province:
        parts.append(f"in {province}")
    elif region:
        parts.append(f"in {region}")
    if contractor:
        parts.append(f"by {contractor}")
    if infra_year:
        parts.append(f"from {infra_year}")
    elif infra_year_start and infra_year_end:
        parts.append(f"from {infra_year_start} to {infra_year_end}")
    if program:
        parts.append(f"under {program}")

    return " ".join(parts) if parts else "the selected filters"


@tool
def ask_clarifying_question(query: str) -> str:
    """
    Use this tool when the user's contract request is broad or underspecified.
    It returns a short, user-friendly clarifying question instead of guessing.
    """

    normalized = " ".join(str(query or "").split()).lower()
    if (
        "same contractor" in normalized
        or "this contractor" in normalized
        or "that contractor" in normalized
        or "the contractor" in normalized
    ):
        return "Which contractor are you referring to?"
    if "detail" in normalized or "lookup" in normalized:
        return "Which contract or project should I look up?"
    if "how many" in normalized or "count" in normalized or "metric" in normalized or "statistics" in normalized:
        return "Which region, contractor, category, or status should I use?"
    return "Which region, contractor, category, or status should I narrow this to?"


def execute_lookup_plan(plan: QueryPlan) -> str:
    return _get_contract_detail_from_lookup_value(plan.lookup_value)


def execute_browse_plan(plan: QueryPlan) -> str:
    return _filter_contracts_from_filters(plan.filters, limit=plan.limit)


def execute_stats_plan(plan: QueryPlan) -> tuple[str, dict[str, object]]:
    payload = _compute_stats_payload(plan.filters, is_availability_query=False)
    return _format_stats_text(payload), payload


def execute_availability_plan(plan: QueryPlan) -> str:
    return _get_contract_statistics_from_filters(plan.filters, is_availability_query=True)


def execute_search_plan(plan: QueryPlan) -> str:
    return search_contracts(_legacy_search_query(plan))


def execute_clarify_plan(plan: QueryPlan) -> str:
    if plan.subject.strip():
        return plan.subject.strip()
    return ask_clarifying_question("clarify")


def _resolve_result_context(
    fallback_intent: str,
    fallback_filters: dict[str, object],
    fallback_subject: str = "",
) -> tuple[str, dict[str, str], str]:
    thread_id = _current_thread_id()
    plan_payload = get_thread_plan(thread_id) if thread_id else {}
    plan_filters = _normalize_result_filters(plan_payload.get("filters", {}))
    plan_subject = str(plan_payload.get("subject", "") or "")
    plan_intent = str(plan_payload.get("intent", "") or "")

    return (
        plan_intent or fallback_intent,
        plan_filters or _normalize_result_filters(fallback_filters),
        plan_subject or fallback_subject,
    )


def _source_value_for_filter(source: dict[str, object], field: str) -> str:
    field_map = {
        "contractor": "contractor",
        "region": "region",
        "province": "province",
        "status": "status",
        "category": "category",
        "program_name": "programName",
        "infra_year": "infraYear",
    }
    value = source.get(field_map.get(field, field))
    return str(value or "").strip().lower()


def _source_matches_filters(source: dict[str, object], filters: dict[str, str]) -> bool:
    for field, expected in filters.items():
        if field in {"infra_year_start", "infra_year_end"}:
            source_year_text = _source_value_for_filter(source, "infra_year")
            expected_text = str(expected or "").strip()
            if not source_year_text or not expected_text:
                return False
            try:
                source_year = int(source_year_text)
                expected_year = int(expected_text)
            except ValueError:
                return False
            if field == "infra_year_start" and source_year < expected_year:
                return False
            if field == "infra_year_end" and source_year > expected_year:
                return False
            continue
        source_value = _source_value_for_filter(source, field)
        expected_value = str(expected or "").strip().lower()
        if not source_value or not expected_value:
            return False
        if field in {"category", "contractor", "region", "province", "program_name"}:
            if expected_value not in source_value and source_value not in expected_value:
                return False
        else:
            if source_value != expected_value:
                return False
    return True


def _get_selected_contract_source(result_state: dict[str, object] | None) -> dict[str, object] | None:
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
            if contract_id == selected_id:
                return source

    if len(displayed_sources) == 1 and isinstance(displayed_sources[0], dict):
        return displayed_sources[0]

    if result_state.get("result_kind") == "contract_detail" and isinstance(displayed_sources[0], dict):
        return displayed_sources[0]

    return None


def _should_exclude_selected_contract() -> tuple[bool, dict[str, object] | None, str | None]:
    thread_id = _current_thread_id()
    if not thread_id:
        return False, None, None

    plan = get_thread_plan(thread_id)
    exclude_selected = bool(plan.get("exclude_selected_contract"))
    if not exclude_selected:
        return False, None, None

    result_state = get_thread_result(thread_id)
    selected_source = _get_selected_contract_source(result_state)
    selected_contract_id = None
    if isinstance(selected_source, dict):
        selected_contract_id = str(selected_source.get("contractId") or "").strip() or None

    return True, selected_source, selected_contract_id


def _build_contract_where_clause(filters: dict[str, str]) -> tuple[str, list[object]]:
    conditions = []
    params: list[object] = []

    for field, value in filters.items():
        if field == "infra_year_start":
            conditions.append("infra_year >= %s")
            params.append(value)
            continue

        if field == "infra_year_end":
            conditions.append("infra_year <= %s")
            params.append(value)
            continue

        if field == "category":
            conditions.append("(description ILIKE %s OR category ILIKE %s)")
            params.append(f"%{value}%")
            params.append(f"%{value}%")
            continue

        if field in FUZZY_FIELDS:
            conditions.append(f"{field} ILIKE %s")
            params.append(f"%{value}%")
        else:
            conditions.append(f"{field} = %s")
            params.append(value)

    return " AND ".join(conditions), params


def _fetch_contract_rows(
    filters: dict[str, str],
    *,
    limit: int,
    count_only: bool = False,
) -> tuple[int, list[dict]]:
    where_clause, params = _build_contract_where_clause(filters)
    if not where_clause:
        return 0, []

    conn = _psycopg2().connect(PG_DSN)
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().DictCursor) as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM contracts WHERE {where_clause}",
                params,
            )
            total_count = int(cur.fetchone()[0])

            if count_only:
                return total_count, []

            cur.execute(
                f"""
                SELECT
                    contract_id, description, category, status,
                    budget, amount_paid, progress, region,
                    province, contractor, infra_year, program_name,
                    completion_date
                FROM contracts
                WHERE {where_clause}
                ORDER BY contract_id ASC
                LIMIT %s;
                """,
                params + [limit],
            )
            rows = [dict(row) for row in cur.fetchall()]
        return total_count, rows
    finally:
        conn.close()


def _summarize_sources(rows: list[dict]) -> list[dict[str, object]]:
    return [
        {
            "description": row["description"],
            "contractId": row["contract_id"],
            "contractor": row["contractor"],
            "region": row["region"],
            "province": row["province"],
            "budget": float(row["budget"]) if row["budget"] else 0.0,
            "progress": row["progress"],
            "status": row["status"],
            "category": row["category"],
            "infraYear": row["infra_year"],
            "programName": row["program_name"],
            "completionDate": str(row.get("completion_date") or "")[:10] or None,
        }
        for row in rows
    ]


def _summarize_stats_contract_sources(rows: list[dict]) -> list[dict[str, object]]:
    return [
        {
            "description": row.get("description"),
            "contractId": row.get("contract_id"),
            "contractor": row.get("contractor"),
            "region": row.get("region"),
            "province": row.get("province"),
            "budget": _coerce_float(row.get("budget")),
            "progress": row.get("progress"),
            "status": row.get("status"),
            "category": row.get("category"),
            "infraYear": row.get("infra_year"),
            "programName": row.get("program_name"),
            "completionDate": str(row.get("completion_date") or "")[:10] or None,
        }
        for row in rows
    ]


def _build_contract_detail_component_payload(component_rows: list[dict]) -> list[dict[str, object]]:
    payload = []
    for row in component_rows:
        raw_json = _row_get(row, "raw_json")
        payload.append(
            {
                "componentOrder": _row_get(row, "component_order"),
                "componentId": _row_get(row, "component_id"),
                "description": _row_get(row, "description"),
                "typeOfWork": _row_get(row, "type_of_work"),
                "infraType": _row_get(row, "infra_type"),
                "region": _row_get(row, "region"),
                "province": _row_get(row, "province"),
                "latitude": _row_get(row, "latitude"),
                "longitude": _row_get(row, "longitude"),
                "rawJson": raw_json if isinstance(raw_json, dict) else raw_json,
            }
        )
    return payload


def _build_contract_detail_source(r, component_rows: list[dict]) -> dict[str, object]:
    budget = _coerce_float(r["budget"])
    amount_paid = _coerce_float(r["amount_paid"])
    award_amount = _coerce_float(r["award_amount"])
    award_to_budget_ratio = (
        (award_amount / budget * 100) if budget > 0 and award_amount > 0 else None
    )
    document_links = _extract_document_links(r.get("raw_json"))
    components = _build_contract_detail_component_payload(component_rows)
    db_fields = {
        "contractId": r["contract_id"],
        "description": r["description"],
        "category": r["category"],
        "status": r["status"],
        "budget": budget,
        "amountPaid": amount_paid,
        "awardAmount": award_amount,
        "awardToBudgetRatio": award_to_budget_ratio,
        "progress": r["progress"],
        "region": r["region"],
        "province": r["province"],
        "latitude": r["latitude"],
        "longitude": r["longitude"],
        "contractor": r["contractor"],
        "advertisementDate": _format_date(r["advertisement_date"]),
        "expiryDate": _format_date(r["expiry_date"]),
        "bidSubmissionDeadline": _format_date(r["bid_submission_deadline"]),
        "startDate": _format_date(r["start_date"]),
        "completionDate": _format_date(r["completion_date"]),
        "infraYear": r["infra_year"],
        "programName": r["program_name"],
        "sourceOfFunds": r["source_of_funds"],
        "contractDuration": _contract_duration(r["start_date"], r["completion_date"]),
    }
    raw_json = r.get("raw_json") if isinstance(r.get("raw_json"), dict) else {}

    return {
        "description": r["description"],
        "contractId": r["contract_id"],
        "contractor": r["contractor"],
        "region": r["region"],
        "province": r["province"],
        "budget": budget,
        "amountPaid": amount_paid,
        "awardAmount": award_amount,
        "awardToBudgetRatio": award_to_budget_ratio,
        "progress": r["progress"],
        "status": r["status"],
        "category": r["category"],
        "infraYear": r["infra_year"],
        "programName": r["program_name"],
        "sourceOfFunds": r["source_of_funds"],
        "advertisementDate": _format_date(r["advertisement_date"]),
        "expiryDate": _format_date(r["expiry_date"]),
        "bidSubmissionDeadline": _format_date(r["bid_submission_deadline"]),
        "startDate": _format_date(r["start_date"]),
        "completionDate": _format_date(r["completion_date"]),
        "contractDuration": _contract_duration(r["start_date"], r["completion_date"]),
        "documentLinks": document_links,
        "components": components,
        "dbFields": db_fields,
        "rawJson": raw_json,
    }


def _format_contract_source_row(row) -> str:
    budget = _coerce_float(_row_get(row, "budget"))
    progress = _row_get(row, "progress")
    progress_text = f"{progress}%" if progress not in (None, "") else "N/A"
    return (
        f"[{_row_get(row, 'contract_id') or 'N/A'}] "
        f"{_truncate_text(_row_get(row, 'description'))}\n"
        f"  Status: {_row_get(row, 'status') or 'N/A'} | "
        f"Budget: PHP {budget:,.0f} | "
        f"Region: {_row_get(row, 'region') or 'N/A'}, "
        f"{_row_get(row, 'province') or 'N/A'}\n"
        f"  Contractor: {_truncate_text(_row_get(row, 'contractor'), 140)} | "
        f"Progress: {progress_text}"
    )


def _exclude_selected_contract_rows(
    rows: list[dict],
    selected_contract_id: str | None,
) -> list[dict]:
    if not selected_contract_id:
        return rows
    return [
        row
        for row in rows
        if str(row.get("contract_id") or "").strip() != selected_contract_id
    ]


def _load_local_contract_record(contract_id: str):
    """
    Fallback for exact lookups when the normalized DB row is missing but the
    source contract JSON is present locally. This keeps ID lookups deterministic
    even when the ingested table is incomplete.
    """

    local_path = Path(__file__).parent / "data" / f"{contract_id}.json"
    if not local_path.exists():
        return None, []

    try:
        with open(local_path) as f:
            payload = json.load(f)
    except Exception:
        return None, []

    data = payload.get("data") or {}
    location = data.get("location") or {}
    coordinates = location.get("coordinates") or {}

    record = {
        "contract_id": data.get("contractId") or contract_id,
        "description": data.get("description"),
        "category": data.get("category"),
        "status": data.get("status"),
        "budget": data.get("budget"),
        "amount_paid": data.get("amountPaid"),
        "award_amount": data.get("procurement", {}).get("awardAmount"),
        "progress": data.get("progress"),
        "region": location.get("region"),
        "province": location.get("province"),
        "latitude": coordinates.get("latitude", data.get("latitude")),
        "longitude": coordinates.get("longitude", data.get("longitude")),
        "contractor": data.get("contractor"),
        "advertisement_date": data.get("procurement", {}).get("advertisementDate"),
        "expiry_date": data.get("expiryDate"),
        "bid_submission_deadline": data.get("procurement", {}).get(
            "bidSubmissionDeadline"
        ),
        "start_date": data.get("startDate"),
        "completion_date": data.get("completionDate"),
        "infra_year": data.get("infraYear"),
        "program_name": data.get("programName"),
        "source_of_funds": data.get("sourceOfFunds"),
        "raw_json": data,
    }

    component_rows = []
    for index, component in enumerate(data.get("components") or [], start=1):
        comp_coords = component.get("coordinates") or {}
        component_rows.append(
            {
                "component_order": index,
                "component_id": component.get("componentId"),
                "description": component.get("description"),
                "type_of_work": component.get("typeOfWork"),
                "infra_type": component.get("infraType"),
                "region": component.get("region"),
                "province": component.get("province"),
                "latitude": comp_coords.get("latitude", component.get("latitude")),
                "longitude": comp_coords.get("longitude", component.get("longitude")),
                "raw_json": component,
            }
        )

    return record, component_rows


def load_contract_detail_sources(contract_ids: list[str]) -> list[dict[str, object]]:
    detail_sources: list[dict[str, object]] = []

    for contract_id in contract_ids:
        normalized_id = str(contract_id or "").strip()
        if not normalized_id:
            continue

        conn = None
        try:
            conn = _psycopg2().connect(PG_DSN)
            with conn.cursor(cursor_factory=_psycopg2_extras().DictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        contract_id, description, category, status,
                        budget, amount_paid, award_amount, progress,
                        region, province, latitude, longitude, contractor,
                        advertisement_date, expiry_date, bid_submission_deadline,
                        start_date, completion_date, infra_year,
                        program_name, source_of_funds, raw_json
                    FROM contracts
                    WHERE contract_id = %s
                    LIMIT 1;
                    """,
                    (normalized_id,),
                )
                row = cur.fetchone()

            if row:
                component_rows = []
                try:
                    with conn.cursor(
                        cursor_factory=_psycopg2_extras().DictCursor
                    ) as comp_cur:
                        comp_cur.execute(
                            """
                            SELECT
                                component_order, component_id, description, type_of_work,
                                infra_type, region, province, latitude, longitude, raw_json
                            FROM contract_components
                            WHERE contract_id = %s
                            ORDER BY component_order ASC, id ASC;
                            """,
                            (normalized_id,),
                        )
                        component_rows = comp_cur.fetchall()
                except Exception as e:
                    print(f"load_contract_detail_sources component lookup error: {e}")

                detail_sources.append(_build_contract_detail_source(row, component_rows))
                continue
        except Exception as e:
            print(f"load_contract_detail_sources DB error: {e}")
        finally:
            if conn is not None:
                conn.close()

        local_record, component_rows = _load_local_contract_record(normalized_id)
        if local_record:
            detail_sources.append(
                _build_contract_detail_source(local_record, component_rows)
            )

    return detail_sources


def _format_contract_lookup_output(
    r, component_rows, value: str, lookup_type: str
) -> str:
    budget = _coerce_float(r["budget"])
    award_amount = _coerce_float(r["award_amount"])
    award_to_budget_ratio = (
        (award_amount / budget * 100) if budget > 0 and award_amount > 0 else None
    )
    award_amount_text = f"PHP {award_amount:,.2f}" if award_amount > 0 else "N/A"
    award_ratio_text = (
        f"{award_to_budget_ratio:.1f}%" if award_to_budget_ratio is not None else "N/A"
    )
    contract_duration = _contract_duration(r["start_date"], r["completion_date"])
    document_links = _extract_document_links(r.get("raw_json"))
    detail_source = _build_contract_detail_source(r, component_rows)

    SOURCE_MARKER = "__SOURCES__"
    sources = [detail_source]

    _record_result_state(
        {
            "result_kind": "contract_detail",
            "intent": "lookup",
            "count": 1,
            "contract_ids": [r["contract_id"]],
            "displayed_contract_ids": [r["contract_id"]],
            "displayed_sources": sources,
            "is_complete_result_set": True,
        }
    )

    detail_block = (
        f"CONTRACT DETAIL RECORD\n"
        f"{'=' * 40}\n"
        f"Description:        {r['description'] or 'N/A'}\n"
        f"Contract ID:        {r['contract_id'] or 'N/A'}\n"
        f"Category:           {r['category'] or 'N/A'}\n"
        f"Status:             {r['status'] or 'N/A'}\n"
        f"Contractor:         {r['contractor'] or 'N/A'}\n"
        f"Region:             {r['region'] or 'N/A'}\n"
        f"Province:           {r['province'] or 'N/A'}\n"
        f"Budget:             PHP {budget:,.2f}\n"
        f"Award Amount:       {award_amount_text}\n"
        f"Award-to-Budget Ratio: {award_ratio_text}\n"
        f"Progress:           {r['progress'] or 'N/A'}%\n"
        f"Infra Year:         {r['infra_year'] or 'N/A'}\n"
        f"Program:            {r['program_name'] or 'N/A'}\n"
        f"Source of Funds:    {r['source_of_funds'] or 'N/A'}\n"
        f"Advertisement Date: {_format_date(r['advertisement_date'])}\n"
        f"Bid Submission Deadline: {_format_date(r['bid_submission_deadline'])}\n"
        f"Start Date:         {_format_date(r['start_date'])}\n"
        f"Completion Date:    {_format_date(r['completion_date'])}\n"
        f"Expiry Date:        {_format_date(r['expiry_date'])}\n"
        f"Contract Duration:  {contract_duration}\n"
    )

    if document_links:
        detail_block += (
            "\nDOCUMENT LINKS\n"
            f"{'=' * 40}\n"
            + "\n".join(
                f"{key}: {url}"
                for key, url in document_links.items()
            )
            + "\n"
        )
    else:
        detail_block += (
            "\nDOCUMENT LINKS\n"
            f"{'=' * 40}\n"
            "The database does not have document links for this contract yet.\n"
        )

    if component_rows:
        detail_block += (
            "\nCONTRACT COMPONENTS\n"
            f"{'=' * 40}\n"
            + "\n".join(
                (
                    f"[{row['component_order']}] {row['component_id'] or 'N/A'}\n"
                    f"Type of Work: {row['type_of_work'] or 'N/A'}\n"
                    f"Description: {row['description'] or 'N/A'}\n"
                    f"Infra Type: {row['infra_type'] or 'N/A'}\n"
                    f"Region: {row['region'] or 'N/A'}\n"
                    f"Province: {row['province'] or 'N/A'}\n"
                    f"Latitude: {row['latitude'] if row['latitude'] is not None else 'N/A'}\n"
                    f"Longitude: {row['longitude'] if row['longitude'] is not None else 'N/A'}\n"
                )
                for row in component_rows
            )
        )

    header = (
        f"Direct lookup result for '{value}' "
        f"({'exact ID match' if lookup_type == 'id' else 'name match'}):\n\n"
    )

    sources_block = f"\n\n{SOURCE_MARKER}{json.dumps(sources)}"
    return header + detail_block + sources_block


@tool
def search_contracts(query: str) -> str:
    """
    Use this tool ONLY when the incoming query starts with 'Find all contracts about'.
    This performs hybrid semantic + keyword search for descriptive project concepts.
    """

    result_intent, result_filters, result_subject = _resolve_result_context(
        "search",
        {},
        "",
    )
    structured_total = structured_match_count(query)
    structured_ids = structured_match_ids(query)
    if structured_total == 0:
        _record_result_state(
            {
                "result_kind": "contract_set",
                "intent": result_intent,
                "filters": result_filters,
                "subject": result_subject,
                "count": 0,
                "contract_ids": [],
                "displayed_contract_ids": [],
                "displayed_sources": [],
                "is_complete_result_set": True,
            }
        )
        return (
            "No matching DPWH contracts found for the structured filters in this query. "
            "Try broadening the location, category, status, or contractor terms."
        )

    try:
        query_vector = embedding.embed_query(query)
    except Exception as e:
        return f"Error: Could not embed query for vector search: {e}"

    # --- Stage 1a: Vector search (wide net) ---
    vector_candidates = []
    try:
        conn = _psycopg2().connect(PG_DSN)
        with conn.cursor(cursor_factory=_psycopg2_extras().DictCursor) as cur:
            cur.execute(
                """
                SELECT
                    c.contract_id, c.description, c.category, c.status,
                    c.budget, c.progress, c.region,
                    c.province, c.contractor, c.infra_year, c.program_name,
                    e.chunk_text
                FROM contract_embeddings e
                JOIN contracts c ON e.contract_id = c.contract_id
                ORDER BY e.embedding <=> %s::vector
                LIMIT 25;
                """,
                (query_vector,),
            )
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f"Error: Database during similarity search: {e}"

    for r in rows:
        vector_candidates.append(
            {
                "chunk_text": r["chunk_text"],
                "contract_id": r["contract_id"],
                "description": r["description"],
                "category": r["category"],
                "status": r["status"],
                "budget": float(r["budget"]) if r["budget"] else 0.0,
                "progress": r["progress"],
                "region": r["region"],
                "province": r["province"],
                "contractor": r["contractor"],
                "infra_year": r["infra_year"],
                "program_name": r["program_name"],
            }
        )

    # --- Stage 1b: BM25 search + RRF merge ---
    merged_candidates = hybrid_search(query, vector_candidates)

    # --- Deduplicate by contract_id (keep first occurrence = highest RRF score) ---
    seen_ids = set()
    unique_candidates = []
    for c in merged_candidates:
        if c["contract_id"] not in seen_ids:
            seen_ids.add(c["contract_id"])
            unique_candidates.append(c)

    if not unique_candidates:
        return "No relevant contracts found in the database"

    if structured_ids is not None:
        unique_candidates = [
            candidate
            for candidate in unique_candidates
            if candidate["contract_id"] in structured_ids
        ]
        if not unique_candidates:
            return (
                "No matching DPWH contracts found for the structured filters in this query. "
                "Try broadening the location, category, status, or contractor terms."
            )

    reranked = rerank(query, unique_candidates, top_k=10)
    exclude_selected_contract, selected_source, selected_contract_id = _should_exclude_selected_contract()
    if exclude_selected_contract and selected_contract_id:
        reranked = [
            candidate
            for candidate in reranked
            if str(candidate.get("contract_id") or "").strip() != selected_contract_id
        ]
        if not reranked:
            contractor_name = (
                str(selected_source.get("contractor") or "").strip()
                if isinstance(selected_source, dict)
                else ""
            )
            if contractor_name:
                return f"No other projects were found for contractor {contractor_name}."
            return "No relevant contracts found in the database"

    SOURCE_MARKER = "__SOURCES__"
    sources = []
    source_rows = []

    for r in reranked:
        sources.append(
            {
                "description": r["description"],
                "contractId": r["contract_id"],
                "contractor": r["contractor"],
                "region": r["region"],
                "province": r["province"],
                "budget": r["budget"],
                "progress": r["progress"],
                "status": r["status"],
                "category": r["category"],
                "infraYear": r["infra_year"],
                "programName": r["program_name"],
            }
        )
        source_rows.append(_format_contract_source_row(r))

    contract_ids = [row["contract_id"] for row in reranked]
    recorded_ids = contract_ids[:RESULT_STATE_ID_CAP]
    result_count = structured_total if structured_total is not None else len(contract_ids)
    if exclude_selected_contract and selected_contract_id and result_count > 0:
        result_count = max(result_count - 1, 0)
    _record_result_state(
        {
            "result_kind": "contract_set",
            "intent": result_intent,
            "filters": result_filters,
            "subject": result_subject,
            "count": result_count,
            "contract_ids": recorded_ids,
            "displayed_contract_ids": contract_ids,
            "displayed_sources": sources,
            "is_complete_result_set": structured_total is not None,
        }
    )

    sources_block = f"\n\n{SOURCE_MARKER}{json.dumps(sources)}"
    if structured_total is not None:
        result_scope = (
            f"Showing top {len(reranked)} of {structured_total:,} matching DPWH contracts."
        )
    else:
        result_scope = (
            f"Showing top {len(reranked)} candidate DPWH contracts. "
            "No reliable structured total count is available for this semantic query."
        )

    content = (
        f"{result_scope}\n\n"
        "Source rows (search-ranked; discuss these contracts directly and do not infer extra analytics):\n\n"
        + "\n\n".join(source_rows)
    )

    return (
        "Here are relevant source rows\n\n"
        + content
        + "\n\nSources:\n"
        + "\n".join(
            f"- {s['description']} | {s['contractId']} | "
            f"{s['contractor']} | {s['region']} | {s['province']}"
            for s in sources
        )
        + sources_block
    )


def _empty_stats_payload(
    filters: dict[str, str],
    *,
    is_availability_query: bool,
    error: str | None = None,
) -> dict[str, object]:
    scope = _build_stats_scope(
        filters.get("region"),
        filters.get("province"),
        filters.get("infra_year"),
        filters.get("infra_year_start"),
        filters.get("infra_year_end"),
        filters.get("status"),
        filters.get("category"),
        filters.get("contractor"),
    )
    payload: dict[str, object] = {
        "total_contracts": 0,
        "total_budget": 0.0,
        "total_award_amount": 0.0,
        "avg_progress": 0.0,
        "award_to_budget_ratio": None,
        "status_breakdown": [],
        "region_breakdown": [],
        "province_breakdown": [],
        "applied_filters": dict(filters),
        "scope_label": scope,
        "is_availability_query": is_availability_query,
    }
    if error:
        payload["error"] = error
    return payload


def _compute_stats_payload(
    filters: dict[str, str],
    is_availability_query: bool,
) -> dict[str, object]:
    params = parse_stats_filters(filters)
    fallback_filters = _normalize_result_filters(
        {
            "region": params["region"],
            "province": params["province"],
            "infra_year": params.get("infra_year"),
            "infra_year_start": params.get("infra_year_start"),
            "infra_year_end": params.get("infra_year_end"),
            "status": params["status"],
            "category": params["category_keyword"],
            "contractor": params["contractor"],
        }
    )
    result_intent, result_filters, _ = _resolve_result_context(
        "availability" if is_availability_query else "stats",
        fallback_filters,
    )
    params = parse_stats_filters(result_filters)
    region = params["region"]
    province = params["province"]
    infra_year = params.get("infra_year")
    infra_year_start = params.get("infra_year_start")
    infra_year_end = params.get("infra_year_end")
    status = params["status"]
    category_keyword = params["category_keyword"]
    contractor = params["contractor"]
    scope = _build_stats_scope(
        region,
        province,
        infra_year,
        infra_year_start,
        infra_year_end,
        status,
        category_keyword,
        contractor,
    )

    conn = None
    try:
        conn = _psycopg2().connect(PG_DSN)
        with conn.cursor() as cur:
            where_clause_sql, sql_params = _build_contract_where_clause(result_filters)
            where_clause = f" WHERE {where_clause_sql}" if where_clause_sql else ""

            # --- Core aggregates ---
            cur.execute(f"SELECT COUNT(*) FROM contracts{where_clause}", sql_params)
            total_contracts = int(cur.fetchone()[0])

            cur.execute(
                f"SELECT COALESCE(SUM(budget), 0) FROM contracts{where_clause}",
                sql_params,
            )
            total_budget = float(cur.fetchone()[0])

            cur.execute(
                f"SELECT COALESCE(SUM(award_amount), 0) FROM contracts{where_clause}",
                sql_params,
            )
            total_award_amount = float(cur.fetchone()[0])

            cur.execute(
                f"SELECT COALESCE(AVG(progress), 0) FROM contracts{where_clause}",
                sql_params,
            )
            avg_progress = float(cur.fetchone()[0])
            award_to_budget_ratio = (
                (total_award_amount / total_budget * 100)
                if total_budget > 0 and total_award_amount > 0
                else None
            )

            # --- Status breakdown ---
            cur.execute(
                f"""
                SELECT status, COUNT(*) 
                FROM contracts{where_clause}
                GROUP BY status 
                ORDER BY COUNT(*) DESC 
                LIMIT 6
                """,
                sql_params,
            )
            status_rows = cur.fetchall()
            status_breakdown = [
                {"status": row[0] or "Unknown", "count": int(row[1])}
                for row in status_rows
            ]

            # --- Top regions (only meaningful for global/province queries) ---
            if not region:
                cur.execute(
                    f"""
                    SELECT region, COUNT(*) 
                    FROM contracts{where_clause}
                    GROUP BY region 
                    ORDER BY COUNT(*) DESC 
                    LIMIT 5
                    """,
                    sql_params,
                )
                region_rows = cur.fetchall()
                region_breakdown = [
                    {"region": row[0] or "Unknown", "count": int(row[1])}
                    for row in region_rows
                ]
            else:
                region_breakdown = []

            cur.execute(
                f"""
                SELECT province, COUNT(*)
                FROM contracts{where_clause}
                GROUP BY province
                ORDER BY COUNT(*) DESC
                LIMIT 10
                """,
                sql_params,
            )
            province_rows = cur.fetchall()
            province_breakdown = [
                {"province": row[0] or "Unknown", "count": int(row[1])}
                for row in province_rows
            ]

            contract_source_rows = []
            if total_contracts > 0:
                cur.execute(
                    f"""
                    SELECT
                        contract_id, description, budget, province,
                        region, status, contractor, progress,
                        category, infra_year, program_name, completion_date
                    FROM contracts{where_clause}
                    ORDER BY budget DESC
                    LIMIT 20;
                    """,
                    sql_params,
                )
                contract_source_rows = [
                    {
                        "contract_id": row[0],
                        "description": row[1],
                        "budget": row[2],
                        "province": row[3],
                        "region": row[4],
                        "status": row[5],
                        "contractor": row[6],
                        "progress": row[7],
                        "category": row[8],
                        "infra_year": row[9],
                        "program_name": row[10],
                        "completion_date": row[11],
                    }
                    for row in cur.fetchall()
                ]

            result_rows = []
            if where_clause_sql:
                result_limit = min(max(total_contracts, 1), RESULT_STATE_ID_CAP)
                cur.execute(
                    f"""
                    SELECT contract_id
                    FROM contracts{where_clause}
                    ORDER BY contract_id ASC
                    LIMIT %s;
                    """,
                    sql_params + [result_limit],
                )
                result_rows = cur.fetchall()

    except Exception as e:
        print(f"Failed to calculate database statistics: {e}")
        return _empty_stats_payload(
            result_filters if "result_filters" in locals() else {},
            is_availability_query=is_availability_query,
            error="Error: unable to process statistical counts on database tables",
        )
    finally:
        if conn is not None:
            conn.close()

    contract_ids = [
        str(row[0])
        for row in result_rows
        if row and row[0] not in (None, "")
    ]
    contract_rows = [
        {
            "contract_id": row.get("contract_id"),
            "description": row.get("description"),
            "budget": _coerce_float(row.get("budget")),
            "province": row.get("province"),
            "region": row.get("region"),
            "status": row.get("status"),
            "contractor": row.get("contractor"),
            "completion_date": row.get("completion_date"),
        }
        for row in contract_source_rows
    ]
    displayed_sources = _summarize_stats_contract_sources(contract_source_rows)
    displayed_contract_ids = [
        str(row["contract_id"])
        for row in contract_source_rows
        if row.get("contract_id") not in (None, "")
    ]
    _record_result_state(
        {
            "result_kind": "contract_set",
            "intent": result_intent,
            "filters": result_filters,
            "count": int(total_contracts),
            "contract_ids": contract_ids[:RESULT_STATE_ID_CAP],
            "displayed_contract_ids": displayed_contract_ids,
            "displayed_sources": displayed_sources,
            "is_complete_result_set": total_contracts <= RESULT_STATE_ID_CAP,
        }
    )

    return {
        "total_contracts": total_contracts,
        "total_budget": total_budget,
        "total_award_amount": total_award_amount,
        "avg_progress": avg_progress,
        "award_to_budget_ratio": award_to_budget_ratio,
        "status_breakdown": status_breakdown,
        "region_breakdown": region_breakdown,
        "province_breakdown": province_breakdown,
        "applied_filters": dict(result_filters),
        "scope_label": scope,
        "is_availability_query": is_availability_query,
        "contract_rows": contract_rows,
        "has_more_contracts": total_contracts > 20,
    }


def _format_stats_text(payload: dict[str, object]) -> str:
    if payload.get("error"):
        return str(payload["error"])

    total_contracts = int(payload.get("total_contracts") or 0)
    total_budget = float(payload.get("total_budget") or 0.0)
    total_award_amount = float(payload.get("total_award_amount") or 0.0)
    avg_progress = float(payload.get("avg_progress") or 0.0)
    award_to_budget_ratio = payload.get("award_to_budget_ratio")
    status_breakdown = payload.get("status_breakdown")
    region_breakdown = payload.get("region_breakdown")
    province_breakdown = payload.get("province_breakdown")
    scope = str(payload.get("scope_label") or "[Global Scope]")

    if payload.get("is_availability_query"):
        availability = "Yes" if total_contracts > 0 else "No"
        return (
            f"Availability Check {scope}:\n"
            f"- Matching Contracts: {total_contracts:,}\n"
            f"- Available: {availability}\n"
            "- Use a listing request if you want to browse matching rows.\n"
        )

    award_ratio_text = (
        f"{float(award_to_budget_ratio):.1f}%"
        if award_to_budget_ratio is not None
        else "N/A"
    )
    def format_breakdown_lines(breakdown: object, label_key: str) -> str:
        if not isinstance(breakdown, list):
            return ""
        lines = []
        for row in breakdown:
            if not isinstance(row, dict):
                continue
            name = row.get(label_key) or "Unknown"
            count = int(row.get("count") or 0)
            pct = (count / total_contracts * 100) if total_contracts else 0.0
            lines.append(f"  - {name}: {count:,} ({pct:.1f}%)")
        return "\n".join(lines)

    def first_breakdown_contributor(
        breakdown: object, label_key: str
    ) -> tuple[str, int] | None:
        if not isinstance(breakdown, list):
            return None
        for row in breakdown:
            if not isinstance(row, dict):
                continue
            return row.get(label_key) or "Unknown", int(row.get("count") or 0)
        return None

    status_breakdown_text = format_breakdown_lines(status_breakdown, "status")

    output = (
        f"Statistics Summary {scope}:\n"
        f"- Total Contracts Matched: {total_contracts:,}\n"
        f"- Combined Budget: PHP {total_budget:,.2f}\n"
        f"- Total Award Amount: PHP {total_award_amount:,.2f}\n"
        f"- Award-to-Budget Ratio: {award_ratio_text}\n"
        f"- Average Progress: {avg_progress:.1f}%\n"
        f"- Status Breakdown:\n{status_breakdown_text or '  - N/A: 0 (0.0%)'}\n"
    )

    region_breakdown_text = format_breakdown_lines(region_breakdown, "region")
    if region_breakdown_text:
        output += f"- Region Breakdown:\n{region_breakdown_text}\n"

    if total_contracts > 0:
        contributor = first_breakdown_contributor(region_breakdown, "region")
        if contributor is None:
            contributor = first_breakdown_contributor(province_breakdown, "province")
        if contributor is not None:
            contributor_name, contributor_count = contributor
            contributor_pct = contributor_count / total_contracts * 100
            output += (
                f"Top contributor: {contributor_name} with "
                f"{contributor_count:,} contracts "
                f"({contributor_pct:.1f}% of total).\n"
            )

    return output


def _get_contract_statistics_from_filters(
    filters: dict[str, str],
    *,
    is_availability_query: bool,
) -> str:
    payload = _compute_stats_payload(
        filters,
        is_availability_query=is_availability_query,
    )
    return _format_stats_text(payload)


@tool
def get_contract_statistics(query: str) -> str:
    """
    Use this tool ONLY when the incoming query starts with 'Calculate metrics where'
    or 'Check availability where'.
    This tool extracts parameters to run SQL COUNT, SUM, and AVG aggregates.
    Supports filtering by region, province, infra_year, infra_year_start,
    infra_year_end, status, category keyword, and contractor name.
    """

    is_availability_query = query.strip().lower().startswith("check availability where")
    stats_query = (
        query.replace("Check availability where", "Filter contracts where", 1)
        if is_availability_query
        else query.replace("Calculate metrics where", "Filter contracts where", 1)
    )
    stats_query = re.sub(r"\s+LIMIT\s+\d+\s*$", "", stats_query, flags=re.IGNORECASE)
    parsed_filters = parse_filter_string(stats_query)
    parsed_filters.pop("all", None)
    normalized_filters = _normalize_result_filters(parsed_filters)
    return _get_contract_statistics_from_filters(
        normalized_filters,
        is_availability_query=is_availability_query,
    )


def _filter_contracts_from_filters(
    filters: dict[str, str],
    *,
    limit: int | None = None,
) -> str:
    result_intent, filters, _ = _resolve_result_context(
        "browse",
        filters,
    )
    limit = int(limit or FILTER_MATCH_LIMIT)
    limit = max(1, min(limit, FILTER_MATCH_LIMIT))
    exclude_selected_contract, selected_source, selected_contract_id = _should_exclude_selected_contract()

    if not filters:
        return (
            "Error: Could not extract any valid filters from the query. "
            "Valid fields are: contractor, region, province, status, category, "
            "infra_year, infra_year_start, infra_year_end, program_name."
        )

    try:
        fetch_limit = limit + 1 if exclude_selected_contract else limit
        total_count, rows = _fetch_contract_rows(filters, limit=fetch_limit)
    except Exception as e:
        print(f"filter_contracts DB error: {e}")
        return "Error: Database failure during filtered query"

    if exclude_selected_contract and selected_contract_id:
        rows = _exclude_selected_contract_rows(rows, selected_contract_id)
        if selected_source and _source_matches_filters(selected_source, filters):
            total_count = max(total_count - 1, 0)

    if not rows:
        applied = _format_filter_phrase(filters)
        _record_result_state(
            {
                "result_kind": "contract_set",
                "intent": result_intent,
                "filters": filters,
                "count": 0,
                "contract_ids": [],
                "displayed_contract_ids": [],
                "displayed_sources": [],
                "is_complete_result_set": True,
            }
        )
        if exclude_selected_contract and filters.get("contractor"):
            return f"No other projects were found for contractor {filters['contractor']}."
        return f"No contracts found matching filters: {applied}"

    SOURCE_MARKER = "__SOURCES__"
    sources = []
    source_rows = []

    sources = _summarize_sources(rows)
    for r in rows:
        source_rows.append(_format_contract_source_row(r))

    _record_result_state(
        {
            "result_kind": "contract_set",
            "intent": result_intent,
            "filters": filters,
            "count": int(total_count),
            "contract_ids": [row["contract_id"] for row in rows][:RESULT_STATE_ID_CAP],
            "displayed_contract_ids": [row["contract_id"] for row in rows],
            "displayed_sources": sources,
            "is_complete_result_set": total_count <= len(rows),
        }
    )

    applied_filters = _format_filter_phrase(filters)
    header = (
        f"Filtered contracts matching {applied_filters} — "
        f"showing {len(rows)} of {total_count:,} total matches "
        f"(capped match window, not ranked):\n\n"
    )

    sources_block = f"\n\n{SOURCE_MARKER}{json.dumps(sources)}"

    return (
        header
        + "Source rows (ordered by contract ID; discuss these contracts directly and do not infer extra analytics):\n\n"
        + "\n\n".join(source_rows)
        + sources_block
    )


@tool
def filter_contracts(query: str) -> str:
    """
    Use this tool ONLY when the incoming query starts with 'Filter contracts where'.
    This performs structured SQL filtering on known contract attributes like
    contractor, region, province, status, category, infra_year,
    infra_year_start, infra_year_end, and program_name.
    Use this for exact or near-exact attribute lookups, NOT for descriptive searches.
    """

    query_without_limit = re.sub(r"\s+LIMIT\s+\d+\s*$", "", query, flags=re.IGNORECASE)
    filters = parse_filter_string(query_without_limit)
    filters.pop("all", None)
    limit_match = re.search(r"\s+LIMIT\s+(\d+)\s*$", query, re.IGNORECASE)
    limit = int(limit_match.group(1)) if limit_match else FILTER_MATCH_LIMIT
    normalized_filters = _normalize_result_filters(filters)
    return _filter_contracts_from_filters(normalized_filters, limit=limit)


def _get_contract_detail_from_lookup_value(value: str) -> str:
    parsed = parse_lookup_string(f"Lookup contract {value}".strip())
    if not parsed:
        _record_empty_contract_detail_state()
        return (
            "Error: Could not extract a contract ID or project name "
            "from the lookup query."
        )

    lookup_type = parsed["lookup_type"]
    value = parsed["value"]
    conn = None
    try:
        conn = _psycopg2().connect(PG_DSN)
        with conn.cursor(cursor_factory=_psycopg2_extras().DictCursor) as cur:
            if lookup_type == "id":
                cur.execute(
                    """
                    SELECT
                        contract_id, description, category, status,
                        budget, amount_paid, award_amount, progress,
                        region, province, latitude, longitude, contractor,
                        advertisement_date, expiry_date, bid_submission_deadline,
                        start_date, completion_date, infra_year,
                        program_name, source_of_funds, raw_json
                    FROM contracts
                    WHERE contract_id = %s
                    LIMIT 1;
                    """,
                    (value,),
                )

            else:
                # Name lookup — try exact match first, fall back to fuzzy
                cur.execute(
                    """
                    SELECT
                        contract_id, description, category, status,
                        budget, amount_paid, award_amount, progress,
                        region, province, latitude, longitude, contractor,
                        advertisement_date, expiry_date, bid_submission_deadline,
                        start_date, completion_date, infra_year,
                        program_name, source_of_funds, raw_json
                    FROM contracts
                    WHERE description ILIKE %s
                    ORDER BY
                        -- Exact match ranks first
                        CASE WHEN LOWER(description) = LOWER(%s) THEN 0 ELSE 1 END,
                        -- Then closest prefix match
                        LENGTH(description) ASC
                    LIMIT 3;
                    """,
                    (f"%{value}%", value),
                )

            rows = cur.fetchall()

        if rows:
            r = rows[0]
            component_rows = []
            try:
                with conn.cursor(cursor_factory=_psycopg2_extras().DictCursor) as comp_cur:
                    comp_cur.execute(
                        """
                        SELECT
                            component_order, component_id, description, type_of_work,
                            infra_type, region, province, latitude, longitude, raw_json
                        FROM contract_components
                        WHERE contract_id = %s
                        ORDER BY component_order ASC, id ASC;
                        """,
                        (r["contract_id"],),
                    )
                    component_rows = comp_cur.fetchall()
            except Exception as e:
                print(f"get_contract_detail component lookup error: {e}")

            return _format_contract_lookup_output(
                r, component_rows, value=value, lookup_type=lookup_type
            )

        # Exact-ID fallback: the source JSON exists locally even if the ingested
        # contracts table does not contain the row yet.
        if lookup_type == "id":
            local_record, component_rows = _load_local_contract_record(value)
            if local_record:
                return _format_contract_lookup_output(
                    local_record, component_rows, value=value, lookup_type=lookup_type
                )

        # Graceful fallback message — agent will then try web search
        _record_empty_contract_detail_state()
        return (
            f"No contract found matching '{value}'. "
            f"The contract ID may not exist or the project name may be spelled differently. "
            f"Try searching with broader terms instead."
        )

    except Exception as e:
        print(f"get_contract_detail DB error: {e}")
        _record_empty_contract_detail_state()
        return "Error: Database failure during contract lookup"
    finally:
        if conn is not None:
            conn.close()


@tool
def get_contract_detail(query: str) -> str:
    """
    Use this tool ONLY when the incoming query starts with 'Lookup contract'.
    This performs a direct database lookup for a specific contract by its ID
    or exact project name. Use this for point lookups, not broad searches.
    """

    parsed = parse_lookup_string(query)
    if not parsed:
        _record_empty_contract_detail_state()
        return (
            "Error: Could not extract a contract ID or project name "
            "from the lookup query."
        )
    return _get_contract_detail_from_lookup_value(parsed["value"])


_PROXIMITY_EXTRACT = _re.compile(
    r"within\s+(\d+(?:\.\d+)?)\s*"
    r"(km|kilometers?|kilometres?|miles?|meters?)\s+of\s+"
    r"(.+?)(?=\s*$|[?.]|,?\s+(?:if\b|with\b|for\b|that\b|and\b))",
    _re.IGNORECASE,
)
_NEAR_WITHIN_EXTRACT = _re.compile(
    r"near(?:by)?\s+(.+?)\s+within\s+(\d+(?:\.\d+)?)\s*"
    r"(km|kilometers?|kilometres?|miles?|meters?)\b",
    _re.IGNORECASE,
)


def _distance_to_km(value: str, unit: str) -> float:
    amount = float(value)
    normalized_unit = unit.lower()
    if normalized_unit.startswith("mile"):
        return amount * 1.609344
    if normalized_unit.startswith("meter"):
        return amount / 1000.0
    return amount


def _clean_reference_name(reference_name: str) -> str:
    cleaned = " ".join(str(reference_name or "").split()).strip(" ?.,")
    cleaned = _re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=_re.IGNORECASE)
    return cleaned


def _parse_proximity_query(query: str) -> tuple[str, float] | None:
    """Returns (reference_name, radius_km) or None if not parseable."""
    m = _PROXIMITY_EXTRACT.search(query)
    if m:
        return _clean_reference_name(m.group(3)), _distance_to_km(m.group(1), m.group(2))
    m = _NEAR_WITHIN_EXTRACT.search(query)
    if m:
        return _clean_reference_name(m.group(1)), _distance_to_km(m.group(2), m.group(3))
    return None


def _format_radius_km(radius_km: float) -> str:
    if radius_km < 1:
        text = f"{radius_km:.3f}".rstrip("0").rstrip(".")
    elif float(radius_km).is_integer():
        text = f"{radius_km:.0f}"
    else:
        text = f"{radius_km:.1f}".rstrip("0").rstrip(".")
    return f"{text} km"


def _reference_search_terms(reference_name: str) -> list[str]:
    base = _clean_reference_name(reference_name)
    if not base:
        return []

    terms = [base]
    generic_tail = _re.sub(
        r"\b(?:project|projects|contract|contracts|site|location|area)\b",
        " ",
        base,
        flags=_re.IGNORECASE,
    )
    generic_tail = _re.sub(r"\s+", " ", generic_tail).strip(" ,")
    if generic_tail and generic_tail.lower() != base.lower():
        terms.append(generic_tail)

    words = [
        word
        for word in _re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", generic_tail or base)
        if word.lower()
        not in {
            "the",
            "project",
            "projects",
            "contract",
            "contracts",
            "near",
            "nearby",
        }
    ]
    if words:
        terms.append(" ".join(words))
        terms.extend(word for word in words if len(word) >= 4)

    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = term.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(term)
    return deduped


def _resolve_reference_project(reference_name: str, category_hint: str | None = None):
    """
    Find the contract most closely matching reference_name.
    Returns a row dict with contract_id, latitude, longitude, province, and description.

    Tries in order:
    1. description ILIKE plus optional category filter
    2. province ILIKE
    Falls back to rows without coordinates if no coordinate-bearing row exists.
    """
    conn = _psycopg2().connect(PG_DSN)
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().DictCursor) as cur:
            for term in _reference_search_terms(reference_name):
                category_attempts = (True, False) if category_hint else (False,)
                for use_category_hint in category_attempts:
                    params: list[object] = [f"%{term}%"]
                    category_clause = ""
                    if category_hint and use_category_hint:
                        category_clause = " AND (category ILIKE %s OR description ILIKE %s)"
                        params += [f"%{category_hint}%", f"%{category_hint}%"]

                    cur.execute(
                        f"""
                        SELECT contract_id, description, latitude, longitude, province, region
                        FROM contracts
                        WHERE description ILIKE %s
                        {category_clause}
                        ORDER BY
                            CASE WHEN latitude IS NOT NULL AND longitude IS NOT NULL THEN 0 ELSE 1 END,
                            LENGTH(description) ASC
                        LIMIT 1;
                        """,
                        params,
                    )
                    row = cur.fetchone()
                    if row:
                        return dict(row)

            for term in _reference_search_terms(reference_name):
                cur.execute(
                    """
                    SELECT contract_id, description, latitude, longitude, province, region
                    FROM contracts
                    WHERE province ILIKE %s
                    ORDER BY
                        CASE WHEN latitude IS NOT NULL AND longitude IS NOT NULL THEN 0 ELSE 1 END
                    LIMIT 1;
                    """,
                    [f"%{term}%"],
                )
                row = cur.fetchone()
                if row:
                    return dict(row)
            return None
    finally:
        conn.close()


def _haversine_search(
    ref_lat: float,
    ref_lon: float,
    radius_km: float,
    exclude_contract_id: str | None,
    category_hint: str | None,
    limit: int = 20,
) -> list[dict]:
    """
    Returns contracts within radius_km of (ref_lat, ref_lon).
    Uses a bounding-box pre-filter for performance, then Haversine for accuracy.
    """
    degree_buffer = max(radius_km / 111.0 * 1.5, 0.1)

    conditions = [
        "latitude IS NOT NULL",
        "longitude IS NOT NULL",
        "latitude BETWEEN %s AND %s",
        "longitude BETWEEN %s AND %s",
    ]
    params: list[object] = [
        ref_lat - degree_buffer,
        ref_lat + degree_buffer,
        ref_lon - degree_buffer,
        ref_lon + degree_buffer,
    ]

    if exclude_contract_id:
        conditions.append("contract_id != %s")
        params.append(exclude_contract_id)

    if category_hint:
        conditions.append("(category ILIKE %s OR description ILIKE %s)")
        params += [f"%{category_hint}%", f"%{category_hint}%"]

    where = " AND ".join(conditions)
    haversine_params = [ref_lat, ref_lon, ref_lat]

    conn = _psycopg2().connect(PG_DSN)
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().DictCursor) as cur:
            cur.execute(
                f"""
                SELECT * FROM (
                    SELECT
                        contract_id, description, category, status, budget,
                        start_date, completion_date, region, province, contractor,
                        latitude, longitude,
                        (6371.0 * acos(
                            LEAST(1.0,
                                cos(radians(%s)) * cos(radians(latitude)) *
                                cos(radians(longitude) - radians(%s)) +
                                sin(radians(%s)) * sin(radians(latitude))
                            )
                        )) AS distance_km
                    FROM contracts
                    WHERE {where}
                ) AS nearby
                WHERE distance_km <= %s
                ORDER BY distance_km ASC
                LIMIT %s;
                """,
                haversine_params + params + [radius_km, limit],
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _province_level_nearby(
    province: str,
    exclude_contract_id: str | None,
    category_hint: str | None,
    limit: int = 20,
) -> list[dict]:
    """Fallback when reference project has no coordinates: search same province."""
    conditions = ["province ILIKE %s"]
    params: list[object] = [f"%{province}%"]
    if exclude_contract_id:
        conditions.append("contract_id != %s")
        params.append(exclude_contract_id)
    if category_hint:
        conditions.append("(category ILIKE %s OR description ILIKE %s)")
        params += [f"%{category_hint}%", f"%{category_hint}%"]
    where = " AND ".join(conditions)
    conn = _psycopg2().connect(PG_DSN)
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().DictCursor) as cur:
            cur.execute(
                f"""
                SELECT contract_id, description, category, status, budget,
                       start_date, completion_date, region, province, contractor,
                       latitude, longitude, NULL::float AS distance_km
                FROM contracts
                WHERE {where}
                ORDER BY budget DESC
                LIMIT %s;
                """,
                params + [limit],
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


@tool
def find_nearby_contracts(query: str) -> str:
    """
    Use this tool when the user asks about contracts near a specific project or location,
    within a given distance, such as "within 10 km of the Miagao project".
    It resolves the reference project from the database, then performs geospatial search.
    """
    parsed = _parse_proximity_query(query)
    if not parsed:
        return (
            "Could not parse a distance and reference project from this query. "
            "Please specify a distance (for example, '10 km') and a project name or location."
        )

    reference_name, radius_km = parsed
    if not reference_name or radius_km <= 0:
        return (
            "Could not parse a valid distance and reference project from this query. "
            "Please specify a positive distance and a project name or location."
        )

    category_hint: str | None = None
    lower_query = query.lower()
    for keyword in ("flood control", "drainage", "road", "bridge", "school", "building", "water"):
        if keyword in lower_query:
            category_hint = keyword
            break

    reference = _resolve_reference_project(reference_name, category_hint)
    if not reference:
        return (
            f"Could not find a contract matching '{reference_name}' in the database. "
            "Try a broader name or check the spelling."
        )

    ref_id = reference.get("contract_id")
    ref_lat = reference.get("latitude")
    ref_lon = reference.get("longitude")
    ref_province = reference.get("province") or ""
    ref_description = reference.get("description") or reference_name

    used_fallback = False
    if ref_lat is not None and ref_lon is not None:
        nearby = _haversine_search(
            float(ref_lat),
            float(ref_lon),
            radius_km,
            exclude_contract_id=ref_id,
            category_hint=category_hint,
        )
    else:
        if not ref_province:
            return (
                f"Found reference project '{ref_description}' but it has no coordinates "
                "and no province, so a proximity search cannot be performed."
            )
        nearby = _province_level_nearby(
            ref_province,
            exclude_contract_id=ref_id,
            category_hint=category_hint,
        )
        used_fallback = True

    if not nearby:
        scope = (
            f"within {_format_radius_km(radius_km)} of {ref_description}"
            if not used_fallback
            else f"in {ref_province} (province-level fallback; reference project has no coordinates)"
        )
        return f"No matching contracts found {scope}."

    SOURCE_MARKER = "__SOURCES__"
    sources = []
    lines = []
    scope_note = (
        f"within {_format_radius_km(radius_km)} of **{ref_description}** ({ref_id})"
        if not used_fallback
        else f"in {ref_province}; note: reference project has no verified coordinates, showing province-level results"
    )
    lines.append(f"Found {len(nearby)} contract(s) {scope_note}:\n")

    for row in nearby:
        budget = _coerce_float(row.get("budget"))
        dist = row.get("distance_km")
        dist_text = f"{float(dist):.1f} km away" if dist is not None else "same province"
        completion = _format_date(row.get("completion_date"))
        start = _format_date(row.get("start_date"))
        lines.append(
            f"[{row['contract_id']}] {_truncate_text(row['description'])}\n"
            f"  Distance: {dist_text}\n"
            f"  Budget: PHP {budget:,.2f}\n"
            f"  Status: {row.get('status') or 'N/A'}\n"
            f"  Province: {row.get('province') or 'N/A'} | Region: {row.get('region') or 'N/A'}\n"
            f"  Start: {start} | Completion: {completion}\n"
            f"  Contractor: {_truncate_text(row.get('contractor') or 'N/A', 120)}"
        )
        sources.append(
            {
                "contractId": row["contract_id"],
                "description": row["description"],
                "contractor": row.get("contractor"),
                "region": row.get("region"),
                "province": row.get("province"),
                "budget": budget,
                "status": row.get("status"),
                "category": row.get("category"),
                "startDate": start,
                "completionDate": completion,
                "distanceKm": float(dist) if dist is not None else None,
            }
        )

    _record_result_state(
        {
            "result_kind": "contract_set",
            "intent": "proximity",
            "filters": {"category": category_hint} if category_hint else {},
            "subject": reference_name,
            "count": len(nearby),
            "contract_ids": [r["contract_id"] for r in nearby],
            "displayed_contract_ids": [r["contract_id"] for r in nearby],
            "displayed_sources": sources,
            "is_complete_result_set": True,
        }
    )

    sources_block = f"\n\n{SOURCE_MARKER}{json.dumps(sources)}"
    return "\n\n".join(lines) + sources_block


def _normalized_plan_filters(plan: QueryPlan) -> dict[str, str]:
    return {
        key: str(value).strip()
        for key, value in plan.filters.items()
        if isinstance(value, str) and str(value).strip()
    }


def analyze_contractor_concentration(plan: QueryPlan) -> dict[str, object]:
    filters = _normalized_plan_filters(plan)
    where_clause, params = _build_contract_where_clause(filters)
    where_sql = f"WHERE {where_clause}" if where_clause else ""
    conn = _psycopg2().connect(PG_DSN)
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().RealDictCursor) as cur:
            cur.execute(
                f"""
                WITH scoped AS (
                    SELECT contractor, COALESCE(budget, 0) AS budget
                    FROM contracts
                    {where_sql}
                ),
                totals AS (
                    SELECT COUNT(*)::float AS total_contracts, COALESCE(SUM(budget), 0)::float AS total_budget
                    FROM scoped
                )
                SELECT
                    scoped.contractor,
                    COUNT(*)::int AS contract_count,
                    COALESCE(SUM(scoped.budget), 0)::float AS total_budget,
                    CASE WHEN totals.total_contracts > 0 THEN COUNT(*)::float / totals.total_contracts ELSE 0 END AS contract_share,
                    CASE WHEN totals.total_budget > 0 THEN COALESCE(SUM(scoped.budget), 0)::float / totals.total_budget ELSE 0 END AS budget_share
                FROM scoped
                CROSS JOIN totals
                GROUP BY scoped.contractor, totals.total_contracts, totals.total_budget
                ORDER BY contract_share DESC, budget_share DESC, scoped.contractor ASC
                LIMIT 25;
                """,
                params,
            )
            rows = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

    flagged_rows = [
        row
        for row in rows
        if float(row.get("contract_share") or 0) > 0.40
        or float(row.get("budget_share") or 0) > 0.40
    ]
    return {
        "analysis_type": "contractor_concentration",
        "filters": filters,
        "rows": rows,
        "flagged_rows": flagged_rows,
    }


def detect_budget_anomalies(plan: QueryPlan) -> dict[str, object]:
    filters = _normalized_plan_filters(plan)
    where_clause, params = _build_contract_where_clause(filters)
    where_sql = f"WHERE {where_clause}" if where_clause else ""
    conn = _psycopg2().connect(PG_DSN)
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT
                    contract_id,
                    description,
                    category,
                    region,
                    infra_year,
                    budget,
                    award_amount,
                    CASE
                        WHEN budget IS NULL OR budget = 0 OR award_amount IS NULL THEN NULL
                        ELSE award_amount / budget
                    END AS award_budget_ratio
                FROM contracts
                {where_sql}
                """,
                params,
            )
            rows = [
                dict(row)
                for row in cur.fetchall()
                if row.get("award_budget_ratio") is not None
                and (
                    float(row.get("award_budget_ratio") or 0) < 0.60
                    or float(row.get("award_budget_ratio") or 0) > 1.05
                )
            ]
    finally:
        conn.close()
    return {"analysis_type": "budget_anomalies", "filters": filters, "rows": rows}


def detect_timeline_anomalies(plan: QueryPlan) -> dict[str, object]:
    filters = _normalized_plan_filters(plan)
    where_clause, params = _build_contract_where_clause(filters)
    where_sql = f"WHERE {where_clause}" if where_clause else ""
    conn = _psycopg2().connect(PG_DSN)
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().RealDictCursor) as cur:
            cur.execute(
                f"""
                WITH scoped AS (
                    SELECT contract_id, description, status, progress, start_date, completion_date, expiry_date
                    FROM contracts
                    {where_sql}
                )
                SELECT * FROM (
                    SELECT 'completion_past_due' AS anomaly_label, * FROM scoped
                    WHERE completion_date < CURRENT_DATE
                      AND status NOT IN ('Completed', 'Terminated', 'Suspended')
                    UNION ALL
                    SELECT 'zero_progress_stale' AS anomaly_label, * FROM scoped
                    WHERE COALESCE(progress, 0) = 0
                      AND start_date < CURRENT_DATE - INTERVAL '12 months'
                    UNION ALL
                    SELECT 'expiry_past_due' AS anomaly_label, * FROM scoped
                    WHERE expiry_date < CURRENT_DATE
                      AND status IN ('On-Going', 'Awarded')
                ) anomalies
                ORDER BY anomaly_label ASC, contract_id ASC
                LIMIT 200;
                """,
                params,
            )
            rows = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()
    return {"analysis_type": "timeline_anomalies", "filters": filters, "rows": rows}


def detect_bidding_anomalies(plan: QueryPlan) -> dict[str, object]:
    filters = _normalized_plan_filters(plan)
    where_clause, params = _build_contract_where_clause(filters)
    contract_where_sql = f"WHERE {where_clause}" if where_clause else ""
    bidder_where_sql = f"AND {where_clause}" if where_clause else ""
    conn = _psycopg2().connect(PG_DSN)
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT
                    c.contract_id,
                    c.description,
                    COUNT(cb.*)::int AS bidder_count
                FROM contracts c
                JOIN contract_bidders cb ON cb.contract_id = c.contract_id
                {contract_where_sql}
                GROUP BY c.contract_id, c.description
                HAVING COUNT(cb.*) = 1
                ORDER BY c.contract_id ASC
                LIMIT 100;
                """,
                params,
            )
            single_bidder_rows = [dict(row) for row in cur.fetchall()]

            cur.execute(
                f"""
                WITH bidder_sets AS (
                    SELECT
                        c.contract_id,
                        string_agg(DISTINCT cb.pcab_id, ',' ORDER BY cb.pcab_id) AS bidder_set
                    FROM contracts c
                    JOIN contract_bidders cb ON cb.contract_id = c.contract_id
                    WHERE cb.pcab_id IS NOT NULL AND BTRIM(cb.pcab_id) <> ''
                    {bidder_where_sql}
                    GROUP BY c.contract_id
                )
                SELECT
                    bidder_set,
                    COUNT(*)::int AS contract_count,
                    array_agg(contract_id ORDER BY contract_id) AS contract_ids
                FROM bidder_sets
                GROUP BY bidder_set
                HAVING COUNT(*) >= 3
                ORDER BY contract_count DESC, bidder_set ASC
                LIMIT 50;
                """,
                params,
            )
            recurring_bidder_sets = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()
    return {
        "analysis_type": "bidding_anomalies",
        "filters": filters,
        "single_bidder_rows": single_bidder_rows,
        "recurring_bidder_sets": recurring_bidder_sets,
    }


def detect_document_gaps(plan: QueryPlan) -> dict[str, object]:
    filters = _normalized_plan_filters(plan)
    where_clause, params = _build_contract_where_clause(filters)
    where_sql = f"WHERE {where_clause}" if where_clause else ""
    conn = _psycopg2().connect(PG_DSN)
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT
                    contract_id,
                    description,
                    raw_json -> 'links' ->> 'contractAgreement' AS contract_agreement,
                    raw_json -> 'links' ->> 'noticeOfAward' AS notice_of_award,
                    raw_json -> 'links' ->> 'noticeToProceed' AS notice_to_proceed
                FROM contracts
                {where_sql};
                """,
                params,
            )
            rows = []
            for row in cur.fetchall():
                payload = dict(row)
                payload["missing_document_count"] = sum(
                    1
                    for key in (
                        "contract_agreement",
                        "notice_of_award",
                        "notice_to_proceed",
                    )
                    if not str(payload.get(key) or "").strip()
                )
                if payload["missing_document_count"] >= 2:
                    rows.append(payload)
    finally:
        conn.close()
    return {"analysis_type": "document_gaps", "filters": filters, "rows": rows}


def find_similar_scope_contracts(reference_id: str, plan: QueryPlan) -> dict[str, object]:
    filters = _normalized_plan_filters(plan)
    conn = _psycopg2().connect(PG_DSN)
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().RealDictCursor) as cur:
            cur.execute(
                """
                SELECT embedding
                FROM contract_embeddings
                WHERE contract_id = %s
                LIMIT 1;
                """,
                (reference_id,),
            )
            reference = cur.fetchone()
            if not reference:
                return {
                    "analysis_type": "similar_scope",
                    "reference_id": reference_id,
                    "filters": filters,
                    "rows": [],
                    "error": "Reference embedding not found.",
                }

            where_clause, params = _build_contract_where_clause(filters)
            extra_clause = f" AND {where_clause}" if where_clause else ""
            cur.execute(
                f"""
                SELECT
                    c.contract_id,
                    c.description,
                    c.category,
                    c.region,
                    c.province,
                    c.contractor,
                    1 - (e.embedding <=> %s::vector) AS similarity_score
                FROM contract_embeddings e
                JOIN contracts c ON c.contract_id = e.contract_id
                WHERE c.contract_id <> %s
                {extra_clause}
                ORDER BY e.embedding <=> %s::vector
                LIMIT 10;
                """,
                [reference["embedding"], reference_id, *params, reference["embedding"]],
            )
            rows = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()
    return {
        "analysis_type": "similar_scope",
        "reference_id": reference_id,
        "filters": filters,
        "rows": rows,
    }


def execute_anomaly_plan(plan: QueryPlan) -> dict[str, object]:
    analysis_type = (plan.analysis_type or "").strip()
    if analysis_type == "contractor_concentration":
        return analyze_contractor_concentration(plan)
    if analysis_type in {"budget_outlier", "budget_anomalies"}:
        return detect_budget_anomalies(plan)
    if analysis_type == "award_anomaly":
        return detect_budget_anomalies(plan)
    if analysis_type in {"timeline_anomaly", "timeline_anomalies"}:
        return detect_timeline_anomalies(plan)
    if analysis_type in {"bidding_anomaly", "bidding_anomalies"}:
        return detect_bidding_anomalies(plan)
    if analysis_type in {"document_gap", "document_gaps"}:
        return detect_document_gaps(plan)
    if analysis_type in {"scope_similarity", "similar_scope"}:
        reference_id = (plan.lookup_value or "").strip()
        if reference_id:
            return find_similar_scope_contracts(reference_id, plan)
        return {
            "analysis_type": "scope_similarity",
            "filters": _normalized_plan_filters(plan),
            "rows": [],
            "error": "A reference contract ID is required for similar-scope analysis.",
        }
    return {
        "analysis_type": analysis_type or "unknown",
        "filters": _normalized_plan_filters(plan),
        "rows": [],
        "error": "Unknown anomaly analysis type.",
    }


tools = [
    ask_clarifying_question,
    search_contracts,
    get_contract_statistics,
    filter_contracts,
    get_contract_detail,
    find_nearby_contracts,
    web_search,
]
