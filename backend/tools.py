import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
from embeddings import LocalAPIEmbeddings
from filter_parser import FUZZY_FIELDS, parse_filter_string
from hybrid_search import hybrid_search, structured_match_count, structured_match_ids
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from lookup_parser import parse_lookup_string
from query_planner import (
    AVAILABILITY_PREFIX,
    BROWSE_PREFIX,
    LOOKUP_PREFIX,
    SEARCH_PREFIX,
    STATS_PREFIX,
    parse_route_query,
)
from query_scope import (
    get_current_thread_id,
    get_thread_plan,
    get_thread_result,
    set_thread_result,
)
from reranker import rerank
from stats_parser import parse_stats_string

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

    conn = psycopg2.connect(PG_DSN)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
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
                    province, contractor, infra_year, program_name
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

    routed = parse_route_query(query)
    result_intent, result_filters, result_subject = _resolve_result_context(
        str(routed["intent"]),
        dict(routed.get("filters", {})),
        str(routed.get("subject", "") or ""),
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
        conn = psycopg2.connect(PG_DSN)
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
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


@tool
def get_contract_statistics(query: str) -> str:
    """
    Use this tool ONLY when the incoming query starts with 'Calculate metrics where'
    or 'Check availability where'.
    This tool extracts parameters to run SQL COUNT, SUM, and AVG aggregates.
    Supports filtering by region, province, infra_year, status, category keyword,
    and contractor name.
    """

    routed = parse_route_query(query)
    is_availability_query = routed["intent"] == "availability"
    params = parse_stats_string(query)

    region = params["region"]
    province = params["province"]
    infra_year = params["infra_year"]
    status = params["status"]
    category_keyword = params["category_keyword"]
    contractor = params["contractor"]
    result_filters = _normalize_result_filters(
        {
            "region": region,
            "province": province,
            "infra_year": infra_year,
            "status": status,
            "category": category_keyword,
            "contractor": contractor,
        }
    )
    result_intent, result_filters, _ = _resolve_result_context(
        "availability" if is_availability_query else str(routed["intent"]),
        result_filters,
    )

    conn = None
    try:
        conn = psycopg2.connect(PG_DSN)
        with conn.cursor() as cur:
            where_clause_sql, sql_params = _build_contract_where_clause(result_filters)
            where_clause = f" WHERE {where_clause_sql}" if where_clause_sql else ""

            # --- Core aggregates ---
            cur.execute(f"SELECT COUNT(*) FROM contracts{where_clause}", sql_params)
            total_contracts = cur.fetchone()[0]

            if is_availability_query:
                _, result_rows = _fetch_contract_rows(
                    result_filters,
                    limit=min(max(total_contracts, 1), RESULT_STATE_ID_CAP),
                )
                _record_result_state(
                    {
                        "result_kind": "contract_set",
                        "intent": result_intent,
                        "filters": result_filters,
                        "count": int(total_contracts),
                        "contract_ids": [row["contract_id"] for row in result_rows][:RESULT_STATE_ID_CAP],
                        "displayed_contract_ids": [],
                        "displayed_sources": [],
                        "is_complete_result_set": total_contracts <= RESULT_STATE_ID_CAP,
                    }
                )
                scope = _build_stats_scope(
                    region,
                    province,
                    infra_year,
                    status,
                    category_keyword,
                    contractor,
                )
                availability = "Yes" if total_contracts > 0 else "No"
                return (
                    f"Availability Check {scope}:\n"
                    f"- Matching Contracts: {total_contracts:,}\n"
                    f"- Available: {availability}\n"
                    "- Use a listing request if you want to browse matching rows.\n"
                )

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
            status_breakdown = ", ".join(
                f"{row[0] or 'Unknown'}: {row[1]:,}" for row in status_rows
            )

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
                region_breakdown = ", ".join(
                    f"{row[0] or 'Unknown'}: {row[1]:,}" for row in region_rows
                )
            else:
                region_breakdown = None

    except Exception as e:
        print(f"Failed to calculate database statistics: {e}")
        return "Error: unable to process statistical counts on database tables"
    finally:
        if conn is not None:
            conn.close()

    scope = _build_stats_scope(
        region,
        province,
        infra_year,
        status,
        category_keyword,
        contractor,
    )

    # --- Award-to-budget ratio ---
    award_to_budget_ratio = (
        (total_award_amount / total_budget * 100)
        if total_budget > 0 and total_award_amount > 0
        else None
    )
    award_ratio_text = (
        f"{award_to_budget_ratio:.1f}%" if award_to_budget_ratio is not None else "N/A"
    )

    _, result_rows = _fetch_contract_rows(
        result_filters,
        limit=min(max(total_contracts, 1), RESULT_STATE_ID_CAP),
    )
    _record_result_state(
        {
            "result_kind": "contract_set",
            "intent": result_intent,
            "filters": result_filters,
            "count": int(total_contracts),
            "contract_ids": [row["contract_id"] for row in result_rows][:RESULT_STATE_ID_CAP],
            "displayed_contract_ids": [],
            "displayed_sources": [],
            "is_complete_result_set": total_contracts <= RESULT_STATE_ID_CAP,
        }
    )

    output = (
        f"Statistics Summary {scope}:\n"
        f"- Total Contracts Matched: {total_contracts:,}\n"
        f"- Combined Budget: PHP {total_budget:,.2f}\n"
        f"- Total Award Amount: PHP {total_award_amount:,.2f}\n"
        f"- Award-to-Budget Ratio: {award_ratio_text}\n"
        f"- Average Progress: {avg_progress:.1f}%\n"
        f"- Status Breakdown: {status_breakdown or 'N/A'}\n"
    )

    if region_breakdown:
        output += f"- Top Regions: {region_breakdown}\n"

    return output


@tool
def filter_contracts(query: str) -> str:
    """
    Use this tool ONLY when the incoming query starts with 'Filter contracts where'.
    This performs structured SQL filtering on known contract attributes like
    contractor, region, province, status, category, infra_year, and program_name.
    Use this for exact or near-exact attribute lookups, NOT for descriptive searches.
    """

    routed = parse_route_query(query)
    filters = routed["filters"] if routed["intent"] == "browse" else parse_filter_string(query)
    result_intent, filters, _ = _resolve_result_context(
        str(routed["intent"]),
        filters,
    )
    limit = int(routed.get("limit") or FILTER_MATCH_LIMIT)
    limit = max(1, min(limit, FILTER_MATCH_LIMIT))
    exclude_selected_contract, selected_source, selected_contract_id = _should_exclude_selected_contract()

    if not filters:
        return (
            "Error: Could not extract any valid filters from the query. "
            "Valid fields are: contractor, region, province, status, category, infra_year, program_name."
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

    lookup_type = parsed["lookup_type"]
    value = parsed["value"]

    conn = None
    try:
        conn = psycopg2.connect(PG_DSN)
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
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
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as comp_cur:
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


tools = [
    ask_clarifying_question,
    search_contracts,
    get_contract_statistics,
    filter_contracts,
    get_contract_detail,
    web_search,
]
