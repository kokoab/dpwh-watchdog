import json
from pathlib import Path

from core.config import postgres_dsn
from features.chat.agent.query_scope import get_current_thread_id, get_thread_plan, get_thread_result, set_thread_result
from features.chat.tools.support import (
    _coerce_float,
    _contract_duration,
    _extract_document_links,
    _format_date,
    _format_filter_phrase,
    _normalize_result_filters,
    _psycopg2,
    _psycopg2_extras,
    _row_get,
    _truncate_text,
)
from contracts.lookup_parser import parse_lookup_string
from langchain.tools import tool

PG_DSN: str = postgres_dsn()
RESULT_STATE_ID_CAP = 100


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

    local_path = Path(__file__).parents[1] / "data" / f"{contract_id}.json"
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
