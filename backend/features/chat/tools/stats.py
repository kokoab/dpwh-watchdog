import json
import re

from core.database import connect
from contracts.filter_parser import parse_filter_string
from contracts.stats_parser import parse_stats_filters
from features.chat.tools.lookup import (
    _fetch_contract_rows,
    _format_contract_source_row,
    _record_result_state,
    _resolve_result_context,
    _should_exclude_selected_contract,
    _source_matches_filters,
    _summarize_sources,
    _summarize_stats_contract_sources,
    _exclude_selected_contract_rows,
)
from features.chat.tools.support import (
    _build_contract_where_clause,
    _build_stats_scope,
    _coerce_float,
    _format_filter_phrase,
    _normalize_result_filters,
)
from langchain.tools import tool

FILTER_MATCH_LIMIT = 10
RESULT_STATE_ID_CAP = 100

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
        conn = connect()
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
