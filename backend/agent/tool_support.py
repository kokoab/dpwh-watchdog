import importlib
from datetime import date, datetime
from typing import Optional

from agent.query_planner import QueryPlan
from rag.filter_parser import FUZZY_FIELDS

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

def _normalized_plan_filters(plan: QueryPlan) -> dict[str, str]:
    return {
        key: str(value).strip()
        for key, value in plan.filters.items()
        if isinstance(value, str) and str(value).strip()
    }
