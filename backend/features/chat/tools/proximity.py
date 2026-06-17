import json
import re as _re

from core.config import postgres_dsn
from contracts.filter_parser import parse_filter_string
from features.chat.tools.lookup import _format_contract_source_row, _record_result_state, _summarize_sources
from features.chat.tools.support import (
    _coerce_float,
    _format_date,
    _psycopg2,
    _psycopg2_extras,
    _truncate_text,
)
from langchain.tools import tool

PG_DSN: str = postgres_dsn()
RESULT_STATE_ID_CAP = 100

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
