from __future__ import annotations
from core.config import postgres_dsn

import importlib
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from functools import lru_cache
from typing import Literal

from rag.lookup_parser import CONTRACT_ID_PATTERNS

PG_DSN: str = postgres_dsn()

QueryIntent = Literal[
    "lookup",
    "availability",
    "browse",
    "stats",
    "search",
    "compare",
    "anomaly",
    "clarify",
    "chat",
    "proximity",
]

ROMAN_NUMERALS = {
    "1": "I",
    "2": "II",
    "3": "III",
    "4": "IV",
    "5": "V",
    "6": "VI",
    "7": "VII",
    "8": "VIII",
    "9": "IX",
    "10": "X",
    "11": "XI",
    "12": "XII",
    "13": "XIII",
    "14": "XIV-A",
    "14a": "IV-A",
    "14-b": "IV-B",
    "14b": "IV-B",
    "15": "XV",
    "16": "XVI",
    "17": "XVII",
}

STATUS_CANONICAL = {
    "under evaluation": "under evaluation",
    "on-going": "On-Going",
    "ongoing": "On-Going",
    "completed": "Completed",
    "delayed": "Delayed",
    "suspended": "Suspended",
    "terminated": "Terminated",
    "awarded": "Awarded",
    "for procurement": "For Procurement",
}

REGION_ALIASES = {
    "metro manila": "National Capital Region",
    "ncr": "National Capital Region",
    "national capital region": "National Capital Region",
    "car": "Cordillera Administrative Region",
    "cordillera administrative region": "Cordillera Administrative Region",
}

GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|nice|cool|good morning|good afternoon|good evening)\b",
    re.IGNORECASE,
)
DOMAIN_PATTERN = re.compile(
    r"\b(contract|contracts|project|projects|contractor|budget|status|region|province|bridge|road|flood|drainage|school|building|water|procurement|award|timeline|bidder|document)\b",
    re.IGNORECASE,
)
LOCATION_PHRASE_PATTERN = re.compile(
    r"\b(?:in|from|near|around|within|at|across)\s+([A-Za-z0-9][A-Za-z0-9 .,&/-]*?)(?=$|\b(?:with|for|by|status|budget|progress|show|list|give|which|what|how many|count|total|sum|average|avg|and status|and contractor|and category|and year)\b)",
    re.IGNORECASE,
)
RELATIVE_YEAR_PATTERN = re.compile(
    r"\b(?:last|past)\s+(\d+)\s+years?\b",
    re.IGNORECASE,
)
LAST_YEAR_PATTERN = re.compile(r"\blast\s+year\b", re.IGNORECASE)
TEMPORAL_LOCATION_PATTERN = re.compile(
    r"^(?:the\s+)?(?:last|past)\s+\d+\s+years?$|^(?:the\s+)?last\s+year$|^(?:the\s+)?past\s+year$",
    re.IGNORECASE,
)
AWARDED_TO_CONTRACTOR_PATTERN = re.compile(
    r"\bawarded\s+to\s+(.+?)(?=$|"
    r"\s+\b(?:in|from|near|around|within|at|across|with|for|by)\b|"
    r"\s+\b(?:status|budget|progress|show|list|give|which|what|how many|count|total|sum|average|avg)\b)",
    re.IGNORECASE,
)
PROXIMITY_PATTERN = re.compile(
    r"\bwithin\s+\d+(?:\.\d+)?\s*(?:km|kilometer|kilometre|miles?|meters?)\b"
    r"|\bnear(?:by)?\s+\S.{2,40}?\s+within\s+\d",
    re.IGNORECASE,
)


def _normalize_text(value: str) -> str:
    lowered = value.lower()
    lowered = lowered.replace("&", " and ")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


@dataclass(frozen=True)
class EntityCatalog:
    regions: tuple[str, ...]
    provinces: tuple[str, ...]
    statuses: tuple[str, ...]
    region_map: dict[str, str] = field(default_factory=dict)
    province_map: dict[str, str] = field(default_factory=dict)
    status_map: dict[str, str] = field(default_factory=dict)


@dataclass
class QueryPlan:
    intent: QueryIntent
    filters: dict[str, str] = field(default_factory=dict)
    subject: str = ""
    lookup_value: str = ""
    limit: int | None = None
    exclude_selected_contract: bool = False
    has_location_phrase: bool = False
    has_unresolved_location_hint: bool = False
    is_follow_up: bool = False
    analysis_type: str = ""

    def to_scope_dict(self) -> dict[str, str]:
        data = dict(self.filters)
        if self.subject:
            data["subject"] = self.subject
        data["intent"] = self.intent
        if self.analysis_type:
            data["analysis_type"] = self.analysis_type
        return data

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _psycopg2():
    return importlib.import_module("psycopg2")


def _load_distinct_values(column: str) -> tuple[str, ...]:
    conn = _psycopg2().connect(PG_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT {column}
                FROM contracts
                WHERE {column} IS NOT NULL AND BTRIM({column}) <> ''
                ORDER BY {column} ASC
                """
            )
            return tuple(row[0] for row in cur.fetchall())
    finally:
        conn.close()


@lru_cache(maxsize=1)
def get_entity_catalog() -> EntityCatalog:
    regions = _load_distinct_values("region")
    provinces = _load_distinct_values("province")
    statuses = _load_distinct_values("status")
    return EntityCatalog(
        regions=regions,
        provinces=provinces,
        statuses=statuses,
        region_map={_normalize_text(value): value for value in regions},
        province_map={_normalize_text(value): value for value in provinces},
        status_map={_normalize_text(value): value for value in statuses},
    )


def _catalog_matches(text: str, catalog_map: dict[str, str]) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    exact = catalog_map.get(normalized)
    if exact:
        return [exact]
    matches = [
        original
        for key, original in catalog_map.items()
        if normalized in key or key in normalized
    ]
    matches.sort(key=len)
    return matches


def _extract_location_candidate(query: str) -> str | None:
    location_match = LOCATION_PHRASE_PATTERN.search(query)
    if not location_match:
        return None
    candidate = location_match.group(1).strip(" ,?.")
    candidate = re.sub(r"\s+", " ", candidate)
    return candidate or None


def _is_temporal_location_candidate(candidate: str | None) -> bool:
    if not candidate:
        return False
    return bool(TEMPORAL_LOCATION_PATTERN.match(candidate.strip()))


def _current_year() -> int:
    return date.today().year


def match_region(query: str, catalog: EntityCatalog | None = None) -> str | None:
    catalog = catalog or get_entity_catalog()
    normalized = _normalize_text(query)
    for alias, canonical in REGION_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return canonical

    region_match = re.search(
        r"\bregion\s+([ivxlcdm]+(?:-[ab])?|\d+(?:-[ab])?|\d+[ab]?)\b",
        normalized,
        re.IGNORECASE,
    )
    if region_match:
        token = region_match.group(1).strip().upper()
        lookup_key = token.lower()
        if lookup_key.isdigit():
            return f"Region {ROMAN_NUMERALS.get(lookup_key, lookup_key.upper())}"
        if lookup_key in ROMAN_NUMERALS:
            return f"Region {ROMAN_NUMERALS[lookup_key]}"
        return f"Region {token}"

    for region in sorted(catalog.regions, key=len, reverse=True):
        if _normalize_text(region) in normalized:
            return region
    return None


def match_province(query: str, catalog: EntityCatalog | None = None) -> tuple[str | None, bool]:
    catalog = catalog or get_entity_catalog()
    normalized = _normalize_text(query)
    for province in sorted(catalog.provinces, key=len, reverse=True):
        if _normalize_text(province) in normalized:
            return province, True

    candidate = _extract_location_candidate(query)
    if not candidate or _is_temporal_location_candidate(candidate):
        return None, False
    matches = _catalog_matches(candidate, catalog.province_map)
    if len(matches) == 1:
        return matches[0], True
    return (candidate.strip().title(), False) if candidate else (None, False)


def match_status(query: str, catalog: EntityCatalog | None = None) -> str | None:
    catalog = catalog or get_entity_catalog()
    normalized = _normalize_text(query)
    for alias, canonical in STATUS_CANONICAL.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return catalog.status_map.get(_normalize_text(canonical), canonical)
    return None


def match_year(query: str) -> str | None:
    match = re.search(r"\b(20\d{2})\b", query)
    return match.group(1) if match else None


def match_year_filters(query: str) -> dict[str, str]:
    explicit_year = match_year(query)
    if explicit_year:
        return {"infra_year": explicit_year}

    relative_match = RELATIVE_YEAR_PATTERN.search(query)
    if relative_match:
        window_size = int(relative_match.group(1))
        if window_size > 0:
            end_year = _current_year()
            start_year = end_year - window_size + 1
            return {
                "infra_year_start": str(start_year),
                "infra_year_end": str(end_year),
            }

    if LAST_YEAR_PATTERN.search(query):
        return {"infra_year": str(_current_year() - 1)}

    return {}


def match_awarded_to_contractor(query: str) -> str | None:
    match = AWARDED_TO_CONTRACTOR_PATTERN.search(query)
    if not match:
        return None
    contractor = match.group(1).strip(" ,?")
    contractor = re.sub(r"\s+", " ", contractor)
    return contractor or None


def has_awarded_to_contractor(query: str) -> bool:
    return match_awarded_to_contractor(query) is not None


def find_lookup_contract_id(query: str) -> str | None:
    for pattern in CONTRACT_ID_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if not match:
            continue
        return match.group(1) if match.lastindex else match.group(0)
    return None


def is_greeting(query: str) -> bool:
    return bool(GREETING_PATTERN.match(query.strip()))


def has_domain_terms(query: str) -> bool:
    return bool(DOMAIN_PATTERN.search(query))


def extract_anchor_filters(query: str) -> dict[str, str]:
    catalog = get_entity_catalog()
    filters: dict[str, str] = {}
    region = match_region(query, catalog)
    province, _ = match_province(query, catalog)
    status = match_status(query, catalog)
    contractor = match_awarded_to_contractor(query)
    year_filters = match_year_filters(query)
    awarded_to_contractor = bool(contractor)

    if region:
        filters["region"] = region
    elif province:
        filters["province"] = province
    if status and not (awarded_to_contractor and status == "Awarded"):
        filters["status"] = status
    if contractor:
        filters["contractor"] = contractor
    filters.update(year_filters)
    return filters


def build_anchor_plan(query: str) -> QueryPlan:
    catalog = get_entity_catalog()
    lookup_value = find_lookup_contract_id(query) or ""
    region = match_region(query, catalog)
    province, province_exact = match_province(query, catalog)
    status = match_status(query, catalog)
    contractor = match_awarded_to_contractor(query)
    year_filters = match_year_filters(query)
    location_candidate = _extract_location_candidate(query)
    temporal_location_hint = _is_temporal_location_candidate(location_candidate)
    awarded_to_contractor = bool(contractor)

    filters: dict[str, str] = {}
    if region:
        filters["region"] = region
    elif province:
        filters["province"] = province
    if status and not (awarded_to_contractor and status == "Awarded"):
        filters["status"] = status
    if contractor:
        filters["contractor"] = contractor
    filters.update(year_filters)

    return QueryPlan(
        intent="lookup" if lookup_value else "chat",
        filters=filters,
        lookup_value=lookup_value,
        has_location_phrase=bool(LOCATION_PHRASE_PATTERN.search(query)),
        has_unresolved_location_hint=bool(
            LOCATION_PHRASE_PATTERN.search(query)
            and not temporal_location_hint
            and not region
            and not province_exact
        ),
    )
