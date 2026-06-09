from __future__ import annotations

import importlib
import os
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Literal

from lookup_parser import CONTRACT_ID_PATTERNS

PG_DSN: str = os.environ.get("PG_DSN") or (
    f"host={os.environ.get('POSTGRES_HOST')} "
    f"port={os.environ.get('POSTGRES_PORT')} "
    f"dbname={os.environ.get('POSTGRES_DB')} "
    f"user={os.environ.get('POSTGRES_USER')} "
    f"password={os.environ.get('POSTGRES_PASSWORD')}"
)

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

    location_match = LOCATION_PHRASE_PATTERN.search(query)
    if not location_match:
        return None, False
    candidate = location_match.group(1).strip(" ,?.")
    candidate = re.sub(r"\s+", " ", candidate)
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
    infra_year = match_year(query)

    if region:
        filters["region"] = region
    elif province:
        filters["province"] = province
    if status:
        filters["status"] = status
    if infra_year:
        filters["infra_year"] = infra_year
    return filters


def build_anchor_plan(query: str) -> QueryPlan:
    catalog = get_entity_catalog()
    lookup_value = find_lookup_contract_id(query) or ""
    region = match_region(query, catalog)
    province, province_exact = match_province(query, catalog)
    status = match_status(query, catalog)
    infra_year = match_year(query)

    filters: dict[str, str] = {}
    if region:
        filters["region"] = region
    elif province:
        filters["province"] = province
    if status:
        filters["status"] = status
    if infra_year:
        filters["infra_year"] = infra_year

    return QueryPlan(
        intent="lookup" if lookup_value else "chat",
        filters=filters,
        lookup_value=lookup_value,
        has_location_phrase=bool(LOCATION_PHRASE_PATTERN.search(query)),
        has_unresolved_location_hint=bool(
            LOCATION_PHRASE_PATTERN.search(query) and not region and not province_exact
        ),
    )
