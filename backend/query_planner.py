from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Literal

import psycopg2

from filter_parser import parse_filter_string
from lookup_parser import CONTRACT_ID_PATTERNS

PG_DSN: str = os.environ.get("PG_DSN") or (
    f"host={os.environ.get('POSTGRES_HOST')} "
    f"port={os.environ.get('POSTGRES_PORT')} "
    f"dbname={os.environ.get('POSTGRES_DB')} "
    f"user={os.environ.get('POSTGRES_USER')} "
    f"password={os.environ.get('POSTGRES_PASSWORD')}"
)

QueryIntent = Literal["lookup", "availability", "browse", "stats", "search", "clarify", "chat"]

LOOKUP_PREFIX = "Lookup contract"
AVAILABILITY_PREFIX = "Check availability where"
BROWSE_PREFIX = "Filter contracts where"
STATS_PREFIX = "Calculate metrics where"
SEARCH_PREFIX = "Find all contracts about"
CLARIFY_PREFIX = "Ask clarifying question"

INTENT_PREFIXES = {
    LOOKUP_PREFIX.lower(): "lookup",
    AVAILABILITY_PREFIX.lower(): "availability",
    BROWSE_PREFIX.lower(): "browse",
    STATS_PREFIX.lower(): "stats",
    SEARCH_PREFIX.lower(): "search",
    CLARIFY_PREFIX.lower(): "clarify",
}

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

REGION_ALIASES = {
    "metro manila": "National Capital Region",
    "ncr": "National Capital Region",
    "national capital region": "National Capital Region",
    "car": "Cordillera Administrative Region",
    "cordillera administrative region": "Cordillera Administrative Region",
}

CONTRACTOR_ALIASES = {
    "sunwst": "SUNWEST",
    "rudhil constuction": "RUDHIL",
    "rudhil construction": "RUDHIL",
}

DOMAIN_TERMS = re.compile(
    r"\b(contract|contracts|project|projects|contractor|budget|status|region|province|bridge|road|flood|drainage|school|building|water|procurement)\b",
    re.IGNORECASE,
)
CONTRACTOR_REFERENCE_TERMS = re.compile(
    r"\b(the contractor|the same contractor|same contractor|this contractor|that contractor|this one|that one|same one)\b",
    re.IGNORECASE,
)
CHAT_TERMS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|nice|cool)\b",
    re.IGNORECASE,
)
LOOKUP_TERMS = re.compile(
    r"\b(lookup|details?|detail|tell me about|status of contract|what is contract)\b",
    re.IGNORECASE,
)
STATS_TERMS = re.compile(
    r"\b(how many|count|counts|total|sum|average|avg|statistics|metrics|breakdown|top|highest|lowest)\b",
    re.IGNORECASE,
)
AVAILABILITY_TERMS = re.compile(
    r"\b(are there|is there|do you have|does .+ have|available|any)\b",
    re.IGNORECASE,
)
BROWSE_TERMS = re.compile(
    r"\b(show|list|give me|which|browse)\b|\bwhat\s+(?:contracts?|projects?)\b|\b(?:contracts?|projects?)\s+are\s+there\b",
    re.IGNORECASE,
)
FOLLOW_UP_TERMS = re.compile(
    r"^(what about|how about|what if|and what about|show them|show those|show these|them|those|these|what about this|what about that)\b",
    re.IGNORECASE,
)
RESULT_REFERENCE_TERMS = re.compile(
    r"\b(show|list)\s+(them|those|these|results|projects|contracts)\b|\bwhat\s+are\s+(those|these)\b|\bthose\s+\d+\s+(projects?|contracts?)\b",
    re.IGNORECASE,
)
LOCATION_CUE_TERMS = re.compile(
    r"\b(in|from|near|around|within|at|across)\b",
    re.IGNORECASE,
)
LOCATION_PHRASE_PATTERN = re.compile(
    r"\b(?:in|from|near|around|within|at|across)\s+([A-Za-z0-9][A-Za-z0-9 .,&/-]*?)(?=$|\b(?:with|for|by|status|budget|progress|show|list|give|which|what|how many|count|total|sum|average|avg|and status|and contractor|and category|and year)\b)",
    re.IGNORECASE,
)
CONTRACTOR_CUE_PATTERNS = [
    re.compile(r"\bby\s+(?:contractor\s+)?([A-Za-z0-9][A-Za-z0-9 .,&()/-]+?)(?=$|\bin\b|\bfrom\b|\bwith\b|\band\b)", re.IGNORECASE),
    re.compile(r"\bdoes\s+([A-Za-z0-9][A-Za-z0-9 .,&()/-]+?)\s+have\b", re.IGNORECASE),
    re.compile(r"\bcontractor\s+([A-Za-z0-9][A-Za-z0-9 .,&()/-]+?)(?=$|\bin\b|\bfrom\b|\bwith\b|\band\b)", re.IGNORECASE),
]
PROGRAM_CUE_PATTERN = re.compile(
    r"\bprogram\s+([A-Za-z0-9][A-Za-z0-9 .,&()/-]+?)(?=$|\bin\b|\bwith\b|\band\b)",
    re.IGNORECASE,
)
TRAILING_FILLER = re.compile(
    r"\b(?:contracts?|projects?|project|contract|there|anything|any|available|please|now|currently)\b",
    re.IGNORECASE,
)
LIMIT_SUFFIX_PATTERN = re.compile(r"\s+LIMIT\s+(\d+)\s*$", re.IGNORECASE)


FILTER_ORDER = (
    "region",
    "province",
    "status",
    "category",
    "contractor",
    "infra_year",
    "program_name",
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
    categories: tuple[str, ...]
    contractors: tuple[str, ...]
    programs: tuple[str, ...]

    region_map: dict[str, str] = field(default_factory=dict)
    province_map: dict[str, str] = field(default_factory=dict)
    status_map: dict[str, str] = field(default_factory=dict)
    contractor_map: dict[str, str] = field(default_factory=dict)
    program_map: dict[str, str] = field(default_factory=dict)


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

    def to_scope_dict(self) -> dict[str, str]:
        data = dict(self.filters)
        if self.subject:
            data["subject"] = self.subject
        data["intent"] = self.intent
        return data


def _load_distinct_values(column: str) -> tuple[str, ...]:
    conn = psycopg2.connect(PG_DSN)
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
    categories = _load_distinct_values("category")
    contractors = _load_distinct_values("contractor")
    programs = _load_distinct_values("program_name")
    return EntityCatalog(
        regions=regions,
        provinces=provinces,
        statuses=statuses,
        categories=categories,
        contractors=contractors,
        programs=programs,
        region_map={_normalize_text(value): value for value in regions},
        province_map={_normalize_text(value): value for value in provinces},
        status_map={_normalize_text(value): value for value in statuses},
        contractor_map={_normalize_text(value): value for value in contractors},
        program_map={_normalize_text(value): value for value in programs},
    )


def _best_catalog_match(text: str, catalog_map: dict[str, str]) -> str | None:
    normalized = _normalize_text(text)
    if not normalized:
        return None
    if normalized in catalog_map:
        return catalog_map[normalized]

    matches = [
        original
        for key, original in catalog_map.items()
        if normalized in key or key in normalized
    ]
    if not matches:
        return None
    matches.sort(key=len, reverse=True)
    return matches[0]


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


def _match_region(query: str, catalog: EntityCatalog) -> str | None:
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
            roman = ROMAN_NUMERALS.get(lookup_key, lookup_key.upper())
            return f"Region {roman}"
        if lookup_key in ROMAN_NUMERALS:
            return f"Region {ROMAN_NUMERALS[lookup_key]}"
        if token.isdigit():
            return f"Region {token}"
        return f"Region {token}"

    for region in sorted(catalog.regions, key=len, reverse=True):
        if _normalize_text(region) in normalized:
            return region

    return None


def _match_status(query: str, catalog: EntityCatalog) -> str | None:
    normalized = _normalize_text(query)
    for alias, canonical in STATUS_CANONICAL.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return catalog.status_map.get(_normalize_text(canonical), canonical)
    return None


def _match_category(query: str) -> str | None:
    normalized = _normalize_text(query)
    for alias in sorted(CATEGORY_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return CATEGORY_ALIASES[alias]
    return None


def _match_year(query: str) -> str | None:
    match = re.search(r"\b(20\d{2})\b", query)
    return match.group(1) if match else None


def _match_location_phrase(query: str) -> str | None:
    match = LOCATION_PHRASE_PATTERN.search(query)
    if not match:
        return None
    candidate = match.group(1).strip(" ,?.")
    candidate = re.sub(r"\s+", " ", candidate)
    return candidate or None


def _match_province(query: str, catalog: EntityCatalog) -> tuple[str | None, bool]:
    for province in sorted(catalog.provinces, key=len, reverse=True):
        if _normalize_text(province) in _normalize_text(query):
            return province, True

    location_phrase = _match_location_phrase(query)
    if not location_phrase:
        return None, False

    matches = _catalog_matches(location_phrase, catalog.province_map)
    if len(matches) == 1:
        return matches[0], True

    return location_phrase.strip().title(), False


def _match_contractor(query: str, catalog: EntityCatalog) -> str | None:
    normalized = _normalize_text(query)
    if CONTRACTOR_REFERENCE_TERMS.search(normalized):
        return None
    for alias, canonical in CONTRACTOR_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return canonical

    for pattern in CONTRACTOR_CUE_PATTERNS:
        match = pattern.search(query)
        if not match:
            continue
        candidate = match.group(1).strip(" ,?.")
        if _is_generic_contractor_reference(candidate):
            return None
        resolved = _best_catalog_match(candidate, catalog.contractor_map)
        return resolved or candidate
    return None


def _is_generic_contractor_reference(value: str) -> bool:
    normalized = _normalize_text(value)
    return normalized in {
        "contractor",
        "the contractor",
        "the same contractor",
        "same contractor",
        "this contractor",
        "that contractor",
        "this one",
        "that one",
        "same one",
    }


def _match_program(query: str, catalog: EntityCatalog) -> str | None:
    match = PROGRAM_CUE_PATTERN.search(query)
    if not match:
        return None
    candidate = match.group(1).strip(" ,?.")
    resolved = _best_catalog_match(candidate, catalog.program_map)
    return resolved or candidate


def _contains_lookup_id(query: str) -> str | None:
    for pattern in CONTRACT_ID_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if not match:
            continue
        return match.group(1) if match.lastindex else match.group(0)
    return None


def _strip_spans(text: str, spans: Iterable[str]) -> str:
    cleaned = text
    for span in spans:
        if not span:
            continue
        cleaned = re.sub(re.escape(span), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[?!.]+$", "", cleaned).strip()
    cleaned = re.sub(r"\b(what about|how about|show me|list|give me|which|what|are there|is there|do you have|does .+ have|find|search for|search|contracts? about|projects? about)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(in|from|near|around|within|at|across)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = TRAILING_FILLER.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    return cleaned


def _has_location_hint(query: str) -> bool:
    return bool(re.search(r"\bregion\b", query, re.IGNORECASE) or LOCATION_CUE_TERMS.search(query))


def _is_generic_subject(subject: str) -> bool:
    normalized = _normalize_text(subject)
    if not normalized:
        return True
    return normalized in {
        "contract",
        "contracts",
        "project",
        "projects",
        "detail",
        "details",
        "more detail",
        "more details",
        "same contractor",
        "this contractor",
        "that contractor",
        "the first one",
    }


def _should_clarify(
    query: str,
    filters: dict[str, str],
    subject: str,
    previous_plan: QueryPlan | None,
) -> bool:
    stripped = query.strip()
    if not stripped:
        return False
    if previous_plan and FOLLOW_UP_TERMS.search(stripped):
        return False
    if _contains_lookup_id(query):
        return False
    if CONTRACTOR_REFERENCE_TERMS.search(query) and not filters and not previous_plan:
        return True
    if LOOKUP_TERMS.search(query) and not filters and _is_generic_subject(subject):
        return True
    if (
        (BROWSE_TERMS.search(query) or AVAILABILITY_TERMS.search(query) or STATS_TERMS.search(query) or DOMAIN_TERMS.search(query))
        and not filters
        and _is_generic_subject(subject)
        and previous_plan is None
    ):
        return True
    return False


def _build_clarification_question(
    query: str,
    filters: dict[str, str],
    subject: str,
    previous_plan: QueryPlan | None,
) -> str:
    if CONTRACTOR_REFERENCE_TERMS.search(query):
        return "Which contractor are you referring to?"
    if LOOKUP_TERMS.search(query):
        return "Which contract or project should I look up?"
    if STATS_TERMS.search(query):
        return "Which region, contractor, category, or status should I use?"
    if BROWSE_TERMS.search(query) or AVAILABILITY_TERMS.search(query) or DOMAIN_TERMS.search(query):
        return "Which region, contractor, category, or status should I narrow this to?"
    if _is_generic_subject(subject):
        return "Which region, contractor, category, or status should I narrow this to?"
    return "Which region, contractor, category, or status should I narrow this to?"


def _is_domain_query(query: str, filters: dict[str, str], subject: str, previous_plan: QueryPlan | None) -> bool:
    return bool(
        DOMAIN_TERMS.search(query)
        or filters
        or subject
        or (previous_plan and FOLLOW_UP_TERMS.search(query.strip()))
    )


def _determine_intent(query: str, filters: dict[str, str], subject: str, previous_plan: QueryPlan | None) -> QueryIntent:
    stripped = query.strip()
    if _should_clarify(query, filters, subject, previous_plan):
        return "clarify"
    if CHAT_TERMS.match(stripped) and not _is_domain_query(query, filters, subject, previous_plan):
        return "chat"
    if _contains_lookup_id(query) or LOOKUP_TERMS.search(query):
        return "lookup"
    if STATS_TERMS.search(query):
        return "stats"
    if BROWSE_TERMS.search(query):
        return "browse"
    if AVAILABILITY_TERMS.search(query):
        return "availability"
    if filters:
        return "browse" if not subject else "search"
    if subject and _is_domain_query(query, filters, subject, previous_plan):
        return "search"
    return "chat"


def _merge_with_previous(plan: QueryPlan, previous_plan: QueryPlan | None, raw_query: str) -> QueryPlan:
    if not previous_plan:
        return plan

    if plan.intent == "chat" and not FOLLOW_UP_TERMS.search(raw_query.strip()):
        return plan

    if not (plan.is_follow_up or FOLLOW_UP_TERMS.search(raw_query.strip()) or not DOMAIN_TERMS.search(raw_query)):
        return plan

    merged = QueryPlan(
        intent=plan.intent if plan.intent != "chat" else previous_plan.intent,
        filters=dict(previous_plan.filters),
        subject=plan.subject or previous_plan.subject,
        lookup_value=plan.lookup_value,
        limit=plan.limit,
        exclude_selected_contract=plan.exclude_selected_contract
        or previous_plan.exclude_selected_contract,
        has_location_phrase=plan.has_location_phrase,
        has_unresolved_location_hint=plan.has_unresolved_location_hint,
        is_follow_up=True,
    )

    for key, value in plan.filters.items():
        merged.filters[key] = value

    if "province" in plan.filters:
        merged.filters.pop("region", None)
    if "region" in plan.filters:
        merged.filters.pop("province", None)

    if not plan.filters.get("region") and not plan.filters.get("province") and not plan.has_unresolved_location_hint:
        for key in ("region", "province"):
            if key in previous_plan.filters and key not in merged.filters:
                merged.filters[key] = previous_plan.filters[key]

    if not plan.subject and plan.intent in {"stats", "availability", "search", "browse"}:
        merged.subject = previous_plan.subject

    return merged


def plan_query(query: str, previous_plan: QueryPlan | None = None) -> QueryPlan:
    catalog = get_entity_catalog()
    region = _match_region(query, catalog)
    province, province_exact = _match_province(query, catalog)
    status = _match_status(query, catalog)
    category = _match_category(query)
    contractor = _match_contractor(query, catalog)
    infra_year = _match_year(query)
    program_name = _match_program(query, catalog)
    lookup_value = _contains_lookup_id(query)
    has_location_phrase = bool(_match_location_phrase(query))

    filters: dict[str, str] = {}
    if region:
        filters["region"] = region
    elif province:
        filters["province"] = province
    if status:
        filters["status"] = status
    if category:
        filters["category"] = category
    if contractor:
        filters["contractor"] = contractor
    if infra_year:
        filters["infra_year"] = infra_year
    if program_name:
        filters["program_name"] = program_name

    stripped_subject = _strip_spans(
        query,
        [region or "", province or "", status or "", contractor or "", infra_year or "", program_name or "", lookup_value or ""],
    )

    subject = stripped_subject if stripped_subject and DOMAIN_TERMS.search(query) else ""
    if category and subject and _normalize_text(subject) == _normalize_text(category):
        subject = ""

    plan = QueryPlan(
        intent=_determine_intent(query, filters, subject, previous_plan),
        filters=filters,
        subject=subject,
        lookup_value=lookup_value or "",
        limit=None,
        has_location_phrase=has_location_phrase,
        has_unresolved_location_hint=bool(_has_location_hint(query) and not region and not province_exact),
        is_follow_up=bool(previous_plan and FOLLOW_UP_TERMS.search(query.strip())),
    )

    if plan.intent == "lookup" and not plan.lookup_value:
        cleaned = _strip_spans(query, [])
        if cleaned:
            plan.lookup_value = cleaned

    if plan.intent == "clarify" and _is_generic_subject(plan.subject):
        plan.subject = _build_clarification_question(query, filters, subject, previous_plan)

    merged = _merge_with_previous(plan, previous_plan, query)
    if merged.intent == "chat" and previous_plan and FOLLOW_UP_TERMS.search(query.strip()):
        merged.intent = previous_plan.intent
    if merged.intent == "search" and not merged.subject and merged.filters:
        merged.intent = "browse"
    return merged


def render_plan(plan: QueryPlan) -> str:
    if plan.intent == "chat":
        return plan.subject or ""

    if plan.intent == "clarify":
        question = plan.subject if plan.subject and not _is_generic_subject(plan.subject) else "Which region, contractor, category, or status should I narrow this to?"
        return f"{CLARIFY_PREFIX}: {question}"

    if plan.intent == "lookup":
        return f"{LOOKUP_PREFIX} {plan.lookup_value}".strip()

    filters = plan.filters
    if plan.intent == "search" and plan.subject and filters.get("category"):
        if _normalize_text(plan.subject) != _normalize_text(filters["category"]):
            filters = {key: value for key, value in filters.items() if key != "category"}

    clauses = [
        f"{key}={filters[key]}"
        for key in FILTER_ORDER
        if filters.get(key)
    ]
    clause_text = " AND ".join(clauses)

    if plan.intent == "availability":
        if not clause_text:
            clause_text = "all=true"
        return f"{AVAILABILITY_PREFIX} {clause_text}"

    if plan.intent == "browse":
        if not clause_text and plan.subject:
            return f"{SEARCH_PREFIX} {plan.subject}"
        browse_query = f"{BROWSE_PREFIX} {clause_text}".strip()
        if plan.limit:
            browse_query += f" LIMIT {plan.limit}"
        return browse_query

    if plan.intent == "stats":
        if not clause_text:
            clause_text = "all=true"
        return f"{STATS_PREFIX} {clause_text}"

    if plan.intent == "search":
        if clause_text:
            return f"{SEARCH_PREFIX} {plan.subject or 'contracts'} where {clause_text}".strip()
        return f"{SEARCH_PREFIX} {plan.subject or 'contracts'}".strip()

    return plan.subject or ""


def detect_intent_from_expanded_query(expanded_query: str) -> QueryIntent:
    lowered = expanded_query.strip().lower()
    for prefix, intent in INTENT_PREFIXES.items():
        if lowered.startswith(prefix):
            return intent
    return "chat"


def parse_route_query(query: str) -> dict[str, object]:
    clean = query.strip()
    lowered = clean.lower()
    limit = None

    limit_match = LIMIT_SUFFIX_PATTERN.search(clean)
    if limit_match:
        limit = int(limit_match.group(1))
        clean = clean[: limit_match.start()].strip()
        lowered = clean.lower()

    if lowered.startswith(LOOKUP_PREFIX.lower()):
        return {"intent": "lookup", "filters": {}, "subject": "", "lookup_value": clean[len(LOOKUP_PREFIX):].strip(), "limit": limit}

    if lowered.startswith(CLARIFY_PREFIX.lower()):
        return {
            "intent": "clarify",
            "filters": {},
            "subject": clean[len(CLARIFY_PREFIX):].lstrip(": ").strip(),
            "lookup_value": "",
            "limit": limit,
        }

    if lowered.startswith(AVAILABILITY_PREFIX.lower()):
        filters = parse_filter_string(f"{BROWSE_PREFIX} {clean[len(AVAILABILITY_PREFIX):].strip()}")
        filters.pop("all", None)
        return {"intent": "availability", "filters": filters, "subject": "", "lookup_value": "", "limit": limit}

    if lowered.startswith(BROWSE_PREFIX.lower()):
        filters = parse_filter_string(clean)
        filters.pop("all", None)
        return {"intent": "browse", "filters": filters, "subject": "", "lookup_value": "", "limit": limit}

    if lowered.startswith(STATS_PREFIX.lower()):
        filters = parse_filter_string(f"{BROWSE_PREFIX} {clean[len(STATS_PREFIX):].strip()}")
        filters.pop("all", None)
        return {"intent": "stats", "filters": filters, "subject": "", "lookup_value": "", "limit": limit}

    if lowered.startswith(SEARCH_PREFIX.lower()):
        rest = clean[len(SEARCH_PREFIX):].strip()
        subject, _, where_clause = rest.partition(" where ")
        filters = {}
        if where_clause:
            filters = parse_filter_string(f"{BROWSE_PREFIX} {where_clause}")
        return {"intent": "search", "filters": filters, "subject": subject.strip(), "lookup_value": "", "limit": limit}

    return {"intent": "chat", "filters": {}, "subject": clean, "lookup_value": "", "limit": limit}
