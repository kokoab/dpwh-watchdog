from __future__ import annotations

import re

from features.chat.agent.query_planner import extract_anchor_filters


CATEGORY_KEYWORDS = {
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
    "road": "road",
    "water system": "water supply",
    "water supply": "water supply",
    "water": "water supply",
    "building": "building",
}


def parse_stats_filters(filters: dict[str, object] | None) -> dict[str, str | None]:
    normalized = filters if isinstance(filters, dict) else {}
    return {
        "region": str(normalized.get("region") or "").strip() or None,
        "province": str(normalized.get("province") or "").strip() or None,
        "infra_year": str(normalized.get("infra_year") or "").strip() or None,
        "infra_year_start": str(normalized.get("infra_year_start") or "").strip() or None,
        "infra_year_end": str(normalized.get("infra_year_end") or "").strip() or None,
        "status": str(normalized.get("status") or "").strip() or None,
        "category_keyword": str(normalized.get("category") or "").strip() or None,
        "contractor": str(normalized.get("contractor") or "").strip() or None,
    }


def parse_stats_string(query: str) -> dict[str, str | None]:
    filters = extract_anchor_filters(query)
    category = None
    lowered = query.lower()
    for keyword, canonical in sorted(CATEGORY_KEYWORDS.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            category = canonical
            break
    if category:
        filters["category"] = category
    return parse_stats_filters(filters)
