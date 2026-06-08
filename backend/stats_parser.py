import re
from typing import Optional

from query_planner import (
    AVAILABILITY_PREFIX,
    SEARCH_PREFIX,
    STATS_PREFIX,
    parse_route_query,
)

# Known status words to detect in the query string. Keep order deterministic:
# "on-going" should be the DB-facing value for casual "ongoing" queries.
STATUS_PATTERNS = [
    ("under evaluation", "under evaluation"),
    ("on-going", "on-going"),
    ("ongoing", "on-going"),
    ("completed", "completed"),
    ("delayed", "delayed"),
    ("suspended", "suspended"),
    ("terminated", "terminated"),
    ("awarded", "awarded"),
]
STATUS_KEYWORDS = {pattern for pattern, _ in STATUS_PATTERNS}

# Known category keywords — expand this list as you learn your data
# The canonical values are what downstream SQL filters should use.
CATEGORY_KEYWORDS = {
    # Bridges
    "Bridges": "bridge",
    "Bridges-Construction-Concrete, Flood Control/Hydraulics/Drainage": "bridge",
    "Bridges-Construction-Concrete, Flood Control/Hydraulics/River Control": "bridge",
    "Bridges-Construction-Concrete, Maintenance Flood Control": "bridge",
    "Bridges-Construction-Concrete, Roads: Construction - PCCP": "bridge",
    "Bridges-Rehabilitation-Concrete, Roads: Rehabilitation - PCCP": "bridge",
    "Bridges: Construction - Concrete (Superstructure) - with Driven Piles, Roads: Construction - PCCP": "bridge",
    "Bridges: Construction - Concrete (Superstructure) - with Driven Piles, Roads: Maintenance": "bridge",
    "Bridges: Construction - Concrete (Superstructure) - with Driven Piles, Roads: Rehabilitation - PCCP": "bridge",
    "Bridges: Construction - Concrete (Superstructure) - without Piles, Roads: Construction - PCCP": "bridge",
    "Bridges: Maintenance": "bridge",
    "Bridges: Rehabilitation - Concrete (Superstructure) - without Piles": "bridge",
    "Bridges: Rehabilitation - Concrete (Superstructure) - without Piles, Roads: Maintenance": "bridge",
    "Bridges: Rehabilitation - Steel (Superstructure) - without Piles": "bridge",
    "Bridges: Retrofitting - Steel (Superstructure) - without Piles": "bridge",
    "GAA 2024 SSP Bridges, Roads": "bridge",
    # Roads & FMR
    "Roads": "road",
    "Roads: Construction - Asphalt, Roads: Construction - PCCP": "road",
    "Roads: Construction - Gravel": "road",
    "Roads: Construction - PCCP": "road",
    "Roads: Rehabilitation - Asphalt, Roads: Rehabilitation - PCCP": "road",
    "Roads: Rehabilitation - PCCP": "road",
    "Flood Control Structures, Roads": "road",
    "Flood Control/Hydraulics/Drainage, Roads-Rehabilitation-Gravel": "road",
    "Flood Control/Hydraulics/Drainage, Roads: Construction - PCCP": "road",
    "Flood Control/Hydraulics/Drainage, Roads: Rehabilitation - Asphalt": "road",
    "Flood Control/Hydraulics/Drainage, Roads: Rehabilitation - PCCP": "road",
    "Flood Control/Hydraulics/River Control, Roads: Construction - PCCP": "road",
    "GAA 2016 DA Farm-to-Market Roads": "road",
    "GAA 2023 DA FMR": "road",
    "GAA 2024 DA FMR": "road",
    "GAA 2025 DA FMR": "road",
    "Maintenance Flood Control, Roads: Construction - PCCP": "road",
    "Maintenance Roads and Bridges, Roads: Construction - PCCP": "road",
    "Maintenance Roads and Bridges, Roads: Construction - PCCP, Roads: Rehabilitation - PCCP": "road",
    # Flood Control
    "Flood Control and Drainage": "flood control",
    "Buildings/Industrial Plant-LOW rise, Flood Control/Hydraulics/Drainage": "flood control",
    "Flood Control/Hydraulics/Dredging, Flood Control/Hydraulics/River Control": "flood control",
    "Flood Control: Construction - Drainage (e.g., Closed and Open Conduits, Spillway)": "flood control",
    # Buildings
    "Building: Completion": "building",
    "Buildings and Facilities": "building",
    "Buildings/Industrial Plant-LOW rise, Maintenance Buildings": "building",
    "Buildings: Construction - without Piles - Low Rise - Concrete (Frame) (1 to 5 Storeys)": "building",
    "Buildings: Construction - without Piles - Low Rise - Steel (Frame) (1 to 5 Storeys)": "building",
    "Buildings: Repair": "building",
    "Buildings: Retrofitting - Low Rise (1 to 5 Storeys)": "building",
    "GAA 2025 SSP Buildings": "building",
    # Schools (DepEd & SUCs)
    "GAA 2016 DepED BEFF": "school",
    "GAA 2017 DepEd BEFF": "school",
    "GAA 2018 DepEd BEFF": "school",
    "GAA 2019 DepEd BEFF": "school",
    "GAA 2022 DepEd BEFF": "school",
    "GAA 2023 DepEd BEFF": "school",
    "GAA 2024 DepEd BEFF": "school",
    "GAA 2024 SUCs Infrastructure Projects": "school",
    "GAA 2025 DEPED BEFF": "school",
    "Multi Purpose Buildings, School Buildings": "school",
    # Water
    "Water Provision and Storage": "water supply",
}

REGION_ALIASES = {
    "metro manila": "National Capital Region",
    "ncr": "National Capital Region",
    "car": "Cordillera Administrative Region",
    "cordillera administrative region": "Cordillera Administrative Region",
    "national capital region": "National Capital Region",
    "cagayan valley": "Region II",
    "davao": "Region XI",
    "cebu": "Region VII",
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
    "14": "XIV",
    "15": "XV",
    "16": "XVI",
    "17": "XVII",
}

# Short, deterministic patterns that cover the free-text query styles used in eval.
CATEGORY_PATTERNS = [
    ("flood control", "flood control"),
    ("covered court", "building"),
    ("multi-purpose building", "building"),
    ("multi purpose building", "building"),
    ("school building", "school"),
    ("school buildings", "school"),
    ("school", "school"),
    ("bridge", "bridge"),
    ("road", "road"),
    ("water system", "water supply"),
    ("water supply", "water supply"),
    ("water", "water supply"),
    ("building", "building"),
]


def parse_stats_string(query: str) -> dict:
    """
    Parses 'Calculate metrics for delayed bridge contracts in Region VIII infra_year=2023'
    into structured kwargs for get_contract_statistics.

    Returns a dict with keys: region, province, infra_year, status,
                               category_keyword, contractor
    All values are Optional[str], None if not found.
    """
    result = {
        "region": None,
        "province": None,
        "infra_year": None,
        "status": None,
        "category_keyword": None,
        "contractor": None,
    }

    routed = parse_route_query(query)
    if routed["intent"] in {"availability", "stats", "browse", "search"}:
        filters = routed["filters"]
        result["region"] = filters.get("region")
        result["province"] = filters.get("province")
        result["infra_year"] = filters.get("infra_year")
        result["status"] = filters.get("status")
        result["category_keyword"] = filters.get("category")
        result["contractor"] = filters.get("contractor")
        if any(result.values()):
            return result

    # Strip the intent prefix
    clean = query.strip()
    clean = re.sub(rf"^(?:{re.escape(STATS_PREFIX)}|calculate metrics for)\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(rf"^{re.escape(AVAILABILITY_PREFIX)}\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(rf"^{re.escape(SEARCH_PREFIX)}\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"[?!.]+$", "", clean).strip()

    clean_lower = clean.lower()

    # --- Extract explicit infra_year=XXXX pattern first ---
    year_match = re.search(r"infra_year=(\d{4})", clean, re.IGNORECASE)
    if year_match:
        result["infra_year"] = year_match.group(1)
        clean = clean[: year_match.start()].strip()

    # --- Extract 4-digit year anywhere in the string ---
    if not result["infra_year"]:
        year_match = re.search(r"\b(20\d{2})\b", clean)
        if year_match:
            result["infra_year"] = year_match.group(1)

    # --- Extract region (Roman numeral, numeric, or named alias) ---
    for alias, canonical in REGION_ALIASES.items():
        if alias in clean_lower:
            result["region"] = canonical
            break

    if not result["region"]:
        region_match = re.search(
            r"region\s+([IVXLCDM]+(?:-[A-Z])?|\d+(?:-[A-Z])?)",
            clean,
            re.IGNORECASE,
        )
        if region_match:
            token = region_match.group(1).strip().upper()
            if token.isdigit():
                token = ROMAN_NUMERALS.get(token, token)
            result["region"] = f"Region {token}"

    # --- Extract province/city (after 'in' keyword, not a region) ---
    province_match = re.search(
        r"\bin\s+(?!region)([A-Za-z\s]+?)(?:\s+infra_year|$|\bregion\b|\bby\b)",
        clean,
        re.IGNORECASE,
    )
    if province_match and not result["region"]:
        candidate = province_match.group(1).strip()
        # Must be at least 3 chars and not a status/category word
        if len(candidate) >= 3 and candidate.lower() not in STATUS_KEYWORDS:
            result["province"] = candidate

    # --- Extract status keywords ---
    for pattern, canonical in STATUS_PATTERNS:
        if pattern in clean_lower:
            result["status"] = canonical
            break

    # --- Extract category keywords using deterministic, human-friendly patterns ---
    for pattern, canonical in CATEGORY_PATTERNS:
        if pattern in clean_lower:
            result["category_keyword"] = canonical
            break

    if not result["category_keyword"]:
        # Fall back to the more exhaustive keyword map in case a new category
        # appears in the dataset. Compare case-insensitively to keep it reliable.
        lower_category_keywords = {
            keyword.lower(): canonical for keyword, canonical in CATEGORY_KEYWORDS.items()
        }
        for keyword in sorted(lower_category_keywords, key=len, reverse=True):
            if keyword in clean_lower:
                result["category_keyword"] = lower_category_keywords[keyword]
                break

    # --- Extract contractor (after 'by contractor' or 'by') ---
    contractor_match = re.search(
        r"by\s+(?:contractor\s+)?([A-Za-z0-9\s&.,]+?)(?:\s+in\b|\s+and\b|$)",
        clean,
        re.IGNORECASE,
    )
    if not contractor_match:
        contractor_match = re.search(
            r"\bdoes\s+([A-Za-z0-9\s&.,]+?)\s+have\b",
            clean,
            re.IGNORECASE,
        )
    if not contractor_match:
        contractor_match = re.search(
            r"\b(?:contractor|contractor=)\s+([A-Za-z0-9\s&.,]+?)(?:\s+in\b|\s+and\b|$)",
            clean,
            re.IGNORECASE,
        )
    if contractor_match:
        candidate = contractor_match.group(1).strip()
        # Make sure it's not a noise word
        if candidate.lower() not in {
            "the",
            "a",
            "an",
            "all",
            "contracts",
            "contract",
            "contractor",
            "the contractor",
            "the same contractor",
            "same contractor",
            "this contractor",
            "that contractor",
            "this one",
            "that one",
            "same one",
        }:
            result["contractor"] = candidate

    return result
