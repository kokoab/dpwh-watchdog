import re

# Common contract ID patterns in DPWH data
# Extend this list as you discover new formats in your DB
CONTRACT_ID_PATTERNS = [
    r"\b\d{4}-[A-Z]{2,6}-\d{4,6}\b",  # 2023-ROW-00145
    r"\b[A-Z]{2,6}-\d{4}-\d{4,6}\b",  # ROW-2023-00145
    r"\b\d{2}-\d{3}-[A-Z]{2,4}-\d+\b",  # 23-001-ROW-145
    r"\b[A-Z]{3,}-\d{5,}\b",  # DPWH-00145
    r"\bcontract\s*#?\s*(\d+)\b",  # contract #4521 or contract 4521
]


def parse_lookup_string(query: str) -> dict:
    """
    Parses 'Lookup contract 2023-ROW-00145' or 'Lookup contract CALA Expressway'
    into {'lookup_type': 'id'|'name', 'value': str}

    Returns None if nothing parseable found.
    """

    # Strip the intent prefix
    clean = re.sub(
        r"^lookup contract\s*", "", query.strip(), flags=re.IGNORECASE
    ).strip()

    if not clean:
        return None

    # Try to match a known contract ID pattern first
    for pattern in CONTRACT_ID_PATTERNS:
        match = re.search(pattern, clean, re.IGNORECASE)
        if match:
            # Use the first capture group if present, else full match
            value = match.group(1) if match.lastindex else match.group(0)
            return {"lookup_type": "id", "value": value.strip()}

    # No ID pattern matched — treat the whole string as a project name
    return {"lookup_type": "name", "value": clean}
