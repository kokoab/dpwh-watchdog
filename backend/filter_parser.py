import re

FIELD_ALIASES = {
    "contractor": "contractor",
    "region": "region",
    "province": "province",
    "status": "status",
    "category": "category",
    "infra_year": "infra_year",
    "infra_year_start": "infra_year_start",
    "infra_year_end": "infra_year_end",
    "year_start": "infra_year_start",
    "year_end": "infra_year_end",
    "year": "infra_year",
    "yr": "infra_year",
    "program": "program_name",
    "program_name": "program_name",
}

FUZZY_FIELDS = {
    "contractor",
    "region",
    "province",
    "category",
    "program_name",
    "status",
}


def parse_filter_string(filter_string: str) -> dict[str, str]:
    clean = re.sub(
        r"^filter contracts where\s", "", filter_string.strip(), flags=re.IGNORECASE
    )

    parts = re.split(r"\s+AND\s+", clean, flags=re.IGNORECASE)

    filters = {}
    for part in parts:
        if "=" not in part:
            continue
        field, _, value = part.partition("=")
        field = field.strip().lower()
        value = value.strip()

        canonical = FIELD_ALIASES.get(field)
        if canonical and value:
            filters[canonical] = value

    return filters
