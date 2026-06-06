import re
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm_expander = ChatOllama(
    model="llama3.1:latest",
    base_url="http://host.docker.internal:11434",
    temperature=0.1,
    top_p=0.3,
)


LOCATION_ALIASES = {
    r"\bmetro manila\b": "National Capital Region",
    r"\bncr\b": "National Capital Region",
    r"\bcar\b": "Cordillera Administrative Region",
    r"\bcordillera administrative region\b": "Cordillera Administrative Region",
    r"\bcagayan valley\b": "Region II",
    r"\bdavao\b": "Region XI",
    r"\bcebu\b": "Region VII",
}
CONTRACTOR_ALIASES = {
    r"\bsunwst\b": "SUNWEST",
    r"\brudhil\s+constuction\b": "RUDHIL",
    r"\brudhil\s+construction\b": "RUDHIL",
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
}

LOOKUP_PATTERN = re.compile(r"\b(?:contract\s*)?\d{2}[A-Z]{1,3}\d{4,6}\b", re.IGNORECASE)
DOMAIN_PATTERN = re.compile(
    r"\b(contract|contracts|project|projects|road|bridge|flood|drainage|school|building|water|seawall|slope|region|province|contractor)\b",
    re.IGNORECASE,
)
STATS_PATTERN = re.compile(
    r"\b(how many|count|total budget|sum|average|avg|statistics|metrics)\b",
    re.IGNORECASE,
)
FILTER_PATTERN = re.compile(
    r"\b(show|list|filter|all)\b.*\b(contract|contracts)\b",
    re.IGNORECASE,
)
SEARCH_PATTERN = re.compile(
    r"\b(project|projects|road|roads|bridge|bridges|flood control|drainage|school|building|buildings|water|seawall|slope protection|mga|sa)\b",
    re.IGNORECASE,
)
EXPANDED_PREFIX_PATTERN = re.compile(
    r"^(Find all contracts about|Calculate metrics for|Filter contracts where|Lookup contract)\b",
    re.IGNORECASE,
)

INTENT_PREFIXES = {
    "find all contracts about": "search",
    "calculate metrics for": "statistics",
    "filter contracts where": "filter",
    "lookup contract": "lookup",
}


def _normalize_locations(query: str) -> str:
    normalized = query
    for pattern, replacement in LOCATION_ALIASES.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\bregion\s+(\d{1,2})\b",
        lambda match: f"Region {ROMAN_NUMERALS.get(match.group(1), match.group(1))}",
        normalized,
        flags=re.IGNORECASE,
    )
    return normalized


def _normalize_contractors(query: str) -> str:
    normalized = query
    for pattern, replacement in CONTRACTOR_ALIASES.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


def _detect_intent(expanded_query: str) -> str:
    lower = expanded_query.strip().lower()
    for prefix, intent in INTENT_PREFIXES.items():
        if lower.startswith(prefix):
            return intent
    return "chat"


def _deterministic_expand(query: str) -> str | None:
    normalized = _normalize_contractors(_normalize_locations(query.strip()))
    if EXPANDED_PREFIX_PATTERN.match(normalized):
        return normalized

    if not DOMAIN_PATTERN.search(normalized):
        return None

    lookup_match = LOOKUP_PATTERN.search(normalized)
    if lookup_match:
        contract_id = lookup_match.group(0)
        contract_id = re.sub(r"^contract\s*", "", contract_id, flags=re.IGNORECASE)
        return f"Lookup contract {contract_id.strip()}"

    if STATS_PATTERN.search(normalized):
        return f"Calculate metrics for {normalized}"

    contractor_filter = re.match(
        r"^\s*([A-Z][A-Z0-9&.,\s]+?)\s+(on-going|ongoing|completed|terminated|for procurement)\s+contracts?\s*$",
        normalized,
        re.IGNORECASE,
    )
    if contractor_filter:
        contractor = contractor_filter.group(1).strip()
        status = contractor_filter.group(2).strip()
        if status.lower() == "ongoing":
            status = "On-Going"
        return f"Filter contracts where contractor={contractor} AND status={status}"

    if FILTER_PATTERN.search(normalized):
        return None

    if SEARCH_PATTERN.search(normalized):
        normalized = re.sub(r"\bmga\s+", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bsa\s+", "in ", normalized, flags=re.IGNORECASE)
        return f"Find all contracts about {normalized}"

    return None


def log_query_expansion(raw_input: str, expanded_output: str, thread_id: str | None = None) -> None:
    log_path = Path(
        os.environ.get(
            "QUERY_EXPAND_LOG_PATH",
            Path(__file__).parent / "logs" / "query_expand.jsonl",
        )
    )
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "raw_input": raw_input,
        "expanded_output": expanded_output,
        "intent": _detect_intent(expanded_output),
    }

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Query expansion log error: {e}")


def query_expand(query: str) -> str:
    deterministic = _deterministic_expand(query)
    if deterministic:
        return deterministic

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a specialized Query Expansion and Search Optimizer AI for the DPWH Watchdog platform.\n\n"
                "### CRITICAL OPERATIONAL RULE\n"
                "- If the user's message is a greeting, small talk, or completely irrelevant to contracts, projects, infrastructure, or locations, you MUST output the user's exact message verbatim. Do not change a single word.\n\n"
                "### SEARCH REWRITING OBJECTIVE\n"
                "### LOCATION ALIASES\n"
                "- 'Metro Manila' -> 'National Capital Region'\n"
                "- 'NCR' -> 'National Capital Region'\n"
                "- 'BARMM' -> 'Bangsamoro Autonomous Region'\n"
                "- 'CAR' -> 'Cordillera Administrative Region'\n"
                "- If the user is asking a quantitative or counting question: Calculate metrics for [Standardized Input]\n"
                "- If the user is asking to filter by known attributes: Filter contracts where [field]=[value] AND [field]=[value]\n"
                "- If the user is referencing a SPECIFIC contract by ID or exact project name: Lookup contract [identifier]\n"
                "- If the user input is a descriptive or conceptual search: Find all contracts about [Standardized Input]\n\n"
                "  - Examples:\n"
                "    - 'how many bridge contracts in region 8' -> Calculate metrics for bridge contracts in Region VIII\n"
                "    - 'total budget for ongoing road projects in davao' -> Calculate metrics for ongoing road contracts in Davao\n"
                "    - 'how many contracts does DMCI have' -> Calculate metrics for contracts by contractor DMCI\n"
                "    - 'sum of delayed flood control projects in 2023' -> Calculate metrics for delayed flood control contracts infra_year=2023\n"
                "### LOOKUP TEMPLATE RULES\n"
                "- Use the Lookup template ONLY when the user provides a specific contract ID pattern "
                "  (e.g. '2023-ROW-00145', 'contract #4521') OR a highly specific proper project name "
                "  (e.g. 'CALA Expressway', 'Leyte Gulf Bridge').\n"
                "- Extract the identifier as-is — do not paraphrase or normalize it.\n"
                "- Examples:\n"
                "  - 'what is contract 2023-ROW-00145' -> Lookup contract 2023-ROW-00145\n"
                "  - 'tell me about the CALA Expressway project' -> Lookup contract CALA Expressway\n"
                "  - 'status of contract #4521' -> Lookup contract 4521\n"
                "  - 'details on Metro Manila Flood Management' -> Lookup contract Metro Manila Flood Management\n"
                "### FILTER TEMPLATE RULES\n"
                "- Only use the Filter template when the user provides concrete, specific attribute values (exact contractor name, specific region/province, a status word like 'delayed' or 'completed', a category, or a year).\n"
                "- Valid filterable fields are: contractor, region, province, status, category, infra_year, program_name\n"
                "- Example: 'show me all delayed contracts in Region VIII' -> Filter contracts where region=Region VIII AND status=delayed\n"
                "- Example: 'contracts by DMCI in 2023' -> Filter contracts where contractor=DMCI AND infra_year=2023\n"
                "- Example: 'all ongoing bridge contracts' -> Filter contracts where category=bridge AND status=ongoing\n\n"
                "### TERMINOLOGY MAPPING RULES\n"
                "- Convert all numeric or casual region names to formal Roman Numerals strictly (e.g., 'region 8' -> 'Region VIII', 'region 10' -> 'Region X').\n"
                "- Clean up shorthand locations to their full official names (e.g., 'cdo' -> 'Cagayan de Oro').\n\n"
                "### OUTPUT FORMAT\n"
                "Output ONLY the final string. Do not add explanations, conversational pleasantries, or markdown formatting blocks.",
            ),
            ("user", "{user_query}"),
        ]
    )

    chain = prompt | llm_expander | StrOutputParser()

    # Execute the chain and clean any accidental white space
    expanded_query = chain.invoke({"user_query": query}).strip()
    return expanded_query
