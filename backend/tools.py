import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
from embeddings import LocalAPIEmbeddings
from filter_parser import FUZZY_FIELDS, parse_filter_string
from hybrid_search import hybrid_search, structured_match_count, structured_match_ids
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from lookup_parser import parse_lookup_string
from reranker import rerank
from stats_parser import parse_stats_string

web_search = DuckDuckGoSearchRun()
embedding = LocalAPIEmbeddings()

PG_DSN: str = os.environ.get("PG_DSN") or (
    f"host={os.environ.get('POSTGRES_HOST')} "
    f"port={os.environ.get('POSTGRES_PORT')} "
    f"dbname={os.environ.get('POSTGRES_DB')} "
    f"user={os.environ.get('POSTGRES_USER')} "
    f"password={os.environ.get('POSTGRES_PASSWORD')}"
)

AVAILABILITY_STATS_MARKER = "availability check:"


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


def _is_availability_stats_query(query: str) -> bool:
    return AVAILABILITY_STATS_MARKER in query.lower()


def _strip_availability_marker(query: str) -> str:
    return re.sub(
        rf"({AVAILABILITY_STATS_MARKER})",
        "",
        query,
        flags=re.IGNORECASE,
    ).replace("  ", " ").strip()


def _build_stats_scope(
    region: Optional[str],
    province: Optional[str],
    infra_year: Optional[str],
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
    if status:
        scope_parts.append(f"Status: {status}")
    if category_keyword:
        scope_parts.append(f"Category: {category_keyword}")
    if contractor:
        scope_parts.append(f"Contractor: {contractor}")
    return f"[{' | '.join(scope_parts)}]" if scope_parts else "[Global Scope]"


def _load_local_contract_record(contract_id: str):
    """
    Fallback for exact lookups when the normalized DB row is missing but the
    source contract JSON is present locally. This keeps ID lookups deterministic
    even when the ingested table is incomplete.
    """

    local_path = Path(__file__).parent / "data" / f"{contract_id}.json"
    if not local_path.exists():
        return None, []

    try:
        with open(local_path) as f:
            payload = json.load(f)
    except Exception:
        return None, []

    data = payload.get("data") or {}
    location = data.get("location") or {}
    coordinates = location.get("coordinates") or {}

    record = {
        "contract_id": data.get("contractId") or contract_id,
        "description": data.get("description"),
        "category": data.get("category"),
        "status": data.get("status"),
        "budget": data.get("budget"),
        "amount_paid": data.get("amountPaid"),
        "award_amount": data.get("procurement", {}).get("awardAmount"),
        "progress": data.get("progress"),
        "region": location.get("region"),
        "province": location.get("province"),
        "latitude": coordinates.get("latitude", data.get("latitude")),
        "longitude": coordinates.get("longitude", data.get("longitude")),
        "contractor": data.get("contractor"),
        "advertisement_date": data.get("procurement", {}).get("advertisementDate"),
        "expiry_date": data.get("expiryDate"),
        "bid_submission_deadline": data.get("procurement", {}).get(
            "bidSubmissionDeadline"
        ),
        "start_date": data.get("startDate"),
        "completion_date": data.get("completionDate"),
        "infra_year": data.get("infraYear"),
        "program_name": data.get("programName"),
        "source_of_funds": data.get("sourceOfFunds"),
        "raw_json": data,
    }

    component_rows = []
    for index, component in enumerate(data.get("components") or [], start=1):
        comp_coords = component.get("coordinates") or {}
        component_rows.append(
            {
                "component_order": index,
                "component_id": component.get("componentId"),
                "description": component.get("description"),
                "type_of_work": component.get("typeOfWork"),
                "infra_type": component.get("infraType"),
                "region": component.get("region"),
                "province": component.get("province"),
                "latitude": comp_coords.get("latitude", component.get("latitude")),
                "longitude": comp_coords.get("longitude", component.get("longitude")),
                "raw_json": component,
            }
        )

    return record, component_rows


def _format_contract_lookup_output(
    r, component_rows, value: str, lookup_type: str
) -> str:
    budget = _coerce_float(r["budget"])
    amount_paid = _coerce_float(r["amount_paid"])
    award_amount = _coerce_float(r["award_amount"])
    award_to_budget_ratio = (
        (award_amount / budget * 100) if budget > 0 and award_amount > 0 else None
    )
    award_amount_text = f"PHP {award_amount:,.2f}" if award_amount > 0 else "N/A"
    award_ratio_text = (
        f"{award_to_budget_ratio:.1f}%" if award_to_budget_ratio is not None else "N/A"
    )
    contract_duration = _contract_duration(r["start_date"], r["completion_date"])

    SOURCE_MARKER = "__SOURCES__"
    sources = [
        {
            "description": r["description"],
            "contractId": r["contract_id"],
            "contractor": r["contractor"],
            "region": r["region"],
            "province": r["province"],
            "budget": budget,
            "amountPaid": amount_paid,
            "awardAmount": award_amount,
            "progress": r["progress"],
            "status": r["status"],
            "category": r["category"],
            "infraYear": r["infra_year"],
            "programName": r["program_name"],
        }
    ]

    detail_block = (
        f"CONTRACT DETAIL RECORD\n"
        f"{'=' * 40}\n"
        f"Description:        {r['description'] or 'N/A'}\n"
        f"Contract ID:        {r['contract_id'] or 'N/A'}\n"
        f"Category:           {r['category'] or 'N/A'}\n"
        f"Status:             {r['status'] or 'N/A'}\n"
        f"Contractor:         {r['contractor'] or 'N/A'}\n"
        f"Region:             {r['region'] or 'N/A'}\n"
        f"Province:           {r['province'] or 'N/A'}\n"
        f"Budget:             PHP {budget:,.2f}\n"
        f"Award Amount:       {award_amount_text}\n"
        f"Award-to-Budget Ratio: {award_ratio_text}\n"
        f"Progress:           {r['progress'] or 'N/A'}%\n"
        f"Infra Year:         {r['infra_year'] or 'N/A'}\n"
        f"Program:            {r['program_name'] or 'N/A'}\n"
        f"Source of Funds:    {r['source_of_funds'] or 'N/A'}\n"
        f"Advertisement Date: {_format_date(r['advertisement_date'])}\n"
        f"Bid Submission Deadline: {_format_date(r['bid_submission_deadline'])}\n"
        f"Start Date:         {_format_date(r['start_date'])}\n"
        f"Completion Date:    {_format_date(r['completion_date'])}\n"
        f"Expiry Date:        {_format_date(r['expiry_date'])}\n"
        f"Contract Duration:  {contract_duration}\n"
    )

    if component_rows:
        detail_block += (
            "\nCONTRACT COMPONENTS\n"
            f"{'=' * 40}\n"
            + "\n".join(
                (
                    f"[{row['component_order']}] {row['component_id'] or 'N/A'}\n"
                    f"Type of Work: {row['type_of_work'] or 'N/A'}\n"
                    f"Description: {row['description'] or 'N/A'}\n"
                    f"Infra Type: {row['infra_type'] or 'N/A'}\n"
                    f"Region: {row['region'] or 'N/A'}\n"
                    f"Province: {row['province'] or 'N/A'}\n"
                    f"Latitude: {row['latitude'] if row['latitude'] is not None else 'N/A'}\n"
                    f"Longitude: {row['longitude'] if row['longitude'] is not None else 'N/A'}\n"
                )
                for row in component_rows
            )
        )

    header = (
        f"Direct lookup result for '{value}' "
        f"({'exact ID match' if lookup_type == 'id' else 'name match'}):\n\n"
    )

    sources_block = f"\n\n{SOURCE_MARKER}{json.dumps(sources)}"
    return header + detail_block + sources_block


@tool
def search_contracts(query: str) -> str:
    """
    Use this tool ONLY when the incoming query starts with 'Find all contracts about'.
    This performs hybrid semantic + keyword search for descriptive project concepts.
    """

    structured_total = structured_match_count(query)
    structured_ids = structured_match_ids(query)
    if structured_total == 0:
        return (
            "No matching DPWH contracts found for the structured filters in this query. "
            "Try broadening the location, category, status, or contractor terms."
        )

    try:
        query_vector = embedding.embed_query(query)
    except Exception as e:
        return f"Error: Could not embed query for vector search: {e}"

    # --- Stage 1a: Vector search (wide net) ---
    vector_candidates = []
    try:
        conn = psycopg2.connect(PG_DSN)
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT
                    c.contract_id, c.description, c.category, c.status,
                    c.budget, c.progress, c.region,
                    c.province, c.contractor, c.infra_year, c.program_name,
                    e.chunk_text
                FROM contract_embeddings e
                JOIN contracts c ON e.contract_id = c.contract_id
                ORDER BY e.embedding <=> %s::vector
                LIMIT 25;
                """,
                (query_vector,),
            )
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f"Error: Database during similarity search: {e}"

    for r in rows:
        vector_candidates.append(
            {
                "chunk_text": r["chunk_text"],
                "contract_id": r["contract_id"],
                "description": r["description"],
                "category": r["category"],
                "status": r["status"],
                "budget": float(r["budget"]) if r["budget"] else 0.0,
                "progress": r["progress"],
                "region": r["region"],
                "province": r["province"],
                "contractor": r["contractor"],
                "infra_year": r["infra_year"],
                "program_name": r["program_name"],
            }
        )

    # --- Stage 1b: BM25 search + RRF merge ---
    merged_candidates = hybrid_search(query, vector_candidates)

    # --- Deduplicate by contract_id (keep first occurrence = highest RRF score) ---
    seen_ids = set()
    unique_candidates = []
    for c in merged_candidates:
        if c["contract_id"] not in seen_ids:
            seen_ids.add(c["contract_id"])
            unique_candidates.append(c)

    if not unique_candidates:
        return "No relevant contracts found in the database"

    if structured_ids is not None:
        unique_candidates = [
            candidate
            for candidate in unique_candidates
            if candidate["contract_id"] in structured_ids
        ]
        if not unique_candidates:
            return (
                "No matching DPWH contracts found for the structured filters in this query. "
                "Try broadening the location, category, status, or contractor terms."
            )

    reranked = rerank(query, unique_candidates, top_k=10)

    # --- Build output (unchanged from before) ---
    SOURCE_MARKER = "__SOURCES__"
    sources = []
    passages = []

    for r in reranked:
        sources.append(
            {
                "description": r["description"],
                "contractId": r["contract_id"],
                "contractor": r["contractor"],
                "region": r["region"],
                "province": r["province"],
                "budget": r["budget"],
                "progress": r["progress"],
                "status": r["status"],
                "category": r["category"],
                "infraYear": r["infra_year"],
                "programName": r["program_name"],
            }
        )
        passages.append(r["chunk_text"])

    sources_block = f"\n\n{SOURCE_MARKER}{json.dumps(sources)}"
    if structured_total is not None:
        result_scope = (
            f"Showing top {len(reranked)} of {structured_total:,} matching DPWH contracts."
        )
    else:
        result_scope = (
            f"Showing top {len(reranked)} candidate DPWH contracts. "
            "No reliable structured total count is available for this semantic query."
        )

    content = f"{result_scope}\n\nHere are the relevant DPWH contracts found:\n\n " + "\n\n---\n\n ".join(
        passages
    )

    return (
        "Here are relevant sources found\n\n"
        + content
        + "\n\nSources:\n"
        + "\n".join(
            f"- {s['description']} | {s['contractId']} | "
            f"{s['contractor']} | {s['region']} | {s['province']}"
            for s in sources
        )
        + sources_block
    )


@tool
def get_contract_statistics(query: str) -> str:
    """
    Use this tool ONLY when the incoming query starts with 'Calculate metrics for'.
    This tool extracts parameters to run SQL COUNT, SUM, and AVG aggregates.
    Supports filtering by region, province, infra_year, status, category keyword,
    and contractor name.
    """

    is_availability_query = _is_availability_stats_query(query)
    effective_query = _strip_availability_marker(query) if is_availability_query else query
    params = parse_stats_string(effective_query)

    region = params["region"]
    province = params["province"]
    infra_year = params["infra_year"]
    status = params["status"]
    category_keyword = params["category_keyword"]
    contractor = params["contractor"]

    try:
        conn = psycopg2.connect(PG_DSN)
        with conn.cursor() as cur:
            # --- Build WHERE clause ---
            conditions = []
            sql_params = []

            if region:
                conditions.append("region ILIKE %s")
                sql_params.append(f"%{region}%")
            if province:
                conditions.append("province ILIKE %s")
                sql_params.append(f"%{province}%")
            if infra_year:
                conditions.append("infra_year = %s")
                sql_params.append(infra_year)
            if status:
                conditions.append("status ILIKE %s")
                sql_params.append(f"%{status}%")
            if category_keyword:
                # Searches both description and category columns
                conditions.append("(description ILIKE %s OR category ILIKE %s)")
                sql_params.append(f"%{category_keyword}%")
                sql_params.append(f"%{category_keyword}%")
            if contractor:
                conditions.append("contractor ILIKE %s")
                sql_params.append(f"%{contractor}%")

            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

            # --- Core aggregates ---
            cur.execute(f"SELECT COUNT(*) FROM contracts{where_clause}", sql_params)
            total_contracts = cur.fetchone()[0]

            if is_availability_query:
                conn.close()
                scope = _build_stats_scope(
                    region,
                    province,
                    infra_year,
                    status,
                    category_keyword,
                    contractor,
                )
                availability = "Yes" if total_contracts > 0 else "No"
                return (
                    f"Availability Check {scope}:\n"
                    f"- Matching Contracts: {total_contracts:,}\n"
                    f"- Available: {availability}\n"
                    "- Use a listing request if you want to browse matching rows.\n"
                )

            cur.execute(
                f"SELECT COALESCE(SUM(budget), 0) FROM contracts{where_clause}",
                sql_params,
            )
            total_budget = float(cur.fetchone()[0])

            cur.execute(
                f"SELECT COALESCE(SUM(award_amount), 0) FROM contracts{where_clause}",
                sql_params,
            )
            total_award_amount = float(cur.fetchone()[0])

            cur.execute(
                f"SELECT COALESCE(AVG(progress), 0) FROM contracts{where_clause}",
                sql_params,
            )
            avg_progress = float(cur.fetchone()[0])

            # --- Status breakdown ---
            cur.execute(
                f"""
                SELECT status, COUNT(*) 
                FROM contracts{where_clause}
                GROUP BY status 
                ORDER BY COUNT(*) DESC 
                LIMIT 6
                """,
                sql_params,
            )
            status_rows = cur.fetchall()
            status_breakdown = ", ".join(
                f"{row[0] or 'Unknown'}: {row[1]:,}" for row in status_rows
            )

            # --- Top regions (only meaningful for global/province queries) ---
            if not region:
                cur.execute(
                    f"""
                    SELECT region, COUNT(*) 
                    FROM contracts{where_clause}
                    GROUP BY region 
                    ORDER BY COUNT(*) DESC 
                    LIMIT 5
                    """,
                    sql_params,
                )
                region_rows = cur.fetchall()
                region_breakdown = ", ".join(
                    f"{row[0] or 'Unknown'}: {row[1]:,}" for row in region_rows
                )
            else:
                region_breakdown = None

        conn.close()

    except Exception as e:
        print(f"Failed to calculate database statistics: {e}")
        return "Error: unable to process statistical counts on database tables"

    scope = _build_stats_scope(
        region,
        province,
        infra_year,
        status,
        category_keyword,
        contractor,
    )

    # --- Award-to-budget ratio ---
    award_to_budget_ratio = (
        (total_award_amount / total_budget * 100)
        if total_budget > 0 and total_award_amount > 0
        else None
    )
    award_ratio_text = (
        f"{award_to_budget_ratio:.1f}%" if award_to_budget_ratio is not None else "N/A"
    )

    output = (
        f"Statistics Summary {scope}:\n"
        f"- Total Contracts Matched: {total_contracts:,}\n"
        f"- Combined Budget: PHP {total_budget:,.2f}\n"
        f"- Total Award Amount: PHP {total_award_amount:,.2f}\n"
        f"- Award-to-Budget Ratio: {award_ratio_text}\n"
        f"- Average Progress: {avg_progress:.1f}%\n"
        f"- Status Breakdown: {status_breakdown or 'N/A'}\n"
    )

    if region_breakdown:
        output += f"- Top Regions: {region_breakdown}\n"

    return output


@tool
def filter_contracts(query: str) -> str:
    """
    Use this tool ONLY when the incoming query starts with 'Filter contracts where'.
    This performs structured SQL filtering on known contract attributes like
    contractor, region, province, status, category, infra_year, and program_name.
    Use this for exact or near-exact attribute lookups, NOT for descriptive searches.
    """

    filters = parse_filter_string(query)

    if not filters:
        return (
            "Error: Could not extract any valid filters from the query. "
            "Valid fields are: contractor, region, province, status, category, infra_year, program_name."
        )

    try:
        conn = psycopg2.connect(PG_DSN)
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            conditions = []
            params = []
            for field, value in filters.items():
                if field in FUZZY_FIELDS:
                    conditions.append(f"{field} ILIKE %s")
                    params.append(f"%{value}%")
                else:
                    # infra_year — exact match
                    conditions.append(f"{field} = %s")
                    params.append(value)

            where_clause = " AND ".join(conditions)

            cur.execute(
                f"""
                SELECT
                    contract_id, description, category, status,
                    budget, amount_paid, progress, region,
                    province, contractor, infra_year, program_name
                FROM contracts
                WHERE {where_clause}
                ORDER BY contract_id ASC
                LIMIT 50;
                """,
                params,
            )
            rows = cur.fetchall()

            # Also get a total count beyond the 50 cap
            cur.execute(
                f"SELECT COUNT(*) FROM contracts WHERE {where_clause}",
                params,
            )
            total_count = cur.fetchone()[0]

        conn.close()
    except Exception as e:
        print(f"filter_contracts DB error: {e}")
        return "Error: Database failure during filtered query"

    if not rows:
        applied = ", ".join(f"{k}={v}" for k, v in filters.items())
        return f"No contracts found matching filters: {applied}"

    SOURCE_MARKER = "__SOURCES__"
    sources = []
    summary_lines = []

    for r in rows:
        sources.append(
            {
                "description": r["description"],
                "contractId": r["contract_id"],
                "contractor": r["contractor"],
                "region": r["region"],
                "province": r["province"],
                "budget": float(r["budget"]) if r["budget"] else 0.0,
                "progress": r["progress"],
                "status": r["status"],
                "category": r["category"],
                "infraYear": r["infra_year"],
                "programName": r["program_name"],
            }
        )
        summary_lines.append(
            f"- {r['description']} | {r['contract_id']} | "
            f"{r['contractor']} | {r['region']} | {r['province']} | "
            f"PHP {float(r['budget']):,.2f} | {r['status']}"
        )

    applied_filters = ", ".join(f"{k}={v}" for k, v in filters.items())
    header = (
        f"Filtered contracts [{applied_filters}] — "
        f"showing {len(rows)} of {total_count:,} total matches "
        f"(capped match window, not ranked):\n\n"
    )

    sources_block = f"\n\n{SOURCE_MARKER}{json.dumps(sources)}"

    return header + "\n".join(summary_lines) + sources_block


@tool
def get_contract_detail(query: str) -> str:
    """
    Use this tool ONLY when the incoming query starts with 'Lookup contract'.
    This performs a direct database lookup for a specific contract by its ID
    or exact project name. Use this for point lookups, not broad searches.
    """

    parsed = parse_lookup_string(query)

    if not parsed:
        return (
            "Error: Could not extract a contract ID or project name "
            "from the lookup query."
        )

    lookup_type = parsed["lookup_type"]
    value = parsed["value"]

    conn = None
    try:
        conn = psycopg2.connect(PG_DSN)
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if lookup_type == "id":
                cur.execute(
                    """
                    SELECT
                        contract_id, description, category, status,
                        budget, amount_paid, award_amount, progress,
                        region, province, latitude, longitude, contractor,
                        advertisement_date, expiry_date, bid_submission_deadline,
                        start_date, completion_date, infra_year,
                        program_name, source_of_funds, raw_json
                    FROM contracts
                    WHERE contract_id = %s
                    LIMIT 1;
                    """,
                    (value,),
                )

            else:
                # Name lookup — try exact match first, fall back to fuzzy
                cur.execute(
                    """
                    SELECT
                        contract_id, description, category, status,
                        budget, amount_paid, award_amount, progress,
                        region, province, latitude, longitude, contractor,
                        advertisement_date, expiry_date, bid_submission_deadline,
                        start_date, completion_date, infra_year,
                        program_name, source_of_funds, raw_json
                    FROM contracts
                    WHERE description ILIKE %s
                    ORDER BY
                        -- Exact match ranks first
                        CASE WHEN LOWER(description) = LOWER(%s) THEN 0 ELSE 1 END,
                        -- Then closest prefix match
                        LENGTH(description) ASC
                    LIMIT 3;
                    """,
                    (f"%{value}%", value),
                )

            rows = cur.fetchall()

        if rows:
            r = rows[0]
            component_rows = []
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as comp_cur:
                    comp_cur.execute(
                        """
                        SELECT
                            component_order, component_id, description, type_of_work,
                            infra_type, region, province, latitude, longitude, raw_json
                        FROM contract_components
                        WHERE contract_id = %s
                        ORDER BY component_order ASC, id ASC;
                        """,
                        (r["contract_id"],),
                    )
                    component_rows = comp_cur.fetchall()
            except Exception as e:
                print(f"get_contract_detail component lookup error: {e}")

            return _format_contract_lookup_output(
                r, component_rows, value=value, lookup_type=lookup_type
            )

        # Exact-ID fallback: the source JSON exists locally even if the ingested
        # contracts table does not contain the row yet.
        if lookup_type == "id":
            local_record, component_rows = _load_local_contract_record(value)
            if local_record:
                return _format_contract_lookup_output(
                    local_record, component_rows, value=value, lookup_type=lookup_type
                )

        # Graceful fallback message — agent will then try web search
        return (
            f"No contract found matching '{value}'. "
            f"The contract ID may not exist or the project name may be spelled differently. "
            f"Try searching with broader terms instead."
        )

    except Exception as e:
        print(f"get_contract_detail DB error: {e}")
        return "Error: Database failure during contract lookup"
    finally:
        if conn is not None:
            conn.close()


tools = [
    search_contracts,
    get_contract_statistics,
    filter_contracts,
    get_contract_detail,
    web_search,
]
