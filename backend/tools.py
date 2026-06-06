import json
import os
from datetime import date, datetime
from typing import Optional

import psycopg2
import psycopg2.extras
from embeddings import LocalAPIEmbeddings
from filter_parser import FUZZY_FIELDS, parse_filter_string
from hybrid_search import hybrid_search
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


def _format_date(value) -> str:
    if value in (None, ""):
        return "N/A"
    if isinstance(value, (date, datetime)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    return str(value)[:10]


def _contract_duration(start_value, completion_value) -> str:
    if not start_value or not completion_value:
        return "N/A"

    start = start_value.date() if isinstance(start_value, datetime) else start_value
    completion = (
        completion_value.date()
        if isinstance(completion_value, datetime)
        else completion_value
    )

    if not isinstance(start, date) or not isinstance(completion, date):
        return "N/A"

    delta_days = (completion - start).days
    if delta_days < 0:
        return "N/A"
    return f"{delta_days} day(s)"


@tool
def search_contracts(query: str) -> str:
    """
    Use this tool ONLY when the incoming query starts with 'Find all contracts about'.
    This performs hybrid semantic + keyword search for descriptive project concepts.
    """

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
                    c.budget, c.amount_paid, c.progress, c.region,
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
                "amount_paid": float(r["amount_paid"]) if r["amount_paid"] else 0.0,
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
                "amountPaid": r["amount_paid"],
                "progress": r["progress"],
                "status": r["status"],
                "category": r["category"],
                "infraYear": r["infra_year"],
                "programName": r["program_name"],
            }
        )
        passages.append(r["chunk_text"])

    sources_block = f"\n\n{SOURCE_MARKER}{json.dumps(sources)}"
    content = "Here are the relevant DPWH contracts found:\n\n " + "\n\n---\n\n ".join(
        passages
    )

    return (
        "Here are relevant sources found"
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

    params = parse_stats_string(query)

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

            cur.execute(
                f"SELECT COALESCE(SUM(budget), 0) FROM contracts{where_clause}",
                sql_params,
            )
            total_budget = float(cur.fetchone()[0])

            cur.execute(
                f"SELECT COALESCE(SUM(amount_paid), 0) FROM contracts{where_clause}",
                sql_params,
            )
            total_paid = float(cur.fetchone()[0])

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

    # --- Build human-readable scope description ---
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

    scope = f"[{' | '.join(scope_parts)}]" if scope_parts else "[Global Scope]"

    # --- Budget utilization rate ---
    utilization = (total_paid / total_budget * 100) if total_budget > 0 else 0.0

    output = (
        f"Statistics Summary {scope}:\n"
        f"- Total Contracts Matched: {total_contracts:,}\n"
        f"- Combined Budget: PHP {total_budget:,.2f}\n"
        f"- Total Amount Paid: PHP {total_paid:,.2f}\n"
        f"- Budget Utilization Rate: {utilization:.1f}%\n"
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
                ORDER BY budget DESC NULLS LAST
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
                "amountPaid": float(r["amount_paid"]) if r["amount_paid"] else 0.0,
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
        f"(sorted by budget descending):\n\n"
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

        if not rows:
            # Graceful fallback message — agent will then try web search
            return (
                f"No contract found matching '{value}'. "
                f"The contract ID may not exist or the project name may be spelled differently. "
                f"Try searching with broader terms instead."
            )

        SOURCE_MARKER = "__SOURCES__"
        sources = []
        detail_blocks = []

        for r in rows:
            budget = float(r["budget"]) if r["budget"] else 0.0
            amount_paid = float(r["amount_paid"]) if r["amount_paid"] else 0.0
            award_amount = float(r["award_amount"]) if r["award_amount"] else 0.0
            utilization = (amount_paid / budget * 100) if budget > 0 else 0.0
            contract_duration = _contract_duration(
                r["start_date"], r["completion_date"]
            )

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

            sources.append(
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
            )

            # Build a rich detail block for the LLM to summarize
            detail_blocks.append(
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
                f"Amount Paid:        PHP {amount_paid:,.2f}\n"
                f"Award Amount:       PHP {award_amount:,.2f}\n"
                f"Utilization Rate:   {utilization:.1f}%\n"
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
                detail_blocks.append(
                    "CONTRACT COMPONENTS\n"
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

        sources_block = f"\n\n{SOURCE_MARKER}{json.dumps(sources)}"

        header = (
            f"Direct lookup result for '{value}' "
            f"({'exact ID match' if lookup_type == 'id' else 'name match'}):\n\n"
        )

        return header + "\n\n".join(detail_blocks) + sources_block

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
