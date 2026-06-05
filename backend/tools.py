import json
import os
from typing import Optional

import psycopg2
import psycopg2.extras
from embeddings import LocalAPIEmbeddings
from filter_parser import FUZZY_FIELDS, parse_filter_string
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from reranker import rerank

web_search = DuckDuckGoSearchRun()
embedding = LocalAPIEmbeddings()

PG_DSN: str = os.environ.get("PG_DSN") or (
    f"host={os.environ.get('POSTGRES_HOST')} "
    f"port={os.environ.get('POSTGRES_PORT')} "
    f"dbname={os.environ.get('POSTGRES_DB')} "
    f"user={os.environ.get('POSTGRES_USER')} "
    f"password={os.environ.get('POSTGRES_PASSWORD')}"
)


@tool
def search_contracts(query: str) -> str:
    """
    Use this tool ONLY when the incoming query starts with 'Find all contracts about'.
    This performs a semantic similarity vector search for descriptive project concepts."""

    try:
        query_vector = embedding.embed_query(query)
    except Exception as e:
        print(f"Failed to fetch embedding microservice: {e}")
        return "Error: Could not embed query for vector search"

    rows = []
    try:
        conn = psycopg2.connect(PG_DSN)
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT 
                    c.contract_id,
                    c.description,
                    c.category,
                    c.status,
                    c.budget,
                    c.amount_paid,
                    c.progress,
                    c.region,
                    c.province,
                    c.contractor,
                    c.infra_year,
                    c.program_name,
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
        print(f"Database query failure: {e}")
        return "Error: Database during similarity search"

    if not rows:
        return "No revelant contracts found in the database"

    candidates = []
    for r in rows:
        candidates.append(
            {
                "chunk_text": r["chunk_text"],
                "description": r["description"],
                "contract_id": r["contract_id"],
                "contractor": r["contractor"],
                "region": r["region"],
                "province": r["province"],
                "budget": float(r["budget"]) if r["budget"] is not None else 0.0,
                "amount_paid": float(
                    r["amount_paid"] if r["amount_paid"] is not None else 0.0
                )
                if r["amount_paid"] is not None
                else 0.0,
                "progress": r["progress"],
                "status": r["status"],
                "category": r["category"],
                "infra_year": r["infra_year"],
                "program_name": r["program_name"],
            }
        )

    seen_ids = set()
    unique_candidates = []

    for c in candidates:
        if c["contract_id"] not in seen_ids:
            seen_ids.add(c["contract_id"])
            unique_candidates.append(c)

    reranked = rerank(query, unique_candidates, 10)

    sources = []
    passages = []
    for r in reranked:
        sources.append(
            {
                "description": r["description"],
                "contract_id": r["contract_id"],
                "contractor": r["contractor"],
                "region": r["region"],
                "province": r["province"],
                "budget": r["budget"],
                "amount_paid": r["amount_paid"],
                "progress": r["progress"],
                "status": r["status"],
                "category": r["category"],
                "infra_year": r["infra_year"],
                "program_name": r["program_name"],
            }
        )
        passages.append(r["chunk_text"])

    # Structural marker read by agent.py to parse streaming source citations
    SOURCE_MARKER = "__SOURCES__"
    sources_block = f"\n\n{SOURCE_MARKER}{json.dumps(sources)}"

    content = "Here are the relevant DPWH contracts found:\n\n " + "\n\n---\n\n ".join(
        passages
    )

    return (
        "Here are relevant sources found"
        + content
        + "\n\nSources:\n"
        + "\n".join(
            f"- {s['description']} | {s['contract_id']} | {s['contractor']} | {s['region']} | {s['province']}"
            for s in sources
        )
        + sources_block
    )


@tool
def get_contract_statistics(
    region: Optional[str] = None,
    province: Optional[str] = None,
    infra_year: Optional[str] = None,
) -> str:
    """
    Use this tool ONLY when the incoming query starts with 'Calculate metrics for'.
    This tool extracts parameters to run SQL COUNT and SUM aggregates across structural columns."""
    try:
        conn = psycopg2.connect(PG_DSN)
        with conn.cursor() as cur:
            conditions = []
            params = []

            if region:
                conditions.append("region ILIKE %s")
                params.append(f"{region}")
            if province:
                conditions.append("province ILIKE %s")
                params.append(f"{province}")
            if infra_year:
                conditions.append("infra_year = %s")
                params.append(infra_year)

            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

            count_query = f"SELECT COUNT (*) FROM contracts{where_clause}"
            cur.execute(count_query, params)
            total_contracts = cur.fetchone()[0]

            budget_query = f"SELECT SUM(budget) FROM contracts {where_clause}"
            cur.execute(budget_query, params)
            total_budget = cur.fetchone()[0] or 0.0

            status_query = f"""
                SELECT status, COUNT(*) 
                FROM contracts 
                {where_clause} 
                GROUP BY status 
                ORDER BY COUNT(*) DESC 
                LIMIT 25;
            """
            cur.execute(status_query, params)
            status_rows = cur.fetchall()
            status_breakdown = ", ".join(
                [f"{row[0] or 'Unknown'}: {row[1]}" for row in status_rows]
            )

        conn.close()

        filter_desc = []
        if region:
            filter_desc.append(f"Region: {region}")
        if province:
            filter_desc.append(f"Province: {province}")
        if infra_year:
            filter_desc.append(f"Infra Year: {infra_year}")
        scope = (
            f"Filters applied -> [{', '.join(filter_desc)}]"
            if filter_desc
            else " (Global Scope)"
        )

        return (
            f"Database Statistics Summary{scope}:\n"
            f"- Total Matches Counted: {total_contracts:,}\n"
            f"- Comvined Filtered Budget: PHP {total_budget:,.2f}\n"
            f"- Breakdown by status: {status_breakdown if status_breakdown else 'None'}"
        )

    except Exception as e:
        print(f"Failed to calculate database statistics: {e}")
        return "Error: unable to process statistical counts on database tables"


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


tools = [
    search_contracts,
    get_contract_statistics,
    filter_contracts,
    web_search,
]
