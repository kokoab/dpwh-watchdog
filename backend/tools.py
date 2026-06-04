import json
import os
from typing import Optional

import psycopg2
import psycopg2.extras
from embeddings import LocalAPIEmbeddings
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun 

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
    Search the local vector database for DPWH (Department of Public Works and Highways)
    contract records, bidding information, procurement history, and infrastructure agreements.
    Use this tool whenever the user asks about specific contract details or local project data.
    """

    try:
        query_vector = embedding.embed_query(query)
    except Exception as e:
        print(f"Failed to fetch embedding microservice: {e}")
        return "Error: Could not embed query for vector search"

    results = []
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
                LIMIT 5;
            """,
                (query_vector,),
            )

            results = cur.fetchall()
        conn.close()
    except Exception as e:
        print(f"Database query failure: {e}")
        return "Error: Database during similarity search"

    if not results:
        return "No revelant contracts found in the database"

    sources = []
    passages = []

    for r in results:
        sources.append(
            {
                "description": r["description"],
                "contractId": r["contract_id"],
                "contractor": r["contractor"],
                "region": r["region"],
                "province": r["province"],
                "budget": float(r["budget"]) if r["budget"] is not None else 0.0,
                "amountPaid": float(r["amount_paid"])
                if r["amount_paid"] is not None
                else 0.0,
                "progress": r["progress"],
                "status": r["status"],
                "category": r["category"],
                "infraYear": r["infra_year"],
                "programName": r["program_name"],
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
            f"- {s['description']} | {s['contractId']} | {s['contractor']} | {s['region']} | {s['province']}"
            for s in sources
        )
        + sources_block
    )


tools = [
    search_contracts,
    web_search,
]
