from core.config import postgres_dsn

import psycopg2
import psycopg2.extras

PG_DSN: str = postgres_dsn()


def bm25_search(query: str, limit: int = 25) -> list[dict]:
    """
    Performs PostgreSQL full-text search using ts_rank_cd (BM25-like ranking).
    Returns up to `limit` candidates as dicts with chunk_text synthesized
    from description + category so the reranker has something to score.
    """

    # Convert query to tsquery
    # plainto_tsquery handles natural language input safely
    # e.g. "bridge contracts Region VIII" -> 'bridge' & 'contracts' & 'Region' & 'VIII'
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
                    c.progress,
                    c.region,
                    c.province,
                    c.contractor,
                    c.infra_year,
                    c.program_name,
                    ts_rank_cd(c.fts_vector, query) AS bm25_score
                FROM contracts c,
                     plainto_tsquery('english', %s) query
                WHERE c.fts_vector @@ query
                ORDER BY bm25_score DESC
                LIMIT %s;
                """,
                (query, limit),
            )
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        print(f"BM25 search error: {e}")
        return []

    results = []
    for r in rows:
        # Synthesize a chunk_text so the cross-encoder reranker
        # has a passage to score — mirrors what contract_embeddings stores
        chunk_text = (
            f"{r['description']}. "
            f"Category: {r['category'] or 'N/A'}. "
            f"Contractor: {r['contractor'] or 'N/A'}. "
            f"Region: {r['region'] or 'N/A'}, {r['province'] or 'N/A'}. "
            f"Status: {r['status'] or 'N/A'}. "
            f"Budget: PHP {float(r['budget']):,.2f}. "
            f"Program: {r['program_name'] or 'N/A'}."
        )
        results.append(
            {
                "chunk_text": chunk_text,
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
                "bm25_score": float(r["bm25_score"]),
            }
        )

    return results
