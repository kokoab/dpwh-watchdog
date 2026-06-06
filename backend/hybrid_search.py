import os
import re

import psycopg2
import psycopg2.extras

from bm25_search import bm25_search
from stats_parser import parse_stats_string

# RRF constant — 60 is the standard value from the original paper
# Higher = diminishes the impact of rank differences
RRF_K = 60

PG_DSN: str = os.environ.get("PG_DSN") or (
    f"host={os.environ.get('POSTGRES_HOST')} "
    f"port={os.environ.get('POSTGRES_PORT')} "
    f"dbname={os.environ.get('POSTGRES_DB')} "
    f"user={os.environ.get('POSTGRES_USER')} "
    f"password={os.environ.get('POSTGRES_PASSWORD')}"
)

CATEGORY_HINT_TERMS = {
    "bridge": ["bridge", "bridges"],
    "road": ["road", "roads"],
    "flood control": ["flood control", "drainage", "river control"],
    "school": ["school", "deped", "beff"],
    "building": ["building", "covered court", "multi-purpose", "multi purpose"],
    "water supply": ["water", "water system", "water supply", "rwcs"],
}


def _synthesize_chunk_text(candidate: dict) -> str:
    return (
        f"{candidate['description']}. "
        f"Category: {candidate['category'] or 'N/A'}. "
        f"Contractor: {candidate['contractor'] or 'N/A'}. "
        f"Region: {candidate['region'] or 'N/A'}, {candidate['province'] or 'N/A'}. "
        f"Status: {candidate['status'] or 'N/A'}. "
        f"Budget: PHP {float(candidate['budget']):,.2f}. "
        f"Program: {candidate['program_name'] or 'N/A'}."
    )


def _metadata_terms(query: str) -> tuple[dict, list[str]]:
    """
    Extract structured hints plus the raw category terms we can search on.
    This keeps the search deterministic and much more precise than vector-only
    retrieval for region/category-style queries.
    """

    hints = parse_stats_string(query)
    terms: list[str] = []

    category = hints.get("category_keyword")
    if category:
        terms.extend(CATEGORY_HINT_TERMS.get(category, [category]))

    # Preserve a couple of very common phrases even when the parser
    # collapses them to a broader category.
    lower_query = query.lower()
    if "covered court" in lower_query and "covered court" not in terms:
        terms.insert(0, "covered court")
    if "multi-purpose" in lower_query and "multi-purpose" not in terms:
        terms.append("multi-purpose")
    if "school building" in lower_query and "school" not in terms:
        terms.append("school")

    # Deduplicate while preserving order.
    seen = set()
    deduped_terms = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            deduped_terms.append(term)

    return hints, deduped_terms


def metadata_search(query: str, limit: int = 25) -> list[dict]:
    """
    Pull extra candidates using explicit region/category/contractor hints.
    This complements vector and BM25 retrieval and is especially helpful for
    precise queries like "school building projects in Region XI".
    """

    hints, terms = _metadata_terms(query)

    if not any(hints.get(key) for key in ("region", "province", "contractor", "status", "category_keyword")):
        return []

    conditions = []
    sql_params = []

    if hints.get("region"):
        conditions.append("c.region ILIKE %s")
        sql_params.append(f"%{hints['region']}%")

    if hints.get("province"):
        conditions.append("c.province ILIKE %s")
        sql_params.append(f"%{hints['province']}%")

    if hints.get("contractor"):
        conditions.append("c.contractor ILIKE %s")
        sql_params.append(f"%{hints['contractor']}%")

    if hints.get("status"):
        conditions.append("c.status ILIKE %s")
        sql_params.append(f"%{hints['status']}%")

    term_clauses = []
    for term in terms:
        term_clauses.append(
            "(c.description ILIKE %s OR c.category ILIKE %s OR c.program_name ILIKE %s)"
        )
        sql_params.extend([f"%{term}%", f"%{term}%", f"%{term}%"])

    if term_clauses:
        conditions.append("(" + " OR ".join(term_clauses) + ")")

    if not conditions:
        return []

    where_clause = " WHERE " + " AND ".join(conditions)

    # Build a simple deterministic score to rank the exact metadata matches first.
    score_parts = []
    score_params = []
    if hints.get("region"):
        score_parts.append("CASE WHEN c.region ILIKE %s THEN 30 ELSE 0 END")
        score_params.append(f"%{hints['region']}%")
    if hints.get("province"):
        score_parts.append("CASE WHEN c.province ILIKE %s THEN 20 ELSE 0 END")
        score_params.append(f"%{hints['province']}%")
    if hints.get("contractor"):
        score_parts.append("CASE WHEN c.contractor ILIKE %s THEN 15 ELSE 0 END")
        score_params.append(f"%{hints['contractor']}%")
    if hints.get("status"):
        score_parts.append("CASE WHEN c.status ILIKE %s THEN 10 ELSE 0 END")
        score_params.append(f"%{hints['status']}%")
    for term in terms:
        score_parts.append(
            "CASE WHEN (c.description ILIKE %s OR c.category ILIKE %s OR c.program_name ILIKE %s) THEN 5 ELSE 0 END"
        )
        score_params.extend([f"%{term}%", f"%{term}%", f"%{term}%"])

    score_sql = " + ".join(score_parts) if score_parts else "0"

    try:
        conn = psycopg2.connect(PG_DSN)
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                f"""
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
                    {score_sql} AS hint_score
                FROM contracts c
                {where_clause}
                ORDER BY hint_score DESC, c.contract_id ASC
                LIMIT %s;
                """,
                tuple(score_params + sql_params + [limit]),
            )
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        print(f"Metadata search error: {e}")
        return []

    results = []
    for r in rows:
        candidate = {
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
        candidate["chunk_text"] = _synthesize_chunk_text(candidate)
        candidate["metadata_score"] = float(r["hint_score"] or 0)
        results.append(candidate)

    return results


def reciprocal_rank_fusion(*ranked_lists: list[dict]) -> list[dict]:
    """
    Merges two ranked lists using Reciprocal Rank Fusion.
    Each candidate scores: sum of 1/(k + rank) across lists it appears in.
    Candidates appearing in both lists get a significant boost.
    """

    scores: dict[str, float] = {}
    candidates: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, candidate in enumerate(ranked, start=1):
            cid = candidate["contract_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            if cid not in candidates:
                candidates[cid] = candidate

    # Sort by fused score descending
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    merged = []
    for cid in sorted_ids:
        candidate = candidates[cid]
        candidate["_rrf_score"] = scores[cid]
        merged.append(candidate)

    return merged


def hybrid_search(query: str, vector_results: list[dict]) -> list[dict]:
    """
    Takes already-fetched vector_results plus runs a fresh BM25 search,
    then merges them with RRF.

    vector_results: list of candidate dicts from your existing vector search
                    (must have contract_id and chunk_text keys)
    Returns: merged list sorted by RRF score, ready for reranker input
    """

    bm25_results = bm25_search(query, limit=25)
    metadata_results = metadata_search(query, limit=25)

    if not bm25_results:
        # BM25 found nothing (very short query, stop words only, etc.)
        # Fall back to vector results alone
        if not metadata_results:
            return vector_results
        if not vector_results:
            return metadata_results
        return reciprocal_rank_fusion(vector_results, metadata_results)

    if not vector_results:
        if not metadata_results:
            return bm25_results
        return reciprocal_rank_fusion(bm25_results, metadata_results)

    if metadata_results:
        merged = reciprocal_rank_fusion(vector_results, bm25_results, metadata_results)
    else:
        merged = reciprocal_rank_fusion(vector_results, bm25_results)
    return merged
