from bm25_search import bm25_search

# RRF constant — 60 is the standard value from the original paper
# Higher = diminishes the impact of rank differences
RRF_K = 60


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
) -> list[dict]:
    """
    Merges two ranked lists using Reciprocal Rank Fusion.
    Each candidate scores: sum of 1/(k + rank) across lists it appears in.
    Candidates appearing in both lists get a significant boost.
    """

    scores: dict[str, float] = {}
    candidates: dict[str, dict] = {}

    # Score vector results
    for rank, candidate in enumerate(vector_results, start=1):
        cid = candidate["contract_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        candidates[cid] = candidate

    # Score BM25 results — candidates in both lists get cumulative score
    for rank, candidate in enumerate(bm25_results, start=1):
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

    if not bm25_results:
        # BM25 found nothing (very short query, stop words only, etc.)
        # Fall back to vector results alone
        return vector_results

    if not vector_results:
        return bm25_results

    merged = reciprocal_rank_fusion(vector_results, bm25_results)
    return merged
