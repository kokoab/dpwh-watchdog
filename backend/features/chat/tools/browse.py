import json

from core.config import postgres_dsn
from contracts.embeddings import LocalAPIEmbeddings
from contracts.filter_parser import parse_filter_string
from contracts.hybrid_search import hybrid_search, structured_match_count, structured_match_ids
from contracts.reranker import rerank
from features.chat.tools.lookup import (
    _exclude_selected_contract_rows,
    _fetch_contract_rows,
    _format_contract_source_row,
    _record_result_state,
    _resolve_result_context,
    _should_exclude_selected_contract,
    _source_matches_filters,
    _summarize_sources,
)
from features.chat.tools.support import (
    _format_filter_phrase,
    _normalize_result_filters,
    _psycopg2,
    _psycopg2_extras,
)
from langchain.tools import tool

embedding = LocalAPIEmbeddings()
PG_DSN: str = postgres_dsn()
FILTER_MATCH_LIMIT = 10
RESULT_STATE_ID_CAP = 100

@tool
def search_contracts(query: str) -> str:
    """
    Use this tool ONLY when the incoming query starts with 'Find all contracts about'.
    This performs hybrid semantic + keyword search for descriptive project concepts.
    """

    result_intent, result_filters, result_subject = _resolve_result_context(
        "search",
        {},
        "",
    )
    structured_total = structured_match_count(query)
    structured_ids = structured_match_ids(query)
    if structured_total == 0:
        _record_result_state(
            {
                "result_kind": "contract_set",
                "intent": result_intent,
                "filters": result_filters,
                "subject": result_subject,
                "count": 0,
                "contract_ids": [],
                "displayed_contract_ids": [],
                "displayed_sources": [],
                "is_complete_result_set": True,
            }
        )
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
        conn = _psycopg2().connect(PG_DSN)
        with conn.cursor(cursor_factory=_psycopg2_extras().DictCursor) as cur:
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
    exclude_selected_contract, selected_source, selected_contract_id = _should_exclude_selected_contract()
    if exclude_selected_contract and selected_contract_id:
        reranked = [
            candidate
            for candidate in reranked
            if str(candidate.get("contract_id") or "").strip() != selected_contract_id
        ]
        if not reranked:
            contractor_name = (
                str(selected_source.get("contractor") or "").strip()
                if isinstance(selected_source, dict)
                else ""
            )
            if contractor_name:
                return f"No other projects were found for contractor {contractor_name}."
            return "No relevant contracts found in the database"

    SOURCE_MARKER = "__SOURCES__"
    sources = []
    source_rows = []

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
        source_rows.append(_format_contract_source_row(r))

    contract_ids = [row["contract_id"] for row in reranked]
    recorded_ids = contract_ids[:RESULT_STATE_ID_CAP]
    result_count = structured_total if structured_total is not None else len(contract_ids)
    if exclude_selected_contract and selected_contract_id and result_count > 0:
        result_count = max(result_count - 1, 0)
    _record_result_state(
        {
            "result_kind": "contract_set",
            "intent": result_intent,
            "filters": result_filters,
            "subject": result_subject,
            "count": result_count,
            "contract_ids": recorded_ids,
            "displayed_contract_ids": contract_ids,
            "displayed_sources": sources,
            "is_complete_result_set": structured_total is not None,
        }
    )

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

    content = (
        f"{result_scope}\n\n"
        "Source rows (search-ranked; discuss these contracts directly and do not infer extra analytics):\n\n"
        + "\n\n".join(source_rows)
    )

    return (
        "Here are relevant source rows\n\n"
        + content
        + "\n\nSources:\n"
        + "\n".join(
            f"- {s['description']} | {s['contractId']} | "
            f"{s['contractor']} | {s['region']} | {s['province']}"
            for s in sources
        )
        + sources_block
    )
