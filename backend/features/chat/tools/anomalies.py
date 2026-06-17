from core.config import postgres_dsn

from features.chat.agent.query_planner import QueryPlan
from features.chat.tools.support import (
    _build_contract_where_clause,
    _normalize_result_filters,
    _normalized_plan_filters,
    _psycopg2,
    _psycopg2_extras,
)

PG_DSN: str = postgres_dsn()

def analyze_contractor_concentration(plan: QueryPlan) -> dict[str, object]:
    filters = _normalized_plan_filters(plan)
    where_clause, params = _build_contract_where_clause(filters)
    where_sql = f"WHERE {where_clause}" if where_clause else ""
    conn = _psycopg2().connect(PG_DSN)
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().RealDictCursor) as cur:
            cur.execute(
                f"""
                WITH scoped AS (
                    SELECT contractor, COALESCE(budget, 0) AS budget
                    FROM contracts
                    {where_sql}
                ),
                totals AS (
                    SELECT COUNT(*)::float AS total_contracts, COALESCE(SUM(budget), 0)::float AS total_budget
                    FROM scoped
                )
                SELECT
                    scoped.contractor,
                    COUNT(*)::int AS contract_count,
                    COALESCE(SUM(scoped.budget), 0)::float AS total_budget,
                    CASE WHEN totals.total_contracts > 0 THEN COUNT(*)::float / totals.total_contracts ELSE 0 END AS contract_share,
                    CASE WHEN totals.total_budget > 0 THEN COALESCE(SUM(scoped.budget), 0)::float / totals.total_budget ELSE 0 END AS budget_share
                FROM scoped
                CROSS JOIN totals
                GROUP BY scoped.contractor, totals.total_contracts, totals.total_budget
                ORDER BY contract_share DESC, budget_share DESC, scoped.contractor ASC
                LIMIT 25;
                """,
                params,
            )
            rows = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

    flagged_rows = [
        row
        for row in rows
        if float(row.get("contract_share") or 0) > 0.40
        or float(row.get("budget_share") or 0) > 0.40
    ]
    return {
        "analysis_type": "contractor_concentration",
        "filters": filters,
        "rows": rows,
        "flagged_rows": flagged_rows,
    }

def detect_budget_anomalies(plan: QueryPlan) -> dict[str, object]:
    filters = _normalized_plan_filters(plan)
    where_clause, params = _build_contract_where_clause(filters)
    where_sql = f"WHERE {where_clause}" if where_clause else ""
    conn = _psycopg2().connect(PG_DSN)
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT
                    contract_id,
                    description,
                    category,
                    region,
                    infra_year,
                    budget,
                    award_amount,
                    CASE
                        WHEN budget IS NULL OR budget = 0 OR award_amount IS NULL THEN NULL
                        ELSE award_amount / budget
                    END AS award_budget_ratio
                FROM contracts
                {where_sql}
                """,
                params,
            )
            rows = [
                dict(row)
                for row in cur.fetchall()
                if row.get("award_budget_ratio") is not None
                and (
                    float(row.get("award_budget_ratio") or 0) < 0.60
                    or float(row.get("award_budget_ratio") or 0) > 1.05
                )
            ]
    finally:
        conn.close()
    return {"analysis_type": "budget_anomalies", "filters": filters, "rows": rows}

def detect_timeline_anomalies(plan: QueryPlan) -> dict[str, object]:
    filters = _normalized_plan_filters(plan)
    where_clause, params = _build_contract_where_clause(filters)
    where_sql = f"WHERE {where_clause}" if where_clause else ""
    conn = _psycopg2().connect(PG_DSN)
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().RealDictCursor) as cur:
            cur.execute(
                f"""
                WITH scoped AS (
                    SELECT contract_id, description, status, progress, start_date, completion_date, expiry_date
                    FROM contracts
                    {where_sql}
                )
                SELECT * FROM (
                    SELECT 'completion_past_due' AS anomaly_label, * FROM scoped
                    WHERE completion_date < CURRENT_DATE
                      AND status NOT IN ('Completed', 'Terminated', 'Suspended')
                    UNION ALL
                    SELECT 'zero_progress_stale' AS anomaly_label, * FROM scoped
                    WHERE COALESCE(progress, 0) = 0
                      AND start_date < CURRENT_DATE - INTERVAL '12 months'
                    UNION ALL
                    SELECT 'expiry_past_due' AS anomaly_label, * FROM scoped
                    WHERE expiry_date < CURRENT_DATE
                      AND status IN ('On-Going', 'Awarded')
                ) anomalies
                ORDER BY anomaly_label ASC, contract_id ASC
                LIMIT 200;
                """,
                params,
            )
            rows = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()
    return {"analysis_type": "timeline_anomalies", "filters": filters, "rows": rows}

def detect_bidding_anomalies(plan: QueryPlan) -> dict[str, object]:
    filters = _normalized_plan_filters(plan)
    where_clause, params = _build_contract_where_clause(filters)
    contract_where_sql = f"WHERE {where_clause}" if where_clause else ""
    bidder_where_sql = f"AND {where_clause}" if where_clause else ""
    conn = _psycopg2().connect(PG_DSN)
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT
                    c.contract_id,
                    c.description,
                    COUNT(cb.*)::int AS bidder_count
                FROM contracts c
                JOIN contract_bidders cb ON cb.contract_id = c.contract_id
                {contract_where_sql}
                GROUP BY c.contract_id, c.description
                HAVING COUNT(cb.*) = 1
                ORDER BY c.contract_id ASC
                LIMIT 100;
                """,
                params,
            )
            single_bidder_rows = [dict(row) for row in cur.fetchall()]

            cur.execute(
                f"""
                WITH bidder_sets AS (
                    SELECT
                        c.contract_id,
                        string_agg(DISTINCT cb.pcab_id, ',' ORDER BY cb.pcab_id) AS bidder_set
                    FROM contracts c
                    JOIN contract_bidders cb ON cb.contract_id = c.contract_id
                    WHERE cb.pcab_id IS NOT NULL AND BTRIM(cb.pcab_id) <> ''
                    {bidder_where_sql}
                    GROUP BY c.contract_id
                )
                SELECT
                    bidder_set,
                    COUNT(*)::int AS contract_count,
                    array_agg(contract_id ORDER BY contract_id) AS contract_ids
                FROM bidder_sets
                GROUP BY bidder_set
                HAVING COUNT(*) >= 3
                ORDER BY contract_count DESC, bidder_set ASC
                LIMIT 50;
                """,
                params,
            )
            recurring_bidder_sets = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()
    return {
        "analysis_type": "bidding_anomalies",
        "filters": filters,
        "single_bidder_rows": single_bidder_rows,
        "recurring_bidder_sets": recurring_bidder_sets,
    }

def detect_document_gaps(plan: QueryPlan) -> dict[str, object]:
    filters = _normalized_plan_filters(plan)
    where_clause, params = _build_contract_where_clause(filters)
    where_sql = f"WHERE {where_clause}" if where_clause else ""
    conn = _psycopg2().connect(PG_DSN)
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT
                    contract_id,
                    description,
                    raw_json -> 'links' ->> 'contractAgreement' AS contract_agreement,
                    raw_json -> 'links' ->> 'noticeOfAward' AS notice_of_award,
                    raw_json -> 'links' ->> 'noticeToProceed' AS notice_to_proceed
                FROM contracts
                {where_sql};
                """,
                params,
            )
            rows = []
            for row in cur.fetchall():
                payload = dict(row)
                payload["missing_document_count"] = sum(
                    1
                    for key in (
                        "contract_agreement",
                        "notice_of_award",
                        "notice_to_proceed",
                    )
                    if not str(payload.get(key) or "").strip()
                )
                if payload["missing_document_count"] >= 2:
                    rows.append(payload)
    finally:
        conn.close()
    return {"analysis_type": "document_gaps", "filters": filters, "rows": rows}

def find_similar_scope_contracts(reference_id: str, plan: QueryPlan) -> dict[str, object]:
    filters = _normalized_plan_filters(plan)
    conn = _psycopg2().connect(PG_DSN)
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().RealDictCursor) as cur:
            cur.execute(
                """
                SELECT embedding
                FROM contract_embeddings
                WHERE contract_id = %s
                LIMIT 1;
                """,
                (reference_id,),
            )
            reference = cur.fetchone()
            if not reference:
                return {
                    "analysis_type": "similar_scope",
                    "reference_id": reference_id,
                    "filters": filters,
                    "rows": [],
                    "error": "Reference embedding not found.",
                }

            where_clause, params = _build_contract_where_clause(filters)
            extra_clause = f" AND {where_clause}" if where_clause else ""
            cur.execute(
                f"""
                SELECT
                    c.contract_id,
                    c.description,
                    c.category,
                    c.region,
                    c.province,
                    c.contractor,
                    1 - (e.embedding <=> %s::vector) AS similarity_score
                FROM contract_embeddings e
                JOIN contracts c ON c.contract_id = e.contract_id
                WHERE c.contract_id <> %s
                {extra_clause}
                ORDER BY e.embedding <=> %s::vector
                LIMIT 10;
                """,
                [reference["embedding"], reference_id, *params, reference["embedding"]],
            )
            rows = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()
    return {
        "analysis_type": "similar_scope",
        "reference_id": reference_id,
        "filters": filters,
        "rows": rows,
    }

