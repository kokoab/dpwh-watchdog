from core.database import connect
from fastapi import APIRouter
from features.chat.tools.support import _psycopg2_extras

router = APIRouter(prefix="/library", tags=["library"])


@router.get("/contracts")
async def get_library_list_contracts():
    try:
        with connect() as conn:
            conn.autocommit = True
            with conn.cursor(cursor_factory=_psycopg2_extras().DictCursor) as cur:
                cur.execute("""
                    SELECT description
                    FROM contracts
                    LIMIT 10
                    """)
                rows = cur.fetchall()
    except Exception as e:
        return f"Error: Database connection error: {e}"

    return rows


@router.get("/contracts/{contract_id}")
async def get_library_contract_details(contract_id: str):
    normalized_id = str(contract_id or "").strip()
    if not normalized_id:
        return "No valid contract id"

    try:
        with connect() as conn:
            conn.autocommit = True
            with conn.cursor(cursor_factory=_psycopg2_extras().DictCursor) as cur:
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
                    (normalized_id,),
                )
                contract_details = cur.fetchone()
    except Exception as e:
        return f"Error: Database connection error: {e}"

    return contract_details
