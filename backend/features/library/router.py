from core.database import connect
from fastapi import APIRouter
from features.chat.tools.support import _psycopg2_extras

router = APIRouter(prefix="/library", tags=["library"])


@router.get("/contracts")
async def library_list_contracts():
    try:
        conn = connect()

        with conn.cursor(cursor_factory=_psycopg2_extras().DictCursor) as cur:
            cur.execute(
                """
                SELECT description
                FROM contracts
                LIMIT 10
                """
            )
            rows = cur.fetchall()
        cur.close()
    except Exception as e:
        return f"Error: Database connection error: {e}"

    return rows
