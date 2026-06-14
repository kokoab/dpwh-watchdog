import psycopg2.extras
from auth.jwt import CurrentUser, verify_supabase_jwt
from core.config import postgres_dsn, super_admin_emails
from core.database import connect
from fastapi import Depends, HTTPException

PG_DSN = postgres_dsn()
SUPER_ADMIN_EMAILS = super_admin_emails()


def db_connect():
    return connect(PG_DSN)


def ensure_profile_and_get_role(user_id: str, email: str | None) -> str:
    normalized_email = (email or "").lower()
    bootstrap_role = "super_admin" if normalized_email in SUPER_ADMIN_EMAILS else "user"

    conn = db_connect()
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    insert into profiles (id, email, role)
                    values (%s, %s, %s)
                    on conflict (id) do update
                    set
                      email = excluded.email,
                      role = case
                        when profiles.role = 'user' and excluded.role = 'super_admin'
                        then 'super_admin'::app_role
                        else profiles.role
                      end,
                      updated_at = now()
                    returning role;
                    """,
                    (user_id, normalized_email, bootstrap_role),
                )
                row = cur.fetchone()
                return row["role"]
    finally:
        conn.close()


def get_current_user(payload: dict = Depends(verify_supabase_jwt)) -> CurrentUser:
    user_id = payload["sub"]
    email = payload.get("email")
    role = ensure_profile_and_get_role(user_id, email)

    return CurrentUser(id=user_id, email=email, role=role)


def require_admin(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if current_user.role not in {"admin", "super_admin"}:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user
