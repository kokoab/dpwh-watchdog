from __future__ import annotations
from core.config import postgres_dsn

import importlib
import re
from typing import Any

PG_DSN: str = postgres_dsn()
_SCHEMA_READY = False


def _psycopg2():
    return importlib.import_module("psycopg2")


def _psycopg2_extras():
    return importlib.import_module("psycopg2.extras")


def _connect():
    return _psycopg2().connect(PG_DSN)


def _ensure_schema_ready() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    initialize_chat_memory_schema()


def initialize_chat_memory_schema() -> None:
    global _SCHEMA_READY
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_threads (
                        thread_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        title TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id BIGSERIAL PRIMARY KEY,
                        thread_id TEXT NOT NULL REFERENCES chat_threads(thread_id) ON DELETE CASCADE,
                        user_id TEXT,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        expanded_query TEXT,
                        intent TEXT,
                        message_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_thread_state (
                        thread_id TEXT PRIMARY KEY REFERENCES chat_threads(thread_id) ON DELETE CASCADE,
                        user_id TEXT,
                        scope JSONB NOT NULL DEFAULT '{}'::jsonb,
                        plan JSONB NOT NULL DEFAULT '{}'::jsonb,
                        result JSONB NOT NULL DEFAULT '{}'::jsonb,
                        selected_contract_id TEXT,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_threads_user_updated
                    ON chat_threads(user_id, updated_at DESC);
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created
                    ON chat_messages(thread_id, created_at DESC, id DESC);
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_messages_user_created
                    ON chat_messages(user_id, created_at DESC, id DESC);
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_messages_search
                    ON chat_messages
                    USING GIN (
                        to_tsvector(
                            'english',
                            COALESCE(content, '') || ' ' || COALESCE(expanded_query, '')
                        )
                    );
                    """
                )
        _SCHEMA_READY = True
    finally:
        conn.close()


def ensure_chat_thread(
    thread_id: str, user_id: str | None = None, title: str | None = None
) -> None:
    _ensure_schema_ready()
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_threads (thread_id, user_id, title)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (thread_id) DO UPDATE
                    SET
                        user_id = COALESCE(EXCLUDED.user_id, chat_threads.user_id),
                        title = COALESCE(chat_threads.title, EXCLUDED.title),
                        updated_at = NOW();
                    """,
                    (thread_id, user_id, title),
                )
    finally:
        conn.close()


def save_chat_message(
    thread_id: str,
    role: str,
    content: str,
    *,
    user_id: str | None = None,
    expanded_query: str | None = None,
    intent: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    _ensure_schema_ready()
    ensure_chat_thread(thread_id, user_id=user_id)
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_messages (
                        thread_id, user_id, role, content, expanded_query, intent, message_metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        thread_id,
                        user_id,
                        role,
                        content,
                        expanded_query,
                        intent,
                        _psycopg2_extras().Json(metadata or {}),
                    ),
                )
                cur.execute(
                    """
                    UPDATE chat_threads
                    SET
                        user_id = COALESCE(%s, user_id),
                        updated_at = NOW()
                    WHERE thread_id = %s;
                    """,
                    (user_id, thread_id),
                )
    finally:
        conn.close()


def get_thread_state(thread_id: str | None) -> dict[str, Any]:
    if not thread_id:
        return {}
    _ensure_schema_ready()

    conn = _connect()
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().RealDictCursor) as cur:
            cur.execute(
                """
                SELECT thread_id, user_id, scope, plan, result, selected_contract_id, updated_at
                FROM chat_thread_state
                WHERE thread_id = %s
                LIMIT 1;
                """,
                (thread_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else {}
    finally:
        conn.close()


def upsert_thread_state(
    thread_id: str,
    *,
    user_id: str | None = None,
    scope: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    selected_contract_id: str | None = None,
) -> None:
    _ensure_schema_ready()
    ensure_chat_thread(thread_id, user_id=user_id)
    scope_payload = scope if scope is not None else {}
    plan_payload = plan if plan is not None else {}
    result_payload = result if result is not None else {}
    has_scope = scope is not None
    has_plan = plan is not None
    has_result = result is not None
    has_selected_contract = selected_contract_id is not None
    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_thread_state (
                        thread_id, user_id, scope, plan, result, selected_contract_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id) DO UPDATE
                    SET
                        user_id = COALESCE(EXCLUDED.user_id, chat_thread_state.user_id),
                        scope = CASE
                            WHEN %s THEN EXCLUDED.scope
                            ELSE chat_thread_state.scope
                        END,
                        plan = CASE
                            WHEN %s THEN EXCLUDED.plan
                            ELSE chat_thread_state.plan
                        END,
                        result = CASE
                            WHEN %s THEN EXCLUDED.result
                            ELSE chat_thread_state.result
                        END,
                        selected_contract_id = CASE
                            WHEN %s THEN EXCLUDED.selected_contract_id
                            ELSE chat_thread_state.selected_contract_id
                        END,
                        updated_at = NOW();
                    """,
                    (
                        thread_id,
                        user_id,
                        _psycopg2_extras().Json(scope_payload),
                        _psycopg2_extras().Json(plan_payload),
                        _psycopg2_extras().Json(result_payload),
                        selected_contract_id,
                        has_scope,
                        has_plan,
                        has_result,
                        has_selected_contract,
                    ),
                )
                cur.execute(
                    """
                    UPDATE chat_threads
                    SET
                        user_id = COALESCE(%s, user_id),
                        updated_at = NOW()
                    WHERE thread_id = %s;
                    """,
                    (user_id, thread_id),
                )
    finally:
        conn.close()


def list_chat_threads(
    user_id: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    _ensure_schema_ready()
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().RealDictCursor) as cur:
            if user_id:
                cur.execute(
                    """
                    SELECT
                        t.thread_id,
                        t.user_id,
                        t.title,
                        t.created_at,
                        t.updated_at,
                        last_message.role AS last_message_role,
                        last_message.content AS last_message_content,
                        last_message.created_at AS last_message_created_at
                    FROM chat_threads t
                    LEFT JOIN LATERAL (
                        SELECT role, content, created_at
                        FROM chat_messages m
                        WHERE m.thread_id = t.thread_id
                        ORDER BY m.created_at DESC, m.id DESC
                        LIMIT 1
                    ) last_message ON TRUE
                    WHERE t.user_id = %s
                    ORDER BY t.updated_at DESC
                    LIMIT %s;
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()
                if rows:
                    return [dict(row) for row in rows]

                # Anonymous browser IDs can change across origins/restarts. Until
                # real auth exists, recover local chat history instead of showing
                # an empty sidebar while rows still exist in Postgres.
                cur.execute(
                    """
                    SELECT
                        t.thread_id,
                        t.user_id,
                        t.title,
                        t.created_at,
                        t.updated_at,
                        last_message.role AS last_message_role,
                        last_message.content AS last_message_content,
                        last_message.created_at AS last_message_created_at
                    FROM chat_threads t
                    LEFT JOIN LATERAL (
                        SELECT role, content, created_at
                        FROM chat_messages m
                        WHERE m.thread_id = t.thread_id
                        ORDER BY m.created_at DESC, m.id DESC
                        LIMIT 1
                    ) last_message ON TRUE
                    ORDER BY t.updated_at DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        t.thread_id,
                        t.user_id,
                        t.title,
                        t.created_at,
                        t.updated_at,
                        last_message.role AS last_message_role,
                        last_message.content AS last_message_content,
                        last_message.created_at AS last_message_created_at
                    FROM chat_threads t
                    LEFT JOIN LATERAL (
                        SELECT role, content, created_at
                        FROM chat_messages m
                        WHERE m.thread_id = t.thread_id
                        ORDER BY m.created_at DESC, m.id DESC
                        LIMIT 1
                    ) last_message ON TRUE
                    ORDER BY t.updated_at DESC
                    LIMIT %s;
                    """,
                    (limit,),
                )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def list_chat_messages(
    thread_id: str,
    *,
    user_id: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    _ensure_schema_ready()
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().RealDictCursor) as cur:
            if user_id:
                cur.execute(
                    """
                    SELECT id, thread_id, user_id, role, content, expanded_query, intent, message_metadata, created_at
                    FROM chat_messages
                    WHERE thread_id = %s AND (user_id = %s OR user_id IS NULL)
                    ORDER BY created_at ASC, id ASC
                    LIMIT %s;
                    """,
                    (thread_id, user_id, limit),
                )
                rows = cur.fetchall()
                if rows:
                    return [dict(row) for row in rows]

                # Same anonymous-ID recovery as list_chat_threads: if the thread
                # exists locally but belongs to a previous generated ID, keep it
                # loadable in this no-auth phase.
                cur.execute(
                    """
                    SELECT id, thread_id, user_id, role, content, expanded_query, intent, message_metadata, created_at
                    FROM chat_messages
                    WHERE thread_id = %s
                    ORDER BY created_at ASC, id ASC
                    LIMIT %s;
                    """,
                    (thread_id, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT id, thread_id, user_id, role, content, expanded_query, intent, message_metadata, created_at
                    FROM chat_messages
                    WHERE thread_id = %s
                    ORDER BY created_at ASC, id ASC
                    LIMIT %s;
                    """,
                    (thread_id, limit),
                )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def find_relevant_messages(
    thread_id: str, query: str, limit: int = 5
) -> list[dict[str, Any]]:
    _ensure_schema_ready()
    terms = [part for part in query.split() if part.strip()]
    is_reference_query = bool(
        re.search(
            r"\b(compare|those|these|them|again|earlier|previous|before|same)\b",
            query,
            re.IGNORECASE,
        )
    )
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=_psycopg2_extras().RealDictCursor) as cur:
            if len(terms) < 3 or is_reference_query:
                cur.execute(
                    """
                    SELECT id, thread_id, user_id, role, content, expanded_query, intent, message_metadata, created_at, 0.0 AS rank
                    FROM chat_messages
                    WHERE
                        thread_id = %s
                        AND (
                            intent IS NOT NULL AND intent <> 'chat'
                            OR message_metadata ? 'result_state'
                            OR message_metadata ? 'plan'
                        )
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s;
                    """,
                    (thread_id, limit),
                )
                return [dict(row) for row in cur.fetchall()]

            cur.execute(
                """
                SELECT
                    id,
                    thread_id,
                    user_id,
                    role,
                    content,
                    expanded_query,
                    intent,
                    message_metadata,
                    created_at,
                    ts_rank_cd(
                        to_tsvector(
                            'english',
                            COALESCE(content, '') || ' ' || COALESCE(expanded_query, '')
                        ),
                        websearch_to_tsquery('english', %s)
                    ) AS rank
                FROM chat_messages
                WHERE
                    thread_id = %s
                    AND (
                        intent IS NOT NULL AND intent <> 'chat'
                        OR message_metadata ? 'result_state'
                        OR message_metadata ? 'plan'
                    )
                ORDER BY rank DESC, created_at DESC, id DESC
                LIMIT %s;
                """,
                (query, thread_id, limit),
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def delete_thread_memory(thread_id: str | None, user_id: str | None) -> None:
    if not thread_id:
        return
    _ensure_schema_ready()

    conn = _connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chat_threads WHERE thread_id = %s AND user_id = %s;",
                    (thread_id, user_id),
                )
    finally:
        conn.close()
