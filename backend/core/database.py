import importlib
import os
import time
from contextlib import contextmanager

from core.config import postgres_dsn

_pool = None


def psycopg2():
    return importlib.import_module("psycopg2")


def psycopg2_pool():
    return importlib.import_module("psycopg2.pool")


def connect(dsn: str | None = None, *, attempts: int = 3):
    db = psycopg2()
    target_dsn = dsn or postgres_dsn()
    attempts = max(1, attempts)
    last_error = None
    for attempt in range(attempts):
        try:
            return db.connect(target_dsn, connect_timeout=10)
        except db.OperationalError as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(0.5 * (attempt + 1))
    raise last_error


def init_pool(dsn: str | None = None):
    global _pool
    if _pool is not None:
        return _pool

    minconn = int(os.environ.get("POSTGRES_POOL_MIN", "1"))
    maxconn = int(os.environ.get("POSTGRES_POOL_MAX", "5"))
    target_dsn = dsn or postgres_dsn()
    _pool = psycopg2_pool().ThreadedConnectionPool(
        minconn,
        maxconn,
        target_dsn,
        connect_timeout=10,
    )
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


@contextmanager
def pooled_connection():
    """Borrow an autocommit connection for short API queries."""
    pool = init_pool()
    conn = pool.getconn()
    previous_autocommit = conn.autocommit
    conn.autocommit = True
    try:
        yield conn
    except Exception:
        raise
    finally:
        conn.autocommit = previous_autocommit
        pool.putconn(conn)
