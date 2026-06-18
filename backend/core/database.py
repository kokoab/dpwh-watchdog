import importlib
import os
import time
from contextlib import contextmanager

from core.config import postgres_dsn

_pool = None


class PooledConnection:
    def __init__(self, pool, conn):
        object.__setattr__(self, "_pool", pool)
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_previous_autocommit", conn.autocommit)
        object.__setattr__(self, "_transaction_closed", False)
        object.__setattr__(self, "_returned", False)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        setattr(self._conn, name, value)

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._conn.__exit__(exc_type, exc, traceback)
        self._transaction_closed = True
        self.close()

    def close(self):
        if self._returned:
            return
        if not self._conn.closed:
            if not self._conn.autocommit and not self._transaction_closed:
                self._conn.rollback()
            self._conn.autocommit = self._previous_autocommit
        self._pool.putconn(self._conn)
        self._returned = True


def psycopg2():
    return importlib.import_module("psycopg2")


def psycopg2_pool():
    return importlib.import_module("psycopg2.pool")


def connect(dsn: str | None = None, *, attempts: int = 3):
    pool = init_pool(dsn=dsn, attempts=attempts)
    return PooledConnection(pool, pool.getconn())


def direct_connect(dsn: str | None = None, *, attempts: int = 3):
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


def init_pool(dsn: str | None = None, *, attempts: int = 3):
    global _pool
    if _pool is not None:
        return _pool

    minconn = int(os.environ.get("POSTGRES_POOL_MIN", "1"))
    maxconn = int(os.environ.get("POSTGRES_POOL_MAX", "5"))
    target_dsn = dsn or postgres_dsn()
    attempts = max(1, attempts)
    last_error = None
    for attempt in range(attempts):
        try:
            _pool = psycopg2_pool().ThreadedConnectionPool(
                minconn,
                maxconn,
                target_dsn,
                connect_timeout=10,
            )
            break
        except psycopg2().OperationalError as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(0.5 * (attempt + 1))
    if _pool is None:
        raise last_error
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
