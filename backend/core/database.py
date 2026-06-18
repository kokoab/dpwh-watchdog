import importlib
import time

from core.config import postgres_dsn


def psycopg2():
    return importlib.import_module("psycopg2")

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
