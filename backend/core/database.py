import importlib

from core.config import postgres_dsn


def psycopg2():
    return importlib.import_module("psycopg2")


def connect(dsn: str | None = None):
    return psycopg2().connect(dsn or postgres_dsn())
