import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def postgres_dsn() -> str:
    return os.environ.get("PG_DSN") or (
        f"host={os.environ.get('POSTGRES_HOST')} "
        f"port={os.environ.get('POSTGRES_PORT')} "
        f"dbname={os.environ.get('POSTGRES_DB')} "
        f"user={os.environ.get('POSTGRES_USER')} "
        f"password={os.environ.get('POSTGRES_PASSWORD')}"
    )


def comma_separated_set(name: str) -> set[str]:
    return {
        value.strip().lower()
        for value in os.environ.get(name, "").split(",")
        if value.strip()
    }


def cors_allowed_origins() -> list[str]:
    return [
        origin.strip()
        for origin in os.environ.get(
            "CORS_ALLOWED_ORIGINS",
            "http://localhost:5173",
        ).split(",")
        if origin.strip()
    ]


def super_admin_emails() -> set[str]:
    return comma_separated_set("SUPER_ADMIN_EMAILS")
