import os
from dataclasses import dataclass
from functools import lru_cache

import httpx
import jwt
from fastapi import Depends, HTTPException, Request, status
from jwt import PyJWKClient

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_JWT_ISSUER = os.environ.get("SUPABASE_JWT_ISSUER")
JWKS_URL = os.environ.get("JWKS_URL")
SUPER_ADMIN_EMAILS = {
    email.strip().lower()
    for email in os.environ.get("SUPER_ADMIN_EMAILS", "").split(",")
    if email.strip()
}


@dataclass(frozen=True)
class CurrentUser:
    id: str
    email: str | None
    role: str


@lru_cache(maxsize=1)
def jwk_client() -> PyJWKClient:
    return PyJWKClient(JWKS_URL)


def _extract_bearer_token(request: Request) -> str:
    auth_header = request.headers.get("Authorization", "")
    scheme, _, token = auth_header.partition(" ")

    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )
    return token


def verify_supabase_jwt(request: Request) -> dict:
    token = _extract_bearer_token(request)

    try:
        signing_key = jwk_client().get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            issuer=SUPABASE_JWT_ISSUER,
            options={"verify_aud": False},
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or expired token"
        )
    if payload.get("role") != "authenticated":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="user is not authenticated"
        )

    return payload
