from __future__ import annotations

from copy import deepcopy


THREAD_SCOPE_STORE: dict[str, dict[str, str]] = {}


def get_thread_scope(thread_id: str | None) -> dict[str, str]:
    if not thread_id:
        return {}
    return deepcopy(THREAD_SCOPE_STORE.get(thread_id, {}))


def set_thread_scope(thread_id: str | None, scope: dict[str, str | None]) -> None:
    if not thread_id:
        return

    cleaned = {
        key: value.strip()
        for key, value in scope.items()
        if isinstance(value, str) and value.strip()
    }
    if cleaned:
        THREAD_SCOPE_STORE[thread_id] = cleaned


def clear_thread_scope(thread_id: str | None) -> None:
    if not thread_id:
        return
    THREAD_SCOPE_STORE.pop(thread_id, None)
