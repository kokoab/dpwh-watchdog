from __future__ import annotations

from copy import deepcopy


THREAD_SCOPE_STORE: dict[str, dict[str, str]] = {}
THREAD_PLAN_STORE: dict[str, dict[str, object]] = {}


def get_thread_scope(thread_id: str | None) -> dict[str, str]:
    if not thread_id:
        return {}
    return deepcopy(THREAD_SCOPE_STORE.get(thread_id, {}))


def get_thread_plan(thread_id: str | None) -> dict[str, object]:
    if not thread_id:
        return {}
    return deepcopy(THREAD_PLAN_STORE.get(thread_id, {}))


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


def set_thread_plan(thread_id: str | None, plan: dict[str, object]) -> None:
    if not thread_id:
        return

    cleaned = {
        key: value
        for key, value in plan.items()
        if value not in (None, "", {}, [])
    }
    if cleaned:
        THREAD_PLAN_STORE[thread_id] = deepcopy(cleaned)

        scope = {
            key: value
            for key, value in cleaned.get("filters", {}).items()
            if isinstance(value, str)
        }
        if cleaned.get("subject"):
            scope["subject"] = str(cleaned["subject"])
        scope["intent"] = str(cleaned.get("intent", "chat"))
        if scope:
            THREAD_SCOPE_STORE[thread_id] = scope


def clear_thread_scope(thread_id: str | None) -> None:
    if not thread_id:
        return
    THREAD_SCOPE_STORE.pop(thread_id, None)
    THREAD_PLAN_STORE.pop(thread_id, None)
