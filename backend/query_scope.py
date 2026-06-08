from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy

from chat_memory import delete_thread_memory, get_thread_state, upsert_thread_state


THREAD_SCOPE_STORE: dict[str, dict[str, str]] = {}
THREAD_PLAN_STORE: dict[str, dict[str, object]] = {}
THREAD_RESULT_STORE: dict[str, dict[str, object]] = {}
CURRENT_THREAD_ID: ContextVar[str | None] = ContextVar("current_thread_id", default=None)


def _hydrate_thread_cache(thread_id: str | None) -> None:
    if not thread_id:
        return

    payload = get_thread_state(thread_id)
    if not payload:
        return

    scope = payload.get("scope")
    if isinstance(scope, dict):
        THREAD_SCOPE_STORE[thread_id] = deepcopy(scope)

    plan = payload.get("plan")
    if isinstance(plan, dict):
        THREAD_PLAN_STORE[thread_id] = deepcopy(plan)

    result = payload.get("result")
    if isinstance(result, dict):
        THREAD_RESULT_STORE[thread_id] = deepcopy(result)


def get_thread_scope(thread_id: str | None) -> dict[str, str]:
    if not thread_id:
        return {}
    if thread_id not in THREAD_SCOPE_STORE:
        _hydrate_thread_cache(thread_id)
    return deepcopy(THREAD_SCOPE_STORE.get(thread_id, {}))


def get_thread_plan(thread_id: str | None) -> dict[str, object]:
    if not thread_id:
        return {}
    if thread_id not in THREAD_PLAN_STORE:
        _hydrate_thread_cache(thread_id)
    return deepcopy(THREAD_PLAN_STORE.get(thread_id, {}))


def get_thread_result(thread_id: str | None) -> dict[str, object]:
    if not thread_id:
        return {}
    if thread_id not in THREAD_RESULT_STORE:
        _hydrate_thread_cache(thread_id)
    return deepcopy(THREAD_RESULT_STORE.get(thread_id, {}))


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
        upsert_thread_state(thread_id, scope=cleaned)


def set_thread_plan(thread_id: str | None, plan: dict[str, object]) -> None:
    if not thread_id:
        return

    cleaned = {
        key: value
        for key, value in plan.items()
        if value not in (None, "", {}, [])
    }
    if not cleaned:
        return

    if str(cleaned.get("intent", "chat")) == "chat":
        existing = get_thread_plan(thread_id)
        if existing and str(existing.get("intent", "chat")) != "chat":
            return
        return

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

    upsert_thread_state(thread_id, scope=scope or None, plan=cleaned)


def set_thread_result(thread_id: str | None, result: dict[str, object]) -> None:
    if not thread_id:
        return

    cleaned = {
        key: value
        for key, value in result.items()
        if value not in (None, "", {}, [])
    }
    if cleaned:
        THREAD_RESULT_STORE[thread_id] = deepcopy(cleaned)
        selected_contract_id = None
        displayed_ids = cleaned.get("displayed_contract_ids")
        if isinstance(displayed_ids, list) and displayed_ids:
            selected_contract_id = str(displayed_ids[0])
        upsert_thread_state(
            thread_id,
            result=cleaned,
            selected_contract_id=selected_contract_id,
        )


def get_current_thread_id() -> str | None:
    return CURRENT_THREAD_ID.get()


def set_current_thread_id(thread_id: str | None):
    return CURRENT_THREAD_ID.set(thread_id)


def reset_current_thread_id(token) -> None:
    CURRENT_THREAD_ID.reset(token)


def clear_current_thread_id() -> None:
    CURRENT_THREAD_ID.set(None)


def clear_thread_cache(thread_id: str | None) -> None:
    if not thread_id:
        return
    THREAD_SCOPE_STORE.pop(thread_id, None)
    THREAD_PLAN_STORE.pop(thread_id, None)
    THREAD_RESULT_STORE.pop(thread_id, None)


def clear_thread_scope(thread_id: str | None) -> None:
    clear_thread_cache(thread_id)
    if not thread_id:
        return
    delete_thread_memory(thread_id)
