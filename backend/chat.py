import json
import uuid
from typing import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent import stream_agent
from chat_memory import (
    ensure_chat_thread,
    list_chat_messages,
    list_chat_threads,
    save_chat_message,
)
from query_planner import detect_intent_from_expanded_query
from query_expand import log_query_expansion, query_expand
from query_scope import get_thread_plan, get_thread_result

router = APIRouter(prefix="/chat")


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None
    user_id: str | None = None


NEXT_STEP_QUESTION = (
    "\n\nWould you like to dive deeper into this contract, compare other projects "
    "by the same contractor, or look at similar projects in the area?"
)


def _format_money(value) -> str:
    try:
        return f"PHP {float(value):,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_progress(value) -> str:
    if value in (None, ""):
        return "N/A"
    return f"{value}%"


def _has_displayed_contract_citation(assistant_text: str, result_state: dict[str, object] | None) -> bool:
    if not isinstance(result_state, dict):
        return True

    sources = result_state.get("displayed_sources")
    if not isinstance(sources, list) or not sources:
        return True

    lowered = assistant_text.lower()
    return any(
        str(source.get("contractId", "")).lower() in lowered
        for source in sources
        if isinstance(source, dict)
    )


def _format_result_listing(result_state: dict[str, object] | None) -> str:
    if not isinstance(result_state, dict) or result_state.get("result_kind") != "contract_set":
        return ""

    sources = result_state.get("displayed_sources")
    if not isinstance(sources, list) or not sources:
        return ""

    filters = result_state.get("filters") if isinstance(result_state.get("filters"), dict) else {}
    category = filters.get("category")
    province = filters.get("province")
    region = filters.get("region")
    subject = f"{category} projects" if category else "contracts"
    location = province or region
    heading = (
        f"The matching {subject} in {location} are:"
        if location
        else f"The matching {subject} are:"
    )

    lines = [heading]
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue

        description = source.get("description") or "Unnamed contract"
        contract_id = source.get("contractId") or "N/A"
        program_name = source.get("programName") or "N/A"
        status = source.get("status") or "N/A"
        budget = _format_money(source.get("budget"))
        location_text = ", ".join(
            part
            for part in [source.get("region"), source.get("province")]
            if part
        ) or "N/A"
        contractor = source.get("contractor") or "N/A"
        progress = _format_progress(source.get("progress"))

        lines.extend(
            [
                "",
                f"{index}. {description} ({contract_id})",
                f"• Description: {description}",
                f"• Program name: {program_name}",
                f"• Status: {status} | Budget: {budget} | Location: {location_text}",
                f"• Contractor: {contractor} | Progress: {progress}",
            ]
        )

    return "\n".join(lines)


def should_append_next_step(intent: str | None, assistant_text: str) -> bool:
    if intent in (None, "", "chat"):
        return False
    if not assistant_text.strip():
        return False
    return "?" not in assistant_text[-500:]


def event_stream(message: str, thread_id: str, user_id: str | None = None) -> Iterator[str]:
    ensure_chat_thread(thread_id, user_id=user_id)
    expanded_message = query_expand(message, thread_id=thread_id)
    log_query_expansion(message, expanded_message, thread_id)
    plan_snapshot = get_thread_plan(thread_id)
    detected_intent = detect_intent_from_expanded_query(expanded_message)
    save_chat_message(
        thread_id,
        "user",
        message,
        user_id=user_id,
        expanded_query=expanded_message,
        intent=detected_intent,
        metadata={"plan": plan_snapshot},
    )

    assistant_chunks: list[str] = []
    latest_result_state: dict[str, object] | None = None

    for event in stream_agent(expanded_message, thread_id):
        if event.get("type") == "token":
            assistant_chunks.append(str(event.get("content", "")))
        elif event.get("type") == "result_state" and isinstance(event.get("content"), dict):
            latest_result_state = event["content"]
        elif event.get("type") == "done":
            assistant_text_so_far = "".join(assistant_chunks).strip()
            if not _has_displayed_contract_citation(assistant_text_so_far, latest_result_state):
                listing = _format_result_listing(latest_result_state)
                if listing:
                    prefix = "\n\n" if assistant_text_so_far else ""
                    content = f"{prefix}{listing}"
                    assistant_chunks.append(content)
                    assistant_text_so_far = "".join(assistant_chunks).strip()
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
            if should_append_next_step(detected_intent, assistant_text_so_far):
                assistant_chunks.append(NEXT_STEP_QUESTION)
                yield f"data: {json.dumps({'type': 'token', 'content': NEXT_STEP_QUESTION})}\n\n"
        yield f"data: {json.dumps(event)}\n\n"

    assistant_text = "".join(assistant_chunks).strip()
    assistant_metadata = {}
    if latest_result_state:
        assistant_metadata["result_state"] = latest_result_state
    else:
        persisted_result_state = get_thread_result(thread_id)
        if isinstance(persisted_result_state, dict) and persisted_result_state:
            assistant_metadata["result_state"] = persisted_result_state

    if assistant_text or assistant_metadata:
        save_chat_message(
            thread_id,
            "assistant",
            assistant_text or "",
            user_id=user_id,
            intent=detected_intent,
            metadata=assistant_metadata,
        )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    thread_id = request.thread_id or str(uuid.uuid4())

    return StreamingResponse(
        event_stream(request.message, thread_id, request.user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Thread-Id": thread_id,
        }
    )


@router.get("/threads")
async def get_chat_threads(user_id: str | None = None, limit: int = 50):
    return {"threads": list_chat_threads(user_id=user_id, limit=max(1, min(limit, 200)))}


@router.get("/threads/{thread_id}/messages")
async def get_chat_thread_messages(
    thread_id: str,
    user_id: str | None = None,
    limit: int = 200,
):
    return {
        "thread_id": thread_id,
        "messages": list_chat_messages(
            thread_id,
            user_id=user_id,
            limit=max(1, min(limit, 500)),
        ),
    }
