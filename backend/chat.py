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


def _stream_token_text(content: str) -> Iterator[str]:
    lines = content.splitlines(keepends=True)
    for line in lines or [content]:
        yield f"data: {json.dumps({'type': 'token', 'content': line})}\n\n"


def should_append_next_step(intent: str | None, assistant_text: str) -> bool:
    if intent in (None, "", "chat", "clarify"):
        return False
    if not assistant_text.strip():
        return False
    recent_text = assistant_text[-500:].lower()
    if "would you like" in recent_text:
        return False
    return "?" not in recent_text


def _format_budget(value: object) -> str:
    if value in (None, ""):
        return "N/A"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"PHP {amount:,.0f}"


def _format_money(value: object) -> str:
    if value in (None, ""):
        return "N/A"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"PHP {amount:,.2f}"


def _format_percent(value: object) -> str:
    if value in (None, ""):
        return "N/A"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{amount:.1f}%"


def _format_value(value: object) -> str:
    if value in (None, ""):
        return "N/A"
    return str(value).strip()


def _build_link_summary(document_links: object) -> str:
    if not isinstance(document_links, dict) or not document_links:
        return "N/A"
    names = [name for name, url in document_links.items() if str(url or "").strip()]
    return ", ".join(names) if names else "N/A"


def _looks_like_tool_call_json(text: str) -> bool:
    stripped = str(text or "").strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return False

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return False

    return (
        isinstance(payload, dict)
        and "name" in payload
        and "parameters" in payload
        and set(payload.keys()).issubset({"name", "parameters"})
    )


def _strip_tool_call_json_text(text: str) -> str:
    if _looks_like_tool_call_json(text):
        return ""

    lines = str(text or "").splitlines(keepends=True)
    if not lines:
        return ""

    cleaned_lines = [line for line in lines if not _looks_like_tool_call_json(line)]
    return "".join(cleaned_lines)


def _build_structured_contract_detail_reply(result_state: dict[str, object]) -> str:
    displayed_sources = result_state.get("displayed_sources")
    if not isinstance(displayed_sources, list) or not displayed_sources:
        return ""

    source = displayed_sources[0]
    if not isinstance(source, dict):
        return ""

    db_fields = source.get("dbFields") if isinstance(source.get("dbFields"), dict) else {}
    components = source.get("components") if isinstance(source.get("components"), list) else []
    document_links = source.get("documentLinks") if isinstance(source.get("documentLinks"), dict) else {}

    description = _format_value(source.get("description") or db_fields.get("description"))
    contract_id = _format_value(source.get("contractId") or db_fields.get("contractId"))
    contractor = _format_value(source.get("contractor") or db_fields.get("contractor"))
    category = _format_value(source.get("category") or db_fields.get("category"))
    status = _format_value(source.get("status") or db_fields.get("status"))
    region = _format_value(source.get("region") or db_fields.get("region"))
    province = _format_value(source.get("province") or db_fields.get("province"))
    budget = _format_money(source.get("budget") or db_fields.get("budget"))
    amount_paid = _format_money(source.get("amountPaid") or db_fields.get("amountPaid"))
    award_amount = _format_money(source.get("awardAmount") or db_fields.get("awardAmount"))
    award_ratio = _format_percent(
        source.get("awardToBudgetRatio") or db_fields.get("awardToBudgetRatio")
    )
    progress = _format_percent(source.get("progress") or db_fields.get("progress"))
    infra_year = _format_value(source.get("infraYear") or db_fields.get("infraYear"))
    program_name = _format_value(source.get("programName") or db_fields.get("programName"))
    source_of_funds = _format_value(
        source.get("sourceOfFunds") or db_fields.get("sourceOfFunds")
    )
    advertisement_date = _format_value(
        source.get("advertisementDate") or db_fields.get("advertisementDate")
    )
    bid_deadline = _format_value(
        source.get("bidSubmissionDeadline") or db_fields.get("bidSubmissionDeadline")
    )
    start_date = _format_value(source.get("startDate") or db_fields.get("startDate"))
    completion_date = _format_value(
        source.get("completionDate") or db_fields.get("completionDate")
    )
    expiry_date = _format_value(source.get("expiryDate") or db_fields.get("expiryDate"))
    contract_duration = _format_value(
        source.get("contractDuration") or db_fields.get("contractDuration")
    )
    link_summary = _build_link_summary(document_links)

    lines = [
        f"{description} ({contract_id})",
        f"• Contractor: {contractor}",
        f"• Category: {category}",
        f"• Status: {status}",
        f"• Budget: {budget}",
        f"• Award Amount: {award_amount}",
        f"• Award-to-Budget Ratio: {award_ratio}",
        f"• Amount Paid: {amount_paid}",
        f"• Progress: {progress}",
        f"• Region: {region}",
        f"• Province: {province}",
        f"• Program: {program_name}",
        f"• Source of Funds: {source_of_funds}",
        f"• Infra Year: {infra_year}",
        f"• Advertisement Date: {advertisement_date}",
        f"• Bid Submission Deadline: {bid_deadline}",
        f"• Start Date: {start_date}",
        f"• Completion Date: {completion_date}",
        f"• Expiry Date: {expiry_date}",
        f"• Contract Duration: {contract_duration}",
        f"• Document Links: {link_summary}",
    ]

    if components:
        lines.append("• Components:")
        for index, component in enumerate(components, start=1):
            if not isinstance(component, dict):
                continue
            component_id = _format_value(component.get("componentId"))
            component_desc = _format_value(component.get("description"))
            component_type = _format_value(component.get("typeOfWork"))
            infra_type = _format_value(component.get("infraType"))
            location = ", ".join(
                part
                for part in (
                    _format_value(component.get("region")),
                    _format_value(component.get("province")),
                )
                if part != "N/A"
            )
            location_text = location if location else "N/A"
            lines.append(
                f"  {index}. {component_id} | {component_type} | {infra_type} | {component_desc} | {location_text}"
            )

    lines.append(NEXT_STEP_QUESTION.strip())
    return "\n".join(lines).strip()


def _build_structured_contract_reply(result_state: dict[str, object]) -> str:
    displayed_sources = result_state.get("displayed_sources")
    if not isinstance(displayed_sources, list) or not displayed_sources:
        return ""

    lines: list[str] = []
    for index, source in enumerate(displayed_sources, start=1):
        if not isinstance(source, dict):
            continue
        description = str(source.get("description") or "N/A").strip()
        contract_id = str(source.get("contractId") or "N/A").strip()
        lines.extend(
            [
                f"{index}. {description} ({contract_id})",
                f"• Contractor: {str(source.get('contractor') or 'N/A').strip()}",
                f"• Status: {str(source.get('status') or 'N/A').strip()}",
                f"• Budget: {_format_budget(source.get('budget'))}",
            ]
        )
        if index != len(displayed_sources):
            lines.append("")

    if not lines:
        return ""

    lines.append("")
    lines.append(NEXT_STEP_QUESTION.strip())
    return "\n".join(lines).strip()


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
    structured_listing_result: dict[str, object] | None = None
    structured_detail_result: dict[str, object] | None = None
    suppress_model_tokens = False

    for event in stream_agent(expanded_message, thread_id):
        if event.get("type") == "token":
            if suppress_model_tokens:
                continue
            token_content = _strip_tool_call_json_text(str(event.get("content", "")))
            if not token_content.strip():
                continue
            assistant_chunks.append(token_content)
            yield f"data: {json.dumps({**event, 'content': token_content})}\n\n"
            continue
        elif event.get("type") == "result_state" and isinstance(event.get("content"), dict):
            latest_result_state = event["content"]
            displayed_sources = latest_result_state.get("displayed_sources")
            if (
                latest_result_state.get("result_kind") == "contract_set"
                and isinstance(displayed_sources, list)
                and displayed_sources
            ):
                structured_listing_result = latest_result_state
                structured_detail_result = None
                suppress_model_tokens = True
            elif (
                latest_result_state.get("result_kind") == "contract_detail"
                and isinstance(displayed_sources, list)
                and displayed_sources
            ):
                structured_detail_result = latest_result_state
                structured_listing_result = None
                suppress_model_tokens = True
            else:
                structured_listing_result = None
                structured_detail_result = None
                suppress_model_tokens = False
            yield f"data: {json.dumps(event)}\n\n"
            continue
        elif event.get("type") == "done":
            assistant_text_so_far = _strip_tool_call_json_text("".join(assistant_chunks)).strip()
            if structured_listing_result:
                assistant_text_so_far = _build_structured_contract_reply(structured_listing_result)
                assistant_chunks = [assistant_text_so_far] if assistant_text_so_far else []
                if assistant_text_so_far:
                    for token_event in _stream_token_text(assistant_text_so_far):
                        yield token_event
                yield f"data: {json.dumps(event)}\n\n"
                continue
            if structured_detail_result:
                assistant_text_so_far = _build_structured_contract_detail_reply(
                    structured_detail_result
                )
                assistant_chunks = [assistant_text_so_far] if assistant_text_so_far else []
                if assistant_text_so_far:
                    for token_event in _stream_token_text(assistant_text_so_far):
                        yield token_event
                yield f"data: {json.dumps(event)}\n\n"
                continue
            if should_append_next_step(detected_intent, assistant_text_so_far):
                assistant_chunks.append(NEXT_STEP_QUESTION)
                for token_event in _stream_token_text(NEXT_STEP_QUESTION):
                    yield token_event
            yield f"data: {json.dumps(event)}\n\n"
            continue

        yield f"data: {json.dumps(event)}\n\n"

    assistant_text = _strip_tool_call_json_text("".join(assistant_chunks)).strip()
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
