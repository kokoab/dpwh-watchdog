import json
import os
import re
import time
import uuid
from typing import Iterator

from agent import stream_agent
from chat_memory import (
    ensure_chat_thread,
    list_chat_messages,
    list_chat_threads,
    save_chat_message,
)
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from query_planner import QueryPlan
from query_planner_llm import plan_message
from query_scope import (
    clear_current_thread_id,
    get_thread_plan,
    get_thread_result,
    set_current_thread_id,
    set_thread_plan,
    set_thread_result,
)
from synthesis import focused_synthesis
from tools import (
    execute_anomaly_plan,
    execute_availability_plan,
    execute_browse_plan,
    execute_clarify_plan,
    execute_lookup_plan,
    execute_search_plan,
    execute_stats_plan,
    load_contract_detail_sources,
)

router = APIRouter(prefix="/chat")


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None
    user_id: str | None = None


NEXT_STEP_QUESTION = (
    "\n\nWould you like to dive deeper into this contract, compare other projects "
    "by the same contractor, or look at similar projects in the area?"
)
DIRECT_TOOL_INTENTS = {"lookup", "browse", "availability", "stats", "clarify", "search", "compare", "anomaly"}
DIRECT_TOOL_BY_INTENT = {
    "lookup": execute_lookup_plan,
    "browse": execute_browse_plan,
    "availability": execute_availability_plan,
    "stats": execute_stats_plan,
    "clarify": execute_clarify_plan,
    "search": execute_search_plan,
}
COMPARE_CLARIFICATION = "Which contracts should I compare?"
STRUCTURED_STREAM_WORDS_PER_CHUNK = max(
    1, int(os.environ.get("STRUCTURED_STREAM_WORDS_PER_CHUNK", "6"))
)
STRUCTURED_STREAM_DELAY_SECONDS = max(
    0.0, float(os.environ.get("STRUCTURED_STREAM_DELAY_SECONDS", "0.20"))
)


def _stream_token_text(content: str) -> Iterator[str]:
    lines = content.splitlines(keepends=True)
    for line in lines or [content]:
        yield f"data: {json.dumps({'type': 'token', 'content': line})}\n\n"


def _iter_structured_stream_chunks(
    content: str, words_per_chunk: int = STRUCTURED_STREAM_WORDS_PER_CHUNK
) -> Iterator[str]:
    for line in content.splitlines(keepends=True) or [content]:
        tokens = re.findall(r"\S+|\s+", line)
        if not tokens:
            yield line
            continue

        chunk_parts: list[str] = []
        word_count = 0
        for token in tokens:
            chunk_parts.append(token)
            if not token.isspace():
                word_count += 1
            if word_count >= words_per_chunk:
                yield "".join(chunk_parts)
                chunk_parts = []
                word_count = 0

        if chunk_parts:
            yield "".join(chunk_parts)


def _stream_structured_token_text(
    content: str, delay_seconds: float = STRUCTURED_STREAM_DELAY_SECONDS
) -> Iterator[str]:
    for chunk in _iter_structured_stream_chunks(content):
        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
        if delay_seconds > 0:
            time.sleep(delay_seconds)


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

    db_fields = (
        source.get("dbFields") if isinstance(source.get("dbFields"), dict) else {}
    )
    components = (
        source.get("components") if isinstance(source.get("components"), list) else []
    )
    document_links = (
        source.get("documentLinks")
        if isinstance(source.get("documentLinks"), dict)
        else {}
    )

    description = _format_value(
        source.get("description") or db_fields.get("description")
    )
    contract_id = _format_value(source.get("contractId") or db_fields.get("contractId"))
    contractor = _format_value(source.get("contractor") or db_fields.get("contractor"))
    category = _format_value(source.get("category") or db_fields.get("category"))
    status = _format_value(source.get("status") or db_fields.get("status"))
    region = _format_value(source.get("region") or db_fields.get("region"))
    province = _format_value(source.get("province") or db_fields.get("province"))
    budget = _format_money(source.get("budget") or db_fields.get("budget"))
    amount_paid = _format_money(source.get("amountPaid") or db_fields.get("amountPaid"))
    award_amount = _format_money(
        source.get("awardAmount") or db_fields.get("awardAmount")
    )
    award_ratio = _format_percent(
        source.get("awardToBudgetRatio") or db_fields.get("awardToBudgetRatio")
    )
    progress = _format_percent(source.get("progress") or db_fields.get("progress"))
    infra_year = _format_value(source.get("infraYear") or db_fields.get("infraYear"))
    program_name = _format_value(
        source.get("programName") or db_fields.get("programName")
    )
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


def _invoke_tool(tool_obj, query: str) -> str:
    if hasattr(tool_obj, "invoke"):
        result = tool_obj.invoke(query)
    else:
        result = tool_obj(query)
    return result if isinstance(result, str) else str(result)


def _build_direct_tool_reply(
    intent: str,
    raw_output: str,
    result_state: dict[str, object] | None,
) -> tuple[str, str]:
    if isinstance(result_state, dict) and result_state:
        displayed_sources = result_state.get("displayed_sources")
        if (
            result_state.get("result_kind") == "contract_detail"
            and isinstance(displayed_sources, list)
            and displayed_sources
        ):
            reply = _build_structured_contract_detail_reply(result_state)
            if reply:
                return reply, "structured"
        if (
            result_state.get("result_kind") == "contract_set"
            and isinstance(displayed_sources, list)
            and displayed_sources
        ):
            reply = _build_structured_contract_reply(result_state)
            if reply:
                return reply, "structured"

    cleaned_output = _strip_tool_call_json_text(raw_output).strip()
    if intent == "clarify" and cleaned_output:
        return cleaned_output, "tool"
    return cleaned_output, "tool"


def _run_direct_compare_turn(
    plan: QueryPlan,
    thread_id: str,
) -> tuple[str, dict[str, object] | None, str]:
    contract_ids = [
        part.strip() for part in str(plan.lookup_value or "").split(",") if part.strip()
    ]
    comparison_query = str(plan.subject or "").strip()

    if len(contract_ids) < 2:
        return COMPARE_CLARIFICATION, None, "tool"

    previous_result_state = get_thread_result(thread_id)
    detail_sources = load_contract_detail_sources(contract_ids)
    if len(detail_sources) < 2:
        return (
            "I could not load enough contract detail records to compare those projects deterministically.",
            None,
            "tool",
        )

    prior_filters = {}
    if isinstance(previous_result_state, dict) and isinstance(
        previous_result_state.get("filters"), dict
    ):
        prior_filters = {
            key: str(value)
            for key, value in previous_result_state["filters"].items()
            if isinstance(value, str)
        }

    comparison_result_state = {
        "result_kind": "contract_compare",
        "intent": "compare",
        "filters": prior_filters,
        "comparison_query": comparison_query,
        "comparison_contract_ids": contract_ids,
        "count": len(detail_sources),
        "contract_ids": contract_ids,
        "displayed_contract_ids": contract_ids,
        "displayed_sources": detail_sources,
        "is_complete_result_set": True,
    }
    set_thread_result(thread_id, comparison_result_state)
    synthesis_output = focused_synthesis(
        comparison_query or "Compare these contracts.",
        {"contracts": detail_sources, "comparison_query": comparison_query},
        thread_id,
    )
    assistant_text = synthesis_output or (
        "I could not produce a comparison summary from the current structured records."
    )
    response_source = "structured"
    return assistant_text, comparison_result_state, response_source


def _run_direct_tool_turn(
    plan: QueryPlan,
    thread_id: str,
) -> tuple[str, dict[str, object] | None, str]:
    if plan.intent == "compare":
        return _run_direct_compare_turn(plan, thread_id)
    if plan.intent == "anomaly":
        tool_output = execute_anomaly_plan(plan)
        assistant_text = focused_synthesis(plan.subject or "Review anomalies in this scope.", tool_output, thread_id)
        return assistant_text, tool_output if isinstance(tool_output, dict) else None, "structured"

    tool_obj = DIRECT_TOOL_BY_INTENT[plan.intent]
    should_capture_result = plan.intent != "clarify"
    set_current_thread_id(thread_id)
    try:
        raw_output = tool_obj(plan)
    finally:
        clear_current_thread_id()

    latest_result_state = (
        get_thread_result(thread_id) if should_capture_result else None
    )
    result_state = (
        latest_result_state
        if isinstance(latest_result_state, dict) and latest_result_state
        else None
    )
    assistant_text, response_source = _build_direct_tool_reply(
        plan.intent,
        raw_output,
        result_state,
    )
    return assistant_text, result_state, response_source


def event_stream(
    message: str, thread_id: str, user_id: str | None = None
) -> Iterator[str]:
    t_start = time.perf_counter()
    t_first_token: float | None = None
    t_result_state: float | None = None

    ensure_chat_thread(thread_id, user_id=user_id)
    plan = plan_message(message, thread_id=thread_id)
    plan_snapshot = plan.to_dict()
    set_thread_plan(thread_id, plan_snapshot)
    save_chat_message(
        thread_id,
        "user",
        message,
        user_id=user_id,
        intent=plan.intent,
        metadata={"plan": plan_snapshot},
    )

    if plan.intent in DIRECT_TOOL_INTENTS:
        assistant_text, latest_result_state, assistant_response_source = (
            _run_direct_tool_turn(
                plan,
                thread_id,
            )
        )
        if latest_result_state:
            yield (
                f"data: {json.dumps({'type': 'result_state', 'content': latest_result_state})}\n\n"
            )
        if assistant_text:
            stream_tokens = (
                _stream_structured_token_text(assistant_text)
                if assistant_response_source == "structured"
                else _stream_token_text(assistant_text)
            )
            for token_event in stream_tokens:
                yield token_event
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

        assistant_metadata = {
            "response_source": assistant_response_source,
            "execution_path": "direct_compare"
            if plan.intent == "compare"
            else "direct_tool",
        }
        if latest_result_state:
            assistant_metadata["result_state"] = latest_result_state
        save_chat_message(
            thread_id,
            "assistant",
            assistant_text or "",
            user_id=user_id,
            intent=plan.intent,
            metadata=assistant_metadata,
        )
        return

    assistant_chunks: list[str] = []
    latest_result_state: dict[str, object] | None = None
    assistant_response_source: str | None = None

    for event in stream_agent(message, thread_id):
        if event.get("type") == "token":
            token_content = _strip_tool_call_json_text(str(event.get("content", "")))
            if not token_content.strip():
                continue
            if t_first_token is None:
                t_first_token = time.perf_counter()
                print(
                    f"[TIMING] First LLM token:      {t_first_token - t_start:.3f}s",
                    flush=True,
                )
            assistant_chunks.append(token_content)
            if assistant_response_source is None:
                assistant_response_source = "llm"
            yield f"data: {json.dumps({**event, 'content': token_content})}\n\n"
            continue

        elif event.get("type") == "result_state" and isinstance(
            event.get("content"), dict
        ):
            t_result_state = time.perf_counter()
            print(
                f"[TIMING] DB result_state ready:  {t_result_state - t_start:.3f}s",
                flush=True,
            )
            latest_result_state = event["content"]
            yield f"data: {json.dumps(event)}\n\n"
            continue

        elif event.get("type") == "done":
            assistant_text_so_far = _strip_tool_call_json_text(
                "".join(assistant_chunks)
            ).strip()

            # LLM path
            t_llm_done = time.perf_counter()
            if t_first_token is not None:
                print(
                    f"[TIMING] LLM reply complete:\n"
                    f"         First token:        {(t_first_token - t_start):.3f}s\n"
                    f"         Full reply done:    {(t_llm_done - t_start):.3f}s total",
                    flush=True,
                )
            if should_append_next_step(plan.intent, assistant_text_so_far):
                assistant_chunks.append(NEXT_STEP_QUESTION)
                for token_event in _stream_token_text(NEXT_STEP_QUESTION):
                    yield token_event
            yield f"data: {json.dumps(event)}\n\n"
            continue

        yield f"data: {json.dumps(event)}\n\n"

    assistant_text = _strip_tool_call_json_text("".join(assistant_chunks)).strip()
    should_persist_assistant_turn = bool(assistant_text or latest_result_state)
    assistant_metadata = {}
    if should_persist_assistant_turn:
        if assistant_response_source is None:
            assistant_response_source = "llm"
        if assistant_response_source:
            assistant_metadata["response_source"] = assistant_response_source
        assistant_metadata["execution_path"] = "llm"
        if latest_result_state:
            assistant_metadata["result_state"] = latest_result_state

    if should_persist_assistant_turn:
        save_chat_message(
            thread_id,
            "assistant",
            assistant_text or "",
            user_id=user_id,
            intent=plan.intent,
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
        },
    )


@router.get("/threads")
async def get_chat_threads(user_id: str | None = None, limit: int = 50):
    return {
        "threads": list_chat_threads(user_id=user_id, limit=max(1, min(limit, 200)))
    }


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
