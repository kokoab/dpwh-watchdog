import json
import time
import uuid
from typing import Iterator

from auth.dependencies import get_current_user
from auth.jwt import CurrentUser
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from features.chat.agent.orchestrator import stream_agent
from features.chat.agent.query_planner import QueryPlan
from features.chat.agent.query_planner_llm import plan_message
from features.chat.agent.query_scope import (
    clear_current_thread_id,
    clear_thread_cache,
    get_thread_result,
    set_current_thread_id,
    set_thread_plan,
    set_thread_result,
)
from features.chat.agent.synthesis import focused_synthesis
from features.chat.memory import (
    delete_thread_memory,
    ensure_chat_thread,
    list_chat_messages,
    list_chat_threads,
    save_chat_message,
)
from features.chat.presenters import (
    _build_structured_contract_detail_reply,
    _build_structured_contract_reply,
    _build_structured_contract_reply_with_dates,
    _comparison_diffs_string,
    _comparison_rankings_string,
    _comparison_table_string,
    _stream_structured_token_text,
    _stream_token_text,
    _strip_generated_comparison_sections,
    _strip_tool_call_json_text,
    should_append_next_step,
)
from features.chat.tools.registry import (
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


NEXT_STEP_QUESTION = (
    "\n\nWould you like to dive deeper into this contract, compare other projects "
    "by the same contractor, or look at similar projects in the area?"
)
DIRECT_TOOL_INTENTS = {
    "lookup",
    "browse",
    "availability",
    "stats",
    "clarify",
    "search",
    "compare",
    "anomaly",
}
DIRECT_TOOL_BY_INTENT = {
    "lookup": execute_lookup_plan,
    "browse": execute_browse_plan,
    "availability": execute_availability_plan,
    "stats": execute_stats_plan,
    "clarify": execute_clarify_plan,
    "search": execute_search_plan,
}
COMPARE_CLARIFICATION = "Which contracts should I compare?"


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


def _is_analytical_stats(plan: QueryPlan, user_message: str) -> bool:
    """True when the stats query needs synthesis instead of a template card."""
    if plan.subject and plan.subject.strip():
        return True
    lower = user_message.lower()
    analytical_signals = [
        "which province",
        "which region",
        "most project",
        "top ",
        "breakdown",
        "and what",
        "and which",
        "explain",
        "analysis",
    ]
    return any(sig in lower for sig in analytical_signals)


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
    from features.chat.agent.comparison_utils import compute_comparison_analytics

    comparison_analytics = compute_comparison_analytics(detail_sources)
    python_table_string = _comparison_table_string(detail_sources)
    python_rankings_string = _comparison_rankings_string(
        comparison_analytics, detail_sources
    )
    python_diffs_string = _comparison_diffs_string(comparison_analytics, detail_sources)

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
        "comparison_analytics": comparison_analytics,
        "is_complete_result_set": True,
    }
    set_thread_result(thread_id, comparison_result_state)
    synthesis_task = (
        "The comparison table, rankings, and differences below are already "
        "formatted — do NOT regenerate them, do NOT output another table. "
        "Write ONLY: (1) one executive summary paragraph 1-3 sentences with "
        "no section header explaining what was found and the most important "
        "difference, then (2) insight bullets using the pre-computed analytics."
    )
    if comparison_query:
        synthesis_task = f"{synthesis_task} {comparison_query}"
    synthesis_payload = {
        "comparison_query": comparison_query,
        "comparison_analytics": comparison_analytics,
        "note": (
            "Table and rankings are pre-built. Write only executive summary "
            "paragraph and insight bullets."
        ),
    }
    synthesis_output = focused_synthesis(
        synthesis_task,
        synthesis_payload,
        thread_id,
    )
    parts = []
    cleaned_synthesis_output = _strip_generated_comparison_sections(synthesis_output)
    if cleaned_synthesis_output:
        parts.append(cleaned_synthesis_output)
    parts.append(python_table_string)
    parts.append("**Rankings**")
    parts.append(python_rankings_string)
    if python_diffs_string:
        parts.append("**Key Differences**")
        parts.append(python_diffs_string)
    assistant_text = "\n\n".join(parts)
    response_source = "structured"
    return assistant_text, comparison_result_state, response_source


def _run_direct_tool_turn(
    plan: QueryPlan,
    thread_id: str,
    user_message: str,
) -> tuple[str, dict[str, object] | None, str]:
    if plan.intent == "compare":
        return _run_direct_compare_turn(plan, thread_id)
    if plan.intent == "anomaly":
        tool_output = execute_anomaly_plan(plan)
        task = (
            f"{plan.subject or 'Review anomalies.'} Present a table of affected "
            "records for 3 or more entries. State count and percentage of total "
            "affected. Begin with an executive summary."
        )
        assistant_text = focused_synthesis(task, tool_output, thread_id)
        return (
            assistant_text,
            tool_output if isinstance(tool_output, dict) else None,
            "structured",
        )

    if plan.intent == "stats":
        set_current_thread_id(thread_id)
        try:
            formatted_text, payload = execute_stats_plan(plan)
        finally:
            clear_current_thread_id()

        latest_result_state = get_thread_result(thread_id)
        result_state = (
            latest_result_state
            if isinstance(latest_result_state, dict) and latest_result_state
            else None
        )
        total = int(payload.get("total_contracts") or 0)
        if 0 < total <= 15 and result_state and result_state.get("displayed_sources"):
            reply = _build_structured_contract_reply_with_dates(result_state)
            if reply:
                return reply, result_state, "structured"
        if _is_analytical_stats(plan, user_message):
            task = (
                f"{user_message} — Present a 1-sentence summary of what was found. "
                "List status, region, and province breakdowns as applicable using bullet points "
                "(e.g. '• Completed: 5 (83.3%)'). "
                "No markdown tables. No pipe characters. End with one insight sentence."
            )
            assistant_text = focused_synthesis(task, payload, thread_id)
            return assistant_text or formatted_text, result_state, "structured"
        return formatted_text, result_state, "tool"

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
    if plan.intent == "availability":
        total = int((result_state or {}).get("count") or 0)
        if 0 < total <= 15 and result_state and result_state.get("displayed_sources"):
            reply = _build_structured_contract_reply_with_dates(result_state)
            if reply:
                return reply, result_state, "structured"
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
                message,
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
async def chat_stream(
    request: ChatRequest, current_user: CurrentUser = Depends(get_current_user)
):
    thread_id = request.thread_id or str(uuid.uuid4())

    return StreamingResponse(
        event_stream(request.message, thread_id, current_user.id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Thread-Id": thread_id,
        },
    )


@router.get("/threads")
async def get_chat_threads(
    limit: int = 50, current_user: CurrentUser = Depends(get_current_user)
):
    return {
        "threads": list_chat_threads(
            user_id=current_user.id, limit=max(1, min(limit, 200))
        )
    }


@router.get("/threads/{thread_id}/messages")
async def get_chat_thread_messages(
    thread_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    limit: int = 200,
):
    return {
        "thread_id": thread_id,
        "messages": list_chat_messages(
            thread_id,
            user_id=current_user.id,
            limit=max(1, min(limit, 500)),
        ),
    }


@router.delete("/threads/{thread_id}")
async def delete_chat_thread(
    thread_id: str, current_user: CurrentUser = Depends(get_current_user)
):
    delete_thread_memory(thread_id, user_id=current_user.id)
    clear_thread_cache(thread_id)
    return "Successfully deleted chat"
