import json
import os
from datetime import date
from typing import Iterator

from chat_memory import list_chat_messages
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from query_scope import (
    clear_current_thread_id,
    get_thread_result,
    set_current_thread_id,
)
from tools import tools

load_dotenv()

CURRENT_DATE = date.today().isoformat()


# def _build_llm():
#     return ChatGroq(
#         model=os.environ.get("GROQ_MODEL"),
#         temperature=float(os.environ.get("GROQ_TEMPERATURE", "0.1")),
#         max_tokens=int(os.environ.get("GROQ_MAX_TOKENS", "8192")),
#         top_p=float(os.environ.get("GROQ_TOP_P", "1")),
#         streaming=True,
#         max_retries=2,
#         timeout=60,
#     )


def _build_llm():
    return ChatOllama(
        model=os.environ.get("OLLAMA_MODEL"),
        temperature=float(os.environ.get("GROQ_TEMPERATURE")),
        max_tokens=int(os.environ.get("GROQ_MAX_TOKENS")),
        top_p=float(os.environ.get("GROQ_TOP_P")),
        max_retries=2,
        timeout=60,
        base_url=os.environ.get("OLLAMA_BASE_URL"),
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            You are the DPWH Watchdog AI assistant.
            Today's date is {CURRENT_DATE}. Use this exact date when judging
            whether a completion date is past due.

            Use tools for contract-specific questions instead of answering from
            memory. If a contract request is broad or underspecified, ask one
            short clarifying question instead of guessing.

            Never output raw tool-call JSON, function-call syntax, or tool names
            in the user-facing answer.

            For greetings or general conversation, respond naturally without
            using tools. If the contract database does not have the answer, you
            may use web search as a fallback.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

_watchdog_agent = None


def _get_watchdog_agent():
    global _watchdog_agent
    if _watchdog_agent is None:
        _watchdog_agent = create_react_agent(
            model=_build_llm(),
            tools=tools,
            prompt=prompt,
        )
    return _watchdog_agent


MAX_AGENT_HISTORY_MESSAGES = max(1, int(os.environ.get("AGENT_HISTORY_LIMIT", "8")))


def _build_agent_messages(user_message: str, thread_id: str) -> list[tuple[str, str]]:
    history = list_chat_messages(thread_id, limit=MAX_AGENT_HISTORY_MESSAGES)
    messages: list[tuple[str, str]] = []

    for message in history:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "")
        expanded_query = str(message.get("expanded_query") or "")
        if role == "user" and expanded_query:
            content = expanded_query
        if role not in {"user", "assistant"} or not content:
            continue
        messages.append((role, content))

    if not messages or messages[-1] != ("user", user_message):
        messages.append(("user", user_message))

    return messages


def _extract_stream_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content

    chunks: list[str] = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    if chunks:
        return "".join(chunks)

    text = getattr(message, "text", "")
    return text if isinstance(text, str) else ""


def stream_agent(user_message: str, thread_id: str) -> Iterator[dict]:
    SOURCE_MARKER = "__SOURCES__"
    emitted_result_state = None
    set_current_thread_id(thread_id)
    watchdog_agent = _get_watchdog_agent()

    try:
        for chunk in watchdog_agent.stream(
            {"messages": _build_agent_messages(user_message, thread_id)},
            stream_mode="messages",
        ):
            msg, metadata = chunk
            node = metadata.get("langgraph_node")
            stream_text = _extract_stream_text(msg)

            if node == "agent" and stream_text:
                yield {"type": "token", "content": stream_text}

            elif node == "tools" and stream_text:
                raw = stream_text
                if SOURCE_MARKER in raw:
                    _, sources_part = raw.split(SOURCE_MARKER, 1)
                    try:
                        sources = json.loads(sources_part)
                        yield {"type": "sources", "content": sources}
                    except json.JSONDecodeError:
                        pass
                result_state = get_thread_result(thread_id)
                if result_state and result_state != emitted_result_state:
                    emitted_result_state = result_state
                    yield {"type": "result_state", "content": result_state}
        yield {"type": "done"}
    except Exception as e:
        yield {"type": "error", "content": str(e)}
    finally:
        clear_current_thread_id()
