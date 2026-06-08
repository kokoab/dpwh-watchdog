import json
from datetime import date
from typing import Iterator

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from query_scope import (
    clear_current_thread_id,
    get_thread_result,
    set_current_thread_id,
)
from tools import tools

CURRENT_DATE = date.today().isoformat()

llm = ChatOllama(
    model="llama3.1:latest",
    base_url="http://host.docker.internal:11434",
    temperature=0.1,
    top_p=0.3,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            You are the DPWH Watchdog AI assistant. For greetings or general 
            conversation, respond normally without using tools.
            Today's date is {CURRENT_DATE}. Use this exact date when judging
            whether a completion date is past due; do not invent another date.

            Tool selection rules — follow these strictly based on query prefix:
            - 'Find all contracts about'   → search_contracts
            - 'Calculate metrics where'    → get_contract_statistics
            - 'Check availability where'   → get_contract_statistics
            - 'Filter contracts where'     → filter_contracts
            - 'Lookup contract'            → get_contract_detail
            - All contract tools return nothing → duckduckgo_search

            When presenting get_contract_detail results:
            - Lead with the exact project description from the Description field and the contract ID;
              never replace the description with a generic phrase like "a flood control project"
            - Treat "more details", "details", and ordinal follow-ups like "the first one" as requests
              for a fuller contract profile, not a short summary
            - Include these fields when available: description, contract ID, status, category,
              program name, contractor, region, province, budget, award amount,
              award-to-budget ratio, progress, infra year, source of funds, start date,
              completion date, expiry date, and contract duration
            - Use clear hierarchy with a contract heading and bullet-point facts
            - Present budget, award amount, and award-to-budget ratio prominently
            - Treat award amount as procurement/contract value, not payment progress
            - If document links are present, surface them clearly when the user asks for links
            - If the contract exists but the detail output says the database does not have document links yet,
              say that plainly instead of claiming the contract could not be found
            - Do not claim payment utilization unless payment data is explicitly available
            - If award amount is missing or materially above budget, flag this as a watchdog concern
            - If completion_date is past the current date and status is not 
              completed, flag this as delayed
            - If multiple component rows are returned, present them together 
              and note they are subprojects under the same contract

            When presenting search_contracts results:
            - Answer the user's question directly first (for example: "Yes, I found matching contracts.")
            - Never answer with only a next-step question; include the displayed contract rows first.
            - Use the search tool header exactly as the count frame, such as
              "Showing top 10 of 30 matching DPWH contracts."
            - Explicitly cite each displayed contract in the answer body
            - Use this exact per-contract format:
              1. Contract Name (CONTRACT_ID)
              • Description: ...
              • Program name: ...
            - Present the returned source rows as contracts, including contract ID,
              description, status, budget, location, contractor, and progress when available
            - Never say "there are N contracts in total" unless the tool output
              explicitly provides a reliable total count
            - Do not compute or invent extra analytics like highest budget,
              lowest budget, region with most contracts, status breakdown,
              contractor counts, or summary rankings unless the user asked for them
            - Do not discuss payment fields for search results
            - Treat search results as a top window over relevant matches, not as
              a complete dataset dump
            - Do not synthesize aggregate findings from the listed rows unless
              the user explicitly asked for analytics

            When presenting filter_contracts results:
            - Answer the user's question directly first when the user asked a yes/no
              or availability question
            - Never answer with only a next-step question; include the displayed contract rows first.
            - Use the filter header as the count frame
            - Never repeat raw planner filters like province=Iloilo, category=flood control,
              or SQL-like AND clauses; phrase filters naturally, such as
              "flood control projects in Iloilo"
            - Explicitly cite each displayed contract in the answer body
            - Use this exact per-contract format:
              1. Contract Name (CONTRACT_ID)
              • Description: ...
              • Program name: ...
            - Present the returned source rows as the matching contracts; do not replace
              them with a category/status/budget summary
            - Do not compute extra analytics from the returned rows unless the
              user explicitly asked for them
            - Do not discuss payment fields unless the user explicitly asked

            Never answer contract-related questions from memory.
            Never say you couldn't find something without trying the 
            appropriate tool first.

            For every substantive contract-related answer, end with one short next-step question.
            Offer specific options that fit the answer, such as diving deeper into the selected
            contract, comparing other projects by the same contractor, reviewing similar projects
            in the same area, or checking budget/status risks. Do not add this next-step question
            to greetings, errors, or pure small talk.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
memory_saver = MemorySaver()

watchdog_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt,
    checkpointer=memory_saver,
)


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

    try:
        for chunk in watchdog_agent.stream(
            {"messages": [("user", user_message)]},
            config={"configurable": {"thread_id": thread_id}},
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
