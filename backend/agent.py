import json
from datetime import date
from typing import Iterator

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
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
            - 'Calculate metrics for'      → get_contract_statistics
            - 'Filter contracts where'     → filter_contracts
            - 'Lookup contract'            → get_contract_detail
            - All contract tools return nothing → duckduckgo_search

            When presenting get_contract_detail results:
            - Lead with the project description and contract ID
            - Present budget, award amount, and award-to-budget ratio prominently
            - Treat award amount as procurement/contract value, not payment progress
            - Do not claim payment utilization unless payment data is explicitly available
            - If award amount is missing or materially above budget, flag this as a watchdog concern
            - If completion_date is past the current date and status is not 
              completed, flag this as delayed
            - If multiple component rows are returned, present them together 
              and note they are subprojects under the same contract

            Never answer contract-related questions from memory.
            Never say you couldn't find something without trying the 
            appropriate tool first.
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


def stream_agent(user_message: str, thread_id: str) -> Iterator[dict]:
    SOURCE_MARKER = "__SOURCES__"

    try:
        for chunk in watchdog_agent.stream(
            {"messages": [("user", user_message)]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="messages",
        ):
            msg, metadata = chunk
            node = metadata.get("langgraph_node")

            if node == "agent" and msg.content:
                yield {"type": "token", "content": msg.content}

            elif node == "tools" and hasattr(msg, "content") and msg.content:
                raw = msg.content
                if SOURCE_MARKER in raw:
                    text_part, sources_part = raw.split(SOURCE_MARKER, 1)
                    try:
                        sources = json.loads(sources_part)
                        yield {"type": "sources", "content": sources}
                    except json.JSONDecodeError:
                        pass
        yield {"type": "done"}
    except Exception as e:
        yield {"type": "error", "content": str(e)}
