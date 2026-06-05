import json
from typing import Iterator

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from tools import tools

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
            """
            You are the DPWH Watchdog AI assistant. For greetings or general conversation, respond normally without using tools.

            Tool selection rules — follow these strictly:
            - If the query starts with 'Find all contracts about': call search_contracts
            - If the query starts with 'Calculate metrics for': call get_contract_statistics\n
                When presenting statistics, always highlight the budget utilization rate
                and flag anything below 30% or above 95% as noteworthy for a watchdog context.\n
            - If the query starts with 'Filter contracts where': call filter_contracts
            - If all contract tools return no results: fall back to duckduckgo_search

            When presenting filter_contracts results, summarize the total count first, then list the top results. Always mention if results were capped (e.g. 'showing 50 of 312 matches'). Present each contract with Description first, then Contract ID, then budget and status.

            Never answer contract-related questions from memory. Never say you couldn't find something without trying the appropriate tool first.

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
