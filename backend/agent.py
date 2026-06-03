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
            "You are the DPWH Watchdog AI assistant. "
            "For greetings or general conversation, respond normally without using tools. "
            "For ANY question about contracts, projects, infrastructure, contractors, or locations, "
            "you MUST call search_contracts first before answering. "
            "Never answer contract-related questions from memory. "
            "If search_contracts returns results, mention at least one contractId and one location field "
            "(region or province) from the tool output in your answer. "
            "If the contract search returns no relevant results, "
            "you MUST then use duckduckgo_search to find information online. "
            "Never say you couldn't find something without trying both tools.",
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
                yield {"type": "token", "token": msg.content}

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

