from langchain_ollama import ChatOllama
from langsmith import Client
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



llm = ChatOllama(
    model="llama3.1:latest",
    base_url = "http://host.docker.internal:11434",
    temperature=0.1,
    top_p=0.3,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are the DPWH Watchdog AI assistant. "
        "You MUST ALWAYS use the search_contracts tool first before answering ANY question. "
        "Never answer from memory. Always search first, then answer based only on what the tool returns. "
        "If the tool returns nothing relevant, say so."
    ),
    MessagesPlaceholder(variable_name="messages"),
])
memory_saver = MemorySaver()

watchdog_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt.messages[0].prompt.template,
    checkpointer=memory_saver
)
