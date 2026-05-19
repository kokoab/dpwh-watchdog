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
        "For greetings or general conversation, respond normally without using tools. "
        "For ANY question about contracts, projects, infrastructure, contractors, or locations, "
        "you MUST call search_contracts first before answering. "
        "Never answer contract-related questions from memory."
        "If the contract search returns no relevant results, "
        "you MUST then use duckduckgo_search to find information online. "
        "Never say you couldn't find something without trying both tools."
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
