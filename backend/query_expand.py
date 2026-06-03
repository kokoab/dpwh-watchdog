from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm_expander = ChatOllama(
    model="llama3.1:latest",
    base_url="http://host.docker.internal:11434",
    temperature=0.1,
    top_p=0.3,
)


def query_expand(query: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a specialized Query Expansion and Search Optimizer AI for the DPWH Watchdog platform. Your task is to analyze user queries and expand them into optimized search strings for the `search_contracts` database tool.\n\n"
                "### WORKFLOW & THINKING PROCESS\n"
                "Before providing the final search query, you MUST break down your thinking process step-by-step inside `<thinking>` tags. Show your analysis of:\n"
                "1. User Intent: What is the user looking for?\n"
                "2. Entities & Locations: Identify regions, provinces, or contractors and standardize them (e.g., convert regions to Roman Numerals).\n"
                "3. Synonyms & Technical Terms: Map conversational words to DPWH terms.\n\n"
                "### SCHEMA TERMINOLOGY REFERENCE\n"
                '- **Regions**: Always format in Roman Numerals (e.g., "Region 8" -> "Region VIII", "Region 10" -> "Region X").\n'
                "- **Provinces**: Full capitalized geographical names.\n"
                '- **Infrastructure Types**: "Flood Control and Drainage", "Revetment", "Roads", "Bridges".\n\n'
                "### OUTPUT FORMAT\n"
                "Your response must strictly follow this XML structure:\n"
                "<thinking>\n"
                "[Your step-by-step reasoning and translation analysis goes here]\n"
                "</thinking>\n"
                "<search_query>[Just the raw, optimized keyword string here]</search_query>",
            ),
            ("user", "{user_query}"),
        ]
    )

    chain = prompt | llm_expander | StrOutputParser()

    raw_ai_response = chain.invoke({"user_query": query})
    return raw_ai_response
