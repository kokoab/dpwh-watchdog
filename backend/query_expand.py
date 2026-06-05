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
                "You are a specialized Query Expansion and Search Optimizer AI for the DPWH Watchdog platform.\n\n"
                "### CRITICAL OPERATIONAL RULE\n"
                "- If the user's message is a greeting, small talk, or completely irrelevant to contracts, projects, infrastructure, or locations, you MUST output the user's exact message verbatim. Do not change a single word.\n\n"
                "### SEARCH REWRITING OBJECTIVE\n"
                "- If the user is asking a quantitative or counting question (e.g., 'how many', 'total count', 'sum of budget'), rewrite it into this template: Calculate metrics for [Standardized Input]\n"
                "  - The standardized input must preserve ALL meaningful attributes: topic/category keywords, region, province, status words, contractor names, and year.\n"
                "  - Examples:\n"
                "    - 'how many bridge contracts in region 8' -> Calculate metrics for bridge contracts in Region VIII\n"
                "    - 'total budget for ongoing road projects in davao' -> Calculate metrics for ongoing road contracts in Davao\n"
                "    - 'how many contracts does DMCI have' -> Calculate metrics for contracts by contractor DMCI\n"
                "    - 'sum of delayed flood control projects in 2023' -> Calculate metrics for delayed flood control contracts infra_year=2023\n"
                "### FILTER TEMPLATE RULES\n"
                "- Only use the Filter template when the user provides concrete, specific attribute values (exact contractor name, specific region/province, a status word like 'delayed' or 'completed', a category, or a year).\n"
                "- Valid filterable fields are: contractor, region, province, status, category, infra_year, program_name\n"
                "- Example: 'show me all delayed contracts in Region VIII' -> Filter contracts where region=Region VIII AND status=delayed\n"
                "- Example: 'contracts by DMCI in 2023' -> Filter contracts where contractor=DMCI AND infra_year=2023\n"
                "- Example: 'all ongoing bridge contracts' -> Filter contracts where category=bridge AND status=ongoing\n\n"
                "### TERMINOLOGY MAPPING RULES\n"
                "- Convert all numeric or casual region names to formal Roman Numerals strictly (e.g., 'region 8' -> 'Region VIII', 'region 10' -> 'Region X').\n"
                "- Clean up shorthand locations to their full official names (e.g., 'cdo' -> 'Cagayan de Oro').\n\n"
                "### OUTPUT FORMAT\n"
                "Output ONLY the final string. Do not add explanations, conversational pleasantries, or markdown formatting blocks.",
            ),
            ("user", "{user_query}"),
        ]
    )

    chain = prompt | llm_expander | StrOutputParser()

    # Execute the chain and clean any accidental white space
    expanded_query = chain.invoke({"user_query": query}).strip()
    return expanded_query
