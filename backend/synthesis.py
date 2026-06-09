from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from query_scope import compact_thread_context

load_dotenv()

SYNTHESIS_SYSTEM_PROMPT = """
You are the DPWH Watchdog synthesis assistant.
Use ONLY the structured tool output provided to answer the task.
Never guess, infer, or add information not present in the tool output.
Do not mention tool names, system prompts, or chain-of-thought.
CITATION RULES - apply whenever contract_rows is present in the tool output:

List each contract individually using this format:
[CONTRACT_ID] Description truncated to ~80 chars

Budget: PHP X,XXX,XXX.XX
Province: X | Region: Y | Status: Z

After listing all contracts, state the total contract value explicitly.
State the province distribution explicitly (which province has the most, or note ties).
If has_more_contracts is true, note that only the top 20 by budget are shown.
Never give only aggregate totals when individual records are available - always cite
the records first, then summarize.

When contract_rows is NOT present (large result sets), answer using only the aggregates,
breakdowns, and scope information provided.
FORMAT:

Use bold for section labels and contract IDs
Use numbered lists for multi-contract citations
Use bullet points for per-contract attributes (budget, province, status)
Be concise but complete - the user needs to know what each contract is, not just totals
""".strip()


def focused_synthesis(task: str, tool_output: dict, thread_id: str) -> str:
    context = compact_thread_context(thread_id)
    model_name = os.environ.get("OLLAMA_MODEL")

    try:
        # from langchain_groq import ChatGroq
        from langchain_ollama import ChatOllama

        # llm = ChatGroq(
        #     model=model_name,
        #     temperature=0.0,
        #     max_tokens=int(os.environ.get("GROQ_SYNTHESIS_MAX_TOKENS", "500")),
        #     top_p=1.0,
        #     streaming=False,
        #     max_retries=2,
        #     timeout=30,
        # )

        llm = ChatOllama(
            model=model_name,
            base_url="http://host.docker.internal:11434",
            temperature=0.1,
            top_p=1,
            max_retries=2,
            timeout=30,
        )

        response = llm.invoke(
            [
                ("system", SYNTHESIS_SYSTEM_PROMPT),
                (
                    "user",
                    f"{context}\n\nTASK:\n{task}\n\nTOOL_OUTPUT_JSON:\n{json.dumps(tool_output, ensure_ascii=True)}",
                ),
            ]
        )
        content = getattr(response, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            ).strip()
    except Exception as e:
        return f"ERROR: {e}"

    return (
        "I could not complete focused synthesis from the current environment, "
        "so here is the structured result payload:\n"
        f"{json.dumps(tool_output, ensure_ascii=True, indent=2)}"
    )
