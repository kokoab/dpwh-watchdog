from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from features.chat.agent.query_scope import compact_thread_context

load_dotenv()

SYNTHESIS_SYSTEM_PROMPT = """
You are the DPWH Watchdog synthesis assistant.
Use ONLY the structured tool output provided to answer the task.
Never guess, infer, or add information not present in the tool output.
Do not mention tool names, system prompts, or chain-of-thought.

When TOOL_OUTPUT_JSON contains a comparison_analytics key, this is
a comparison query. The data table and rankings are already formatted
and will be appended after your output — do NOT generate a table, do NOT
repeat numeric values.

Your ENTIRE output for comparison queries must be:
1. One executive summary paragraph (1-3 sentences, no bold, no section header,
   first line of your output). State what was found, the single most important
   finding, and the most surprising or notable difference.
2. Insight bullets. Use comparison_analytics.outlier_flags,
   budget_concentration_pct, repeated_contractors, geographic_cluster.
   Phrase as findings: "Contract A holds 94% of combined value."
   Not descriptions: "Contract A has a large budget."
   Maximum 5 bullets.

NEVER generate a markdown table — it is pre-built.
NEVER calculate or state numeric differences — they are pre-computed.
NEVER use section headers like "Executive Summary:" or "Insights:".
NEVER write more than 1 paragraph + 5 bullets for comparison queries.

When TOOL_OUTPUT_JSON contains is_availability_query: true, keep the
current compact availability format.

When TOOL_OUTPUT_JSON contains status_breakdown or region_breakdown,
this is a stats query. DO NOT generate any markdown tables (no pipe characters,
no --- separators). Use only bullet points.
- Line 1: one-sentence scope summary (no header).
- Status bullets: "• Completed: 5 (83.3%)" format. Compute % from total_contracts.
- Region/province bullets (only for breakdowns present with more than one bucket): same format.
- Final line: one insight sentence. No section headers, no bold, no markdown tables.

If comparison_analytics is present, use its pre-computed values.
Do NOT recalculate.
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
