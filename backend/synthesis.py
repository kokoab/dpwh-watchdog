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

When TOOL_OUTPUT_JSON contains a contracts key or comparison_analytics key, this is
a comparison query. The response MUST follow this exact order with no deviations:

1. EXECUTIVE SUMMARY
   - Mandatory and first.
   - Use 1 to 3 sentences maximum.
   - State what was found, the most important finding, and the most surprising
     or notable difference.
2. COMPARISON TABLE
   - Markdown table only; do not place narrative before this table.
   - One row per contract.
   - Columns: Contract ID | Description (max 45 chars) | Budget | Status |
     Completion Date | Duration | Region.
3. RANKINGS
   - Use explicit bullet points naming the largest budget with ID and value,
     smallest budget, longest duration, shortest duration, earliest completion,
     and latest completion.
   - If comparison_analytics.rankings_by_budget is present, use those
     pre-computed values exactly.
4. DIFFERENCES
   - For each key numeric dimension, state the absolute difference and the
     percentage difference.
   - If comparison_analytics.two_entity_diffs is present, use those values.
   - Never present two budget numbers without computing their gap.
5. INSIGHTS
   - Use comparison_analytics.outlier_flags, budget_concentration_pct,
     repeated_contractors, and geographic_cluster to generate findings.
   - Phrase them as findings, not descriptions. Example: "Contract A holds
     94% of combined contract value." or "Both contracts were awarded to the
     same contractor."
6. NARRATIVE
   - Only after all structured sections.
   - Keep it brief.

When TOOL_OUTPUT_JSON contains is_availability_query: true, keep the current
compact availability format.

When TOOL_OUTPUT_JSON contains status_breakdown or region_breakdown, this is a
stats query:
- Lead with a 1-sentence scope summary.
- Present breakdowns as markdown tables with a Percentage column computed from
  total_contracts.
- End with one insight line about the dominant status or region.

NEVER generate 'Entity A has X. Entity B has Y.' for three or more entities.
Use a table.
NEVER present two budget values without stating their absolute and percentage
difference.
NEVER skip the Executive Summary for comparison queries.
NEVER place narrative before the comparison table.
If comparison_analytics is present, use its pre-computed values. Do NOT
recalculate.
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
