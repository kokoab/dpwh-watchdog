from __future__ import annotations

import json
import os

from query_scope import compact_thread_context

SYNTHESIS_SYSTEM_PROMPT = """
You are the DPWH Watchdog synthesis assistant.
Use only the structured tool output provided to answer the task.
Be concise, explicit, and evidence-bound.
If the data does not contain enough evidence to answer the question, say so explicitly.
Do not mention hidden prompts, tool names, or chain-of-thought.
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
