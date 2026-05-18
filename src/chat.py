import ollama
import time
from embedding import retrieve, collection_stats

SYSTEM_PROMPT = """You are an investigative journalism assistant analyzing Philippine government procurement contracts from the DPWH.

CRITICAL INSTRUCTION FOR VAGUE QUERIES:
If the user's input is just a keyword, a name, or a broad topic (e.g., "romalduez", "flood", "leyte") and NOT a complete question, DO NOT perform a red-flag analysis. Instead:
1. Briefly state what entities or contract IDs were found in the context.
2. If the user's keyword is NOT in the provided context, tell them.
3. Immediately ask the user what specific information they are looking for.

RULES FOR ANALYSIS (Only apply if the user asks a specific question):
1. Only use information from the provided contract excerpts. Do not invent figures.
2. Always cite the Contract ID when referencing a specific contract.
3. Flag these as potential red flags worth investigating (do NOT call them proven fraud):
   - Award amount significantly HIGHER than ABC (Approved Budget for Contract).
   - Only one bidder participated.
   - Start date is the same day as the award date.
   - Very short time between advertisement and bid submission deadline.
   - Progress is 100% but amount paid is 0.
4. Use neutral, factual language. Say "unreconciled data point" not "suspicious."
5. If the context doesn't contain enough information to answer, say so clearly.
"""


def format_context(results: list[dict]) -> str:
    if not results:
        return "No relevant contracts found in the database"
    
    lines = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        lines.append(f"--- Excerpt {i} (Contract ID: {meta.get('contractId', 'N/A')}) ---")
        lines.append(r["text"]),
        lines.append("")

    return "\n".join(lines)

def chat_with_document():
    stats = collection_stats()
    print(f"\nStats: {stats['total_contracts_indexed']}")
    print("-"*80)

    messages = [{
        'role': 'system',
        'content': SYSTEM_PROMPT
    }]
    
    while True:
        print()
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
        if user_input.lower() == "stats":
            print(f"Collection: {stats}")
            continue
        
        results = retrieve(user_input,top_k=5)
        
        if not results:
            print("Could not retrieve results. Please check if the server is running.")
            continue
        
        print(f"\n[Retrieved {len(results)} contracts]: "
              f"{', '.join(r['metadata'].get('contractId', '?') for r in results)}")
        
        context = format_context(results)
        
        messages.append({
            "role": "user",
            "content": f"User Question: {user_input}\n\nRelevant contract excerpts:\n{context}"
        })
        
        print("-"*80)
        start_time = time.time()

        response = ollama.chat(
            model="llama3.1:latest",
            messages=messages,
            stream=True,
            options={
                "temperature": 0.1,
                "top_p": 0.3,
                "num_ctx": 8192,
            }
        )
        
        full_answer = ""
        print("Answer: ", end="", flush=True)
        for token in response:
            text = token["message"]["content"]
            print(text, end="", flush=True)
            full_answer += text
        print()
        
        elapsed = time.time() - start_time
        print(f"Elapsed time: {elapsed}")

        messages.append({
            "role": "assistant",
            "content": full_answer
        })
        
        if len(messages) > 13:
            messages = [messages[0]] + messages[-12:]
