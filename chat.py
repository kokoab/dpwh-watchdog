import ollama
import time
from embedding import retrieve


def chat_with_document(document_text):
    messages = [
        {
            'role': 'system',
            'content': "You are an investigative journalism assistant analyzing government procurement text.\n"
                       "Your job is to identify discrepancies, unverified figures, mismatched numbers, "
                       "or missing details (like dates or signatures) using ONLY the provided text.\n"
                       "CRITICAL RULES:\n"
                       "1. Base your comparison strictly on the figures and words inside the excerpts.\n"
                       "2. If you find a data mismatch (e.g., conflicting costs), report it as a factual variance.\n"
                       "3. Do NOT invent outside context. Do NOT use the word 'suspicious' unless the text says it; "
                       "instead, flag it as an 'unreconciled data point' or 'missing information'."
                       "When quoting a number or fact, print the exact excerpt number it came from so I can verify it."
        }
    ]

    while True:
        print("-" * 80)
        user_input = input("You: ")

        if user_input.lower() in ['exit', 'bye']:
            break

        chunk = retrieve(user_input, 3)
        formatted_chunks = [f"{i + 1}. {text}" for i, text in enumerate(chunk)]

        messages.append({
            'role': 'user',
            'content': f"User Query: {user_input}\n\n Relevant excerpts:\n {formatted_chunks}"
        })

        print("-" * 80)
        start_time = time.time()

        response = ollama.chat(
            model='llama3.1:latest',
            messages=messages,
            stream=True,
            options={
                "temperature": 0.1,
                "top_p": 0.3
            }
        )

        end_time = time.time()

        response_time = end_time - start_time

        full_answer = ""
        print(f"Thought for: {response_time}")
        print(f"Answer: ", end="", flush=True)
        for token in response:
            text = token['message']['content']
            print(text, end="", flush=True)
            full_answer += text
        print()
        messages.append({
            'role': 'assistant',
            'content': full_answer
        })
