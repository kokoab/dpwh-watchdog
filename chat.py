import ollama
import time
from embedding import retrieve

def chat_with_document(document_text):
    messages=[
        {
            'role': 'system',
            'content': f"You are an investigative journalism assistant. Answer the user's questions based ONLY on the following government procurement document. If the document doesn't contain the answer, say 'The document does not specify."
        }
    ]
    
    while True:
        print("-"*80)
        user_input = input("You: ")

        if user_input.lower() in ['exit', 'bye']:
            break
        
        chunk = retrieve(user_input, 3)
        formatted_chunks = [f"{i+1}. {text}" for i, text in enumerate(chunk)]
        
        messages.append({
            'role': 'user',
            'content': f"User Query: {user_input}\n\n Relevant excerpts:\n {formatted_chunks}"
        })

        print("-"*80)
        start_time = time.time()

        response = ollama.chat(
            model='llama3.2:1b',
            messages=messages,
            stream=True
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
        print("-"*80)
        messages.append ({
            'role': 'assistant',
            'content': full_answer
        })



