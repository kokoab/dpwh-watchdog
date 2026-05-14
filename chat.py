import ollama
import time
from embedding import retrieve

def chat_with_document(document_text):
    messages=[
        {
            'role': 'system',
            'content': f"You are an investigative journalism assistant. Answer the user's questions based ONLY on the following government procurement document. If the document doesn't contain the answer, say 'The document does not specify.'\n\nDocument Text:\n{document_text}"
        }
    ]
    
    while True:
        print("-"*80)
        user_input = input("You: ")

        if user_input.lower in ['exit', 'bye']:
            break
        
        user_input_chunk = retrieve(user_input)
        user_input_embeddings = [f"{i+1}. {text}" for i, text in enumerate(user_input_chunk)]
        
        messages.append({
            'role': 'user',
            'content': f"User Query: {user_input}\n Result Embeddings: {user_input_embeddings}"
        })

        print("-"*80)
        start_time = time.time()

        response = ollama.chat(
            model='llama3.1:latest',
            messages=messages
        )

        end_time = time.time()

        response_time = end_time - start_time
        
        answer = response['message']['content']
        print(f"Thought for: {response_time}")
        print(f"Answer: {answer}")
        print("-"*80)
        messages.append ({
            'role': 'assistant',
            'content': answer
        })



