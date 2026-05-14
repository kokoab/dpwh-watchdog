import ollama
import time

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
        
        messages.append({
            'role': 'user',
            'content': user_input
        })

        print("-"*80)
        start_time = time.time()

        response = ollama.chat(
            model='llama3.1:latest',
            messages=messages
        )

        end_time = time.time()

        response_time = end_time - start_time



