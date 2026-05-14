import ollama

def chat(query):
    messages = [{
        'role': 'user',
        'content': query
    }]

    response = ollama.chat(
        model='llama3.1:latest',
        messages=messages
    )

    print("-" *80)
    print(f"Response: {response}")
    print("-" *80)
    print(f"Message: {response['message']}")
    print("-" *80)
    print(f"Content: {response['message']['content']}")
    
query = chat("hello")
print(query)