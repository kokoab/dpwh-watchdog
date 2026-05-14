import requests
from chunking import chunks
import chromadb

URL = "http://127.0.0.1:8000/embed"
BATCH_SIZE = 64

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="my_pdf_docs")

def get_embeddings(text_list):
    payload = {
        "inputs": text_list
    }
    response = requests.post(URL, json=payload)
    if response.status_code == 200:
        return response.json()["embedding"]
    return None

def index_docs():
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        prefixed_batch = [f"passage: {t}" for t in batch]
        
        vectors = get_embeddings(prefixed_batch)
        
        if vectors:
            collection.add(
                documents=batch,
                embeddings=vectors,
                ids=[f"id_{j}" for j in range(i, i + len(batch))]
            )
    print("Indexing complete")
            
def retrieve(query, top_k=3):
    prefixed_query = f"query: {query}"

    q_emb = get_embeddings([prefixed_query])[0]
    
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )
    
    return results["documents"][0]
        
if __name__ == "__main__":
    index_docs()

    user_query = "what is the location on the first page?"
    context_chunk = retrieve(user_query)

    print("\nTop matches found:")
    for i, text in enumerate(context_chunk):
        print(f"{i+1}. {text}\n")

