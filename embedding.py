import requests
import chromadb
import os

URL = "http://127.0.0.1:8000/embed"
BATCH_SIZE = 64

CHROMA_PATH = "./chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="dpwh_contracts",
    metadata={"hnsw:space": "cosine"}
    
)

def get_embeddings(text_list: list[str]) -> list[list[float]] | None:
    try:
        response = requests.post(URL, json={"inputs": text_list}, timeout=30)
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            print(f"Embedding server error: {response.status_code} {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("ERROR: Server not running. Start with: uvicorn api:app")
        return None
        

def index_docs(chunks: list[dict]) -> None:
    existing_ids = set()
    
    if collection.count()>0:
        existing = collection.get(ids=[c["id"] for c in chunks], include=[])
        existing_ids = set(existing["ids"])

    new_chunks = [c for c in chunks if c ["id"] not in existing_ids]
    
    if not new_chunks:
        print("All chunks already indexed.")
        return

    print(f"Indexing {len(new_chunks)} new chunks. Skipping: {len(existing_ids)}")
    
    success = 0
    
    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i: i + BATCH_SIZE]

        texts = [f"passage: {c['text']}" for c in batch]
        vectors = get_embeddings(texts)

        if vectors is None:
            print(f"Skipping batch {i // BATCH_SIZE + 1} - embedding failed")
            continue
        
        collection.add(
            documents=[c["text"] for c in batch],
            embeddings=vectors,
            metadatas=[c["metadata"] for c in batch],
            ids=[c["id"] for c in batch]
        )
        success += len(batch)
        print(f"Indexed {success}/{len(new_chunks)}", end="\r")
        
    print(f"\nIndexing complete. Collection now has {collection.count()} contracts")
    
            
def retrieve(query: str, top_k: int=5, filters: dict = None) -> list[dict]:
    prefixed_query = f"query: {query}"
    q_emb = get_embeddings([prefixed_query])

    if q_emb is None:
        return []

    kwargs = {
        "query_embeddings": [q_emb[0]],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"]
    }
    
    if filters:
        kwargs["where"] = filters
        
    results = collection.query(**kwargs)
    
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "text": doc,
            "metadatas": meta,
            "distances": round(dist, 4)
        })
        
    return output

def collection_stats() -> str:
    count = collection.count()
    return {
        "total_contracts_indexed": count,
        "collection_name": collection.name
    }
