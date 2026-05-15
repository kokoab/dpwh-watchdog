import requests
import chromadb
import os
import asyncio
import aiohttp

URL = "http://127.0.0.1:8000/embed"
BATCH_SIZE = 512
CONCURRENT_REQUESTS = 4 

CHROMA_PATH = "./chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="dpwh_contracts",
    metadata={"hnsw:space": "cosine"}
    
)

async def fetch_embeddings_async(session: aiohttp.ClientSession, text_list: list[str]) -> list[list[float]] | None:
    try:
        async with session.post(URL, json={"inputs": text_list,}, timeout=aiohttp.ClientTimeout(total=60)) as response:
            if response.status == 200:
                data = await response.json()
                return data["embedding"]
            else:
                print(f"Embedding server error: {response.status}")
    except Exception as e:
        print(f"ERROR: Server connection failed: {e}")
        return None
            

def get_embeddings_sync(text_list: list[str]) -> list[list[float]] | None:
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
        print("Checking database for existing documents")
        all_ids = [c["id"] for c in chunks]

        GET_BATCH_SIZE = 20000
        for i in range(0, len(all_ids), GET_BATCH_SIZE):
            batchs_ids = all_ids[i: i + GET_BATCH_SIZE]
            existing = collection.get(ids=batchs_ids, include=[])
            existing_ids.update(existing["ids"])

    new_chunks = [c for c in chunks if c ["id"] not in existing_ids]
    
    if not new_chunks:
        print("All chunks already indexed.")
        return

    print(f"Indexing {len(new_chunks)} new chunks. Skipping: {len(existing_ids)}")
    
    async def process_all_chunks():
        success = 0
        
        connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
        async with aiohttp.ClientSession(connector=connector) as session:
            step_size = BATCH_SIZE * CONCURRENT_REQUESTS
            
            for i in range(0, len(new_chunks), step_size):
                super_batch = new_chunks[i: i + step_size]
                
                tasks = []
                sub_batch = []

                for j in range(0, len(super_batch), BATCH_SIZE):
                    batch = super_batch[j: j + BATCH_SIZE]
                    sub_batch.append(batch)
                    texts = [f"passage: {c['text']}" for c in batch]

                    tasks.append(fetch_embeddings_async(session, texts))
                
                results = await asyncio.gather(*tasks)

                for batch, vectors in zip(sub_batch, results):
                    if vectors:
                        collection.add(
                            documents=[c["text"] for c in batch],
                            embeddings=vectors,
                            metadatas=[c["metadata"] for c in batch],
                            ids=[c["id"] for c in batch]
                        )
                        success += len(batch)
                print(f"Indexed {success}/{len(new_chunks)}", end="\r")
                
    asyncio.run(process_all_chunks())
    print(f"\nIndexing complete. Collection now has {collection.count()} contracts")
            
def retrieve(query: str, top_k: int=5, filters: dict = None) -> list[dict]:
    prefixed_query = f"query: {query}"
    q_emb = get_embeddings_sync([prefixed_query])

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
            "metadata": meta,
            "distance": round(dist, 4)
        })
        
    return output

def collection_stats() -> str:
    count = collection.count()
    return {
        "total_contracts_indexed": count,
        "collection_name": collection.name
    }
