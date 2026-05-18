import chromadb
import requests

CHROMA_PATH = "./chroma_db"
URL = "http://127.0.0.1:8000/embed"

def get_query_embedding(text: str):
    # E5-small requires "query: " prefix. BGE-M3 does NOT need a prefix!
    # If using BGE-M3, remove "query: " and just pass the text.
    response = requests.post(URL, json={"inputs": [text]}, timeout=30)
    return response.json()["embedding"][0]

def test_query(query: str):
    print(f"\n{'='*50}")
    print(query)
    print(f"{'='*50}")
    
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name="dpwh_contracts")
    
    q_emb = get_query_embedding(query)
    
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=5,
        include=["metadatas", "distances"]
    )
    
    # Print the top 5 results
    for i in range(5):
        try:
            contract_id = results["metadatas"][0][i].get("contractId", "Unknown")
            contractor = results["metadatas"][0][i].get("contractor", "Unknown")
            distance = results["distances"][0][i]
            
            # Lower distance means mathematically closer/better!
            print(f"Rank {i+1} | Distance: {distance:.4f} | ID: {contract_id}")
            print(f"   -> Contractor: {contractor[:60]}...")
        except IndexError:
            break

if __name__ == "__main__":
    # Put your Golden Queries here!
    test_query("Projects awarded to B. Vicencio Builders in Leyte.")
    test_query("Flood control projects that are 100% completed but amount paid is 0.")