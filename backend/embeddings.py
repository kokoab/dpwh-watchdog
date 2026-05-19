from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import requests
from pathlib import Path
from typing import Iterator
from chunking import load_document
import gc


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "dpwh_contracts"
DATA_DIR = "./data"
URL = "http://127.0.0.1:8000/embed"

INGEST_BATCH_SIZE = 128

class LocalAPIEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(URL, json={"inputs": texts}, timeout=120)
        response.raise_for_status()
        return response.json()["embedding"]
    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([f"query: {text}"])[0]
    
def save_to_chroma(documents: list[Document]):
    embeddings = LocalAPIEmbeddings()
    
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    
    ids = [doc.metadata["contractId"] for doc in documents]
    vector_store.add_documents(documents=documents, ids=ids)
    print(f"Saved {len(documents)} documents to ChromaDB")

def query_chroma(query: str, top_k: int = 5) -> list[Document]:
    embeddings = LocalAPIEmbeddings()
    
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    results = vector_store.similarity_search(query, k=top_k)
    return results

def buffer_contracts(data_dir: Path) -> Iterator[list[dict]]:
    path = Path(data_dir)
    buffer = []
    total = 0
    
    for json_file in sorted(path.glob("*.json")):
        contracts = load_document(json_file)
        for c in contracts:
            buffer.append(c)
            if len(buffer) >= INGEST_BATCH_SIZE:
                total += len(buffer)
                yield buffer
                buffer = []
    if buffer:
        total += len(buffer)
        yield buffer
        
    print(f"Total contracts streamed: {total:,}")
        
def ingest_all(data_dir: Path) -> None:
    vector_store = Chroma (
        collection_name=COLLECTION_NAME,
        embedding_function=LocalAPIEmbeddings(),
        persist_directory=CHROMA_PATH
        
    )
    
    batch_num = 0
    total_added = 0
    
    for raw_batch in buffer_contracts(data_dir):
        batch_num += 1
        
        docs = raw_batch
        ids = [doc.metadata["contractId"] for doc in docs]
        
        if not docs:
            continue

        existing = vector_store.get(ids=ids, include=["metadatas"])
        # existing_set = set(existing)

        existing_map = {
            meta["contractId"]: meta
            for meta in existing["metadatas"]
        }
        
        new_docs = []
        upgrade_docs = []
        # new_ids = [d.metadata["contractId"] for d in new_docs]

        for d in docs:
            cid = d.metadata["contractId"]
            if cid not in existing_map:
                new_docs.append(d)
            elif d.metadata["hasDetail"] and not existing_map[cid].get("hasDetail"): 
                upgrade_docs.append(d)

        skipped = len(docs) - len(new_docs) - len(upgrade_docs)

        if new_docs:
            save_to_chroma(new_docs)
            total_added += len(new_docs)
            
        if upgrade_docs:
            upgrade_ids = [d.metadata["contractId"] for d in upgrade_docs]
            vector_store.update_documents(ids=upgrade_ids, documents=upgrade_docs)
            print(f"Upgraded {len(upgrade_docs)} shallow records to detailed")

        print(
            f"Batch: {batch_num:>4} | "
            f"Processed: {len(raw_batch):>5} | "
            f"Added: {len(new_docs):>5} | "
            f"Skipped: {skipped} | "
            f"Total Stored: {total_added:>7,}"
        )
        
        del raw_batch, docs, new_docs, upgrade_docs, existing, existing_map
        gc.collect()
        
    print(f"\nIngestion Complete: {total_added} new contracts stored.")
    print(f"Collection Total: {vector_store._collection.count():,}")
        

if __name__ == "__main__":
    ingest_all(Path(DATA_DIR))




