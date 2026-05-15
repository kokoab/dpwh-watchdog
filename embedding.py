import requests
import chromadb
import asyncio
import aiohttp

URL = "http://127.0.0.1:8000/embed"
BATCH_SIZE = 256
CONCURRENT_REQUESTS = 4
RETRY_SPLIT_THRESHOLD = 500
REQUEST_RETRY_ATTEMPTS = 3
REQUEST_RETRY_BASE_DELAY = 0.25

CHROMA_PATH = "./chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="dpwh_contracts",
    metadata={"hnsw:space": "cosine"}
)

async def fetch_embeddings_async(
    session: aiohttp.ClientSession,
    text_list: list[str],
    attempt: int = 0,
) -> list[list[float]] | None:
    try:
        async with session.post(URL, json={"inputs": text_list}, timeout=aiohttp.ClientTimeout(total=60)) as response:
            if response.status == 200:
                data = await response.json()
                return data["embedding"]
            if response.status >= RETRY_SPLIT_THRESHOLD and len(text_list) > 1:
                midpoint = len(text_list) // 2
                left = await fetch_embeddings_async(session, text_list[:midpoint])
                right = await fetch_embeddings_async(session, text_list[midpoint:])
                if left is None or right is None:
                    return None
                return left + right
            else:
                print(f"Embedding server error: {response.status}")
    except Exception as e:
        print(f"ERROR: Server connection failed: {e}")
        if attempt + 1 < REQUEST_RETRY_ATTEMPTS:
            await asyncio.sleep(REQUEST_RETRY_BASE_DELAY * (2 ** attempt))
            return await fetch_embeddings_async(session, text_list, attempt + 1)
        if len(text_list) > 1:
            midpoint = len(text_list) // 2
            left = await fetch_embeddings_async(session, text_list[:midpoint])
            right = await fetch_embeddings_async(session, text_list[midpoint:])
            if left is None or right is None:
                return None
            return left + right
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


def _is_detailed(text: str) -> bool:
    """
    Detect if an indexed chunk already has rich detail fields.
    'Number of Bidders' only appears in detailed records, never in
    shallow dump records — safe to use as the marker.
    """
    return "Number of Bidders" in text


def index_docs(chunks: list[dict]) -> None:
    """
    Index chunks with smart upgrade logic:

      New contract (never seen)           → add
      Existing shallow + incoming rich    → upsert (overwrite with better data)
      Existing detailed OR incoming shallow → skip

    Running ingest on detail files after dump files will automatically
    upgrade every shallow record with full bidder/procurement data.
    """
    if not chunks:
        return

    # Step 1: find which IDs already exist and grab their current text
    all_ids = [c["id"] for c in chunks]
    existing_docs = {}  # id → current indexed text

    if collection.count() > 0:
        print("Checking database for existing documents...")
        GET_BATCH_SIZE = 20000
        for i in range(0, len(all_ids), GET_BATCH_SIZE):
            batch_ids = all_ids[i: i + GET_BATCH_SIZE]
            existing = collection.get(ids=batch_ids, include=["documents"])
            for eid, edoc in zip(existing["ids"], existing["documents"]):
                existing_docs[eid] = edoc

    # Step 2: sort into new / upgrade / skip
    new_chunks = []
    upgrade_chunks = []
    skipped = 0

    for chunk in chunks:
        cid = chunk["id"]
        if cid not in existing_docs:
            new_chunks.append(chunk)
        elif _is_detailed(chunk["text"]) and not _is_detailed(existing_docs[cid]):
            # Incoming is rich, stored is shallow — upgrade it
            upgrade_chunks.append(chunk)
        else:
            skipped += 1

    total_to_write = len(new_chunks) + len(upgrade_chunks)

    if total_to_write == 0:
        print(f"Nothing to do — {skipped} contracts already fully indexed.")
        return

    print(f"New: {len(new_chunks)} | Upgrades: {len(upgrade_chunks)} | Skipped: {skipped}")

    # Step 3: embed and write
    async def process_chunks(chunks_to_write: list[dict], mode: str):
        success = 0
        connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
        async with aiohttp.ClientSession(connector=connector) as session:
            step_size = BATCH_SIZE * CONCURRENT_REQUESTS

            for i in range(0, len(chunks_to_write), step_size):
                super_batch = chunks_to_write[i: i + step_size]
                tasks = []
                sub_batches = []

                for j in range(0, len(super_batch), BATCH_SIZE):
                    batch = super_batch[j: j + BATCH_SIZE]
                    sub_batches.append(batch)
                    texts = [f"passage: {c['text']}" for c in batch]
                    tasks.append(fetch_embeddings_async(session, texts))

                results = await asyncio.gather(*tasks)

                for batch, vectors in zip(sub_batches, results):
                    if not vectors:
                        continue
                    if mode == "add":
                        collection.add(
                            documents=[c["text"] for c in batch],
                            embeddings=vectors,
                            metadatas=[c["metadata"] for c in batch],
                            ids=[c["id"] for c in batch],
                        )
                    else:
                        collection.upsert(
                            documents=[c["text"] for c in batch],
                            embeddings=vectors,
                            metadatas=[c["metadata"] for c in batch],
                            ids=[c["id"] for c in batch],
                        )
                    success += len(batch)

                print(f"  [{mode}] {success}/{len(chunks_to_write)}", end="\r")

        print()

    if new_chunks:
        print(f"Adding {len(new_chunks)} new contracts...")
        asyncio.run(process_chunks(new_chunks, mode="add"))

    if upgrade_chunks:
        print(f"Upgrading {len(upgrade_chunks)} shallow records with full detail...")
        asyncio.run(process_chunks(upgrade_chunks, mode="upsert"))

    print(f"Done. Collection now has {collection.count()} contracts.")


def retrieve(query: str, top_k: int = 5, filters: dict = None) -> list[dict]:
    prefixed_query = f"query: {query}"
    q_emb = get_embeddings_sync([prefixed_query])

    if q_emb is None:
        return []

    kwargs = {
        "query_embeddings": [q_emb[0]],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
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
            "distance": round(dist, 4),
        })

    return output


def collection_stats() -> dict:
    count = collection.count()
    return {
        "total_contracts_indexed": count,
        "collection_name": collection.name,
    }