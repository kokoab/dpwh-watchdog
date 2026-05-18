import requests
import chromadb
import asyncio
import aiohttp

URL = "http://127.0.0.1:8000/embed"

# CRITICAL: must be <= MAX_BATCH_TEXTS in api.py
# api.py sets 64 on MPS, 1024 on CPU — keep this at 32 to stay safely under both
BATCH_SIZE = 128

CONCURRENT_REQUESTS = 4
REQUEST_RETRY_ATTEMPTS = 3
REQUEST_RETRY_BASE_DELAY = 0.5

CHROMA_PATH = "./chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="dpwh_contracts", metadata={"hnsw:space": "cosine"}
)


async def fetch_embeddings_async(
    session: aiohttp.ClientSession,
    text_list: list[str],
    attempt: int = 0,
) -> list[list[float]] | None:
    """
    Fetch embeddings for a batch. Retries with backoff.
    Does NOT split recursively — that caused the connection flood.
    """
    try:
        async with session.post(
            URL,
            json={"inputs": text_list},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data["embedding"]
            body = await response.text()
            print(f"\nServer error {response.status}: {body[:120]}")
            return None

    except Exception as e:
        if attempt + 1 < REQUEST_RETRY_ATTEMPTS:
            wait = REQUEST_RETRY_BASE_DELAY * (2**attempt)
            print(
                f"\nConnection failed (attempt {attempt + 1}), retrying in {wait:.1f}s: {e}"
            )
            await asyncio.sleep(wait)
            return await fetch_embeddings_async(session, text_list, attempt + 1)
        print(f"\nGiving up after {REQUEST_RETRY_ATTEMPTS} attempts: {e}")
        return None


def get_embeddings_sync(text_list: list[str]) -> list[list[float]] | None:
    try:
        response = requests.post(URL, json={"inputs": text_list}, timeout=60)
        if response.status_code == 200:
            return response.json()["embedding"]
        print(f"Embedding server error: {response.status_code} {response.text}")
        return None
    except requests.exceptions.ConnectionError:
        print("ERROR: Embedding server not running. Start: uvicorn api:app --reload")
        return None


def _is_detailed(text: str) -> bool:
    """'Number of Bidders' only exists in detailed records, never in shallow dumps."""
    return "Number of Bidders" in text


def index_docs(chunks: list[dict]) -> None:
    if not chunks:
        return

    # ── Step 1: check existing DB ─────────────────────────────────────────────
    all_ids = [c["id"] for c in chunks]
    existing_docs = {}

    if collection.count() > 0:
        print("Checking database for existing documents...")
        GET_BATCH = 20000
        for i in range(0, len(all_ids), GET_BATCH):
            batch_ids = all_ids[i : i + GET_BATCH]
            result = collection.get(ids=batch_ids, include=["documents"])
            for eid, edoc in zip(result["ids"], result["documents"]):
                existing_docs[eid] = edoc

    # ── Step 2: classify chunks ───────────────────────────────────────────────
    new_chunks = []
    upgrade_chunks = []
    skipped = 0

    for chunk in chunks:
        cid = chunk["id"]
        if cid not in existing_docs:
            new_chunks.append(chunk)
        elif _is_detailed(chunk["text"]) and not _is_detailed(existing_docs[cid]):
            upgrade_chunks.append(chunk)
        else:
            skipped += 1

    if len(new_chunks) + len(upgrade_chunks) == 0:
        print(f"Nothing to do — {skipped} already fully indexed.")
        return

    print(
        f"New: {len(new_chunks)} | Upgrades: {len(upgrade_chunks)} | Skipped: {skipped}"
    )

    # ── Step 3: embed + add new contracts ────────────────────────────────────
    if new_chunks:
        print(f"Adding {len(new_chunks)} new contracts...")

        async def add_new():
            success = 0
            failed = 0
            connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
            async with aiohttp.ClientSession(connector=connector) as session:
                for i in range(0, len(new_chunks), BATCH_SIZE):
                    batch = new_chunks[i : i + BATCH_SIZE]
                    vectors = await fetch_embeddings_async(
                        session, [c["text"] for c in batch]
                    )

                    if vectors:
                        collection.add(
                            documents=[c["text"] for c in batch],
                            embeddings=vectors,
                            metadatas=[c["metadata"] for c in batch],
                            ids=[c["id"] for c in batch],
                        )
                        success += len(batch)
                    else:
                        failed += len(batch)

                    print(
                        f"  [add] {success}/{len(new_chunks)}  ({failed} failed)",
                        end="\r",
                    )
            print()

        asyncio.run(add_new())

    # ── Step 4: upgrade shallow → detailed WITHOUT re-embedding ───────────────
    # collection.update() replaces document text + metadata but keeps the
    # existing embedding vector. Zero API calls, runs in seconds not minutes.


    if upgrade_chunks:
        print(f"Upgrading {len(upgrade_chunks)} shallow records (no re-embedding)...")
        UPDATE_BATCH = 5000
        upgraded = 0
        for i in range(0, len(upgrade_chunks), UPDATE_BATCH):
            batch = upgrade_chunks[i : i + UPDATE_BATCH]

            # Get existing embeddings and reuse them explicitly
            existing = collection.get(ids=[c["id"] for c in batch], include=["embeddings"])

            collection.update(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                embeddings=existing["embeddings"],  # reuse existing vectors
                metadatas=[c["metadata"] for c in batch],
            )
            upgraded += len(batch)
            print(f"  [upgrade] {upgraded}/{len(upgrade_chunks)}", end="\r")
        print(f"\n  Done — {upgraded} records upgraded instantly.")

    print(f"Collection now has {collection.count()} contracts.")


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

    return [
        {"text": doc, "metadata": meta, "distance": round(dist, 4)}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def collection_stats() -> dict:
    return {
        "total_contracts_indexed": collection.count(),
        "collection_name": collection.name,
    }
