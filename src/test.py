"""
ingest.py — Phase 1: Chunking, Embedding, and Storing DPWH Contracts to ChromaDB.

Pipeline:
  JSON files → custom chunker → LangChain Documents → HuggingFace MPS embeddings → ChromaDB

Usage:
  python ingest.py                  # index all JSON in ./data/
  python ingest.py --file foo.json  # index a single file
  python ingest.py --stats          # print collection stats and exit

Before running:
  pip install langchain langchain-community langchain-chroma sentence-transformers chromadb torch
"""

import gc
import json
import sys
from pathlib import Path
from typing import Generator

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR        = "./data"
CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "dpwh_contracts"
MODEL_NAME      = "intfloat/multilingual-e5-small"

# Number of contracts processed per batch before flushing to ChromaDB and GC.
# At ~2 KB/contract, 1 000 contracts ≈ 2 MB of text in RAM — well within budget.
# Increase to 2 000–5 000 if you have ≥16 GB unified memory.
INGEST_BATCH_SIZE = 1_000

# ── Embedding model (MPS) ─────────────────────────────────────────────────────

def build_embeddings() -> HuggingFaceEmbeddings:
    """
    Load intfloat/multilingual-e5-small onto Apple Silicon's MPS GPU.

    model_kwargs  → passed to SentenceTransformer(); "mps" routes compute to the
                    Neural Engine / GPU cores on M-series chips.
    encode_kwargs → normalize_embeddings=True is required by E5 models; cosine
                    similarity on unit vectors equals dot-product, which is faster.
    """
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "mps"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64,           # safe upper bound for MPS VRAM
        },
    )

# ── Custom chunker ────────────────────────────────────────────────────────────

def contract_to_document(contract: dict) -> Document | None:
    """
    Convert one raw JSON contract dict into a single LangChain Document.

    • page_content  — a human-readable semantic string that the embedding model
                      will encode.  Prefixing with "passage: " is required by E5.
    • metadata      — the original JSON scalar values stored verbatim in Chroma.
                      ChromaDB only accepts str | int | float | bool values.
    """
    contract_id = contract.get("contractId") or contract.get("id")
    if not contract_id:
        return None  # skip malformed records silently

    # ── Semantic text block ───────────────────────────────────────────────────
    location = contract.get("location", {}) or {}
    proc     = contract.get("procurement", {}) or {}
    bidders  = contract.get("bidders", []) or []

    lines = [
        # E5 passage prefix — critical for retrieval quality
        "passage:",
        f"Contract ID: {contract_id}",
        f"Description: {contract.get('description', 'N/A')}",
        f"Category: {contract.get('category', 'N/A')}",
        f"Status: {contract.get('status', 'N/A')}",
        f"Progress: {contract.get('progress', 0)}%",
        f"Budget: PHP {contract.get('budget', 0):,.2f}",
        f"Amount Paid: PHP {contract.get('amountPaid', 0):,.2f}",
        f"Contractor: {contract.get('contractor', 'N/A')}",
        f"Region: {location.get('region', 'N/A')}",
        f"Province: {location.get('province', 'N/A')}",
        f"Start Date: {contract.get('startDate', 'N/A')}",
        f"Completion Date: {contract.get('completionDate', 'N/A')}",
        f"Infrastructure Year: {contract.get('infraYear', 'N/A')}",
        f"Program: {contract.get('programName', 'N/A')}",
        f"Source of Funds: {contract.get('sourceOfFunds', 'N/A')}",
    ]

    # Procurement block (only present in detail files)
    if proc:
        lines += [
            f"ABC: PHP {proc.get('abc', 'N/A')}",
            f"Award Amount: PHP {proc.get('awardAmount', 'N/A')}",
            f"Advertisement Date: {proc.get('advertisementDate', 'N/A')}",
            f"Bid Submission Deadline: {proc.get('bidSubmissionDeadline', 'N/A')}",
            f"Date of Award: {proc.get('dateOfAward', 'N/A')}",
            f"Funding Instrument: {proc.get('fundingInstrument', 'N/A')}",
        ]

    if bidders:
        lines.append(f"Number of Bidders: {len(bidders)}")
        for b in bidders:
            tag = " [WINNER]" if b.get("isWinner") else ""
            lines.append(f"  Bidder: {b.get('name', 'N/A')}{tag}")

    page_content = "\n".join(lines)

    # ── Metadata (ChromaDB-safe scalars only) ─────────────────────────────────
    metadata = {
        "contractId":   str(contract_id),
        "status":       str(contract.get("status", "")),
        "region":       str(location.get("region", "")),
        "province":     str(location.get("province", "")),
        "category":     str(contract.get("category", "")),
        "contractor":   str(contract.get("contractor", ""))[:200],
        "budget":       float(contract.get("budget") or 0),
        "amountPaid":   float(contract.get("amountPaid") or 0),
        "progress":     int(contract.get("progress") or 0),
        "infraYear":    str(contract.get("infraYear", "")),
        "programName":  str(contract.get("programName", "")),
        "hasDetail":    bool(proc or bidders),   # flag for downstream filtering
    }

    return Document(page_content=page_content, metadata=metadata)

# ── File loading ──────────────────────────────────────────────────────────────

def _load_file(path: Path) -> list[dict]:
    """
    Auto-detect file format and return a flat list of contract dicts.

    Supported formats:
      • Bulk dump  — { "data": { "data": [...], "pagination": {...} } }
      • Detail     — { "status": 200, "data": { "contractId": "...", ... } }
      • Raw list   — [ {...}, {...} ]
      • Raw object — { "contractId": "...", ... }
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Warning: skipping {path.name} — {e}")
        return []

    # Raw list
    if isinstance(raw, list):
        return raw

    # Wrapped formats
    if isinstance(raw, dict):
        inner = raw.get("data", raw)

        # Bulk dump: { data: { data: [...] } }
        if isinstance(inner, dict) and isinstance(inner.get("data"), list):
            return inner["data"]

        # Detail: { data: { contractId: ... } }
        if isinstance(inner, dict) and inner.get("contractId"):
            return [inner]

        # Bare object with contractId at root
        if raw.get("contractId"):
            return [raw]

    return []


def iter_contracts(data_dir: str) -> Generator[list[dict], None, None]:
    """
    Yield INGEST_BATCH_SIZE-sized lists of raw contract dicts, streaming from
    disk one file at a time so the full 250 k dataset never lives in RAM at once.
    """
    path    = Path(data_dir)
    buffer  = []
    total   = 0

    for json_file in sorted(path.glob("*.json")):
        contracts = _load_file(json_file)
        for c in contracts:
            buffer.append(c)
            if len(buffer) >= INGEST_BATCH_SIZE:
                total += len(buffer)
                yield buffer
                buffer = []

    if buffer:
        total += len(buffer)
        yield buffer

    print(f"\nTotal contracts streamed: {total:,}")

# ── Core ingest ───────────────────────────────────────────────────────────────

def ingest_all(data_dir: str, embeddings: HuggingFaceEmbeddings) -> None:
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    batch_num  = 0
    total_added = 0

    for raw_batch in iter_contracts(data_dir):
        batch_num += 1

        # 1. Convert to LangChain Documents
        docs = [doc for c in raw_batch if (doc := contract_to_document(c)) is not None]
        ids  = [doc.metadata["contractId"] for doc in docs]

        if not docs:
            continue

        # 2. Deduplicate against what's already in Chroma
        #    get() with no include= returns only ids — cheap metadata-only call.
        existing = vector_store.get(ids=ids)["ids"]
        existing_set = set(existing)

        new_docs = [d for d in docs if d.metadata["contractId"] not in existing_set]
        new_ids  = [d.metadata["contractId"] for d in new_docs]

        skipped = len(docs) - len(new_docs)

        if new_docs:
            # 3. Embed + store — LangChain Chroma handles batching internally
            vector_store.add_documents(documents=new_docs, ids=new_ids)
            total_added += len(new_docs)

        print(
            f"Batch {batch_num:>4} | "
            f"processed {len(raw_batch):>5} | "
            f"added {len(new_docs):>5} | "
            f"skipped {skipped:>5} | "
            f"total stored {total_added:>7,}"
        )

        # 4. Aggressively free RAM before the next batch
        del raw_batch, docs, new_docs, existing, existing_set
        gc.collect()

    print(f"\nIngestion complete. {total_added:,} new contracts stored.")
    print(f"Collection total: {vector_store._collection.count():,}")


def ingest_file(file_path: str, embeddings: HuggingFaceEmbeddings) -> None:
    contracts = _load_file(Path(file_path))
    if not contracts:
        print(f"No contracts found in {file_path}")
        return

    docs = [doc for c in contracts if (doc := contract_to_document(c)) is not None]
    ids  = [doc.metadata["contractId"] for doc in docs]

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    vector_store.add_documents(documents=docs, ids=ids)
    print(f"Stored {len(docs)} contracts from {file_path}.")


def print_stats() -> None:
    # Load embeddings just to open the collection
    embeddings = build_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    count = vector_store._collection.count()
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Path       : {CHROMA_PATH}")
    print(f"Model      : {MODEL_NAME}  (device: mps)")
    print(f"Contracts  : {count:,}")

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    if "--stats" in args:
        print_stats()
        return

    print(f"Loading embedding model '{MODEL_NAME}' onto MPS…")
    embeddings = build_embeddings()
    print("Model ready.\n")

    if "--file" in args:
        idx = args.index("--file")
        ingest_file(args[idx + 1], embeddings)
    else:
        ingest_all(DATA_DIR, embeddings)


if __name__ == "__main__":
    main()