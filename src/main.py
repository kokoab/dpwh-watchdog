"""
main.py — entry point for the DPWH Watchdog RAG pipeline.
 
Usage:
  python main.py ingest                         # index all JSON in ./data/
  python main.py ingest --detail                # index per-contract detail JSONs
  python main.py ingest --file dump.json        # index a single file
  python main.py chat                           # start the chat interface
  python main.py stats                          # show collection stats
 
Before running:
  1. Start the embedding server:     uvicorn api:app --reload
  2. Put your JSON files in ./data/
  3. Run:                             python main.py ingest
  4. Then:                            python main.py chat
"""

import sys
from pathlib import Path
import gc

from create_chunking import (
    load_contracts_from_dump,
    load_contracts_from_detail,
    prepare_chunks
)
from create_embedding import index_docs, collection_stats
from chat import chat_with_document

DATA_DIR = "./data"


def cmd_ingest(args: list[str]):
    detail_mode = "--detail" in args

    if "--file" in args:
        idx = args.index("--file")
        file_path = args[idx + 1]
        if detail_mode:
            contracts = load_contracts_from_detail(file_path)
        else:
            contracts = load_contracts_from_dump(file_path)

        if not contracts:
            print("No contracts found. Check your directory.")
            return

        chunks = prepare_chunks(contracts, detail=detail_mode)
        index_docs(chunks)
        return

    json_files = list(Path(DATA_DIR).glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {DATA_DIR}")
        return

    print(f"Found {len(json_files)} files. Beginning batched ingestion...")

    BATCH_SIZE = 2048
    for i in range(0, len(json_files), BATCH_SIZE):
        batch_files = json_files[i: i + BATCH_SIZE]
        print(f"\n--- Loading File Batch {i + 1} to {min(i + BATCH_SIZE, len(json_files))} of {len(json_files)} ---")

        batch_contracts = []

        for file in batch_files:
            if detail_mode:
                c = load_contracts_from_detail(str(file))
            else:
                c = load_contracts_from_dump(str(file))

            if c:
                if isinstance(c, list):
                    batch_contracts.extend(c)
                else:
                    batch_contracts.append(c)

        if not batch_contracts:
            continue

        print(f"Prepared {len(batch_contracts)} contracts to process...")
        chunks = prepare_chunks(batch_contracts, detail=detail_mode)
        index_docs(chunks)

        del batch_contracts
        del chunks
        gc.collect()


def cmd_stats():
    stats = collection_stats()

    print(f"Collection name: {stats['collection_name']}")
    print(f"Contracts Indexed: {stats['total_contracts_indexed']}")


def cmd_chat():
    stats = collection_stats()
    if stats["total_contracts_indexed"] == 0:
        prepare_chunks("No contracts indexed yet. Run: python main.py ingest")
        return
    chat_with_document()


def main():
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        return

    command = args[0]

    if command == "ingest":
        cmd_ingest(args[1:])
    elif command == "chat":
        cmd_chat()
    elif command == "stats":
        cmd_stats()
    else:
        print(f"Unknown Command: {command!r}")
        print(__doc__)


if __name__ == "__main__":
    main()
