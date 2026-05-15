import sys
from pathlib import Path

from chunking import (
    load_contracts_from_dump,
    load_contracts_from_detail,
    load_all_contracts,
    prepare_chunks
)
from embedding import index_docs, collection_stats
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
    else:
        contracts = load_all_contracts(DATA_DIR, detail=detail_mode)
    
    if not contracts:
        print("No contracts found. Check your directory.")
        return
    
    chunks = prepare_chunks(contracts, detail=detail_mode)
    index_docs(chunks)
    
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
        print("Unknown Command")

if __name__ == "__main__":
    main()

