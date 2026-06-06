"""
Usage:
  python main.py ingest   # index all JSON in ./data/
  python main.py chat     # start the chat interface
"""

import json
import os
import sys

import httpx
from embeddings import ingest_all

DATA_DIR = "./data"
API_URL = os.environ.get("CHAT_API_URL", "http://localhost:8000")


def cmd_ingest():
    try:
        ingest_all(DATA_DIR)

        print("Ingesting data...")

    except httpx.ConnectError as e:
        print(f"Cannot connect to the LLM: {e}")
    except httpx.ConnectTimeout as e:
        print(f"Connection timed out: {e}")
    except httpx.NetworkError as e:
        print(f"Network Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")


def cmd_chat():
    try:
        while True:
            user_input = input("You: ")
            if not user_input:
                continue
            if user_input.lower() in ["exit", "bye"]:
                break

            url = f"{API_URL}/chat/stream"
            payload = {"message": user_input, "thread_id": "main-session"}

            try:
                with httpx.stream("POST", url, json=payload, timeout=None) as resp:
                    if resp.status_code != 200:
                        print(f"Server error: {resp.status_code}")
                        continue

                    buffer = []
                    for line in resp.iter_lines():
                        if line is None:
                            continue
                        if line == "":
                            for inner_line in buffer:
                                if inner_line.startswith("data"):
                                    data = inner_line[len("data:") :].strip()
                                    if not data:
                                        continue
                                    try:
                                        event = json.loads(data)
                                    except Exception:
                                        continue
                                    etype = event.get("type")
                                    if etype == "token":
                                        print(
                                            event.get("token", ""), end="", flush=True
                                        )
                                    elif etype == "sources":
                                        sources = event.get("content")
                                        if sources:
                                            print("\n\nSources: ")
                                            for source in sources:
                                                parts = [
                                                    f"Contract ID: {source.get('contractId', 'Unkown')}",
                                                    f"Contractor: {source.get('contractor', 'Unknown')}",
                                                    f"Region: {source.get('region', 'Unknown')}",
                                                    f"Province: {source.get('province', 'Unknown')}",
                                                    f"Budger: {source.get('budget', 'Unknown')}",
                                                    f"Infrastructure Year: {source.get('infraYear', 'Unknown')}",
                                                    f"Program Name: {source.get('programName', 'Unknown')}\n\n",
                                                ]
                                                print("-" + "\n ".join(parts))
                                            print()
                                    elif etype == "done":
                                        pass
                                    elif etype == "error":
                                        print("\n[error]", event.get("content"))
                            buffer = []
                        else:
                            buffer.append(line)
                    print()
            except httpx.RequestError as e:
                print(f"Cannot connect to chat server: {e}")
    except KeyboardInterrupt:
        print("\nExiting chat.")
    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    args = sys.argv[1:]

    if not args:
        return __doc__

    command = args[0]

    if command == "ingest":
        cmd_ingest()
    elif command == "chat":
        cmd_chat()
    else:
        print(f"Unknown command: {command!r}")
        print(__doc__)


if __name__ == "__main__":
    main()
