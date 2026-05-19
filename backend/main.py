"""
Usage:
  python main.py ingest   # index all JSON in ./data/
  python main.py chat     # start the chat interface
"""

from embeddings import ingest_all
import sys
from agent import watchdog_agent

DATA_DIR = "./data"

def cmd_ingest():
    ingest_all(DATA_DIR)

    print("Ingesting data...")

def cmd_chat():
    while True:
        user_input = input("You: ")
        
        if not user_input:
            continue

        if user_input.lower() in ["exit", "bye"]:
            break
        
        for chunk in watchdog_agent.stream (
            {"messages": [("user", user_input)]},
            config = {"configurable": {"thread_id": "main-session"}},
            stream_mode="values",
        ):
            msg = chunk["messages"][-1]
            msg.pretty_print()
        
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

    