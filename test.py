import chromadb

# Connect to your local database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("dpwh_contracts")

# Fetch the exact contract that the LLM messed up
contract_id = "21IM0067"
results = collection.get(ids=[contract_id])

if results and results["documents"]:
    print(f"--- Indexed Text for {contract_id} ---")
    print(results["documents"][0])
else:
    print(f"Contract {contract_id} not found in the database.")