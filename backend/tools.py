from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from embeddings import LocalAPIEmbeddings
import json

CHROMA_PATH = "./chroma_db"
web_search = DuckDuckGoSearchRun()
embedding = LocalAPIEmbeddings()
COLLECTION_NAME = "dpwh_contracts"

@tool
def search_contracts(query: str) -> str:
    """
    Search the local vector database for DPWH (Department of Public Works and Highways) 
    contract records, bidding information, procurement history, and infrastructure agreements.
    Use this tool whenever the user asks about specific contract details or local project data.
    """    

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding,
        collection_name=COLLECTION_NAME
    )
    results = db.similarity_search(query, k=5)
    
    if not results:
        print("No relevant contracts in the database")

    sources = []
    passages = []

    for r in results:
        m = r.metadata
        sources.append({
            "contractId": m.get("contractId"),
            "contractor": m.get("contractor"),
            "region": m.get("region"),
            "province": m.get("province"),
            "budget": m.get("budget"),
            "amountPaid": m.get("amountPaid"),
            "progress": m.get("progress"),
            "status": m.get("status"),
            "category": m.get("category"),
            "infraYear": m.get("infraYear"),
            "programName": m.get("programName"),
        })
        passages.append(r.page_content)

    sources_block = f"\n\n__SOURCES__:{json.dumps(sources)}"
    content = f"Here are the relevant DPWH contracts found:\n\n {"\n\n---\n\n ".join(passages)}"

    return content + sources_block
        
    
    
    # in tools.py, change the return line
    return "Here are the relevant DPWH contracts found:\n\n" + "\n\n---\n\n".join([r.page_content for r in results])

tools = [
    search_contracts,
    web_search,
]