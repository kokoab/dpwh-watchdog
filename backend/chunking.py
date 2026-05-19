import json
from pathlib import Path
from langchain_core.documents import Document

# path_file = "./data/dump-page-1-100.json"
DATA_DIR = "./data"


def load_document(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        raw_json = json.load(file)

    contract_dump = raw_json.get("data", {}).get("data", [])

    documents = []
    for contract in contract_dump:
        doc = contract_to_document(contract, file_path)
        if doc:
            documents.append(doc)
            
    return documents


def contract_to_document(contract: dict, file_path: Path) -> Document | None:
    contract_id = contract.get("contractId")
    if not contract_id:
        return None

    location = contract.get("location", {}) or {}
    proc = {}
    bidders = {}

    lines = [
        "passage:",
        f"Contract ID: {contract_id}",
        f"Description: {contract.get('description', 'N/A')}",
        f"Category: {contract.get('category', 'N/A')}",
        f"Status: {contract.get('status', 'N/A')}",
        f"Progress: {contract.get('progress', 0)}%",
        f"Budget: {contract.get('budget', 0)}",
        f"Amount Paid: {contract.get('amountPaid', 0)}",
        f"Contractor: {contract.get('contractor', 'N/A')}",
        f"Region: {location.get('region', 'N/A')}",
        f"Province: {location.get('province', 'N/A')}",
        f"Start Date: {contract.get('startDate', 'N/A')}",
        f"Completion Date: {contract.get('completionDate', 'N/A')}",
        f"Infrastructure Year: {contract.get('infraYear', 'N/A')}",
        f"Program: {contract.get('programName', 'N/A')}",
        f"Source of Funds: {contract.get('sourceOfFunds', 'N/A')}",
    ]

    page_content = "\n".join(lines)

    metadata = {
        "contractId": str(contract_id),
        "status": str(contract.get("status") or "Unknown"),
        "region": str(location.get("region") or "Unknown"),
        "province": str(location.get("province") or "Unknown"),
        "category": str(contract.get("category") or "Unknown"),
        "contractor": str(contract.get("contractor") or "Unknown")[:200],
        "budget": float(contract.get("budget") or 0.0),
        "amountPaid": float(contract.get("amountPaid") or 0.0),
        "progress": int(contract.get("progress") or 0),
        "infraYear": str(contract.get("infraYear") or "Unknown"),
        "programName": str(contract.get("programName") or "Unknown"),
        "hasDetail": bool(proc or bidders),
        "source": str(file_path),
    }

    return Document(page_content=page_content, metadata=metadata)
