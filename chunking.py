import json 
import os
from pathlib import Path

def contract_to_text(contract: dict) -> str:
    """
    Convert a contract dict into a readable text passage.
    This is what gets embedded — make it human-readable so the
    embedding model understands it the same way a person would.
    """
    location = contract.get("location", {})
    region = location.get("region", "Unknown region")
    province = location.get("province", "Unknown province")
 
    lines = [
        f"Contract ID: {contract.get('contractId', 'N/A')}",
        f"Description: {contract.get('description', 'N/A')}",
        f"Category: {contract.get('category', 'N/A')}",
        f"Status: {contract.get('status', 'N/A')}",
        f"Progress: {contract.get('progress', 0)}%",
        f"Budget: PHP {contract.get('budget', 0):,.2f}",
        f"Amount Paid: PHP {contract.get('amountPaid', 0):,.2f}",
        f"Contractor: {contract.get('contractor', 'N/A')}",
        f"Region: {region}",
        f"Province: {province}",
        f"Start Date: {contract.get('startDate', 'N/A')}",
        f"Completion Date: {contract.get('completionDate', 'N/A')}",
        f"Infrastructure Year: {contract.get('infraYear', 'N/A')}",
        f"Program: {contract.get('programName', 'N/A')}",
        f"Source of Funds: {contract.get('sourceOfFunds', 'N/A')}",
    ]
 
    return "\n".join(lines)
 
 
def contract_to_text_detailed(contract: dict) -> str:
    """
    For the richer per-contract detail JSON (from projects-data/).
    Includes bidders, procurement timeline, award amounts, and PDF links.
    """
    # Start with the base fields
    base = contract_to_text(contract)
    extras = []
 
    # Procurement details
    proc = contract.get("procurement", {})
    if proc:
        extras.append(f"Approved Budget for Contract (ABC): PHP {proc.get('abc', 'N/A')}")
        extras.append(f"Award Amount: PHP {proc.get('awardAmount', 'N/A')}")
        extras.append(f"Advertisement Date: {proc.get('advertisementDate', 'N/A')}")
        extras.append(f"Bid Submission Deadline: {proc.get('bidSubmissionDeadline', 'N/A')}")
        extras.append(f"Date of Award: {proc.get('dateOfAward', 'N/A')}")
        extras.append(f"Funding: {proc.get('fundingInstrument', 'N/A')}")
 
    # Bidders
    bidders = contract.get("bidders", [])
    if bidders:
        extras.append(f"Number of Bidders: {len(bidders)}")
        for b in bidders:
            winner_tag = " [WINNER]" if b.get("isWinner") else ""
            extras.append(f"  Bidder: {b.get('name', 'N/A')}{winner_tag}")
 
    # Document links
    links = contract.get("links", {})
    if links:
        for link_type, url in links.items():
            if url:
                extras.append(f"Document ({link_type}): {url}")
 
    return base + "\n" + "\n".join(extras)
 
 
def extract_metadata(contract: dict) -> dict:
    """
    Pull out the fields that ChromaDB stores as filterable metadata.
    These let you filter queries like: only Region VIII, only > 10M budget.
    ChromaDB metadata values must be str, int, or float — no dicts or lists.
    """
    location = contract.get("location", {})
    return {
        "contractId": str(contract.get("contractId", "")),
        "status": str(contract.get("status", "")),
        "region": str(location.get("region", "")),
        "province": str(location.get("province", "")),
        "category": str(contract.get("category", "")),
        "contractor": str(contract.get("contractor", ""))[:200],  # ChromaDB has field length limits
        "budget": float(contract.get("budget") or 0),
        "infraYear": str(contract.get("infraYear", "")),
        "programName": str(contract.get("programName", "")),
        "progress": int(contract.get("progress") or 0),
    }
 
def load_contracts_from_dump(json_path: str) -> list[dict]:
    """
    Load contracts from the bulk dump format:
    { "data": { "data": [ ...contracts... ] } }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
 
    contracts = raw.get("data", {}).get("data", [])
    print(f"Loaded {len(contracts)} contracts from {Path(json_path).name}")
    return contracts
 
def load_contracts_from_detail(json_path: str) -> list[dict]:
    """
    Load from a single per-contract detail file:
    { "data": { ...contract... } }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
 
    contract = raw.get("data", {})
    return [contract] if contract else []


def load_all_contracts(data_dir: str, detail: bool = False) -> list[dict]:
    """
    Walk a directory and load all JSON files.
    Set detail=True if the folder contains per-contract detail files.
    Set detail=False (default) for bulk dump files.
    """
    contracts = []
    path = Path(data_dir)
 
    for json_file in sorted(path.glob("*.json")):
        try:
            if detail:
                contracts.extend(load_contracts_from_detail(str(json_file)))
            else:
                contracts.extend(load_contracts_from_dump(str(json_file)))
        except Exception as e:
            print(f"  Warning: failed to load {json_file.name}: {e}")
 
    print(f"Total contracts loaded: {len(contracts)}")
    return contracts
 
 
def prepare_chunks(contracts: list[dict], detail: bool = False) -> list[dict]:
    """
    Convert contracts into chunk dicts ready for indexing.
    Each chunk dict has: text, metadata, id.
    """
    chunks = []
    seen_ids = set()
 
    for contract in contracts:
        contract_id = contract.get("contractId")
        if not contract_id or contract_id in seen_ids:
            continue
        seen_ids.add(contract_id)
 
        text = contract_to_text_detailed(contract) if detail else contract_to_text(contract)
        metadata = extract_metadata(contract)
 
        chunks.append({
            "id": contract_id,
            "text": text,
            "metadata": metadata,
        })
 
    print(f"Prepared {len(chunks)} chunks (deduplicated)")
    return chunks

 
