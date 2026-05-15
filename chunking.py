import json 
import os
from pathlib import Path

def contract_to_text(contract: dict) -> str:
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
    base = contract_to_text(contract)
    extras = []
 
    proc = contract.get("procurement", {})
    if proc:
        extras.append(f"Approved Budget for Contract (ABC): PHP {proc.get('abc', 'N/A')}")
        extras.append(f"Award Amount: PHP {proc.get('awardAmount', 'N/A')}")
        extras.append(f"Advertisement Date: {proc.get('advertisementDate', 'N/A')}")
        extras.append(f"Bid Submission Deadline: {proc.get('bidSubmissionDeadline', 'N/A')}")
        extras.append(f"Date of Award: {proc.get('dateOfAward', 'N/A')}")
        extras.append(f"Funding: {proc.get('fundingInstrument', 'N/A')}")
 
    bidders = contract.get("bidders", [])
    if bidders:
        extras.append(f"Number of Bidders: {len(bidders)}")
        for b in bidders:
            winner_tag = " [WINNER]" if b.get("isWinner") else ""
            extras.append(f"  Bidder: {b.get('name', 'N/A')}{winner_tag}")
 
    links = contract.get("links", {})
    if links:
        for link_type, url in links.items():
            if url:
                extras.append(f"Document ({link_type}): {url}")
 
    return base + "\n" + "\n".join(extras)
 
 
def extract_metadata(contract: dict) -> dict:
    location = contract.get("location", {})
    return {
        "contractId": str(contract.get("contractId", "")),
        "status": str(contract.get("status", "")),
        "region": str(location.get("region", "")),
        "province": str(location.get("province", "")),
        "category": str(contract.get("category", "")),
        "contractor": str(contract.get("contractor", ""))[:200],
        "budget": float(contract.get("budget") or 0),
        "infraYear": str(contract.get("infraYear", "")),
        "programName": str(contract.get("programName", "")),
        "progress": int(contract.get("progress") or 0),
    }


def detect_and_load(json_path: str) -> list[dict]:
    """
    Auto-detect whether a JSON file is a bulk dump or a single contract detail.

    Bulk dump:   { "data": { "data": [...contracts...], "pagination": {...} } }
    Detail file: { "status": 200, "data": { "contractId": "...", ... } }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    inner = raw.get("data", {})

    # Bulk dump: data.data is a list of contracts
    if isinstance(inner, dict) and isinstance(inner.get("data"), list):
        return inner["data"]

    # Single detail file: data is a dict with a contractId key
    if isinstance(inner, dict) and inner.get("contractId"):
        return [inner]

    # Unrecognised — skip silently
    return []


def load_contracts_from_dump(json_path: str) -> list[dict]:
    contracts = detect_and_load(json_path)
    print(f"Loaded {len(contracts)} contracts from {Path(json_path).name}")
    return contracts


def load_contracts_from_detail(json_path: str) -> list[dict]:
    return detect_and_load(json_path)


def load_all_contracts(data_dir: str, detail: bool = False) -> list[dict]:
    """
    Walk a directory and auto-detect each file's format.
    Mixes dump files and detail files safely in the same folder.
    """
    contracts = []
    dump_files = 0
    detail_files = 0
    path = Path(data_dir)

    for json_file in sorted(path.glob("*.json")):
        try:
            loaded = detect_and_load(str(json_file))
            if not loaded:
                continue
            if len(loaded) > 1:
                dump_files += 1
            else:
                detail_files += 1
            contracts.extend(loaded)
        except Exception as e:
            print(f"  Warning: failed to load {json_file.name}: {e}")

    print(f"Total contracts loaded: {len(contracts)} "
          f"({dump_files} dump files + {detail_files} detail files)")
    return contracts
 
 
def prepare_chunks(contracts: list[dict], detail: bool = False) -> list[dict]:
    chunks = []
    seen_ids = set()
 
    for contract in contracts:
        contract_id = contract.get("contractId")
        if not contract_id or contract_id in seen_ids:
            continue
        seen_ids.add(contract_id)

        # Auto-detect richness per contract regardless of flag
        is_detailed = bool(contract.get("bidders") or contract.get("procurement"))
        text = contract_to_text_detailed(contract) if (detail or is_detailed) else contract_to_text(contract)
        metadata = extract_metadata(contract)
 
        chunks.append({
            "id": contract_id,
            "text": text,
            "metadata": metadata,
        })
 
    print(f"Prepared {len(chunks)} chunks (deduplicated)")
    return chunks