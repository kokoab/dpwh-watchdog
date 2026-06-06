import json
from pathlib import Path
from langchain_core.documents import Document

# path_file = "./data/dump-page-1-100.json"
DATA_DIR = "./data"


def _summarize_components(components: list[dict], limit: int = 8) -> str:
    valid_components = [c for c in components if isinstance(c, dict)]
    if not valid_components:
        return "No component records"

    lines = []
    for idx, component in enumerate(valid_components[:limit], start=1):
        component_id = component.get("componentId") or f"component-{idx}"
        description = component.get("description") or "N/A"
        type_of_work = component.get("typeOfWork") or component.get("infraType") or "N/A"
        region = component.get("region") or "N/A"
        province = component.get("province") or "N/A"
        lines.append(
            f"{idx}. {component_id} | {type_of_work} | {description} | {region} | {province}"
        )

    remaining = len(valid_components) - min(len(valid_components), limit)
    if remaining > 0:
        lines.append(f"... and {remaining} more component(s)")

    return "\n".join(lines)


def load_document(file_path: Path) -> list[Document]:
    with open(file_path, "r", encoding="utf-8") as file:
        raw_json = json.load(file)

    inner = raw_json.get("data", {})

    # dump files
    if isinstance(inner.get("data"), list):
        contract_dump = inner["data"]
    # detail files
    elif inner.get("contractId"):
        contract_dump = [inner]
    else:
        contract_dump = []
    
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
    proc = contract.get("procurement", {}) or {}
    bidders = contract.get("bidders", []) or []
    components = contract.get("components", []) or []
    component_summary = _summarize_components(components)

    lines = [
        "passage:",
        f"Contract ID: {contract_id}",
        f"Description: {contract.get('description', 'N/A')}",
        f"Category: {contract.get('category', 'N/A')}",
        f"Status: {contract.get('status', 'N/A')}",
        f"Progress: {contract.get('progress', 0)}%",
        f"Budget: {contract.get('budget', 0)}",
        f"Award Amount: {proc.get('awardAmount', 'N/A')}",
        f"Contractor: {contract.get('contractor', 'N/A')}",
        f"Region: {location.get('region', 'N/A')}",
        f"Province: {location.get('province', 'N/A')}",
        f"Start Date: {contract.get('startDate', 'N/A')}",
        f"Completion Date: {contract.get('completionDate', 'N/A')}",
        f"Infrastructure Year: {contract.get('infraYear', 'N/A')}",
        f"Program: {contract.get('programName', 'N/A')}",
        f"Source of Funds: {contract.get('sourceOfFunds', 'N/A')}",
        "Component Summary:",
        component_summary,
    ]

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

    metadata = {
        "contractId": str(contract_id),
        "status": str(contract.get("status") or "Unknown"),
        "region": str(location.get("region") or "Unknown"),
        "province": str(location.get("province") or "Unknown"),
        "category": str(contract.get("category") or "Unknown"),
        "contractor": str(contract.get("contractor") or "Unknown")[:200],
        "budget": float(contract.get("budget") or 0.0),
        "progress": int(contract.get("progress") or 0),
        "infraYear": str(contract.get("infraYear") or "Unknown"),
        "programName": str(contract.get("programName") or "Unknown"),
        "hasDetail": bool(proc or bidders or components),
        "source": str(file_path),
    }

    return Document(page_content=page_content, metadata=metadata)
