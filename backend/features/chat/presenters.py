import json
import os
import re
import time
from datetime import date, datetime
from typing import Iterator

STRUCTURED_STREAM_WORDS_PER_CHUNK = max(
    1, int(os.environ.get("STRUCTURED_STREAM_WORDS_PER_CHUNK", "6"))
)
STRUCTURED_STREAM_DELAY_SECONDS = max(
    0.0, float(os.environ.get("STRUCTURED_STREAM_DELAY_SECONDS", "0.20"))
)

def _stream_token_text(content: str) -> Iterator[str]:
    lines = content.splitlines(keepends=True)
    for line in lines or [content]:
        yield f"data: {json.dumps({'type': 'token', 'content': line})}\n\n"

def _iter_structured_stream_chunks(
    content: str, words_per_chunk: int = STRUCTURED_STREAM_WORDS_PER_CHUNK
) -> Iterator[str]:
    for line in content.splitlines(keepends=True) or [content]:
        if line.strip().startswith("|") and line.count("|") >= 3:
            yield line
            continue

        tokens = re.findall(r"\S+|\s+", line)
        if not tokens:
            yield line
            continue

        chunk_parts: list[str] = []
        word_count = 0
        for token in tokens:
            chunk_parts.append(token)
            if not token.isspace():
                word_count += 1
            if word_count >= words_per_chunk:
                yield "".join(chunk_parts)
                chunk_parts = []
                word_count = 0

        if chunk_parts:
            yield "".join(chunk_parts)

def _stream_structured_token_text(
    content: str, delay_seconds: float = STRUCTURED_STREAM_DELAY_SECONDS
) -> Iterator[str]:
    for chunk in _iter_structured_stream_chunks(content):
        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
        if delay_seconds > 0:
            time.sleep(delay_seconds)

def should_append_next_step(intent: str | None, assistant_text: str) -> bool:
    if intent in (None, "", "chat", "clarify"):
        return False
    if not assistant_text.strip():
        return False
    recent_text = assistant_text[-500:].lower()
    if "would you like" in recent_text:
        return False
    return "?" not in recent_text

def _format_budget(value: object) -> str:
    if value in (None, ""):
        return "N/A"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"PHP {amount:,.0f}"

def _compact_budget(value: object) -> str:
    if value in (None, ""):
        return "N/A"
    try:
        amount = float(
            str(value).replace(",", "").replace("PHP", "").replace("₱", "").strip()
        )
    except (TypeError, ValueError):
        return "N/A"

    if amount < 1_000:
        return f"₱{amount:.0f}"
    if amount < 1_000_000:
        return f"₱{amount / 1_000:.0f}K"
    if amount < 1_000_000_000:
        millions = f"{amount / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"₱{millions}M"
    return f"₱{amount / 1_000_000_000:.2f}B"

def _format_money(value: object) -> str:
    if value in (None, ""):
        return "N/A"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"PHP {amount:,.2f}"

def _format_percent(value: object) -> str:
    if value in (None, ""):
        return "N/A"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{amount:.1f}%"

def _format_value(value: object) -> str:
    if value in (None, ""):
        return "N/A"
    return str(value).strip()

def _coerce_numeric(value: object) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").replace("PHP", "").strip())
    except (TypeError, ValueError):
        return 0.0

def _truncate_table_text(value: object, limit: int = 45) -> str:
    text = " ".join(str(value or "N/A").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."

def _markdown_cell(value: object) -> str:
    return str(value if value not in (None, "") else "N/A").replace("|", "\\|")

def _source_raw_value(source: dict[str, object], key: str) -> object:
    value = source.get(key)
    if value not in (None, ""):
        return value
    db_fields = source.get("dbFields")
    if isinstance(db_fields, dict):
        return db_fields.get(key)
    return value

def _source_id(source: dict[str, object]) -> str:
    return _format_value(_source_raw_value(source, "contractId"))

def _source_progress_text(source: dict[str, object]) -> str:
    value = _source_raw_value(source, "progress")
    if value in (None, "", "N/A"):
        return "N/A"

    text = str(value).strip()
    if text.upper() == "N/A":
        return "N/A"
    if text.endswith("%"):
        return text
    try:
        amount = float(text)
    except (TypeError, ValueError):
        return text
    if amount.is_integer():
        return f"{int(amount)}%"
    return f"{amount:.1f}%"

def _source_budget_text(source: dict[str, object]) -> str:
    value = _source_raw_value(source, "budget")
    if value in (None, "", "N/A"):
        return "N/A"
    return _compact_budget(value)

def _contract_scope_text(result_state: dict[str, object]) -> str:
    filters = result_state.get("filters")
    if not isinstance(filters, dict):
        return "matching contracts"

    category = str(filters.get("category") or "").strip()
    province = str(filters.get("province") or "").strip()
    region = str(filters.get("region") or "").strip()
    status = str(filters.get("status") or "").strip()

    subject = f"{category} contracts" if category else "contracts"
    qualifiers: list[str] = []
    if province:
        qualifiers.append(f"in {province}")
    elif region:
        qualifiers.append(f"in {region}")
    if status:
        qualifiers.append(f"with status {status}")

    if qualifiers:
        return f"matching {subject} {' '.join(qualifiers)}"
    return f"matching {subject}"

def _contract_table_string(valid_sources: list[dict[str, object]]) -> str:
    lines = [
        "|Contract ID|Description|Budget|Status|Completion Date|Progress|Office/Province|",
        "|---|---|---:|---|---|---:|---|",
    ]
    for source in valid_sources:
        office_or_province = (
            _source_raw_value(source, "province")
            or _source_raw_value(source, "region")
            or "N/A"
        )
        row = [
            _source_id(source),
            _truncate_table_text(_source_raw_value(source, "description"), 56),
            _source_budget_text(source),
            _format_value(_source_raw_value(source, "status")),
            _format_value(_source_raw_value(source, "completionDate")),
            _source_progress_text(source),
            _format_value(office_or_province),
        ]
        lines.append("|" + "|".join(_markdown_cell(value) for value in row) + "|")
    return "\n".join(lines)

def _parse_date_value(value: object) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        return datetime.fromisoformat(text[:10]).date()
    except (TypeError, ValueError):
        return None

def _computed_duration_text(source: dict[str, object]) -> str:
    duration = _source_raw_value(source, "contractDuration")
    if duration not in (None, ""):
        return _format_value(duration)

    start = _parse_date_value(_source_raw_value(source, "startDate"))
    completion = _parse_date_value(_source_raw_value(source, "completionDate"))
    if start and completion:
        return f"{(completion - start).days} days"
    return "N/A"

def _comparison_table_string(detail_sources: list[dict[str, object]]) -> str:
    lines = [
        "|Contract ID|Description|Budget|Status|Completion Date|Duration|Region|",
        "|---|---|---:|---|---|---|---|",
    ]
    for source in detail_sources:
        row = [
            _source_id(source),
            _truncate_table_text(_source_raw_value(source, "description"), 28),
            _compact_budget(_source_raw_value(source, "budget")),
            _format_value(_source_raw_value(source, "status")),
            _format_value(_source_raw_value(source, "completionDate")),
            _computed_duration_text(source),
            _format_value(_source_raw_value(source, "region")),
        ]
        lines.append("|" + "|".join(_markdown_cell(value) for value in row) + "|")
    return "\n".join(lines)

def _number_text(value: object) -> str:
    if value in (None, ""):
        return "N/A"
    return str(value)

def _percentage_difference_text(value: object) -> str:
    if value in (None, ""):
        return "N/A difference"
    return f"{value}% difference"

def _format_whole_php(value: object) -> str:
    try:
        return f"PHP {float(value):,.0f}"
    except (TypeError, ValueError):
        return "PHP 0"

def _completion_rankings(
    detail_sources: list[dict[str, object]],
) -> tuple[tuple[str, str], tuple[str, str]]:
    completions: list[tuple[date, str, str]] = []
    for source in detail_sources:
        parsed = _parse_date_value(_source_raw_value(source, "completionDate"))
        if parsed is None:
            continue
        completions.append((parsed, _source_id(source), parsed.isoformat()))

    if not completions:
        fallback_id = _source_id(detail_sources[0]) if detail_sources else "N/A"
        return (fallback_id, "N/A"), (fallback_id, "N/A")

    completions.sort(key=lambda item: item[0])
    earliest = completions[0]
    latest = completions[-1]
    return (earliest[1], earliest[2]), (latest[1], latest[2])

def _comparison_rankings_string(
    comparison_analytics: dict[str, object],
    detail_sources: list[dict[str, object]],
) -> str:
    lines: list[str] = []
    budget_rankings = comparison_analytics.get("rankings_by_budget")
    if isinstance(budget_rankings, list) and budget_rankings:
        largest_budget = budget_rankings[0]
        smallest_budget = budget_rankings[-1]
        if isinstance(largest_budget, dict):
            lines.append(
                "- Largest budget: "
                f"{_format_value(largest_budget.get('id'))} — "
                f"{_compact_budget(largest_budget.get('budget'))}"
            )
        if isinstance(smallest_budget, dict):
            lines.append(
                "- Smallest budget: "
                f"{_format_value(smallest_budget.get('id'))} — "
                f"{_compact_budget(smallest_budget.get('budget'))}"
            )

    duration_rankings = comparison_analytics.get("rankings_by_duration_days")
    if isinstance(duration_rankings, list):
        duration_values = [
            item
            for item in duration_rankings
            if isinstance(item, dict) and item.get("duration_days") is not None
        ]
        if duration_values:
            longest_duration = duration_values[0]
            shortest_duration = duration_values[-1]
            lines.append(
                "- Longest duration: "
                f"{_format_value(longest_duration.get('id'))} — "
                f"{longest_duration.get('duration_days')} days"
            )
            lines.append(
                "- Shortest duration: "
                f"{_format_value(shortest_duration.get('id'))} — "
                f"{shortest_duration.get('duration_days')} days"
            )

    earliest, latest = _completion_rankings(detail_sources)
    lines.append(f"- Earliest completion: {earliest[0]} — {earliest[1]}")
    lines.append(f"- Latest completion: {latest[0]} — {latest[1]}")
    return "\n".join(lines)

def _comparison_diffs_string(
    comparison_analytics: dict[str, object], detail_sources: list[dict[str, object]]
) -> str:
    if len(detail_sources) != 2:
        return ""

    diffs = comparison_analytics.get("two_entity_diffs")
    if not isinstance(diffs, dict) or not diffs:
        return ""

    lines = [
        "- Budget gap: "
        f"{_format_whole_php(diffs.get('budget_abs_diff'))} "
        f"({_percentage_difference_text(diffs.get('budget_pct_diff'))})"
    ]
    if diffs.get("duration_diff_days") is not None:
        lines.append(f"- Duration gap: {diffs.get('duration_diff_days')} days")
    lines.append(
        "- Progress gap: "
        f"{_number_text(diffs.get('progress_diff_pct'))} percentage points"
    )
    return "\n".join(lines)

def _strip_generated_comparison_sections(text: str) -> str:
    skipped_prefixes = (
        "- largest budget:",
        "- smallest budget:",
        "- longest duration:",
        "- shortest duration:",
        "- earliest completion:",
        "- latest completion:",
        "- budget gap:",
        "- duration gap:",
        "- progress gap:",
    )
    cleaned_lines: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        normalized = stripped.lower().strip("*: ")
        if stripped.startswith("|") and stripped.count("|") >= 3:
            continue
        if normalized in {
            "comparison table",
            "executive summary",
            "rankings",
            "differences",
            "key differences",
            "insights",
            "narrative",
        }:
            continue
        if stripped.lower().startswith(skipped_prefixes):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

def _build_link_summary(document_links: object) -> str:
    if not isinstance(document_links, dict) or not document_links:
        return "N/A"
    names = [name for name, url in document_links.items() if str(url or "").strip()]
    return ", ".join(names) if names else "N/A"

def _looks_like_tool_call_json(text: str) -> bool:
    stripped = str(text or "").strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return False

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return False

    return (
        isinstance(payload, dict)
        and "name" in payload
        and "parameters" in payload
        and set(payload.keys()).issubset({"name", "parameters"})
    )

def _strip_tool_call_json_text(text: str) -> str:
    if _looks_like_tool_call_json(text):
        return ""

    lines = str(text or "").splitlines(keepends=True)
    if not lines:
        return ""

    cleaned_lines = [line for line in lines if not _looks_like_tool_call_json(line)]
    return "".join(cleaned_lines)

def _build_structured_contract_detail_reply(result_state: dict[str, object]) -> str:
    displayed_sources = result_state.get("displayed_sources")
    if not isinstance(displayed_sources, list) or not displayed_sources:
        return ""

    source = displayed_sources[0]
    if not isinstance(source, dict):
        return ""

    db_fields = (
        source.get("dbFields") if isinstance(source.get("dbFields"), dict) else {}
    )
    components = (
        source.get("components") if isinstance(source.get("components"), list) else []
    )
    document_links = (
        source.get("documentLinks")
        if isinstance(source.get("documentLinks"), dict)
        else {}
    )

    description = _format_value(
        source.get("description") or db_fields.get("description")
    )
    contract_id = _format_value(source.get("contractId") or db_fields.get("contractId"))
    contractor = _format_value(source.get("contractor") or db_fields.get("contractor"))
    category = _format_value(source.get("category") or db_fields.get("category"))
    status = _format_value(source.get("status") or db_fields.get("status"))
    region = _format_value(source.get("region") or db_fields.get("region"))
    province = _format_value(source.get("province") or db_fields.get("province"))
    budget = _format_money(source.get("budget") or db_fields.get("budget"))
    amount_paid = _format_money(source.get("amountPaid") or db_fields.get("amountPaid"))
    award_amount = _format_money(
        source.get("awardAmount") or db_fields.get("awardAmount")
    )
    award_ratio = _format_percent(
        source.get("awardToBudgetRatio") or db_fields.get("awardToBudgetRatio")
    )
    progress = _format_percent(source.get("progress") or db_fields.get("progress"))
    infra_year = _format_value(source.get("infraYear") or db_fields.get("infraYear"))
    program_name = _format_value(
        source.get("programName") or db_fields.get("programName")
    )
    source_of_funds = _format_value(
        source.get("sourceOfFunds") or db_fields.get("sourceOfFunds")
    )
    advertisement_date = _format_value(
        source.get("advertisementDate") or db_fields.get("advertisementDate")
    )
    bid_deadline = _format_value(
        source.get("bidSubmissionDeadline") or db_fields.get("bidSubmissionDeadline")
    )
    start_date = _format_value(source.get("startDate") or db_fields.get("startDate"))
    completion_date = _format_value(
        source.get("completionDate") or db_fields.get("completionDate")
    )
    expiry_date = _format_value(source.get("expiryDate") or db_fields.get("expiryDate"))
    contract_duration = _format_value(
        source.get("contractDuration") or db_fields.get("contractDuration")
    )
    link_summary = _build_link_summary(document_links)

    lines = [
        f"{description} ({contract_id})",
        f"• Contractor: {contractor}",
        f"• Category: {category}",
        f"• Status: {status}",
        f"• Budget: {budget}",
        f"• Award Amount: {award_amount}",
        f"• Award-to-Budget Ratio: {award_ratio}",
        f"• Amount Paid: {amount_paid}",
        f"• Progress: {progress}",
        f"• Region: {region}",
        f"• Province: {province}",
        f"• Program: {program_name}",
        f"• Source of Funds: {source_of_funds}",
        f"• Infra Year: {infra_year}",
        f"• Advertisement Date: {advertisement_date}",
        f"• Bid Submission Deadline: {bid_deadline}",
        f"• Start Date: {start_date}",
        f"• Completion Date: {completion_date}",
        f"• Expiry Date: {expiry_date}",
        f"• Contract Duration: {contract_duration}",
        f"• Document Links: {link_summary}",
    ]

    if components:
        lines.append("• Components:")
        for index, component in enumerate(components, start=1):
            if not isinstance(component, dict):
                continue
            component_id = _format_value(component.get("componentId"))
            component_desc = _format_value(component.get("description"))
            component_type = _format_value(component.get("typeOfWork"))
            infra_type = _format_value(component.get("infraType"))
            location = ", ".join(
                part
                for part in (
                    _format_value(component.get("region")),
                    _format_value(component.get("province")),
                )
                if part != "N/A"
            )
            location_text = location if location else "N/A"
            lines.append(
                f"  {index}. {component_id} | {component_type} | {infra_type} | {component_desc} | {location_text}"
            )

    lines.append(NEXT_STEP_QUESTION.strip())
    return "\n".join(lines).strip()

def _build_structured_contract_reply(result_state: dict[str, object]) -> str:
    displayed_sources = result_state.get("displayed_sources")
    if not isinstance(displayed_sources, list) or not displayed_sources:
        return ""

    valid_sources = [s for s in displayed_sources if isinstance(s, dict)]
    if not valid_sources:
        return ""

    try:
        total_available = int(result_state.get("count") or len(valid_sources))
    except (TypeError, ValueError):
        total_available = len(valid_sources)

    highest_source = max(
        valid_sources,
        key=lambda s: _coerce_numeric(_source_raw_value(s, "budget")),
    )
    highest_id = _source_id(highest_source)
    highest_budget = _source_budget_text(highest_source)
    scope = _contract_scope_text(result_state)
    displayed_count = len(valid_sources)

    if displayed_count == total_available:
        summary_count = f"Found {total_available:,}"
        insight_count = f"Showing all {displayed_count:,}"
    else:
        summary_count = f"Found {total_available:,}; showing {displayed_count:,}"
        insight_count = f"Showing {displayed_count:,} of {total_available:,}"

    return "\n\n".join(
        [
            (
                f"**Executive summary:** {summary_count} {scope}. "
                "The table lists the displayed contracts with budgets, status, "
                "completion dates, progress, and office/province."
            ),
            _contract_table_string(valid_sources),
            (
                f"**Insight:** {insight_count} matching contracts. "
                f"Highest listed budget: {highest_id} at {highest_budget}."
            ),
        ]
    ).strip()

def _build_structured_contract_reply_with_dates(result_state: dict[str, object]) -> str:
    return _build_structured_contract_reply(result_state)

