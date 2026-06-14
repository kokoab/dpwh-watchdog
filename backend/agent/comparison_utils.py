from __future__ import annotations

from collections import Counter
from datetime import date, datetime


def _coerce_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        cleaned = (
            str(value)
            .replace(",", "")
            .replace("PHP", "")
            .replace("%", "")
            .strip()
        )
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def _safe_number(value: object) -> float:
    number = _coerce_float(value)
    return number if number is not None else 0.0


def _coerce_date(value: object) -> date | None:
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


def _duration_days(source: dict) -> int | None:
    try:
        start = _coerce_date(source.get("startDate"))
        completion = _coerce_date(source.get("completionDate"))
        if not start or not completion:
            return None
        return (completion - start).days
    except Exception:
        return None


def _description_snippet(value: object, limit: int = 45) -> str:
    text = " ".join(str(value or "N/A").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _source_id(source: dict) -> str:
    return str(source.get("contractId") or "N/A").strip() or "N/A"


def _unique_values(sources: list[dict], key: str) -> list[str]:
    values: list[str] = []
    seen = set()
    for source in sources:
        value = str(source.get(key) or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _percentage_difference(first: float, second: float) -> float | None:
    lower = min(first, second)
    if lower <= 0:
        return None
    return round(abs(first - second) / lower * 100, 1)


def _compute_two_entity_diffs(sources: list[dict]) -> dict[str, object]:
    first, second = sources
    first_budget = _safe_number(first.get("budget"))
    second_budget = _safe_number(second.get("budget"))
    first_duration = _duration_days(first)
    second_duration = _duration_days(second)
    first_progress = _safe_number(first.get("progress"))
    second_progress = _safe_number(second.get("progress"))

    return {
        "budget_abs_diff": abs(first_budget - second_budget),
        "budget_pct_diff": _percentage_difference(first_budget, second_budget),
        "duration_diff_days": (
            abs(first_duration - second_duration)
            if first_duration is not None and second_duration is not None
            else None
        ),
        "progress_diff_pct": round(abs(first_progress - second_progress), 1),
    }


def _compute_outlier_flags(sources: list[dict], budgets: list[float]) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    today = date.today()
    mean_budget = sum(budgets) / len(budgets) if budgets else 0.0

    for source, budget in zip(sources, budgets, strict=False):
        contract_id = _source_id(source)
        progress = _coerce_float(source.get("progress"))
        completion = _coerce_date(source.get("completionDate"))

        try:
            if (
                progress is not None
                and progress == 0
                and completion is not None
                and completion < today
            ):
                flags.append(
                    {
                        "id": contract_id,
                        "flag_type": "zero_progress_overdue",
                        "detail": "Progress is 0 and completion date is in the past.",
                    }
                )
        except Exception:
            pass

        if mean_budget > 0 and budget > mean_budget * 1.5:
            flags.append(
                {
                    "id": contract_id,
                    "flag_type": "budget_above_1_5x_mean",
                    "detail": (
                        f"Budget PHP {budget:,.2f} is above 1.5x "
                        f"the set mean of PHP {mean_budget:,.2f}."
                    ),
                }
            )

    return flags


def _compute_repeated_contractors(sources: list[dict]) -> list[str]:
    normalized_to_original: dict[str, str] = {}
    normalized_names: list[str] = []

    for source in sources:
        contractor = str(source.get("contractor") or "").strip()
        if not contractor:
            continue
        normalized = " ".join(contractor.lower().split())
        normalized_to_original.setdefault(normalized, contractor)
        normalized_names.append(normalized)

    counts = Counter(normalized_names)
    return [
        normalized_to_original[name]
        for name, count in counts.items()
        if count > 1
    ]


def compute_comparison_analytics(sources: list[dict]) -> dict:
    safe_sources = [source for source in sources if isinstance(source, dict)]
    budgets = [_safe_number(source.get("budget")) for source in safe_sources]
    total_combined_budget = sum(budgets)
    largest_budget = max(budgets) if budgets else 0.0
    durations = [_duration_days(source) for source in safe_sources]
    regions = _unique_values(safe_sources, "region")
    provinces = _unique_values(safe_sources, "province")
    has_all_regions = all(
        str(source.get("region") or "").strip() for source in safe_sources
    )
    has_all_provinces = all(
        str(source.get("province") or "").strip() for source in safe_sources
    )

    rankings_by_budget = sorted(
        [
            {
                "id": _source_id(source),
                "description_snippet": _description_snippet(source.get("description")),
                "budget": budget,
            }
            for source, budget in zip(safe_sources, budgets, strict=False)
        ],
        key=lambda item: item["budget"],
        reverse=True,
    )
    rankings_by_duration_days = sorted(
        [
            {
                "id": _source_id(source),
                "description_snippet": _description_snippet(source.get("description")),
                "duration_days": duration,
            }
            for source, duration in zip(safe_sources, durations, strict=False)
        ],
        key=lambda item: (
            item["duration_days"] is not None,
            item["duration_days"] if item["duration_days"] is not None else -1,
        ),
        reverse=True,
    )
    rankings_by_progress = sorted(
        [
            {
                "id": _source_id(source),
                "progress": _safe_number(source.get("progress")),
            }
            for source in safe_sources
        ],
        key=lambda item: item["progress"],
        reverse=True,
    )

    return {
        "rankings_by_budget": rankings_by_budget,
        "rankings_by_duration_days": rankings_by_duration_days,
        "rankings_by_progress": rankings_by_progress,
        "total_combined_budget": total_combined_budget,
        "budget_concentration_pct": (
            round(largest_budget / total_combined_budget * 100, 1)
            if total_combined_budget > 0
            else 0.0
        ),
        "two_entity_diffs": (
            _compute_two_entity_diffs(safe_sources)
            if len(safe_sources) == 2
            else {}
        ),
        "outlier_flags": _compute_outlier_flags(safe_sources, budgets),
        "geographic_cluster": {
            "all_same_region": (
                bool(safe_sources) and has_all_regions and len(regions) == 1
            ),
            "all_same_province": (
                bool(safe_sources) and has_all_provinces and len(provinces) == 1
            ),
            "regions": regions,
        },
        "repeated_contractors": _compute_repeated_contractors(safe_sources),
    }
