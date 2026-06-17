from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from features.chat.agent.query_planner import QueryPlan
from features.chat.tools.anomalies import (
    analyze_contractor_concentration,
    detect_bidding_anomalies,
    detect_budget_anomalies,
    detect_document_gaps,
    detect_timeline_anomalies,
    find_similar_scope_contracts,
)
from features.chat.tools.browse import search_contracts
from features.chat.tools.lookup import get_contract_detail, load_contract_detail_sources, _get_contract_detail_from_lookup_value, _summarize_sources
from features.chat.tools.proximity import find_nearby_contracts, _parse_proximity_query
from features.chat.tools.stats import filter_contracts, get_contract_statistics, _compute_stats_payload, _filter_contracts_from_filters, _format_stats_text, _get_contract_statistics_from_filters
from features.chat.tools.support import _build_contract_where_clause, _build_stats_scope, _legacy_search_query

web_search = DuckDuckGoSearchRun()


@tool
def ask_clarifying_question(query: str) -> str:
    """
    Use this tool when the user's contract request is broad or underspecified.
    It returns a short, user-friendly clarifying question instead of guessing.
    """

    normalized = " ".join(str(query or "").split()).lower()
    if (
        "same contractor" in normalized
        or "this contractor" in normalized
        or "that contractor" in normalized
        or "the contractor" in normalized
    ):
        return "Which contractor are you referring to?"
    if "detail" in normalized or "lookup" in normalized:
        return "Which contract or project should I look up?"
    if "how many" in normalized or "count" in normalized or "metric" in normalized or "statistics" in normalized:
        return "Which region, contractor, category, or status should I use?"
    return "Which region, contractor, category, or status should I narrow this to?"


def execute_lookup_plan(plan: QueryPlan) -> str:
    return _get_contract_detail_from_lookup_value(plan.lookup_value)


def execute_browse_plan(plan: QueryPlan) -> str:
    return _filter_contracts_from_filters(plan.filters, limit=plan.limit)


def execute_stats_plan(plan: QueryPlan) -> tuple[str, dict[str, object]]:
    payload = _compute_stats_payload(plan.filters, is_availability_query=False)
    return _format_stats_text(payload), payload


def execute_availability_plan(plan: QueryPlan) -> str:
    return _get_contract_statistics_from_filters(plan.filters, is_availability_query=True)


def execute_search_plan(plan: QueryPlan) -> str:
    return search_contracts(_legacy_search_query(plan))


def execute_clarify_plan(plan: QueryPlan) -> str:
    if plan.subject.strip():
        return plan.subject.strip()
    return ask_clarifying_question("clarify")


def execute_anomaly_plan(plan: QueryPlan) -> dict[str, object]:
    anomaly_kind = plan.analysis_type or "budget"
    if anomaly_kind in {"budget_outlier", "award_anomaly"}:
        anomaly_kind = "budget"
    elif anomaly_kind == "timeline_anomaly":
        anomaly_kind = "timeline"
    elif anomaly_kind == "bidding_anomaly":
        anomaly_kind = "bidding"
    elif anomaly_kind == "document_gap":
        anomaly_kind = "document_gaps"
    elif anomaly_kind == "scope_similarity":
        anomaly_kind = "similar_scope"
    if anomaly_kind == "budget":
        return detect_budget_anomalies(plan)
    if anomaly_kind == "timeline":
        return detect_timeline_anomalies(plan)
    if anomaly_kind == "contractor_concentration":
        return analyze_contractor_concentration(plan)
    if anomaly_kind == "similar_scope":
        return find_similar_scope_contracts(plan.lookup_value, plan)
    if anomaly_kind == "document_gaps":
        return detect_document_gaps(plan)
    if anomaly_kind == "bidding":
        return detect_bidding_anomalies(plan)
    return detect_budget_anomalies(plan)


tools = [
    search_contracts,
    filter_contracts,
    get_contract_detail,
    get_contract_statistics,
    find_nearby_contracts,
    ask_clarifying_question,
    web_search,
]
