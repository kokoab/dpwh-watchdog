import importlib.util
import json
import sys
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch


def _resolve_backend_module_path(filename: str) -> Path:
    candidates = [
        Path(f"backend/{filename}").resolve(),
        Path(filename).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@dataclass
class QueryPlan:
    intent: str
    filters: dict[str, str] = field(default_factory=dict)
    subject: str = ""
    lookup_value: str = ""
    limit: int | None = None
    exclude_selected_contract: bool = False
    has_location_phrase: bool = False
    has_unresolved_location_hint: bool = False
    is_follow_up: bool = False
    analysis_type: str = ""


def _load_tools_module():
    psycopg2_mod = types.ModuleType("psycopg2")
    psycopg2_mod.connect = lambda *args, **kwargs: None
    extras_mod = types.ModuleType("psycopg2.extras")
    extras_mod.DictCursor = object
    extras_mod.RealDictCursor = object
    extras_mod.Json = lambda value: value
    psycopg2_mod.extras = extras_mod

    embeddings_mod = types.ModuleType("rag.embeddings")
    embeddings_mod.LocalAPIEmbeddings = lambda: object()

    filter_parser_mod = types.ModuleType("rag.filter_parser")
    filter_parser_mod.FUZZY_FIELDS = {"region", "province", "category", "contractor", "program_name"}
    filter_parser_mod.parse_filter_string = lambda query: {}

    hybrid_search_mod = types.ModuleType("rag.hybrid_search")
    hybrid_search_mod.hybrid_search = lambda query, candidates: candidates
    hybrid_search_mod.structured_match_count = lambda query: 0
    hybrid_search_mod.structured_match_ids = lambda query: None

    langchain_tools_mod = types.ModuleType("langchain.tools")
    langchain_tools_mod.tool = lambda fn: fn

    langchain_chroma_mod = types.ModuleType("langchain_chroma")
    langchain_chroma_mod.Chroma = object

    langchain_community_tools_mod = types.ModuleType("langchain_community.tools")
    langchain_community_tools_mod.DuckDuckGoSearchRun = lambda: object()

    lookup_parser_mod = types.ModuleType("rag.lookup_parser")
    lookup_parser_mod.parse_lookup_string = lambda query: {"lookup_type": "id", "value": query}

    query_planner_mod = types.ModuleType("agent.query_planner")
    query_planner_mod.QueryPlan = QueryPlan

    query_scope_mod = types.ModuleType("agent.query_scope")
    query_scope_mod.get_current_thread_id = lambda: None
    query_scope_mod.get_thread_plan = lambda thread_id=None: {}
    query_scope_mod.get_thread_result = lambda thread_id=None: {}
    query_scope_mod.set_thread_result = lambda thread_id, payload: None

    reranker_mod = types.ModuleType("rag.reranker")
    reranker_mod.rerank = lambda query, candidates, top_k=10: candidates[:top_k]

    stats_parser_mod = types.ModuleType("rag.stats_parser")
    stats_parser_mod.parse_stats_filters = lambda filters: {
        "region": filters.get("region"),
        "province": filters.get("province"),
        "infra_year": filters.get("infra_year"),
        "infra_year_start": filters.get("infra_year_start"),
        "infra_year_end": filters.get("infra_year_end"),
        "status": filters.get("status"),
        "category_keyword": filters.get("category"),
        "contractor": filters.get("contractor"),
    }

    modules = {
        "psycopg2": psycopg2_mod,
        "psycopg2.extras": extras_mod,
        "rag.embeddings": embeddings_mod,
        "rag.filter_parser": filter_parser_mod,
        "rag.hybrid_search": hybrid_search_mod,
        "langchain.tools": langchain_tools_mod,
        "langchain_chroma": langchain_chroma_mod,
        "langchain_community.tools": langchain_community_tools_mod,
        "rag.lookup_parser": lookup_parser_mod,
        "agent.query_planner": query_planner_mod,
        "agent.query_scope": query_scope_mod,
        "rag.reranker": reranker_mod,
        "rag.stats_parser": stats_parser_mod,
    }
    old_modules = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        module_path = _resolve_backend_module_path("agent/tools.py")
        spec = importlib.util.spec_from_file_location("tools_plan_test_mod", module_path)
        assert spec and spec.loader
        tools_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tools_mod)
    finally:
        for name, old_value in old_modules.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value
    return tools_mod


class ToolsPlanExecutionTests(unittest.TestCase):
    def test_direct_plan_executors_use_structured_helpers(self) -> None:
        tools_mod = _load_tools_module()
        plan = QueryPlan(
            intent="browse",
            filters={"province": "Leyte", "category": "road"},
            lookup_value="21GF0024",
            limit=5,
        )

        with (
            patch.object(tools_mod, "_get_contract_detail_from_lookup_value", return_value="lookup") as lookup_mock,
            patch.object(tools_mod, "_filter_contracts_from_filters", return_value="browse") as browse_mock,
            patch.object(tools_mod, "_compute_stats_payload", return_value={"ok": True}) as stats_payload_mock,
            patch.object(tools_mod, "_format_stats_text", return_value="stats") as stats_format_mock,
            patch.object(tools_mod, "_get_contract_statistics_from_filters", return_value="availability") as availability_mock,
        ):
            self.assertEqual(tools_mod.execute_lookup_plan(plan), "lookup")
            self.assertEqual(tools_mod.execute_browse_plan(plan), "browse")
            self.assertEqual(tools_mod.execute_stats_plan(plan), ("stats", {"ok": True}))
            self.assertEqual(tools_mod.execute_availability_plan(plan), "availability")

        lookup_mock.assert_called_once_with("21GF0024")
        browse_mock.assert_called_once_with({"province": "Leyte", "category": "road"}, limit=5)
        stats_payload_mock.assert_called_once_with(
            {"province": "Leyte", "category": "road"},
            is_availability_query=False,
        )
        stats_format_mock.assert_called_once_with({"ok": True})
        availability_mock.assert_called_once_with(
            {"province": "Leyte", "category": "road"},
            is_availability_query=True,
        )

    def test_build_contract_where_clause_supports_year_ranges(self) -> None:
        tools_mod = _load_tools_module()

        where_clause, params = tools_mod._build_contract_where_clause(
            {
                "contractor": "TOPMOST DEVELOPMENT & MKTG. CORP.",
                "infra_year_start": "2022",
                "infra_year_end": "2026",
            }
        )

        self.assertIn("contractor ILIKE %s", where_clause)
        self.assertIn("infra_year >= %s", where_clause)
        self.assertIn("infra_year <= %s", where_clause)
        self.assertEqual(
            params,
            ["%TOPMOST DEVELOPMENT & MKTG. CORP.%", "2022", "2026"],
        )

    def test_build_stats_scope_renders_year_ranges(self) -> None:
        tools_mod = _load_tools_module()

        scope = tools_mod._build_stats_scope(
            None,
            None,
            None,
            "2022",
            "2026",
            "Awarded",
            None,
            "TOPMOST DEVELOPMENT & MKTG. CORP.",
        )

        self.assertIn("Years: 2022-2026", scope)
        self.assertIn("Status: Awarded", scope)
        self.assertIn("Contractor: TOPMOST DEVELOPMENT & MKTG. CORP.", scope)

    def test_summarize_sources_adds_completion_date_and_preserves_province_suffix(self) -> None:
        tools_mod = _load_tools_module()

        sources = tools_mod._summarize_sources(
            [
                {
                    "contract_id": "A001",
                    "description": "Flood control project",
                    "category": "Flood Control",
                    "status": "Completed",
                    "budget": 1500.0,
                    "amount_paid": None,
                    "progress": 100,
                    "region": "Region VIII",
                    "province": "Leyte 5th DEO",
                    "contractor": "Builder A",
                    "infra_year": "2025",
                    "program_name": "Regular Infra",
                    "completion_date": "2025-03-30",
                }
            ]
        )

        self.assertEqual(sources[0]["completionDate"], "2025-03-30")
        self.assertEqual(sources[0]["province"], "Leyte 5th DEO")

    def test_parse_proximity_query_stops_before_followup_sentence(self) -> None:
        tools_mod = _load_tools_module()

        parsed = tools_mod._parse_proximity_query(
            "Are there any other flood control projects within 10 km of the Miagao project? "
            "If so, compare their budgets and completion dates."
        )

        self.assertEqual(parsed, ("Miagao project", 10.0))

    def test_find_nearby_contracts_formats_results_and_records_state(self) -> None:
        tools_mod = _load_tools_module()
        captured_state = {}

        def capture_result_state(thread_id, payload):
            captured_state["thread_id"] = thread_id
            captured_state["payload"] = payload

        nearby_rows = [
            {
                "contract_id": "N001",
                "description": "Flood control structure near Miagao",
                "category": "flood control",
                "status": "Completed",
                "budget": 2000.0,
                "start_date": "2025-01-15",
                "completion_date": "2025-06-30",
                "region": "Region VI",
                "province": "Iloilo",
                "contractor": "Builder A",
                "latitude": 10.6,
                "longitude": 122.2,
                "distance_km": 3.25,
            }
        ]

        with (
            patch.object(
                tools_mod,
                "_resolve_reference_project",
                return_value={
                    "contract_id": "REF001",
                    "description": "Miagao flood control project",
                    "latitude": 10.65,
                    "longitude": 122.23,
                    "province": "Iloilo",
                    "region": "Region VI",
                },
            ) as resolve_mock,
            patch.object(tools_mod, "_haversine_search", return_value=nearby_rows) as search_mock,
            patch.object(tools_mod, "get_current_thread_id", return_value="thread-proximity"),
            patch.object(tools_mod, "set_thread_result", side_effect=capture_result_state),
        ):
            output = tools_mod.find_nearby_contracts(
                "Are there any other flood control projects within 10 km of the Miagao project? "
                "If so, compare their budgets and completion dates."
            )

        self.assertIn("Found 1 contract(s) within 10 km", output)
        self.assertIn("Budget: PHP 2,000.00", output)
        self.assertIn("Completion: 2025-06-30", output)
        self.assertIn("__SOURCES__", output)
        resolve_mock.assert_called_once_with("Miagao project", "flood control")
        search_mock.assert_called_once_with(
            10.65,
            122.23,
            10.0,
            exclude_contract_id="REF001",
            category_hint="flood control",
        )

        result_state = captured_state["payload"]
        self.assertEqual(captured_state["thread_id"], "thread-proximity")
        self.assertEqual(result_state["intent"], "proximity")
        self.assertEqual(result_state["displayed_contract_ids"], ["N001"])
        self.assertEqual(result_state["displayed_sources"][0]["contractId"], "N001")
        self.assertEqual(result_state["displayed_sources"][0]["completionDate"], "2025-06-30")

        sources_json = output.split("__SOURCES__", 1)[1]
        self.assertEqual(json.loads(sources_json)[0]["distanceKm"], 3.25)

    def test_find_nearby_contracts_uses_province_fallback_without_coordinates(self) -> None:
        tools_mod = _load_tools_module()

        with (
            patch.object(
                tools_mod,
                "_resolve_reference_project",
                return_value={
                    "contract_id": "REF002",
                    "description": "Miagao reference project",
                    "latitude": None,
                    "longitude": None,
                    "province": "Iloilo",
                    "region": "Region VI",
                },
            ),
            patch.object(tools_mod, "_province_level_nearby", return_value=[
                {
                    "contract_id": "P001",
                    "description": "Province fallback flood control",
                    "category": "flood control",
                    "status": "On-Going",
                    "budget": 3000.0,
                    "start_date": None,
                    "completion_date": None,
                    "region": "Region VI",
                    "province": "Iloilo",
                    "contractor": "Builder B",
                    "latitude": None,
                    "longitude": None,
                    "distance_km": None,
                }
            ]) as fallback_mock,
        ):
            output = tools_mod.find_nearby_contracts(
                "Find flood control contracts within 10 km of Miagao project"
            )

        self.assertIn("province-level results", output)
        fallback_mock.assert_called_once_with(
            "Iloilo",
            exclude_contract_id="REF002",
            category_hint="flood control",
        )

    def test_anomaly_executor_accepts_prompt_schema_analysis_types(self) -> None:
        tools_mod = _load_tools_module()

        with (
            patch.object(tools_mod, "detect_budget_anomalies", return_value={"ok": "budget"}) as budget_mock,
            patch.object(tools_mod, "detect_timeline_anomalies", return_value={"ok": "timeline"}) as timeline_mock,
            patch.object(tools_mod, "detect_bidding_anomalies", return_value={"ok": "bidding"}) as bidding_mock,
            patch.object(tools_mod, "detect_document_gaps", return_value={"ok": "documents"}) as documents_mock,
            patch.object(tools_mod, "find_similar_scope_contracts", return_value={"ok": "scope"}) as scope_mock,
        ):
            self.assertEqual(
                tools_mod.execute_anomaly_plan(QueryPlan(intent="anomaly", analysis_type="budget_outlier")),
                {"ok": "budget"},
            )
            self.assertEqual(
                tools_mod.execute_anomaly_plan(QueryPlan(intent="anomaly", analysis_type="award_anomaly")),
                {"ok": "budget"},
            )
            self.assertEqual(
                tools_mod.execute_anomaly_plan(QueryPlan(intent="anomaly", analysis_type="timeline_anomaly")),
                {"ok": "timeline"},
            )
            self.assertEqual(
                tools_mod.execute_anomaly_plan(QueryPlan(intent="anomaly", analysis_type="bidding_anomaly")),
                {"ok": "bidding"},
            )
            self.assertEqual(
                tools_mod.execute_anomaly_plan(QueryPlan(intent="anomaly", analysis_type="document_gap")),
                {"ok": "documents"},
            )
            self.assertEqual(
                tools_mod.execute_anomaly_plan(
                    QueryPlan(intent="anomaly", analysis_type="scope_similarity", lookup_value="21GF0024")
                ),
                {"ok": "scope"},
            )

        budget_mock.assert_called_with(unittest.mock.ANY)
        self.assertEqual(budget_mock.call_count, 2)
        timeline_mock.assert_called_once_with(unittest.mock.ANY)
        bidding_mock.assert_called_once_with(unittest.mock.ANY)
        documents_mock.assert_called_once_with(unittest.mock.ANY)
        scope_mock.assert_called_once_with("21GF0024", unittest.mock.ANY)


if __name__ == "__main__":
    unittest.main()
