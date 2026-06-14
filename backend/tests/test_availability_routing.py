import unittest
import types
from unittest.mock import patch

from rag.query_expand import _detect_intent, query_expand
from agent.query_scope import clear_thread_scope, get_thread_plan, set_thread_result

CATALOG = types.SimpleNamespace(
    regions=(
        "Region VI",
        "Region VIII",
        "Region XI",
        "National Capital Region",
        "Cordillera Administrative Region",
        "Negros Island Region",
    ),
    provinces=("Iloilo", "Leyte", "Davao City DEO"),
    statuses=("On-Going", "Completed"),
    region_map={
        "region vi": "Region VI",
        "region viii": "Region VIII",
        "region xi": "Region XI",
        "national capital region": "National Capital Region",
        "ncr": "National Capital Region",
        "car": "Cordillera Administrative Region",
        "negros island region": "Negros Island Region",
    },
    province_map={
        "iloilo": "Iloilo",
        "leyte": "Leyte",
        "davao city deo": "Davao City DEO",
    },
    status_map={"ongoing": "On-Going", "on-going": "On-Going", "completed": "Completed"},
)


class DeterministicRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._patchers = [
            patch(
                "langchain_groq.ChatGroq",
                side_effect=RuntimeError("force fallback planner"),
            ),
            patch("agent.query_planner.get_entity_catalog", return_value=CATALOG),
            patch("agent.query_scope.upsert_thread_state"),
            patch("agent.query_scope.get_thread_state", return_value={}),
            patch("agent.query_scope.delete_thread_memory"),
        ]
        for patcher in self._patchers:
            patcher.start()

    def tearDown(self) -> None:
        for thread_id in (
            "availability-region-6",
            "browse-negros",
            "stats-car",
            "search-leyte",
            "browse-iloilo",
            "follow-up-region",
            "follow-up-location-leak",
            "browse-davao-city-deo",
            "browse-ncr",
            "result-reference-seven",
            "result-reference-show-them",
            "result-reference-region-switch",
            "result-reference-first-one",
            "same-contractor-detail",
            "same-contractor-plain-reference",
            "compare-visible-three",
            "compare-named-projects",
            "compare-missing-context",
            "clarify-broad-query",
            "clarify-same-contractor",
        ):
            clear_thread_scope(thread_id)
        for patcher in reversed(self._patchers):
            patcher.stop()

    def test_any_ongoing_road_projects_routes_to_availability(self) -> None:
        expanded = query_expand(
            "any ongoing road projects in region xi?",
            thread_id="availability-region-6",
        )
        self.assertEqual(
            expanded,
            "Check availability where region=Region XI AND status=On-Going AND category=road",
        )
        self.assertEqual(_detect_intent(expanded), "availability")

    def test_list_flood_control_projects_in_iloilo_routes_to_browse(self) -> None:
        expanded = query_expand(
            "list flood control projects in iloilo",
            thread_id="browse-iloilo",
        )
        self.assertEqual(
            expanded,
            "Filter contracts where province=Iloilo AND category=flood control",
        )
        self.assertEqual(_detect_intent(expanded), "browse")

    def test_how_many_completed_bridges_in_car_routes_to_stats(self) -> None:
        expanded = query_expand(
            "how many completed bridges in CAR",
            thread_id="stats-car",
        )
        self.assertEqual(
            expanded,
            "Calculate metrics where region=Cordillera Administrative Region AND status=Completed AND category=bridge",
        )
        self.assertEqual(_detect_intent(expanded), "stats")

    def test_school_buildings_in_davao_city_deo_routes_to_availability(self) -> None:
        expanded = query_expand(
            "are there school buildings in davao city deo?",
            thread_id="browse-davao-city-deo",
        )
        self.assertEqual(
            expanded,
            "Check availability where province=Davao City DEO AND category=school",
        )
        self.assertEqual(_detect_intent(expanded), "availability")

    def test_negros_island_region_is_not_overwritten_by_prior_scope(self) -> None:
        query_expand(
            "show contracts in ncr",
            thread_id="follow-up-location-leak",
        )
        expanded = query_expand(
            "give me contracts for Negros Island Region",
            thread_id="follow-up-location-leak",
        )
        self.assertEqual(
            expanded,
            "Filter contracts where region=Negros Island Region",
        )

    def test_search_query_keeps_raw_location_without_scope_leak(self) -> None:
        query_expand(
            "show contracts in national capital region",
            thread_id="search-leyte",
        )
        expanded = query_expand(
            "road widening and drainage near Leyte",
            thread_id="search-leyte",
        )
        self.assertEqual(
            expanded,
            "Find all contracts about road widening and drainage where province=Leyte AND category=road",
        )
        self.assertEqual(_detect_intent(expanded), "search")

    def test_follow_up_region_replaces_prior_region_and_keeps_shape(self) -> None:
        query_expand(
            "show road projects in region viii",
            thread_id="follow-up-region",
        )
        expanded = query_expand(
            "what about region vi?",
            thread_id="follow-up-region",
        )
        self.assertEqual(
            expanded,
            "Filter contracts where region=Region VI AND category=road",
        )

    def test_show_contracts_in_ncr_routes_to_browse(self) -> None:
        expanded = query_expand(
            "show contracts in national capital region",
            thread_id="browse-ncr",
        )
        self.assertEqual(
            expanded,
            "Filter contracts where region=National Capital Region",
        )
        self.assertEqual(_detect_intent(expanded), "browse")

    def test_result_reference_routes_to_browse_with_limit(self) -> None:
        set_thread_result(
            "result-reference-seven",
            {
                "result_kind": "contract_set",
                "intent": "availability",
                "filters": {
                    "region": "Region XI",
                    "status": "On-Going",
                    "category": "road",
                },
                "count": 7,
                "contract_ids": ["A", "B", "C"],
            },
        )

        expanded = query_expand(
            "what are those 7 projects?",
            thread_id="result-reference-seven",
        )

        self.assertEqual(
            expanded,
            "Filter contracts where region=Region XI AND status=On-Going AND category=road LIMIT 7",
        )
        self.assertEqual(_detect_intent(expanded), "browse")

    def test_show_them_reuses_last_result_filters(self) -> None:
        set_thread_result(
            "result-reference-show-them",
            {
                "result_kind": "contract_set",
                "intent": "availability",
                "filters": {
                    "province": "Iloilo",
                    "category": "flood control",
                },
                "count": 4,
            },
        )

        expanded = query_expand(
            "show them",
            thread_id="result-reference-show-them",
        )

        self.assertEqual(
            expanded,
            "Filter contracts where province=Iloilo AND category=flood control LIMIT 4",
        )
        self.assertEqual(_detect_intent(expanded), "browse")

    def test_follow_up_region_switch_keeps_prior_shape(self) -> None:
        set_thread_result(
            "result-reference-region-switch",
            {
                "result_kind": "contract_set",
                "intent": "browse",
                "filters": {
                    "region": "Region XI",
                    "status": "On-Going",
                    "category": "road",
                },
                "count": 7,
                "displayed_contract_ids": ["20L00044", "21LD0082"],
            },
        )
        expanded = query_expand(
            "how about in region 8?",
            thread_id="result-reference-region-switch",
        )

        self.assertEqual(
            expanded,
            "Filter contracts where region=Region VIII AND status=On-Going AND category=road",
        )
        self.assertEqual(_detect_intent(expanded), "browse")

    def test_ordinal_lookup_uses_displayed_result_order(self) -> None:
        set_thread_result(
            "result-reference-first-one",
            {
                "result_kind": "contract_set",
                "intent": "browse",
                "filters": {
                    "region": "Region VIII",
                    "status": "On-Going",
                    "category": "road",
                },
                "count": 10,
                "displayed_contract_ids": ["17LI0023", "19I00121", "19I00064"],
            },
        )

        expanded = query_expand(
            "give me details about the first one",
            thread_id="result-reference-first-one",
        )

        self.assertEqual(expanded, "Lookup contract 17LI0023")
        self.assertEqual(_detect_intent(expanded), "lookup")

    def test_same_contractor_follow_up_resolves_from_detail_context(self) -> None:
        set_thread_result(
            "same-contractor-detail",
            {
                "result_kind": "contract_detail",
                "intent": "lookup",
                "count": 1,
                "contract_ids": ["21GF0024"],
                "displayed_contract_ids": ["21GF0024"],
                "displayed_sources": [
                    {
                        "description": "CONSTRUCTION/IMPROVEMENT OF SAN JOAQUIN SHORELINE PROTECTION, SAN JOAQUIN, ILOILO",
                        "contractId": "21GF0024",
                        "contractor": "ABRIGHT BUILDERS CORPORATION (46487)",
                        "region": "Region VI",
                        "province": "Iloilo 1st DEO",
                        "budget": 5929936.5,
                        "awardAmount": 5929936.5,
                        "status": "Completed",
                        "category": "Flood Control and Drainage",
                    }
                ],
            },
        )

        expanded = query_expand(
            "what other projects does the same contractor have?",
            thread_id="same-contractor-detail",
        )

        self.assertEqual(
            expanded,
            "Filter contracts where contractor=ABRIGHT BUILDERS CORPORATION (46487)",
        )
        self.assertEqual(_detect_intent(expanded), "browse")
        self.assertTrue(get_thread_plan("same-contractor-detail").get("exclude_selected_contract"))

    def test_plain_contractor_reference_resolves_from_detail_context(self) -> None:
        set_thread_result(
            "same-contractor-plain-reference",
            {
                "result_kind": "contract_detail",
                "intent": "lookup",
                "count": 1,
                "contract_ids": ["21GF0024"],
                "displayed_contract_ids": ["21GF0024"],
                "displayed_sources": [
                    {
                        "description": "CONSTRUCTION/IMPROVEMENT OF SAN JOAQUIN SHORELINE PROTECTION, SAN JOAQUIN, ILOILO",
                        "contractId": "21GF0024",
                        "contractor": "ABRIGHT BUILDERS CORPORATION (46487)",
                        "region": "Region VI",
                        "province": "Iloilo 1st DEO",
                        "budget": 5929936.5,
                        "awardAmount": 5929936.5,
                        "status": "Completed",
                        "category": "Flood Control and Drainage",
                    }
                ],
            },
        )

        expanded = query_expand(
            "what other contracts does the contractor have?",
            thread_id="same-contractor-plain-reference",
        )

        self.assertEqual(
            expanded,
            "Filter contracts where contractor=ABRIGHT BUILDERS CORPORATION (46487)",
        )
        self.assertEqual(_detect_intent(expanded), "browse")
        self.assertTrue(
            get_thread_plan("same-contractor-plain-reference").get("exclude_selected_contract")
        )

    def test_broad_contract_query_routes_to_clarification(self) -> None:
        expanded = query_expand(
            "show me contracts",
            thread_id="clarify-broad-query",
        )

        self.assertEqual(
            expanded,
            "Ask clarifying question: Which region, contractor, category, or status should I narrow this to?",
        )
        self.assertEqual(_detect_intent(expanded), "clarify")

    def test_same_contractor_without_context_asks_for_clarification(self) -> None:
        expanded = query_expand(
            "what other projects does the same contractor have?",
            thread_id="clarify-same-contractor",
        )

        self.assertEqual(
            expanded,
            "Ask clarifying question: Which contractor are you referring to?",
        )
        self.assertEqual(_detect_intent(expanded), "clarify")

    def test_compare_visible_three_routes_to_compare_intent(self) -> None:
        set_thread_result(
            "compare-visible-three",
            {
                "result_kind": "contract_set",
                "intent": "browse",
                "filters": {"province": "Iloilo", "category": "flood control"},
                "count": 3,
                "displayed_contract_ids": ["21GF0024", "21GJ0002", "24GF0054"],
                "displayed_sources": [
                    {
                        "description": "CONSTRUCTION/IMPROVEMENT OF SAN JOAQUIN SHORELINE PROTECTION, SAN JOAQUIN, ILOILO",
                        "contractId": "21GF0024",
                    },
                    {
                        "description": "CONSTRUCTION OF SLOPE PROTECTION STRUCTURE - CONSTRUCTION OF SLOPE PROTECTION ALONG ILOILO CITY FLOODWAY, (BUHANG BRIDGE TO RADIAL BR. R/S) JARO, ILOILO CITY",
                        "contractId": "21GJ0002",
                    },
                    {
                        "description": "CONSTRUCTION OF MIAGAO POBLACION FLOOD CONTROL STRUCTURES INCLUDING ACCESS ROAD, MIAGAO, ILOILO",
                        "contractId": "24GF0054",
                    },
                ],
            },
        )

        expanded = query_expand(
            "Compare these three projects. Why is the Iloilo City Floodway project more expensive than the San Joaquin shoreline protection project?",
            thread_id="compare-visible-three",
        )

        self.assertEqual(
            expanded,
            "Compare contracts 21GF0024,21GJ0002,24GF0054: Compare these three projects. Why is the Iloilo City Floodway project more expensive than the San Joaquin shoreline protection project?",
        )
        self.assertEqual(_detect_intent(expanded), "compare")

    def test_compare_named_projects_resolves_subset(self) -> None:
        set_thread_result(
            "compare-named-projects",
            {
                "result_kind": "contract_set",
                "intent": "browse",
                "filters": {"province": "Iloilo", "category": "flood control"},
                "count": 3,
                "displayed_contract_ids": ["21GF0024", "21GJ0002", "24GF0054"],
                "displayed_sources": [
                    {
                        "description": "CONSTRUCTION/IMPROVEMENT OF SAN JOAQUIN SHORELINE PROTECTION, SAN JOAQUIN, ILOILO",
                        "contractId": "21GF0024",
                    },
                    {
                        "description": "CONSTRUCTION OF SLOPE PROTECTION STRUCTURE - CONSTRUCTION OF SLOPE PROTECTION ALONG ILOILO CITY FLOODWAY, (BUHANG BRIDGE TO RADIAL BR. R/S) JARO, ILOILO CITY",
                        "contractId": "21GJ0002",
                    },
                    {
                        "description": "CONSTRUCTION OF MIAGAO POBLACION FLOOD CONTROL STRUCTURES INCLUDING ACCESS ROAD, MIAGAO, ILOILO",
                        "contractId": "24GF0054",
                    },
                ],
            },
        )

        expanded = query_expand(
            "Compare the Iloilo City Floodway project and the San Joaquin shoreline protection project.",
            thread_id="compare-named-projects",
        )

        self.assertEqual(
            expanded,
            "Compare contracts 21GF0024,21GJ0002: Compare the Iloilo City Floodway project and the San Joaquin shoreline protection project.",
        )
        self.assertEqual(_detect_intent(expanded), "compare")

    def test_compare_without_result_context_asks_for_clarification(self) -> None:
        expanded = query_expand(
            "Compare these three projects.",
            thread_id="compare-missing-context",
        )

        self.assertEqual(
            expanded,
            "Ask clarifying question: Which contracts should I compare?",
        )
        self.assertEqual(_detect_intent(expanded), "clarify")


if __name__ == "__main__":
    unittest.main()
