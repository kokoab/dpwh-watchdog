import unittest

from query_expand import _detect_intent, query_expand
from query_scope import clear_thread_scope, get_thread_plan, set_thread_result


class DeterministicRoutingTests(unittest.TestCase):
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
            "clarify-broad-query",
            "clarify-same-contractor",
        ):
            clear_thread_scope(thread_id)

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
            "Find all contracts about road widening and drainage where province=Leyte",
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


if __name__ == "__main__":
    unittest.main()
