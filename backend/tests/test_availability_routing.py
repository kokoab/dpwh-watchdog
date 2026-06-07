import unittest

from query_expand import _detect_intent, query_expand
from query_scope import clear_thread_scope


class AvailabilityRoutingTests(unittest.TestCase):
    def tearDown(self) -> None:
        for thread_id in (
            "availability-region-8",
            "availability-any-region-8",
            "availability-road-region-8",
            "about-region-8",
            "browse-region-8",
            "browse-tacloban",
            "scope-follow-up",
        ):
            clear_thread_scope(thread_id)

    def test_do_you_have_available_contracts_routes_to_statistics(self) -> None:
        expanded = query_expand(
            "do you have any contracts available in region 8",
            thread_id="availability-region-8",
        )
        self.assertEqual(
            expanded,
            "Calculate metrics for availability check: contracts in Region VIII",
        )
        self.assertEqual(_detect_intent(expanded), "statistics")

    def test_are_there_any_contracts_routes_to_statistics(self) -> None:
        expanded = query_expand(
            "are there any contracts in region 8",
            thread_id="availability-any-region-8",
        )
        self.assertEqual(
            expanded,
            "Calculate metrics for availability check: contracts in Region VIII",
        )
        self.assertEqual(_detect_intent(expanded), "statistics")

    def test_is_there_a_road_contract_routes_to_statistics(self) -> None:
        expanded = query_expand(
            "is there a road contract in region 8",
            thread_id="availability-road-region-8",
        )
        self.assertEqual(
            expanded,
            "Calculate metrics for availability check: road contract in Region VIII",
        )
        self.assertEqual(_detect_intent(expanded), "statistics")

    def test_contracts_about_follow_up_stays_search_and_carries_scope(self) -> None:
        query_expand(
            "do you have any contracts available in region 8",
            thread_id="scope-follow-up",
        )

        expanded = query_expand(
            "are there contracts about road construction?",
            thread_id="scope-follow-up",
        )
        self.assertEqual(
            expanded,
            "Find all contracts about road construction in Region VIII",
        )
        self.assertEqual(_detect_intent(expanded), "search")

    def test_explicit_region_override_beats_prior_scope(self) -> None:
        query_expand(
            "do you have any contracts available in region 8",
            thread_id="scope-follow-up",
        )

        expanded = query_expand(
            "are there contracts about road construction in region 6?",
            thread_id="scope-follow-up",
        )
        self.assertEqual(
            expanded,
            "Find all contracts about road construction in Region VI",
        )

    def test_show_me_all_contracts_stays_filter(self) -> None:
        expanded = query_expand(
            "show me all contracts in region 8",
            thread_id="availability-region-8",
        )
        self.assertEqual(expanded, "Filter contracts where region=Region VIII")
        self.assertEqual(_detect_intent(expanded), "filter")

    def test_contracts_about_region_routes_to_search_not_filter(self) -> None:
        expanded = query_expand(
            "are there any contracts about region 8?",
            thread_id="about-region-8",
        )
        self.assertEqual(expanded, "Find all contracts about Region VIII")
        self.assertEqual(_detect_intent(expanded), "search")

    def test_what_contracts_are_there_routes_to_filter(self) -> None:
        expanded = query_expand(
            "what contracts are there in region 8?",
            thread_id="browse-region-8",
        )
        self.assertEqual(expanded, "Filter contracts where region=Region VIII")
        self.assertEqual(_detect_intent(expanded), "filter")

    def test_what_projects_are_there_in_city_routes_to_filter(self) -> None:
        expanded = query_expand(
            "what projects are there in tacloban?",
            thread_id="browse-tacloban",
        )
        self.assertEqual(expanded, "Filter contracts where province=tacloban")
        self.assertEqual(_detect_intent(expanded), "filter")


if __name__ == "__main__":
    unittest.main()
