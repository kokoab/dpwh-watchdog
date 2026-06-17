import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, "backend")

from features.chat.agent.query_planner import build_anchor_plan, extract_anchor_filters


CATALOG = types.SimpleNamespace(
    regions=("Region VIII",),
    provinces=("Leyte",),
    statuses=("Awarded", "On-Going"),
    region_map={"region viii": "Region VIII"},
    province_map={"leyte": "Leyte"},
    status_map={"awarded": "Awarded", "on going": "On-Going", "ongoing": "On-Going"},
)


class QueryPlannerAnchorTests(unittest.TestCase):
    def test_relative_year_window_does_not_become_fake_province(self) -> None:
        with (
            patch("features.chat.agent.query_planner.get_entity_catalog", return_value=CATALOG),
            patch("features.chat.agent.query_planner._current_year", return_value=2026),
        ):
            filters = extract_anchor_filters("show me projects in the last 5 years")

        self.assertNotIn("province", filters)
        self.assertEqual(filters.get("infra_year_start"), "2022")
        self.assertEqual(filters.get("infra_year_end"), "2026")

    def test_relative_year_window_and_real_province_can_coexist(self) -> None:
        with (
            patch("features.chat.agent.query_planner.get_entity_catalog", return_value=CATALOG),
            patch("features.chat.agent.query_planner._current_year", return_value=2026),
        ):
            filters = extract_anchor_filters("show me projects in leyte in the last 5 years")

        self.assertEqual(filters.get("province"), "Leyte")
        self.assertEqual(filters.get("infra_year_start"), "2022")
        self.assertEqual(filters.get("infra_year_end"), "2026")

    def test_explicit_year_takes_precedence_over_relative_window(self) -> None:
        with (
            patch("features.chat.agent.query_planner.get_entity_catalog", return_value=CATALOG),
            patch("features.chat.agent.query_planner._current_year", return_value=2026),
        ):
            filters = extract_anchor_filters("show me projects awarded in 2024 from the last 5 years")

        self.assertEqual(filters.get("infra_year"), "2024")
        self.assertNotIn("infra_year_start", filters)
        self.assertNotIn("infra_year_end", filters)

    def test_non_location_relative_time_does_not_mark_unresolved_location(self) -> None:
        with (
            patch("features.chat.agent.query_planner.get_entity_catalog", return_value=CATALOG),
            patch("features.chat.agent.query_planner._current_year", return_value=2026),
        ):
            plan = build_anchor_plan("show me projects in the last 5 years")

        self.assertFalse(plan.has_unresolved_location_hint)
        self.assertNotIn("province", plan.filters)

    def test_awarded_to_extracts_contractor_without_lifecycle_status(self) -> None:
        query = "projects awarded to TOPMOST DEVELOPMENT & MKTG. CORP."
        with patch("features.chat.agent.query_planner.get_entity_catalog", return_value=CATALOG):
            filters = extract_anchor_filters(query)

        self.assertEqual(filters.get("contractor"), "TOPMOST DEVELOPMENT & MKTG. CORP.")
        self.assertNotIn("status", filters)

    def test_broken_query_maps_to_expected_filters(self) -> None:
        query = (
            "Show me all projects awarded to TOPMOST DEVELOPMENT & MKTG. CORP. "
            "in the last 5 years. What is the total contract value, and which provinces "
            "received the most projects?"
        )
        with (
            patch("features.chat.agent.query_planner.get_entity_catalog", return_value=CATALOG),
            patch("features.chat.agent.query_planner._current_year", return_value=2026),
        ):
            filters = extract_anchor_filters(query)

        self.assertNotIn("province", filters)
        self.assertEqual(filters.get("contractor"), "TOPMOST DEVELOPMENT & MKTG. CORP.")
        self.assertEqual(filters.get("infra_year_start"), "2022")
        self.assertEqual(filters.get("infra_year_end"), "2026")
        self.assertNotIn("status", filters)


if __name__ == "__main__":
    unittest.main()
