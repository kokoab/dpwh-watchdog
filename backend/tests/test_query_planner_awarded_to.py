import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, "backend")

psycopg2 = types.ModuleType("psycopg2")
psycopg2.connect = lambda *args, **kwargs: None
psycopg2_extras = types.ModuleType("psycopg2.extras")
psycopg2.extras = psycopg2_extras

langchain_groq = types.ModuleType("langchain_groq")
langchain_groq.ChatGroq = object

_old_psycopg2 = sys.modules.get("psycopg2")
_old_psycopg2_extras = sys.modules.get("psycopg2.extras")
_old_langchain_groq = sys.modules.get("langchain_groq")
sys.modules["psycopg2"] = psycopg2
sys.modules["psycopg2.extras"] = psycopg2_extras
sys.modules["langchain_groq"] = langchain_groq
try:
    import query_planner
    from query_planner_llm import plan_message
finally:
    if _old_psycopg2 is None:
        sys.modules.pop("psycopg2", None)
    else:
        sys.modules["psycopg2"] = _old_psycopg2

    if _old_psycopg2_extras is None:
        sys.modules.pop("psycopg2.extras", None)
    else:
        sys.modules["psycopg2.extras"] = _old_psycopg2_extras

    if _old_langchain_groq is None:
        sys.modules.pop("langchain_groq", None)
    else:
        sys.modules["langchain_groq"] = _old_langchain_groq


def _catalog():
    return types.SimpleNamespace(
        regions=("Region VIII",),
        provinces=("Leyte",),
        statuses=("Awarded",),
        region_map={"region viii": "Region VIII"},
        province_map={"leyte": "Leyte"},
        status_map={"awarded": "Awarded"},
    )


def _fallback_plan(query: str):
    with (
        patch("query_planner.get_entity_catalog", return_value=_catalog()),
        patch("query_planner._current_year", return_value=2026),
        patch("query_planner_llm.get_thread_plan", return_value={}),
        patch("query_planner_llm.get_thread_result", return_value={}),
        patch("query_planner_llm.compact_thread_context", return_value="CONTEXT:"),
        patch.dict(sys.modules, {"langchain_groq": langchain_groq}),
        patch("langchain_groq.ChatGroq", side_effect=RuntimeError("force fallback")),
    ):
        return plan_message(query, thread_id="awarded-to-tests")


class QueryPlannerAwardedToTests(unittest.TestCase):
    def test_awarded_to_topmost_last_five_years_keeps_contractor_and_year_window(self) -> None:
        plan = _fallback_plan(
            "Show me all projects awarded to TOPMOST DEVELOPMENT & MKTG. CORP. "
            "in the last 5 years"
        )

        self.assertEqual(plan.filters.get("contractor"), "TOPMOST DEVELOPMENT & MKTG. CORP.")
        self.assertEqual(plan.filters.get("infra_year_start"), "2022")
        self.assertEqual(plan.filters.get("infra_year_end"), "2026")
        self.assertNotIn("status", plan.filters)

    def test_projects_awarded_to_contractor_does_not_add_awarded_status(self) -> None:
        plan = _fallback_plan("projects awarded to ABC Corp")

        self.assertEqual(plan.filters.get("contractor"), "ABC Corp")
        self.assertNotIn("status", plan.filters)

    def test_last_five_years_window_uses_current_year_inclusively(self) -> None:
        with patch("query_planner._current_year", return_value=2026):
            filters = query_planner.match_year_filters("last 5 years")

        self.assertEqual(filters, {"infra_year_start": "2022", "infra_year_end": "2026"})

    def test_last_year_uses_previous_calendar_year(self) -> None:
        with patch("query_planner._current_year", return_value=2026):
            filters = query_planner.match_year_filters("last year")

        self.assertEqual(filters, {"infra_year": "2025"})


if __name__ == "__main__":
    unittest.main()
