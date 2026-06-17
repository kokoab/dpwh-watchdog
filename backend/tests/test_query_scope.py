import unittest
import sys
import types
from unittest.mock import patch

sys.path.insert(0, "backend")

psycopg2 = types.ModuleType("psycopg2")
psycopg2.connect = lambda *args, **kwargs: None
extras = types.ModuleType("psycopg2.extras")
extras.Json = lambda value: value
extras.RealDictCursor = object
psycopg2.extras = extras
_old_psycopg2 = sys.modules.get("psycopg2")
_old_psycopg2_extras = sys.modules.get("psycopg2.extras")
sys.modules["psycopg2"] = psycopg2
sys.modules["psycopg2.extras"] = extras
try:
    from features.chat.agent.query_scope import compact_thread_context
finally:
    if _old_psycopg2 is None:
        sys.modules.pop("psycopg2", None)
    else:
        sys.modules["psycopg2"] = _old_psycopg2

    if _old_psycopg2_extras is None:
        sys.modules.pop("psycopg2.extras", None)
    else:
        sys.modules["psycopg2.extras"] = _old_psycopg2_extras


class CompactThreadContextTests(unittest.TestCase):
    def test_compact_thread_context_summarizes_result_and_plan(self) -> None:
        result_state = {
            "result_kind": "contract_set",
            "count": 7,
            "filters": {"province": "Iloilo", "category": "flood control"},
            "displayed_sources": [
                {
                    "contractId": "21GF0024",
                    "description": (
                        "CONSTRUCTION/IMPROVEMENT OF SAN JOAQUIN SHORELINE "
                        "PROTECTION, SAN JOAQUIN, ILOILO"
                    ),
                    "contractor": "ABRIGHT BUILDERS CORPORATION (46487)",
                },
                {
                    "contractId": "21GJ0002",
                    "description": "CONSTRUCTION OF SLOPE PROTECTION STRUCTURE",
                    "contractor": "BOAZ AND JACHIN CONSTRUCTION SUPPLY & SERVICES",
                },
            ],
        }
        plan_state = {"intent": "browse", "filters": {"province": "Iloilo"}}

        with (
            patch("features.chat.agent.query_scope.get_thread_result", return_value=result_state),
            patch("features.chat.agent.query_scope.get_thread_plan", return_value=plan_state),
        ):
            context = compact_thread_context("thread-1")

        self.assertIn("result_kind: contract_set", context)
        self.assertIn("result_count: 7", context)
        self.assertIn('active_filters: {"category": "flood control", "province": "Iloilo"}', context)
        self.assertIn('"index": 0, "id": "21GF0024"', context)
        self.assertIn("Contractor: ABRIGHT BUILDERS CORPORATION (46487)", context)
        self.assertIn("ABRIGHT BUILDERS CORPORATION (46487)", context)
        self.assertIn("BOAZ AND JACHIN CONSTRUCTION SUPPLY & SERVICES", context)
        self.assertIn('"index": 1, "id": "21GJ0002"', context)
        self.assertIn('selected_contract: {"id": "21GF0024"', context)
        self.assertIn("last_intent: browse", context)

    def test_compact_thread_context_handles_empty_state(self) -> None:
        with (
            patch("features.chat.agent.query_scope.get_thread_result", return_value={}),
            patch("features.chat.agent.query_scope.get_thread_plan", return_value={}),
        ):
            context = compact_thread_context("thread-2")

        self.assertEqual(
            context,
            "\n".join(
                [
                    "result_kind: none",
                    "result_count: 0",
                    "active_filters: {}",
                    "displayed_contracts: []",
                    "selected_contract: null",
                    "last_intent: none",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
