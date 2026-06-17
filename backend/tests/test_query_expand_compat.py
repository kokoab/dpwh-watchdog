import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, "backend")

psycopg2 = types.ModuleType("psycopg2")
psycopg2.connect = lambda *args, **kwargs: None
extras = types.ModuleType("psycopg2.extras")
extras.Json = lambda value: value
extras.RealDictCursor = object
psycopg2.extras = extras


class _ExplodingGroq:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("force fallback planner")


langchain_groq = types.ModuleType("langchain_groq")
langchain_groq.ChatGroq = _ExplodingGroq

_old_psycopg2 = sys.modules.get("psycopg2")
_old_psycopg2_extras = sys.modules.get("psycopg2.extras")
_old_langchain_groq = sys.modules.get("langchain_groq")
sys.modules["psycopg2"] = psycopg2
sys.modules["psycopg2.extras"] = extras
sys.modules["langchain_groq"] = langchain_groq
try:
    from contracts.query_expand import _detect_intent, query_expand
    from features.chat.agent.query_scope import clear_thread_cache, set_thread_result
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
        "cordillera administrative region": "Cordillera Administrative Region",
        "negros island region": "Negros Island Region",
    },
    province_map={
        "iloilo": "Iloilo",
        "leyte": "Leyte",
        "davao city deo": "Davao City DEO",
    },
    status_map={"on going": "On-Going", "completed": "Completed"},
)


class QueryExpandCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        patchers = [
            patch("features.chat.agent.query_planner.get_entity_catalog", return_value=CATALOG),
            patch("features.chat.agent.query_scope.upsert_thread_state"),
            patch("features.chat.agent.query_scope.get_thread_state", return_value={}),
            patch("langchain_groq.ChatGroq", side_effect=RuntimeError("force fallback planner")),
        ]
        self._patchers = patchers
        for patcher in patchers:
            patcher.start()

    def tearDown(self) -> None:
        for thread_id in (
            "availability-region",
            "browse-iloilo",
            "stats-car",
            "search-leyte",
            "follow-up-region",
            "result-reference",
            "ordinal-lookup",
            "same-contractor",
            "clarify-broad",
            "compare-visible",
            "compare-named",
        ):
            clear_thread_cache(thread_id)
        for patcher in reversed(self._patchers):
            patcher.stop()

    def test_core_routing_shapes(self) -> None:
        expanded = query_expand("any ongoing road projects in region xi?", thread_id="availability-region")
        self.assertEqual(
            expanded,
            "Check availability where region=Region XI AND status=On-Going AND category=road",
        )
        self.assertEqual(_detect_intent(expanded), "availability")

        expanded = query_expand("list flood control projects in iloilo", thread_id="browse-iloilo")
        self.assertEqual(
            expanded,
            "Filter contracts where province=Iloilo AND category=flood control",
        )
        self.assertEqual(_detect_intent(expanded), "browse")

        expanded = query_expand("how many completed bridges in CAR", thread_id="stats-car")
        self.assertEqual(
            expanded,
            "Calculate metrics where region=Cordillera Administrative Region AND status=Completed AND category=bridge",
        )
        self.assertEqual(_detect_intent(expanded), "stats")

    def test_search_follow_up_and_result_reference_cases(self) -> None:
        query_expand("show contracts in national capital region", thread_id="search-leyte")
        expanded = query_expand("road widening and drainage near Leyte", thread_id="search-leyte")
        self.assertEqual(
            expanded,
            "Find all contracts about road widening and drainage where province=Leyte AND category=road",
        )
        self.assertEqual(_detect_intent(expanded), "search")

        query_expand("show road projects in region viii", thread_id="follow-up-region")
        expanded = query_expand("what about region vi?", thread_id="follow-up-region")
        self.assertEqual(
            expanded,
            "Filter contracts where region=Region VI AND category=road",
        )

        set_thread_result(
            "result-reference",
            {
                "result_kind": "contract_set",
                "intent": "availability",
                "filters": {"province": "Iloilo", "category": "flood control"},
                "count": 4,
                "displayed_contract_ids": ["A", "B", "C", "D"],
            },
        )
        expanded = query_expand("show them", thread_id="result-reference")
        self.assertEqual(
            expanded,
            "Filter contracts where province=Iloilo AND category=flood control LIMIT 4",
        )

    def test_context_reference_and_compare_cases(self) -> None:
        set_thread_result(
            "ordinal-lookup",
            {
                "result_kind": "contract_set",
                "intent": "browse",
                "filters": {"region": "Region VIII", "status": "On-Going", "category": "road"},
                "count": 10,
                "displayed_contract_ids": ["17LI0023", "19I00121", "19I00064"],
            },
        )
        expanded = query_expand("give me details about the first one", thread_id="ordinal-lookup")
        self.assertEqual(expanded, "Lookup contract 17LI0023")

        set_thread_result(
            "same-contractor",
            {
                "result_kind": "contract_detail",
                "intent": "lookup",
                "count": 1,
                "displayed_contract_ids": ["21GF0024"],
                "displayed_sources": [
                    {
                        "description": "CONSTRUCTION/IMPROVEMENT OF SAN JOAQUIN SHORELINE PROTECTION, SAN JOAQUIN, ILOILO",
                        "contractId": "21GF0024",
                        "contractor": "ABRIGHT BUILDERS CORPORATION (46487)",
                    }
                ],
            },
        )
        expanded = query_expand(
            "what other contracts does the contractor have?",
            thread_id="same-contractor",
        )
        self.assertEqual(
            expanded,
            "Filter contracts where contractor=ABRIGHT BUILDERS CORPORATION (46487)",
        )

        expanded = query_expand("show me contracts", thread_id="clarify-broad")
        self.assertEqual(
            expanded,
            "Ask clarifying question: Which region, contractor, category, or status should I narrow this to?",
        )

        set_thread_result(
            "compare-visible",
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
        expanded = query_expand("Compare these three projects.", thread_id="compare-visible")
        self.assertEqual(
            expanded,
            "Compare contracts 21GF0024,21GJ0002,24GF0054: Compare these three projects.",
        )

        set_thread_result(
            "compare-named",
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
            thread_id="compare-named",
        )
        self.assertEqual(
            expanded,
            "Compare contracts 21GF0024,21GJ0002: Compare the Iloilo City Floodway project and the San Joaquin shoreline protection project.",
        )


if __name__ == "__main__":
    unittest.main()
