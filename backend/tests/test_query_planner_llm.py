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
    from query_planner import QueryPlan
    import query_planner_llm
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


class PlannerMessageTests(unittest.TestCase):
    def test_payload_analysis_type_is_normalized_to_executor_contract(self) -> None:
        plan = query_planner_llm._plan_from_payload({"intent": "anomaly", "analysis_type": "document_gap"})
        self.assertEqual(plan.analysis_type, "document_gap")

    def test_contract_id_short_circuits_to_lookup(self) -> None:
        with patch("query_planner.get_entity_catalog", return_value=types.SimpleNamespace(
            regions=(),
            provinces=(),
            statuses=(),
            region_map={},
            province_map={},
            status_map={},
        )):
            plan = plan_message("show contract 21GF0024", thread_id="t1")

        self.assertEqual(plan.intent, "lookup")
        self.assertEqual(plan.lookup_value, "21GF0024")

    def test_greeting_without_domain_terms_short_circuits_to_chat(self) -> None:
        with patch("query_planner.get_entity_catalog", return_value=types.SimpleNamespace(
            regions=(),
            provinces=(),
            statuses=(),
            region_map={},
            province_map={},
            status_map={},
        )):
            plan = plan_message("hello there", thread_id="t2")

        self.assertEqual(plan.intent, "chat")

    def test_llm_failure_falls_back_to_deterministic_browse_plan(self) -> None:
        catalog = types.SimpleNamespace(
            regions=("Region VIII",),
            provinces=("Leyte",),
            statuses=("On-Going",),
            region_map={"region viii": "Region VIII"},
            province_map={"leyte": "Leyte"},
            status_map={"on going": "On-Going", "ongoing": "On-Going"},
        )
        with (
            patch("query_planner.get_entity_catalog", return_value=catalog),
            patch("query_planner_llm.get_thread_plan", return_value={}),
            patch("query_planner_llm.get_thread_result", return_value={}),
            patch("query_planner_llm.compact_thread_context", return_value="CONTEXT:"),
            patch("langchain_groq.ChatGroq", side_effect=RuntimeError("boom")),
        ):
            plan = plan_message("show road projects in leyte", thread_id="t3")

        self.assertIsInstance(plan, QueryPlan)
        self.assertEqual(plan.intent, "browse")
        self.assertEqual(plan.filters.get("province"), "Leyte")
        self.assertEqual(plan.filters.get("category"), "road")

    def test_history_recovery_restores_result_context_for_show_them(self) -> None:
        catalog = types.SimpleNamespace(
            regions=("Region XI",),
            provinces=(),
            statuses=("On-Going",),
            region_map={"region xi": "Region XI"},
            province_map={},
            status_map={"on going": "On-Going", "ongoing": "On-Going"},
        )
        recovered_messages = [
            {
                "message_metadata": {
                    "plan": {
                        "intent": "availability",
                        "filters": {
                            "region": "Region XI",
                            "status": "On-Going",
                            "category": "road",
                        },
                    },
                    "result_state": {
                        "result_kind": "contract_set",
                        "intent": "availability",
                        "filters": {
                            "region": "Region XI",
                            "status": "On-Going",
                            "category": "road",
                        },
                        "count": 7,
                        "displayed_contract_ids": ["20L00044", "21LD0082"],
                    },
                }
            }
        ]
        thread_plan_state = {}
        thread_result_state = {}

        def fake_set_thread_plan(thread_id, payload):
            thread_plan_state.clear()
            thread_plan_state.update(payload)

        def fake_set_thread_result(thread_id, payload):
            thread_result_state.clear()
            thread_result_state.update(payload)

        with (
            patch("query_planner.get_entity_catalog", return_value=catalog),
            patch("query_planner_llm.get_thread_plan", side_effect=lambda thread_id: dict(thread_plan_state)),
            patch("query_planner_llm.get_thread_result", side_effect=lambda thread_id: dict(thread_result_state)),
            patch("query_planner_llm.find_relevant_messages", return_value=recovered_messages),
            patch("query_planner_llm.set_thread_plan", side_effect=fake_set_thread_plan),
            patch("query_planner_llm.set_thread_result", side_effect=fake_set_thread_result),
            patch("langchain_groq.ChatGroq", side_effect=RuntimeError("boom")),
        ):
            plan = plan_message("show them", thread_id="t4")

        self.assertEqual(plan.intent, "browse")
        self.assertEqual(plan.filters.get("region"), "Region XI")
        self.assertEqual(plan.filters.get("status"), "On-Going")
        self.assertEqual(plan.filters.get("category"), "road")

    def test_fallback_same_contractor_anomaly_uses_prompt_canonical_values(self) -> None:
        catalog = types.SimpleNamespace(
            regions=(),
            provinces=(),
            statuses=(),
            region_map={},
            province_map={},
            status_map={},
        )
        result_state = {
            "displayed_sources": [
                {
                    "contractId": "21GF0024",
                    "description": "CONSTRUCTION/IMPROVEMENT OF SAN JOAQUIN SHORELINE PROTECTION",
                    "contractor": "ABRIGHT BUILDERS CORPORATION (46487)",
                }
            ]
        }
        with (
            patch("query_planner.get_entity_catalog", return_value=catalog),
            patch("query_planner_llm.get_thread_result", return_value=result_state),
            patch("query_planner_llm.get_thread_plan", return_value={}),
            patch("langchain_groq.ChatGroq", side_effect=RuntimeError("boom")),
        ):
            plan = plan_message("find suspicious missing documents for this contractor", thread_id="t5")

        self.assertEqual(plan.intent, "anomaly")
        self.assertEqual(plan.analysis_type, "document_gap")


if __name__ == "__main__":
    unittest.main()
