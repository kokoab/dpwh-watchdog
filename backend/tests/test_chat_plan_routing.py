import importlib.util
import json
import sys
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import Mock


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

    def to_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "filters": dict(self.filters),
            "subject": self.subject,
            "lookup_value": self.lookup_value,
            "limit": self.limit,
            "exclude_selected_contract": self.exclude_selected_contract,
            "has_location_phrase": self.has_location_phrase,
            "has_unresolved_location_hint": self.has_unresolved_location_hint,
            "is_follow_up": self.is_follow_up,
            "analysis_type": self.analysis_type,
        }


def _load_chat_module(plan: QueryPlan, *, anomaly_output=None, compare_output="compare synthesis") -> tuple[types.ModuleType, list[dict]]:
    saved_messages: list[dict] = []
    thread_result: dict[str, object] = {}

    agent_mod = types.ModuleType("agent")
    agent_mod.stream_agent = lambda message, thread_id: iter([{"type": "done"}])

    chat_memory_mod = types.ModuleType("chat_memory")
    chat_memory_mod.ensure_chat_thread = lambda *args, **kwargs: None
    chat_memory_mod.list_chat_messages = lambda *args, **kwargs: []
    chat_memory_mod.list_chat_threads = lambda *args, **kwargs: []
    chat_memory_mod.save_chat_message = lambda *args, **kwargs: saved_messages.append(
        {"args": args, "kwargs": kwargs}
    )

    fastapi_mod = types.ModuleType("fastapi")

    class APIRouter:
        def __init__(self, *args, **kwargs):
            pass

        def post(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def get(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    fastapi_mod.APIRouter = APIRouter

    fastapi_responses_mod = types.ModuleType("fastapi.responses")

    class StreamingResponse:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fastapi_responses_mod.StreamingResponse = StreamingResponse

    pydantic_mod = types.ModuleType("pydantic")

    class BaseModel:
        pass

    pydantic_mod.BaseModel = BaseModel

    query_expand_mod = types.ModuleType("query_expand")
    query_expand_mod._detect_intent = lambda text: "chat"
    query_expand_mod.log_query_expansion = lambda *args, **kwargs: None
    query_expand_mod.query_expand = lambda query, thread_id=None: query

    query_planner_mod = types.ModuleType("query_planner")
    query_planner_mod.QueryPlan = QueryPlan

    query_planner_llm_mod = types.ModuleType("query_planner_llm")
    query_planner_llm_mod.plan_message = lambda message, thread_id=None: plan

    query_scope_mod = types.ModuleType("query_scope")
    query_scope_mod.clear_current_thread_id = lambda: None
    query_scope_mod.get_thread_plan = lambda thread_id=None: {}
    query_scope_mod.get_thread_result = lambda thread_id=None: dict(thread_result)
    query_scope_mod.set_current_thread_id = lambda thread_id=None: None
    query_scope_mod.set_thread_plan = lambda thread_id, payload: None
    query_scope_mod.set_thread_result = lambda thread_id, payload: (
        thread_result.clear(),
        thread_result.update(payload),
    )

    synthesis_mod = types.ModuleType("synthesis")
    synthesis_mod.focused_synthesis = lambda task, tool_output, thread_id: compare_output

    tools_mod = types.ModuleType("tools")
    tools_mod.execute_lookup_plan = lambda plan: "lookup"
    tools_mod.execute_browse_plan = lambda plan: "browse"
    tools_mod.execute_availability_plan = lambda plan: "availability"
    tools_mod.execute_stats_plan = lambda plan: (
        "stats",
        {
            "total_contracts": 3,
            "total_budget": 1000.0,
            "province_breakdown": [{"province": "Cebu", "count": 2}],
        },
    )
    tools_mod.execute_clarify_plan = lambda plan: plan.subject
    tools_mod.execute_search_plan = lambda plan: "search"
    tools_mod.execute_anomaly_plan = lambda plan: anomaly_output or {
        "analysis_type": plan.analysis_type,
        "rows": [{"contract_id": "A"}],
    }
    tools_mod.load_contract_detail_sources = lambda contract_ids: [
        {"contractId": contract_id, "description": f"Contract {contract_id}"}
        for contract_id in contract_ids
    ]

    modules = {
        "agent": agent_mod,
        "chat_memory": chat_memory_mod,
        "fastapi": fastapi_mod,
        "fastapi.responses": fastapi_responses_mod,
        "pydantic": pydantic_mod,
        "query_expand": query_expand_mod,
        "query_planner": query_planner_mod,
        "query_planner_llm": query_planner_llm_mod,
        "query_scope": query_scope_mod,
        "synthesis": synthesis_mod,
        "tools": tools_mod,
    }

    old_modules = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        module_path = _resolve_backend_module_path("chat.py")
        spec = importlib.util.spec_from_file_location("chat_plan_test_mod", module_path)
        assert spec and spec.loader
        chat_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chat_mod)
    finally:
        for name, old_value in old_modules.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value

    return chat_mod, saved_messages


class ChatPlanRoutingTests(unittest.TestCase):
    def _events(self, payloads: list[str]) -> list[dict[str, object]]:
        return [json.loads(payload[len("data: "):]) for payload in payloads]

    def test_event_stream_routes_anomaly_plan_through_focused_synthesis(self) -> None:
        plan = QueryPlan(
            intent="anomaly",
            filters={"region": "Region XI"},
            subject="Find suspicious bidding patterns.",
            analysis_type="bidding_anomalies",
        )
        chat_mod, saved_messages = _load_chat_module(
            plan,
            anomaly_output={"analysis_type": "bidding_anomalies", "rows": [{"contract_id": "21A"}]},
            compare_output="Anomaly synthesis reply",
        )

        events = self._events(list(chat_mod.event_stream("find suspicious bidding patterns", "thread-1")))

        self.assertEqual(events[0]["type"], "result_state")
        self.assertEqual(events[-1]["type"], "done")
        streamed = "".join(str(event["content"]) for event in events if event["type"] == "token")
        self.assertEqual(streamed, "Anomaly synthesis reply")
        self.assertEqual(saved_messages[0]["kwargs"]["intent"], "anomaly")
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["execution_path"], "direct_tool")
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["response_source"], "structured")

    def test_event_stream_routes_compare_plan_through_focused_synthesis(self) -> None:
        plan = QueryPlan(
            intent="compare",
            lookup_value="21GF0024,21GJ0002",
            subject="Compare these contracts.",
        )
        chat_mod, saved_messages = _load_chat_module(
            plan,
            compare_output="Compare synthesis reply",
        )

        events = self._events(list(chat_mod.event_stream("compare these contracts", "thread-2")))

        streamed = "".join(str(event["content"]) for event in events if event["type"] == "token")
        self.assertIn("Compare synthesis reply", streamed)
        self.assertIn("|Contract ID|Description|Budget|Status|Completion Date|Duration|Region|", streamed)
        self.assertIn("**Rankings**", streamed)
        self.assertEqual(saved_messages[0]["kwargs"]["intent"], "compare")
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["execution_path"], "direct_compare")
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["response_source"], "structured")

    def test_is_analytical_stats_detects_subject_and_question_signals(self) -> None:
        plan = QueryPlan(intent="stats")
        chat_mod, _ = _load_chat_module(plan)

        self.assertTrue(
            chat_mod._is_analytical_stats(
                QueryPlan(intent="stats", subject="contract value by province"),
                "how many projects",
            )
        )
        self.assertTrue(
            chat_mod._is_analytical_stats(
                QueryPlan(intent="stats"),
                "which province received the most projects?",
            )
        )
        self.assertFalse(
            chat_mod._is_analytical_stats(
                QueryPlan(intent="stats"),
                "how many completed bridges in CAR",
            )
        )

    def test_direct_stats_turn_calls_focused_synthesis_for_analytical_question(self) -> None:
        plan = QueryPlan(intent="stats", filters={"contractor": "TOPMOST"})
        chat_mod, _ = _load_chat_module(plan)
        synthesis_mock = Mock(return_value="Synthesized stats reply")
        chat_mod.focused_synthesis = synthesis_mock

        assistant_text, result_state, source = chat_mod._run_direct_tool_turn(
            plan,
            "stats-thread",
            "which province received the most projects?",
        )

        self.assertEqual(assistant_text, "Synthesized stats reply")
        self.assertIsNone(result_state)
        self.assertEqual(source, "structured")
        synthesis_mock.assert_called_once()
        synthesis_task = synthesis_mock.call_args.args[0]
        self.assertIn("which province received the most projects?", synthesis_task)
        self.assertIn("No markdown tables", synthesis_task)
        self.assertIn("No pipe characters", synthesis_task)
        self.assertEqual(synthesis_mock.call_args.args[2], "stats-thread")
        self.assertIn("province_breakdown", synthesis_mock.call_args.args[1])

    def test_direct_stats_turn_skips_focused_synthesis_for_template_question(self) -> None:
        plan = QueryPlan(intent="stats", filters={"region": "CAR"})
        chat_mod, _ = _load_chat_module(plan)
        synthesis_mock = Mock(return_value="Should not be used")
        chat_mod.focused_synthesis = synthesis_mock

        assistant_text, result_state, source = chat_mod._run_direct_tool_turn(
            plan,
            "stats-thread",
            "how many completed bridges in CAR",
        )

        self.assertEqual(assistant_text, "stats")
        self.assertIsNone(result_state)
        self.assertEqual(source, "tool")
        synthesis_mock.assert_not_called()

    def test_structured_contract_reply_preserves_deo_suffixes(self) -> None:
        plan = QueryPlan(intent="browse")
        chat_mod, _ = _load_chat_module(plan)

        reply = chat_mod._build_structured_contract_reply(
            {
                "count": 2,
                "displayed_sources": [
                    {
                        "contractId": "A001",
                        "description": "Flood control project",
                        "budget": 1000.0,
                        "status": "Completed",
                        "province": "Leyte 2nd DEO",
                    },
                    {
                        "contractId": "A002",
                        "description": "Drainage project",
                        "budget": 500.0,
                        "status": "On-Going",
                        "province": "Tacloban City DEO",
                    },
                ],
            }
        )

        self.assertIn("Leyte 2nd DEO", reply)
        self.assertIn("Tacloban City DEO", reply)
        self.assertIn("**Executive summary:**", reply)
        self.assertIn("|Contract ID|Description|Budget|Status|Completion Date|Progress|Office/Province|", reply)
        self.assertIn("|A001|Flood control project|₱1K|Completed|N/A|N/A|Leyte 2nd DEO|", reply)
        self.assertIn("Highest listed budget: A001 at ₱1K", reply)
        self.assertNotIn("1. **A001**", reply)

    def test_structured_contract_reply_with_dates_includes_completion(self) -> None:
        plan = QueryPlan(intent="stats")
        chat_mod, _ = _load_chat_module(plan)

        reply = chat_mod._build_structured_contract_reply_with_dates(
            {
                "count": 1,
                "displayed_sources": [
                    {
                        "contractId": "A003",
                        "description": "Largest flood control project",
                        "budget": 1500.0,
                        "status": "Completed",
                        "progress": 100,
                        "completionDate": "2025-03-30",
                    },
                ],
            }
        )

        self.assertIn("**Executive summary:**", reply)
        self.assertIn("|Contract ID|Description|Budget|Status|Completion Date|Progress|Office/Province|", reply)
        self.assertIn("|A003|Largest flood control project|₱2K|Completed|2025-03-30|100%|N/A|", reply)
        self.assertIn("2025-03-30", reply)
        self.assertIn("100%", reply)
        self.assertNotIn("1. **A003**", reply)

    def test_direct_browse_contract_set_uses_table_formatter(self) -> None:
        plan = QueryPlan(intent="browse")
        chat_mod, _ = _load_chat_module(plan)
        result_state = {
            "result_kind": "contract_set",
            "count": 1,
            "filters": {"province": "Leyte", "category": "flood control"},
            "displayed_sources": [
                {
                    "contractId": "ABC123",
                    "description": "Construction of drainage canal",
                    "budget": 950002,
                    "status": "Completed",
                    "progress": 100,
                    "completionDate": "2021-06-15",
                    "province": "Leyte",
                }
            ],
        }

        def execute_browse(_plan):
            chat_mod.set_thread_result("browse-thread", result_state)
            return "raw browse output"

        chat_mod.DIRECT_TOOL_BY_INTENT["browse"] = execute_browse

        assistant_text, latest_result_state, source = chat_mod._run_direct_tool_turn(
            plan,
            "browse-thread",
            "are there any flood control projects around Tacloban City? what are their budgets and completion dates",
        )

        self.assertEqual(source, "structured")
        self.assertEqual(latest_result_state, result_state)
        self.assertIn("**Executive summary:** Found 1 matching flood control contracts in Leyte.", assistant_text)
        self.assertIn("|ABC123|Construction of drainage canal|₱950K|Completed|2021-06-15|100%|Leyte|", assistant_text)
        self.assertIn("Highest listed budget: ABC123 at ₱950K", assistant_text)
        self.assertNotIn("raw browse output", assistant_text)


if __name__ == "__main__":
    unittest.main()
