import importlib.util
import json
import sys
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path


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
    tools_mod.execute_stats_plan = lambda plan: "stats"
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
        self.assertEqual(streamed, "Compare synthesis reply")
        self.assertEqual(saved_messages[0]["kwargs"]["intent"], "compare")
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["execution_path"], "direct_compare")
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["response_source"], "structured")


if __name__ == "__main__":
    unittest.main()
