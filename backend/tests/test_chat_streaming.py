import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessageChunk

import agent
import chat
from query_planner import QueryPlan


def _sse_event(payload: str) -> dict[str, object]:
    prefix = "data: "
    if not payload.startswith(prefix):
        raise AssertionError(f"Unexpected SSE payload: {payload!r}")
    return json.loads(payload[len(prefix):])


class ChatStreamingTests(unittest.TestCase):
    RESULT_STATE = {
        "result_kind": "contract_set",
        "intent": "browse",
        "filters": {"province": "Leyte", "category": "flood control"},
        "displayed_sources": [
            {
                "description": "Construction of drainage canal",
                "contractId": "ABC123",
                "programName": "Regular Infra",
                "status": "Completed",
                "budget": 950002,
                "region": "Region VIII",
                "province": "Leyte",
                "contractor": "RJIR Enterprises",
                "progress": 100,
                "completionDate": "2021-06-15",
            }
        ],
    }
    EXPECTED_REPLY = (
        "1. **ABC123** Construction of drainage canal — ₱950K — Completed — 2021-06-15 (Leyte)\n\n"
        "Showing 1 of 1 available contracts. Highest budget: ABC123 at ₱950K."
    )

    def _run_event_stream(
        self,
        streamed_events: list[dict[str, object]],
        *,
        plan_snapshot: dict[str, object] | None = None,
        detected_intent: str = "browse",
    ):
        saved_messages = []

        def capture_save(*args, **kwargs):
            saved_messages.append({"args": args, "kwargs": kwargs})

        plan_payload = plan_snapshot or {"intent": detected_intent}
        plan = QueryPlan(
            intent=str(plan_payload.get("intent", detected_intent)),
            filters=dict(plan_payload.get("filters", {})),
            subject=str(plan_payload.get("subject", "") or ""),
            lookup_value=str(plan_payload.get("lookup_value", "") or ""),
            limit=plan_payload.get("limit"),
            exclude_selected_contract=bool(plan_payload.get("exclude_selected_contract", False)),
            has_location_phrase=bool(plan_payload.get("has_location_phrase", False)),
            has_unresolved_location_hint=bool(plan_payload.get("has_unresolved_location_hint", False)),
            is_follow_up=bool(plan_payload.get("is_follow_up", False)),
            analysis_type=str(plan_payload.get("analysis_type", "") or ""),
        )

        with (
            patch("chat.ensure_chat_thread"),
            patch("chat.plan_message", return_value=plan),
            patch("chat.set_thread_plan"),
            patch("chat.save_chat_message", side_effect=capture_save),
            patch("chat.stream_agent", return_value=iter(streamed_events)),
        ):
            payloads = list(chat.event_stream("show flood control projects in leyte", "stream-thread"))

        return [_sse_event(payload) for payload in payloads], saved_messages

    def _run_direct_event_stream(
        self,
        *,
        detected_intent: str,
        direct_reply: str,
        result_state: dict[str, object] | None = None,
        response_source: str = "structured",
    ):
        saved_messages = []

        def capture_save(*args, **kwargs):
            saved_messages.append({"args": args, "kwargs": kwargs})

        plan = QueryPlan(intent=detected_intent)

        with (
            patch("chat.ensure_chat_thread"),
            patch("chat.plan_message", return_value=plan),
            patch("chat.set_thread_plan"),
            patch("chat.save_chat_message", side_effect=capture_save),
            patch(
                "chat._run_direct_tool_turn",
                return_value=(direct_reply, result_state, response_source),
            ),
            patch("chat.stream_agent", side_effect=AssertionError("stream_agent should not be called")),
        ):
            payloads = list(chat.event_stream("show flood control projects in leyte", "stream-thread"))

        return [_sse_event(payload) for payload in payloads], saved_messages

    def test_browse_turn_uses_direct_tool_path(self) -> None:
        events, saved_messages = self._run_direct_event_stream(
            detected_intent="browse",
            direct_reply=self.EXPECTED_REPLY,
            result_state=self.RESULT_STATE,
        )

        self.assertEqual(events[0]["type"], "result_state")
        self.assertEqual(events[-1]["type"], "done")

        streamed_text = "".join(
            str(event["content"])
            for event in events
            if event["type"] == "token"
        )
        self.assertEqual(streamed_text, self.EXPECTED_REPLY)
        self.assertEqual(saved_messages[1]["args"][2], self.EXPECTED_REPLY)
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["response_source"], "structured")
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["execution_path"], "direct_tool")

    def test_lookup_turn_uses_direct_tool_path(self) -> None:
        detail_reply = (
            "CONSTRUCTION/IMPROVEMENT OF SAN JOAQUIN SHORELINE PROTECTION, SAN JOAQUIN, ILOILO (21GF0024)\n"
            "• Contractor: ABRIGHT BUILDERS CORPORATION (46487)\n\n"
            "Would you like to dive deeper into this contract, compare other projects by the same contractor, or look at similar projects in the area?"
        )
        events, saved_messages = self._run_direct_event_stream(
            detected_intent="lookup",
            direct_reply=detail_reply,
            result_state={
                "result_kind": "contract_detail",
                "displayed_sources": [{"contractId": "21GF0024"}],
            },
        )

        streamed_text = "".join(str(event["content"]) for event in events if event["type"] == "token")
        self.assertIn("21GF0024", streamed_text)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["execution_path"], "direct_tool")

    def test_availability_turn_uses_direct_tool_path(self) -> None:
        direct_reply = (
            "Availability Check [Region: Region XI | Status: On-Going | Category: road]:\n"
            "- Matching Contracts: 7\n"
            "- Available: Yes\n"
            "- Use a listing request if you want to browse matching rows.\n"
        )
        events, saved_messages = self._run_direct_event_stream(
            detected_intent="availability",
            direct_reply=direct_reply,
            result_state={
                "result_kind": "contract_set",
                "count": 7,
                "filters": {"region": "Region XI", "status": "On-Going", "category": "road"},
            },
            response_source="tool",
        )

        streamed_text = "".join(str(event["content"]) for event in events if event["type"] == "token")
        self.assertIn("Matching Contracts: 7", streamed_text)
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["execution_path"], "direct_tool")

    def test_stats_turn_uses_direct_tool_path(self) -> None:
        direct_reply = (
            "Statistics Summary [Province: Leyte]:\n"
            "- Total Contracts Matched: 12\n"
            "- Combined Budget: PHP 10,000,000.00\n"
        )
        events, saved_messages = self._run_direct_event_stream(
            detected_intent="stats",
            direct_reply=direct_reply,
            result_state={
                "result_kind": "contract_set",
                "count": 12,
                "filters": {"province": "Leyte"},
            },
            response_source="tool",
        )

        streamed_text = "".join(str(event["content"]) for event in events if event["type"] == "token")
        self.assertIn("Total Contracts Matched: 12", streamed_text)
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["execution_path"], "direct_tool")

    def test_clarify_turn_uses_direct_tool_path(self) -> None:
        events, saved_messages = self._run_direct_event_stream(
            detected_intent="clarify",
            direct_reply="Which contractor are you referring to?",
            response_source="tool",
        )

        streamed_text = "".join(str(event["content"]) for event in events if event["type"] == "token")
        self.assertEqual(streamed_text, "Which contractor are you referring to?")
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["execution_path"], "direct_tool")

    def test_search_turn_uses_direct_tool_path(self) -> None:
        events, saved_messages = self._run_direct_event_stream(
            detected_intent="search",
            direct_reply=self.EXPECTED_REPLY,
            result_state=self.RESULT_STATE,
        )

        streamed_text = "".join(str(event["content"]) for event in events if event["type"] == "token")
        self.assertEqual(streamed_text, self.EXPECTED_REPLY)
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["execution_path"], "direct_tool")

    def test_compare_turn_uses_direct_compare_path(self) -> None:
        compare_reply = (
            "Comparing these contracts:\n"
            "1. CONSTRUCTION/IMPROVEMENT OF SAN JOAQUIN SHORELINE PROTECTION, SAN JOAQUIN, ILOILO (21GF0024)\n"
            "• Budget: PHP 5,929,936.50\n\n"
            "Observable comparison points:\n"
            "• The higher listed budget is for CONSTRUCTION OF SLOPE PROTECTION STRUCTURE (21GJ0002).\n"
        )
        compare_state = {
            "result_kind": "contract_compare",
            "intent": "compare",
            "comparison_query": "Compare these three projects.",
            "displayed_contract_ids": ["21GF0024", "21GJ0002", "24GF0054"],
            "displayed_sources": [
                {
                    "description": "CONSTRUCTION/IMPROVEMENT OF SAN JOAQUIN SHORELINE PROTECTION, SAN JOAQUIN, ILOILO",
                    "contractId": "21GF0024",
                    "budget": 5929936.5,
                    "awardAmount": 5929936.5,
                    "progress": 100,
                    "status": "Completed",
                    "contractor": "ABRIGHT BUILDERS CORPORATION (46487)",
                    "startDate": "2021-04-06",
                    "completionDate": "2021-06-15",
                    "contractDuration": "70 day(s)",
                    "documentLinks": {},
                    "components": [],
                },
                {
                    "description": "CONSTRUCTION OF SLOPE PROTECTION STRUCTURE",
                    "contractId": "21GJ0002",
                    "budget": 14132569,
                    "awardAmount": 14132569,
                    "progress": 100,
                    "status": "Completed",
                    "contractor": "BOAZ AND JACHIN CONSTRUCTION SUPPLY & SERVICES",
                    "startDate": "2021-01-01",
                    "completionDate": "2021-06-01",
                    "contractDuration": "151 day(s)",
                    "documentLinks": {"advertisement": "https://example.com"},
                    "components": [],
                },
            ],
        }
        events, saved_messages = self._run_direct_event_stream(
            detected_intent="compare",
            direct_reply=compare_reply,
            result_state=compare_state,
        )

        streamed_text = "".join(str(event["content"]) for event in events if event["type"] == "token")
        self.assertIn("Comparing these contracts:", streamed_text)
        self.assertNotIn("2. CONSTRUCTION OF SLOPE PROTECTION STRUCTURE", streamed_text[:120])
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["execution_path"], "direct_compare")
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["response_source"], "structured")

    def test_direct_tool_no_results_stays_non_llm(self) -> None:
        events, saved_messages = self._run_direct_event_stream(
            detected_intent="browse",
            direct_reply="No contracts found matching filters: flood control projects in Leyte",
            result_state={
                "result_kind": "contract_set",
                "count": 0,
                "filters": {"province": "Leyte", "category": "flood control"},
            },
            response_source="tool",
        )

        streamed_text = "".join(str(event["content"]) for event in events if event["type"] == "token")
        self.assertEqual(
            streamed_text,
            "No contracts found matching filters: flood control projects in Leyte",
        )
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["execution_path"], "direct_tool")
        self.assertEqual(events[-1]["type"], "done")

    def test_event_stream_does_not_duplicate_model_next_step_prompt(self) -> None:
        events, saved_messages = self._run_event_stream(
            [
                {"type": "token", "content": "Here are the details.\n\nWould you like to:"},
                {"type": "done"},
            ],
            plan_snapshot={"intent": "chat"},
            detected_intent="chat",
        )

        streamed_text = "".join(
            str(event["content"])
            for event in events
            if event["type"] == "token"
        )
        self.assertEqual(streamed_text, "Here are the details.\n\nWould you like to:")
        self.assertEqual(saved_messages[1]["args"][2], streamed_text)
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["response_source"], "llm")

    def test_event_stream_does_not_append_next_step_for_clarifying_questions(self) -> None:
        events, saved_messages = self._run_direct_event_stream(
            detected_intent="clarify",
            direct_reply="Which contractor are you referring to?",
            response_source="tool",
        )

        streamed_text = "".join(
            str(event["content"])
            for event in events
            if event["type"] == "token"
        )
        self.assertEqual(streamed_text, "Which contractor are you referring to?")
        self.assertEqual(saved_messages[1]["args"][2], streamed_text)

    def test_event_stream_strips_raw_tool_call_json_from_assistant_text(self) -> None:
        events, saved_messages = self._run_event_stream(
            [
                {
                    "type": "token",
                    "content": "{\"name\": \"get_contract_statistics\", \"parameters\": {\"query\": \"contractor=ABRIGHT BUILDERS CORPORATION\"}}",
                },
                {"type": "done"},
            ],
            plan_snapshot={"intent": "chat"},
            detected_intent="chat",
        )

        self.assertEqual([event["type"] for event in events], ["done"])
        self.assertEqual(len(saved_messages), 1)


class AgentStreamingTextTests(unittest.TestCase):
    def test_extract_stream_text_from_plain_string_chunk(self) -> None:
        message = AIMessageChunk(content="hello")
        self.assertEqual(agent._extract_stream_text(message), "hello")

    def test_extract_stream_text_from_content_blocks(self) -> None:
        message = AIMessageChunk(
            content=[
                {"type": "text", "text": "hello"},
                {"type": "tool_call_chunk", "name": "filter_contracts"},
                {"type": "text", "text": " world"},
            ]
        )
        self.assertEqual(agent._extract_stream_text(message), "hello world")


if __name__ == "__main__":
    unittest.main()
