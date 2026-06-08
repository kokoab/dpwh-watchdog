import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessageChunk

import agent
import chat


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
            }
        ],
    }
    EXPECTED_REPLY = (
        "1. Construction of drainage canal (ABC123)\n"
        "• Contractor: RJIR Enterprises\n"
        "• Status: Completed\n"
        "• Budget: PHP 950,002\n\n"
        "Would you like to dive deeper into this contract, compare other projects by the same contractor, or look at similar projects in the area?"
    )

    def _run_event_stream(
        self,
        streamed_events: list[dict[str, object]],
        *,
        expanded_query: str = "Filter contracts where province=Leyte AND category=flood control",
        plan_snapshot: dict[str, object] | None = None,
        detected_intent: str = "browse",
    ):
        saved_messages = []

        def capture_save(*args, **kwargs):
            saved_messages.append({"args": args, "kwargs": kwargs})

        with (
            patch("chat.ensure_chat_thread"),
            patch("chat.query_expand", return_value=expanded_query),
            patch("chat.log_query_expansion"),
            patch("chat.get_thread_plan", return_value=plan_snapshot or {"intent": detected_intent}),
            patch("chat.detect_intent_from_expanded_query", return_value=detected_intent),
            patch("chat.save_chat_message", side_effect=capture_save),
            patch("chat.stream_agent", return_value=iter(streamed_events)),
        ):
            payloads = list(chat.event_stream("show flood control projects in leyte", "stream-thread"))

        return [_sse_event(payload) for payload in payloads], saved_messages

    def _run_direct_event_stream(
        self,
        *,
        expanded_query: str,
        detected_intent: str,
        direct_reply: str,
        result_state: dict[str, object] | None = None,
        response_source: str = "structured",
    ):
        saved_messages = []

        def capture_save(*args, **kwargs):
            saved_messages.append({"args": args, "kwargs": kwargs})

        with (
            patch("chat.ensure_chat_thread"),
            patch("chat.query_expand", return_value=expanded_query),
            patch("chat.log_query_expansion"),
            patch("chat.get_thread_plan", return_value={"intent": detected_intent}),
            patch("chat.detect_intent_from_expanded_query", return_value=detected_intent),
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
            expanded_query="Filter contracts where province=Leyte AND category=flood control",
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
            expanded_query="Lookup contract 21GF0024",
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
            expanded_query="Check availability where region=Region XI AND status=On-Going AND category=road",
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
            expanded_query="Calculate metrics where province=Leyte",
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
            expanded_query="Ask clarifying question: Which contractor are you referring to?",
            detected_intent="clarify",
            direct_reply="Which contractor are you referring to?",
            response_source="tool",
        )

        streamed_text = "".join(str(event["content"]) for event in events if event["type"] == "token")
        self.assertEqual(streamed_text, "Which contractor are you referring to?")
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["execution_path"], "direct_tool")

    def test_search_turn_uses_direct_tool_path(self) -> None:
        events, saved_messages = self._run_direct_event_stream(
            expanded_query="Find all contracts about flood control where province=Leyte",
            detected_intent="search",
            direct_reply=self.EXPECTED_REPLY,
            result_state=self.RESULT_STATE,
        )

        streamed_text = "".join(str(event["content"]) for event in events if event["type"] == "token")
        self.assertEqual(streamed_text, self.EXPECTED_REPLY)
        self.assertEqual(saved_messages[1]["kwargs"]["metadata"]["execution_path"], "direct_tool")

    def test_direct_tool_no_results_stays_non_llm(self) -> None:
        events, saved_messages = self._run_direct_event_stream(
            expanded_query="Filter contracts where province=Leyte AND category=flood control",
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
            ]
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
            expanded_query="Ask clarifying question: Which contractor are you referring to?",
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
            expanded_query="tell me something",
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
