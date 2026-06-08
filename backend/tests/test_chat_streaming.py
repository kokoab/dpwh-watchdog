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

    def _run_event_stream(self, streamed_events: list[dict[str, object]]):
        saved_messages = []

        def capture_save(*args, **kwargs):
            saved_messages.append({"args": args, "kwargs": kwargs})

        with (
            patch("chat.ensure_chat_thread"),
            patch("chat.query_expand", return_value="Filter contracts where province=Leyte AND category=flood control"),
            patch("chat.log_query_expansion"),
            patch("chat.get_thread_plan", return_value={"intent": "browse"}),
            patch("chat.detect_intent_from_expanded_query", return_value="browse"),
            patch("chat.save_chat_message", side_effect=capture_save),
            patch("chat.stream_agent", return_value=iter(streamed_events)),
        ):
            payloads = list(chat.event_stream("show flood control projects in leyte", "stream-thread"))

        return [_sse_event(payload) for payload in payloads], saved_messages

    def test_event_stream_keeps_llm_tokens_after_result_state(self) -> None:
        events, saved_messages = self._run_event_stream(
            [
                {"type": "token", "content": "Here are the matching projects: "},
                {"type": "result_state", "content": self.RESULT_STATE},
                {"type": "token", "content": "1. Construction of drainage canal (ABC123). Want details?"},
                {"type": "done"},
            ]
        )

        self.assertEqual(
            [event["type"] for event in events],
            ["token", "result_state", "token", "done"],
        )
        self.assertEqual(
            [event["content"] for event in events if event["type"] == "token"],
            [
                "Here are the matching projects: ",
                "1. Construction of drainage canal (ABC123). Want details?",
            ],
        )
        self.assertEqual(saved_messages[1]["args"][2], (
            "Here are the matching projects: "
            "1. Construction of drainage canal (ABC123). Want details?"
        ))

    def test_event_stream_does_not_append_structured_fallback_when_contract_ids_are_missing(self) -> None:
        events, saved_messages = self._run_event_stream(
            [
                {"type": "token", "content": "I found matching projects. Want the full list?"},
                {"type": "result_state", "content": self.RESULT_STATE},
                {"type": "done"},
            ]
        )

        self.assertEqual([event["type"] for event in events[:2]], ["token", "result_state"])

        streamed_text = "".join(
            str(event["content"])
            for event in events
            if event["type"] == "token"
        )
        self.assertIn("I found matching projects. Want the full list?", streamed_text)
        self.assertNotIn("The matching flood control projects in Leyte are:", streamed_text)
        self.assertNotIn("1. Construction of drainage canal (ABC123)", streamed_text)
        self.assertEqual(saved_messages[1]["args"][2], streamed_text)

    def test_event_stream_does_not_append_structured_fallback_when_llm_is_empty(self) -> None:
        events, saved_messages = self._run_event_stream(
            [
                {"type": "result_state", "content": self.RESULT_STATE},
                {"type": "done"},
            ]
        )

        self.assertEqual([event["type"] for event in events], ["result_state", "done"])
        self.assertEqual(saved_messages[1]["args"][2], "")

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
