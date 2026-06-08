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

    def test_event_stream_replaces_generic_llm_text_with_structured_contract_rows(self) -> None:
        events, saved_messages = self._run_event_stream(
            [
                {"type": "result_state", "content": self.RESULT_STATE},
                {"type": "token", "content": "I found matching projects. Want the full list?"},
                {"type": "done"},
            ]
        )

        self.assertEqual(events[0]["type"], "result_state")
        self.assertEqual(events[-1]["type"], "done")

        streamed_text = "".join(
            str(event["content"])
            for event in events
            if event["type"] == "token"
        )
        self.assertEqual(streamed_text, self.EXPECTED_REPLY)
        self.assertNotIn("Description:", streamed_text)
        self.assertEqual(saved_messages[1]["args"][2], self.EXPECTED_REPLY)

    def test_event_stream_emits_structured_reply_when_llm_is_empty(self) -> None:
        events, saved_messages = self._run_event_stream(
            [
                {"type": "result_state", "content": self.RESULT_STATE},
                {"type": "done"},
            ]
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

    def test_event_stream_does_not_append_next_step_for_clarifying_questions(self) -> None:
        events, saved_messages = self._run_event_stream(
            [
                {"type": "token", "content": "Which contractor are you referring to?"},
                {"type": "done"},
            ],
            expanded_query="Ask clarifying question: Which contractor are you referring to?",
            plan_snapshot={"intent": "clarify"},
            detected_intent="clarify",
        )

        streamed_text = "".join(
            str(event["content"])
            for event in events
            if event["type"] == "token"
        )
        self.assertEqual(streamed_text, "Which contractor are you referring to?")
        self.assertEqual(saved_messages[1]["args"][2], streamed_text)

    def test_event_stream_replaces_generic_llm_text_with_structured_contract_detail(self) -> None:
        detail_state = {
            "result_kind": "contract_detail",
            "intent": "lookup",
            "count": 1,
            "displayed_contract_ids": ["21GF0024"],
            "displayed_sources": [
                {
                    "description": "CONSTRUCTION/IMPROVEMENT OF SAN JOAQUIN SHORELINE PROTECTION, SAN JOAQUIN, ILOILO",
                    "contractId": "21GF0024",
                    "contractor": "ABRIGHT BUILDERS CORPORATION (46487)",
                    "region": "Region VI",
                    "province": "Iloilo 1st DEO",
                    "budget": 5929936.5,
                    "amountPaid": 0,
                    "awardAmount": 5929936.5,
                    "awardToBudgetRatio": 100.0,
                    "progress": 100,
                    "status": "Completed",
                    "category": "Flood Control and Drainage",
                    "infraYear": 2021,
                    "programName": "Regular Infra",
                    "sourceOfFunds": "Regular Infra - GAA 2021 LP",
                    "advertisementDate": "2020-11-18",
                    "bidSubmissionDeadline": "2020-12-10",
                    "startDate": "2021-04-06",
                    "completionDate": "2021-06-15",
                    "expiryDate": "2021-07-05",
                    "contractDuration": "70 day(s)",
                    "documentLinks": {
                        "advertisement": "https://dcs.infrawatch.ph/advertisement/21GF0024/21GF0024.rar",
                        "noticeOfAward": "https://dcs.infrawatch.ph/notice_of_award/21GF0024/21GF0024_-_ABRIGHT_-_NOA.pdf",
                        "noticeToProceed": "https://dcs.infrawatch.ph/notice_to_proceed/21GF0024/21GF0024_-_ABRIGHT_-_CONTRACT.PDF",
                        "contractAgreement": "https://dcs.infrawatch.ph/contract_agreement/21GF0024/21GF0024_-_ABRIGHT_-_CONTRACT.PDF",
                    },
                    "components": [
                        {
                            "componentId": "P00549905VS-CW1",
                            "typeOfWork": "Construction of Flood Mitigation Structure",
                            "infraType": "Flood Control and Drainage",
                            "description": "Construction of Flood Mitigation Structure - Construction / Improvement of San Joaquin Shoreline Protection, San Joaquin, Iloilo",
                            "region": "Region VI",
                            "province": "ILOILO",
                        }
                    ],
                    "dbFields": {
                        "contractId": "21GF0024",
                        "description": "CONSTRUCTION/IMPROVEMENT OF SAN JOAQUIN SHORELINE PROTECTION, SAN JOAQUIN, ILOILO",
                        "category": "Flood Control and Drainage",
                        "status": "Completed",
                        "budget": 5929936.5,
                        "amountPaid": 0,
                        "awardAmount": 5929936.5,
                        "awardToBudgetRatio": 100.0,
                        "progress": 100,
                        "region": "Region VI",
                        "province": "Iloilo 1st DEO",
                        "latitude": 10.5884489,
                        "longitude": 122.1433233,
                        "contractor": "ABRIGHT BUILDERS CORPORATION (46487)",
                        "advertisementDate": "2020-11-18",
                        "bidSubmissionDeadline": "2020-12-10",
                        "startDate": "2021-04-06",
                        "completionDate": "2021-06-15",
                        "expiryDate": "2021-07-05",
                        "infraYear": 2021,
                        "programName": "Regular Infra",
                        "sourceOfFunds": "Regular Infra - GAA 2021 LP",
                        "contractDuration": "70 day(s)",
                    },
                    "rawJson": {
                        "description": "CONSTRUCTION/IMPROVEMENT OF SAN JOAQUIN SHORELINE PROTECTION, SAN JOAQUIN, ILOILO",
                        "links": {
                            "advertisement": "https://dcs.infrawatch.ph/advertisement/21GF0024/21GF0024.rar"
                        },
                    },
                }
            ],
        }

        events, saved_messages = self._run_event_stream(
            [
                {"type": "result_state", "content": detail_state},
                {"type": "token", "content": "Here is a brief summary."},
                {"type": "done"},
            ],
            expanded_query="Lookup contract 21GF0024",
            plan_snapshot={"intent": "lookup"},
            detected_intent="lookup",
        )

        streamed_text = "".join(
            str(event["content"])
            for event in events
            if event["type"] == "token"
        )
        self.assertIn(
            "CONSTRUCTION/IMPROVEMENT OF SAN JOAQUIN SHORELINE PROTECTION, SAN JOAQUIN, ILOILO (21GF0024)",
            streamed_text,
        )
        self.assertIn("• Award Amount: PHP 5,929,936.50", streamed_text)
        self.assertIn(
            "• Document Links: advertisement, noticeOfAward, noticeToProceed, contractAgreement",
            streamed_text,
        )
        self.assertIn(
            "Open the contract drawer to view more details.",
            streamed_text,
        )
        self.assertIn(
            "Would you like to dive deeper into this contract, compare other projects by the same contractor, or look at similar projects in the area?",
            streamed_text,
        )
        self.assertNotIn("Here is a brief summary.", streamed_text)
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
