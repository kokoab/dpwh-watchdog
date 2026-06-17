import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.dependencies import get_current_user
from auth.jwt import CurrentUser
from features.chat.router import router
from features.chat.memory import delete_thread_memory, ensure_chat_thread, save_chat_message
from contracts.query_expand import query_expand
from features.chat.agent.query_scope import clear_thread_cache, clear_thread_scope


class DurableChatMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._groq_patcher = patch(
            "langchain_groq.ChatGroq",
            side_effect=RuntimeError("force fallback planner"),
        )
        self._groq_patcher.start()

    def tearDown(self) -> None:
        self._groq_patcher.stop()
        for thread_id in (
            "durable-follow-up",
            "durable-chat-turn",
            "durable-history-recovery",
            "durable-compare-recovery",
            "history-api-a",
            "history-api-b",
        ):
            clear_thread_scope(thread_id)

    def test_follow_up_survives_cache_clear(self) -> None:
        thread_id = "durable-follow-up"
        query_expand("show road projects in region viii", thread_id=thread_id)

        clear_thread_cache(thread_id)

        expanded = query_expand("what about region vi?", thread_id=thread_id)
        self.assertEqual(
            expanded,
            "Filter contracts where region=Region VI AND category=road",
        )

    def test_chat_turn_does_not_erase_last_domain_plan(self) -> None:
        thread_id = "durable-chat-turn"
        query_expand("show road projects in region viii", thread_id=thread_id)
        query_expand("thanks", thread_id=thread_id)

        clear_thread_cache(thread_id)

        expanded = query_expand("what about region vi?", thread_id=thread_id)
        self.assertEqual(
            expanded,
            "Filter contracts where region=Region VI AND category=road",
        )

    def test_show_them_recovers_older_result_from_message_history(self) -> None:
        thread_id = "durable-history-recovery"
        ensure_chat_thread(thread_id)
        save_chat_message(
            thread_id,
            "user",
            "any ongoing road projects in region xi?",
            expanded_query="Check availability where region=Region XI AND status=On-Going AND category=road",
            intent="availability",
            metadata={
                "plan": {
                    "intent": "availability",
                    "filters": {
                        "region": "Region XI",
                        "status": "On-Going",
                        "category": "road",
                    },
                    "subject": "",
                    "lookup_value": "",
                    "limit": None,
                    "has_location_phrase": True,
                    "has_unresolved_location_hint": False,
                    "is_follow_up": False,
                }
            },
        )
        save_chat_message(
            thread_id,
            "assistant",
            "There are 7 matching contracts.",
            intent="availability",
            metadata={
                "result_state": {
                    "result_kind": "contract_set",
                    "intent": "availability",
                    "filters": {
                        "region": "Region XI",
                        "status": "On-Going",
                        "category": "road",
                    },
                    "count": 7,
                    "contract_ids": ["20L00044", "21LD0082"],
                    "displayed_contract_ids": ["20L00044", "21LD0082"],
                }
            },
        )

        expanded = query_expand("show them", thread_id=thread_id)
        self.assertEqual(
            expanded,
            "Filter contracts where region=Region XI AND status=On-Going AND category=road LIMIT 7",
        )

    def test_history_endpoints_filter_threads_by_user(self) -> None:
        delete_thread_memory("history-api-a", user_id="user-a")
        delete_thread_memory("history-api-b", user_id="user-b")

        ensure_chat_thread("history-api-a", user_id="user-a")
        save_chat_message("history-api-a", "user", "show projects in ncr", user_id="user-a")
        ensure_chat_thread("history-api-b", user_id="user-b")
        save_chat_message("history-api-b", "user", "show projects in region vi", user_id="user-b")

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: CurrentUser(
            id="user-a", email="user-a@example.com", role="user"
        )
        client = TestClient(app)

        threads_response = client.get("/chat/threads")
        self.assertEqual(threads_response.status_code, 200)
        threads = threads_response.json()["threads"]
        self.assertEqual(len(threads), 1)
        self.assertEqual(threads[0]["thread_id"], "history-api-a")

        messages_response = client.get(
            "/chat/threads/history-api-a/messages",
        )
        self.assertEqual(messages_response.status_code, 200)
        messages = messages_response.json()["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "show projects in ncr")

    def test_compare_these_three_recovers_result_context_from_history(self) -> None:
        thread_id = "durable-compare-recovery"
        ensure_chat_thread(thread_id)
        save_chat_message(
            thread_id,
            "user",
            "show me flood control projects in iloilo",
            expanded_query="Filter contracts where province=Iloilo AND category=flood control",
            intent="browse",
            metadata={
                "plan": {
                    "intent": "browse",
                    "filters": {
                        "province": "Iloilo",
                        "category": "flood control",
                    },
                    "subject": "",
                    "lookup_value": "",
                    "limit": None,
                    "has_location_phrase": True,
                    "has_unresolved_location_hint": False,
                    "is_follow_up": False,
                }
            },
        )
        save_chat_message(
            thread_id,
            "assistant",
            "Here are the matching contracts.",
            intent="browse",
            metadata={
                "result_state": {
                    "result_kind": "contract_set",
                    "intent": "browse",
                    "filters": {
                        "province": "Iloilo",
                        "category": "flood control",
                    },
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
                }
            },
        )

        expanded = query_expand(
            "Compare these three projects.",
            thread_id=thread_id,
        )

        self.assertEqual(
            expanded,
            "Compare contracts 21GF0024,21GJ0002,24GF0054: Compare these three projects.",
        )


if __name__ == "__main__":
    unittest.main()
