import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from chat import router
from chat_memory import ensure_chat_thread, save_chat_message
from query_expand import query_expand
from query_scope import clear_thread_cache, clear_thread_scope


class DurableChatMemoryTests(unittest.TestCase):
    def tearDown(self) -> None:
        for thread_id in (
            "durable-follow-up",
            "durable-chat-turn",
            "durable-history-recovery",
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
        ensure_chat_thread("history-api-a", user_id="user-a")
        save_chat_message("history-api-a", "user", "show projects in ncr", user_id="user-a")
        ensure_chat_thread("history-api-b", user_id="user-b")
        save_chat_message("history-api-b", "user", "show projects in region vi", user_id="user-b")

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        threads_response = client.get("/chat/threads", params={"user_id": "user-a"})
        self.assertEqual(threads_response.status_code, 200)
        threads = threads_response.json()["threads"]
        self.assertEqual(len(threads), 1)
        self.assertEqual(threads[0]["thread_id"], "history-api-a")

        messages_response = client.get(
            "/chat/threads/history-api-a/messages",
            params={"user_id": "user-a"},
        )
        self.assertEqual(messages_response.status_code, 200)
        messages = messages_response.json()["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "show projects in ncr")


if __name__ == "__main__":
    unittest.main()
