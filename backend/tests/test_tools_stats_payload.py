import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from test_tools_plan_execution import _call_tool, _load_tools_module


class FakeCursor:
    def __init__(self) -> None:
        self.result = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, query: str, params=None) -> None:
        sql = " ".join(query.lower().split())
        if "group by status" in sql:
            self.result = [("Completed", 2), ("On-Going", 1)]
        elif "group by region" in sql:
            self.result = [("Region VII", 3)]
        elif "group by province" in sql:
            self.result = [("Cebu", 2), ("Bohol", 1)]
        elif "sum(budget)" in sql:
            self.result = [(3000.0,)]
        elif "sum(award_amount)" in sql:
            self.result = [(2700.0,)]
        elif "avg(progress)" in sql:
            self.result = [(75.0,)]
        elif "select contract_id, description, budget, province" in sql:
            self.result = [
                (
                    "A003",
                    "Largest project",
                    1500.0,
                    "Bohol",
                    "Region VII",
                    "Completed",
                    "Contractor C",
                    100.0,
                    "Road",
                    "2024",
                    "Program C",
                    "2025-03-30",
                ),
                (
                    "A002",
                    "Second project",
                    1000.0,
                    "Cebu",
                    "Region VII",
                    "On-Going",
                    "Contractor B",
                    75.0,
                    "Bridge",
                    "2023",
                    "Program B",
                    "2024-12-15",
                ),
                (
                    "A001",
                    "Third project",
                    500.0,
                    "Cebu",
                    "Region VII",
                    "Completed",
                    "Contractor A",
                    50.0,
                    "Road",
                    "2022",
                    "Program A",
                    None,
                ),
            ]
        elif "select contract_id" in sql:
            self.result = [("A001",), ("A002",), ("A003",)]
        elif "select count(*) from contracts" in sql:
            self.result = [(3,)]
        else:
            raise AssertionError(f"Unexpected SQL: {query}")

    def fetchone(self):
        return self.result[0]

    def fetchall(self):
        return self.result


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_obj = FakeCursor()
        self.closed = False

    def cursor(self, *args, **kwargs):
        return self.cursor_obj

    def close(self) -> None:
        self.closed = True


class FakePsycopg2:
    def __init__(self) -> None:
        self.connect_calls = 0
        self.connection = FakeConnection()

    def connect(self, *args, **kwargs):
        self.connect_calls += 1
        return self.connection


class ToolsStatsPayloadTests(unittest.TestCase):
    def test_compute_stats_payload_returns_structured_breakdowns(self) -> None:
        tools_mod = _load_tools_module()
        stats_mod = sys.modules["features.chat.tools.stats"]
        fake_psycopg2 = FakePsycopg2()

        with patch.object(stats_mod, "connect", side_effect=fake_psycopg2.connect):
            payload = stats_mod._compute_stats_payload(
                {"province": "Cebu"},
                is_availability_query=False,
            )

        required_keys = {
            "total_contracts",
            "total_budget",
            "total_award_amount",
            "avg_progress",
            "award_to_budget_ratio",
            "status_breakdown",
            "region_breakdown",
            "province_breakdown",
            "applied_filters",
            "scope_label",
            "is_availability_query",
            "contract_rows",
            "has_more_contracts",
        }
        self.assertTrue(required_keys.issubset(payload.keys()))
        self.assertEqual(payload["total_contracts"], 3)
        self.assertEqual(payload["province_breakdown"][0], {"province": "Cebu", "count": 2})
        self.assertEqual(payload["contract_rows"][0]["contract_id"], "A003")
        self.assertEqual(payload["contract_rows"][0]["budget"], 1500.0)
        self.assertFalse(payload["has_more_contracts"])
        self.assertIsInstance(payload["province_breakdown"], list)
        self.assertEqual(fake_psycopg2.connect_calls, 1)

        formatted = stats_mod._format_stats_text(payload)
        self.assertIn("Total Contracts", formatted)
        self.assertIn("Status Breakdown", formatted)

    def test_compute_stats_payload_records_displayed_sources_from_contract_rows(self) -> None:
        tools_mod = _load_tools_module()
        stats_mod = sys.modules["features.chat.tools.stats"]
        fake_psycopg2 = FakePsycopg2()
        captured_state = {}

        def capture_result_state(payload):
            captured_state["payload"] = payload

        with (
            patch.object(stats_mod, "connect", side_effect=fake_psycopg2.connect),
            patch.object(stats_mod, "_record_result_state", side_effect=capture_result_state),
        ):
            stats_mod._compute_stats_payload(
                {"province": "Cebu"},
                is_availability_query=False,
            )

        result_state = captured_state["payload"]
        self.assertEqual(result_state["displayed_contract_ids"], ["A003", "A002", "A001"])
        self.assertEqual(result_state["displayed_sources"][0]["contractId"], "A003")
        self.assertEqual(result_state["displayed_sources"][0]["programName"], "Program C")
        self.assertEqual(result_state["displayed_sources"][0]["completionDate"], "2025-03-30")

    def test_compute_stats_payload_marks_availability_queries(self) -> None:
        tools_mod = _load_tools_module()
        stats_mod = sys.modules["features.chat.tools.stats"]
        fake_psycopg2 = FakePsycopg2()

        with patch.object(stats_mod, "connect", side_effect=fake_psycopg2.connect):
            payload = stats_mod._compute_stats_payload(
                {"region": "Region VII"},
                is_availability_query=True,
            )

        self.assertTrue(payload["is_availability_query"])
        self.assertIsInstance(payload["province_breakdown"], list)
        self.assertIn("Availability Check", stats_mod._format_stats_text(payload))
        self.assertEqual(fake_psycopg2.connect_calls, 1)

    def test_public_statistics_tool_still_returns_formatted_string(self) -> None:
        tools_mod = _load_tools_module()
        stats_mod = sys.modules["features.chat.tools.stats"]

        with (
            patch.object(stats_mod, "parse_filter_string", return_value={"province": "Cebu"}),
            patch.object(stats_mod, "_compute_stats_payload", return_value={"ok": True}) as compute_mock,
            patch.object(stats_mod, "_format_stats_text", return_value="formatted stats") as format_mock,
        ):
            output = _call_tool(
                stats_mod.get_contract_statistics,
                "Calculate metrics where province=Cebu",
            )

        self.assertEqual(output, "formatted stats")
        compute_mock.assert_called_once_with(
            {"province": "Cebu"},
            is_availability_query=False,
        )
        format_mock.assert_called_once_with({"ok": True})


if __name__ == "__main__":
    unittest.main()
