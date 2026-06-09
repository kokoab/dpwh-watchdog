import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from test_tools_plan_execution import _load_tools_module


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
        fake_psycopg2 = FakePsycopg2()

        with patch.object(tools_mod, "_psycopg2", return_value=fake_psycopg2):
            payload = tools_mod._compute_stats_payload(
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
        }
        self.assertTrue(required_keys.issubset(payload.keys()))
        self.assertEqual(payload["total_contracts"], 3)
        self.assertEqual(payload["province_breakdown"][0], {"province": "Cebu", "count": 2})
        self.assertIsInstance(payload["province_breakdown"], list)
        self.assertEqual(fake_psycopg2.connect_calls, 1)

        formatted = tools_mod._format_stats_text(payload)
        self.assertIn("Total Contracts", formatted)
        self.assertIn("Status Breakdown", formatted)

    def test_compute_stats_payload_marks_availability_queries(self) -> None:
        tools_mod = _load_tools_module()
        fake_psycopg2 = FakePsycopg2()

        with patch.object(tools_mod, "_psycopg2", return_value=fake_psycopg2):
            payload = tools_mod._compute_stats_payload(
                {"region": "Region VII"},
                is_availability_query=True,
            )

        self.assertTrue(payload["is_availability_query"])
        self.assertIsInstance(payload["province_breakdown"], list)
        self.assertIn("Availability Check", tools_mod._format_stats_text(payload))
        self.assertEqual(fake_psycopg2.connect_calls, 1)

    def test_public_statistics_tool_still_returns_formatted_string(self) -> None:
        tools_mod = _load_tools_module()

        with (
            patch.object(tools_mod, "parse_filter_string", return_value={"province": "Cebu"}),
            patch.object(tools_mod, "_compute_stats_payload", return_value={"ok": True}) as compute_mock,
            patch.object(tools_mod, "_format_stats_text", return_value="formatted stats") as format_mock,
        ):
            output = tools_mod.get_contract_statistics(
                "Calculate metrics where province=Cebu"
            )

        self.assertEqual(output, "formatted stats")
        compute_mock.assert_called_once_with(
            {"province": "Cebu"},
            is_availability_query=False,
        )
        format_mock.assert_called_once_with({"ok": True})


if __name__ == "__main__":
    unittest.main()
