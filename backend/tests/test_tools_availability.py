import os
import unittest

import psycopg2

from tools import filter_contracts, get_contract_statistics

PG_DSN: str = os.environ.get("PG_DSN") or (
    f"host={os.environ.get('POSTGRES_HOST')} "
    f"port={os.environ.get('POSTGRES_PORT')} "
    f"dbname={os.environ.get('POSTGRES_DB')} "
    f"user={os.environ.get('POSTGRES_USER')} "
    f"password={os.environ.get('POSTGRES_PASSWORD')}"
)


def _count_matches(where_clause: str, params: tuple[object, ...]) -> int:
    conn = psycopg2.connect(PG_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM contracts WHERE {where_clause}", params)
            return int(cur.fetchone()[0])
    finally:
        conn.close()


class AvailabilityToolOutputTests(unittest.TestCase):
    def test_region_xi_road_availability_is_count_only(self) -> None:
        expected = _count_matches(
            "region ILIKE %s AND status ILIKE %s AND (description ILIKE %s OR category ILIKE %s)",
            ("%Region XI%", "%On-Going%", "%road%", "%road%"),
        )
        output = get_contract_statistics.invoke(
            "Check availability where region=Region XI AND status=On-Going AND category=road"
        )

        self.assertIn(
            "Availability Check [Region: Region XI | Status: On-Going | Category: road]:",
            output,
        )
        self.assertIn(f"- Matching Contracts: {expected:,}", output)
        self.assertIn("- Available: Yes" if expected > 0 else "- Available: No", output)
        self.assertNotIn("Combined Budget", output)
        self.assertNotIn("Award Amount", output)
        self.assertNotIn("Average Progress", output)
        self.assertNotIn("Status Breakdown", output)

    def test_stats_output_uses_structured_prefix(self) -> None:
        expected = _count_matches(
            "region ILIKE %s AND status ILIKE %s AND (description ILIKE %s OR category ILIKE %s)",
            (
                "%Cordillera Administrative Region%",
                "%Completed%",
                "%bridge%",
                "%bridge%",
            ),
        )
        output = get_contract_statistics.invoke(
            "Calculate metrics where region=Cordillera Administrative Region AND status=Completed AND category=bridge"
        )

        self.assertIn(
            "Statistics Summary [Region: Cordillera Administrative Region | Status: Completed | Category: bridge]:",
            output,
        )
        self.assertIn(f"- Total Contracts Matched: {expected:,}", output)
        self.assertIn("Combined Budget", output)
        self.assertIn("Average Progress", output)

    def test_filter_output_uses_national_scope_filter(self) -> None:
        expected = _count_matches(
            "region ILIKE %s",
            ("%National Capital Region%",),
        )
        output = filter_contracts.invoke(
            "Filter contracts where region=National Capital Region"
        )

        self.assertIn(f"showing 10 of {expected:,} total matches", output)
        self.assertIn("(capped match window, not ranked)", output)
        self.assertIn("Source rows (ordered by contract ID", output)
        self.assertNotIn("sorted by budget descending", output)


if __name__ == "__main__":
    unittest.main()
