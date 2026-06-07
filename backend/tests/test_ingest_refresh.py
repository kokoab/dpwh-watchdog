import unittest
from unittest.mock import Mock

from ingest_pipeline import compile_rag_chunk_text, filter_existing_contracts


class IngestRefreshTests(unittest.TestCase):
    def test_chunk_text_no_longer_contains_amount_paid(self) -> None:
        chunk_text = compile_rag_chunk_text(
            {
                "contractId": "TEST-1",
                "description": "Road project",
                "budget": 1000,
                "amountPaid": 250,
                "location": {"region": "Region VIII"},
            },
            "TEST-1",
        )

        self.assertNotIn("Amount Paid:", chunk_text)

    def test_refresh_existing_bypasses_existing_id_filter(self) -> None:
        conn = Mock()
        records = [{"contract_id": "A"}, {"contract_id": "B"}]

        result = filter_existing_contracts(conn, records, refresh_existing=True)

        self.assertEqual(result, records)
        conn.cursor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
