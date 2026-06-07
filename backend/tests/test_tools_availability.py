import unittest

from tools import filter_contracts, get_contract_statistics


class AvailabilityToolOutputTests(unittest.TestCase):
    def test_region_viii_availability_is_count_only(self) -> None:
        output = get_contract_statistics.invoke(
            "Calculate metrics for availability check: contracts in Region VIII"
        )

        self.assertIn("Availability Check [Region: Region VIII]:", output)
        self.assertIn("- Matching Contracts: 62", output)
        self.assertIn("- Available: Yes", output)
        self.assertNotIn("Combined Budget", output)
        self.assertNotIn("Award Amount", output)
        self.assertNotIn("Average Progress", output)
        self.assertNotIn("Status Breakdown", output)

    def test_region_viii_road_availability_is_count_only(self) -> None:
        output = get_contract_statistics.invoke(
            "Calculate metrics for availability check: road contracts in Region VIII"
        )

        self.assertIn(
            "Availability Check [Region: Region VIII | Category: road]:", output
        )
        self.assertIn("- Matching Contracts: 30", output)
        self.assertIn("- Available: Yes", output)
        self.assertNotIn("Combined Budget", output)

    def test_filter_output_uses_neutral_framing(self) -> None:
        output = filter_contracts.invoke("Filter contracts where region=Region VIII")

        self.assertIn("showing 10 of 62 total matches", output)
        self.assertIn("(capped match window, not ranked)", output)
        self.assertIn("Source rows (ordered by contract ID", output)
        self.assertIn("[16I00019]", output)
        self.assertNotIn("sorted by budget descending", output)
        self.assertNotIn("NETWORK DEVELOPMENT - CONSTRUCTION OF BY-PASS", output)


if __name__ == "__main__":
    unittest.main()
