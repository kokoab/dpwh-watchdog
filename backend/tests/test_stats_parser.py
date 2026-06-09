import sys
import unittest

sys.path.insert(0, "backend")

from filter_parser import parse_filter_string
from stats_parser import parse_stats_filters


class StatsParserTests(unittest.TestCase):
    def test_parse_stats_filters_keeps_year_window_fields(self) -> None:
        filters = parse_stats_filters(
            {
                "contractor": "TOPMOST DEVELOPMENT & MKTG. CORP.",
                "infra_year_start": "2022",
                "infra_year_end": "2026",
                "status": "Awarded",
            }
        )

        self.assertEqual(filters["contractor"], "TOPMOST DEVELOPMENT & MKTG. CORP.")
        self.assertEqual(filters["infra_year_start"], "2022")
        self.assertEqual(filters["infra_year_end"], "2026")
        self.assertEqual(filters["status"], "Awarded")

    def test_parse_filter_string_accepts_year_window_aliases(self) -> None:
        filters = parse_filter_string(
            "Filter contracts where contractor=TOPMOST DEVELOPMENT & MKTG. CORP. "
            "AND year_start=2022 AND year_end=2026"
        )

        self.assertEqual(filters["contractor"], "TOPMOST DEVELOPMENT & MKTG. CORP.")
        self.assertEqual(filters["infra_year_start"], "2022")
        self.assertEqual(filters["infra_year_end"], "2026")


if __name__ == "__main__":
    unittest.main()
