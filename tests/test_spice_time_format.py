import unittest
from datetime import datetime, timedelta, timezone

from backend.spice_time import format_spice_utc


class SpiceTimeFormatTests(unittest.TestCase):
    def test_ist_midnight_converts_to_previous_day_utc(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        value = datetime(2026, 11, 14, 0, 0, 0, tzinfo=ist)
        self.assertEqual(format_spice_utc(value), "2026 NOV 13 18:30:00 UTC")

    def test_fractional_seconds_are_preserved(self):
        value = datetime(2026, 11, 13, 18, 30, 0, 125000, tzinfo=timezone.utc)
        self.assertEqual(format_spice_utc(value), "2026 NOV 13 18:30:00.125 UTC")

    def test_naive_datetime_is_rejected(self):
        value = datetime(2026, 11, 13, 18, 30, 0)
        with self.assertRaises(ValueError):
            format_spice_utc(value)


if __name__ == "__main__":
    unittest.main()
