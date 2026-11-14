from __future__ import annotations
import unittest
from pathlib import Path

from backend.dashboard_data import snapshot_slider_bounds


class AppRegressionTests(unittest.TestCase):
    def test_snapshot_slider_is_disabled_for_single_sample_ist_day(self):
        # The app consumes this helper before creating a Streamlit slider.
        self.assertIsNone(snapshot_slider_bounds(1))

    def test_snapshot_slider_is_available_for_multi_sample_days(self):
        self.assertEqual(snapshot_slider_bounds(288), (0, 287))


if __name__ == "__main__":
    unittest.main()
