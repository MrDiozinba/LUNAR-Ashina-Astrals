from __future__ import annotations
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from backend.decision_support import build_delay_sweep, compare_horizon_profiles

ROOT = Path(__file__).resolve().parents[1]


class ComparisonColumnRegressionTests(unittest.TestCase):
    def test_comparison_returns_the_column_used_by_site_comparison_page(self):
        az = np.arange(0.05, 360.0, 0.1)
        one = pd.DataFrame({
            "azimuth_center_deg": az,
            "combined_horizon_elevation_deg": np.full(3600, 1.25),
        })
        five = pd.DataFrame({
            "azimuth_center_deg": az,
            "combined_horizon_elevation_deg": np.full(3600, 0.75),
        })
        difference, stats = compare_horizon_profiles(one, five)
        self.assertIn("horizon_1m_minus_5m_deg", difference.columns)
        self.assertTrue(np.allclose(difference["horizon_1m_minus_5m_deg"], 0.5))
        self.assertAlmostEqual(stats["median_deg"], 0.5)

    def test_app_uses_canonical_comparison_column(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('difference["horizon_1m_minus_5m_deg"]', source)
        self.assertNotIn('difference["horizon_depression_5m_vs_1m_deg"]', source)


class DelayGeometryContextTests(unittest.TestCase):
    @staticmethod
    def timeline(hours: int = 24) -> pd.DataFrame:
        utc = pd.date_range("2026-11-13T18:30:00Z", periods=hours * 12 + 1, freq="5min")
        n = len(utc)
        # Visibility is intentionally saturated while body geometry changes.
        sun_el = np.linspace(1.6, 1.2, n)
        earth_el = np.linspace(-5.3, -3.0, n)
        return pd.DataFrame({
            "timestamp_utc": utc,
            "timestamp_ist": utc.tz_convert("Asia/Kolkata"),
            "sun_visible_fraction_approx": np.ones(n),
            "earth_visible_fraction_approx": np.zeros(n),
            "sun_visibility_state": ["full"] * n,
            "earth_visibility_state": ["hidden"] * n,
            "sun_center_elevation_deg": sun_el,
            "sun_terrain_horizon_deg": np.full(n, -3.0),
            "sun_angular_semidiameter_deg": np.full(n, 0.27),
            "earth_center_elevation_deg": earth_el,
            "earth_terrain_horizon_deg": np.full(n, 2.0),
            "earth_angular_semidiameter_deg": np.full(n, 0.9),
        })

    def test_flat_visibility_metrics_include_nonflat_geometry_margins(self):
        sweep = build_delay_sweep(self.timeline(), 6, step_minutes=5)
        self.assertEqual(sweep["sun_mean_disk_fraction_pct"].nunique(), 1)
        self.assertEqual(sweep["earth_mean_disk_fraction_pct"].nunique(), 1)
        self.assertEqual(sweep["both_any_visible_pct"].nunique(), 1)
        self.assertIn("sun_min_lower_limb_clearance_deg", sweep.columns)
        self.assertIn("earth_max_upper_limb_clearance_deg", sweep.columns)
        self.assertGreater(sweep["sun_min_lower_limb_clearance_deg"].nunique(), 1)
        self.assertGreater(sweep["earth_max_upper_limb_clearance_deg"].nunique(), 1)
        self.assertGreater(sweep["earth_max_upper_limb_clearance_deg"].iloc[-1],
                           sweep["earth_max_upper_limb_clearance_deg"].iloc[0])

    def test_app_explains_flat_visibility_and_renders_geometry_margin_chart(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("The visibility curves are flat because every candidate window", source)
        self.assertIn('sweep["sun_min_lower_limb_clearance_deg"]', source)
        self.assertIn('sweep["earth_max_upper_limb_clearance_deg"]', source)


if __name__ == "__main__":
    unittest.main()
