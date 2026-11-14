from __future__ import annotations
import unittest
import numpy as np
import pandas as pd

from backend.decision_support import (
    build_delay_sweep, classify_objective, compare_horizon_profiles,
    compare_timeline_visibility, longest_runs, sample_horizon_difference_at_azimuths,
    select_half_open_window, summarize_horizon_sectors, compare_timeline_to_flat_horizon,
)
from backend.dashboard_data import load_scientific_bundle


class RealScientificBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_scientific_bundle(".")

    def test_actual_range_coverage_and_cap_pinned_counts(self):
        s1 = summarize_horizon_sectors(self.bundle["horizons"][1], 45_000.0)
        s5 = summarize_horizon_sectors(self.bundle["horizons"][5], 45_000.0)
        self.assertEqual(int(s1["Range-limited azimuth bins"].sum()), 3600)
        self.assertEqual(int(s5["Range-limited azimuth bins"].sum()), 3600)
        self.assertEqual(int(s1["Combined cap-pinned bins"].sum()), 151)
        self.assertEqual(int(s5["Combined cap-pinned bins"].sum()), 225)
        self.assertEqual(int(s1["80 m controlling bins"].sum()), 767)
        self.assertEqual(int(s5["80 m controlling bins"].sum()), 1013)
        self.assertNotIn("Peak skyline (°)", s1.columns)

    def test_all_3600_range_limited_is_data_derived_not_hard_coded(self):
        h = self.bundle["horizons"][1].copy()
        h.loc[0, "regional_80m_coverage_end_m"] = 44_000.0
        summary = summarize_horizon_sectors(h, 45_000.0)
        self.assertEqual(int(summary["Range-limited azimuth bins"].sum()), 3599)
        self.assertEqual(int(summary["Coverage-short azimuth bins"].sum()), 1)

    def test_cap_pin_tolerance_is_explicit(self):
        h = self.bundle["horizons"][1].copy()
        pinned = h.index[h["combined_distance_at_horizon_m"] >= 44_999.5]
        self.assertGreaterEqual(len(pinned), 2)
        h.loc[pinned[0], "combined_distance_at_horizon_m"] = 44_999.9
        h.loc[pinned[1], "combined_distance_at_horizon_m"] = 44_999.0
        summary = summarize_horizon_sectors(h, 45_000.0, cap_pin_tolerance_m=0.5)
        self.assertEqual(int(summary["Combined cap-pinned bins"].sum()), 150)

    def test_body_track_weighted_observer_height_difference(self):
        sun = sample_horizon_difference_at_azimuths(
            self.bundle["horizons"][1], self.bundle["horizons"][5],
            self.bundle["timelines"][1]["sun_azimuth_deg"].to_numpy(),
        )
        earth = sample_horizon_difference_at_azimuths(
            self.bundle["horizons"][1], self.bundle["horizons"][5],
            self.bundle["timelines"][1]["earth_azimuth_deg"].to_numpy(),
        )
        self.assertAlmostEqual(sun["minimum_deg"], 0.0050752, places=4)
        self.assertAlmostEqual(earth["minimum_deg"], 2.53352, places=3)

    def test_flat_horizon_sensitivity_is_reported_for_this_interval(self):
        result = compare_timeline_to_flat_horizon(self.bundle["timelines"][1])
        self.assertEqual(result["sun"]["terrain_vs_flat_classification_changed_rows"], 0)
        self.assertEqual(result["earth"]["terrain_vs_flat_classification_changed_rows"], 0)
        self.assertEqual(result["sun"]["terrain_vs_flat_fraction_changed_rows"], 0)
        self.assertEqual(result["earth"]["terrain_vs_flat_fraction_changed_rows"], 0)

    def test_circular_horizon_interpolation_wraps_across_zero_degrees(self):
        az = np.arange(0.05, 360.0, 0.1)
        h1 = pd.DataFrame({"azimuth_center_deg": az, "combined_horizon_elevation_deg": np.zeros(len(az))})
        h5 = h1.copy()
        h1.loc[0, "combined_horizon_elevation_deg"] = 0.0
        h1.loc[len(h1) - 1, "combined_horizon_elevation_deg"] = 2.0
        result = sample_horizon_difference_at_azimuths(h1, h5, [359.99])
        self.assertAlmostEqual(result["mean_deg"], 1.2, places=6)

    def test_observer_height_comparison_is_rowwise(self):
        result = compare_timeline_visibility(self.bundle["timelines"][1], self.bundle["timelines"][5])
        self.assertEqual(result["samples"], 577)
        self.assertEqual(result["sun_state_different_rows"], 0)
        self.assertEqual(result["earth_state_different_rows"], 0)
        self.assertEqual(result["sun_fraction_different_rows"], 0)
        self.assertEqual(result["earth_fraction_different_rows"], 0)

    def test_actual_circular_interpolation_cross_file_join(self):
        # load_scientific_bundle already performs this on all 4 body tracks.
        self.assertTrue(all(self.bundle["hash_checks"].values()))


class DelaySweepBehaviourTests(unittest.TestCase):
    @staticmethod
    def timeline(hours=48):
        utc = pd.date_range("2026-11-13T18:30:00Z", periods=hours * 12 + 1, freq="5min")
        return pd.DataFrame({
            "timestamp_utc": utc,
            "timestamp_ist": utc.tz_convert("Asia/Kolkata"),
            "sun_visible_fraction_approx": np.ones(len(utc)),
            "earth_visible_fraction_approx": np.zeros(len(utc)),
            "sun_visibility_state": ["full"] * len(utc),
            "earth_visibility_state": ["hidden"] * len(utc),
            "sun_center_elevation_deg": np.linspace(1.6, 1.2, len(utc)),
            "sun_terrain_horizon_deg": np.full(len(utc), -3.0),
            "sun_angular_semidiameter_deg": np.full(len(utc), 0.27),
            "earth_center_elevation_deg": np.linspace(-5.3, -3.0, len(utc)),
            "earth_terrain_horizon_deg": np.full(len(utc), 2.0),
            "earth_angular_semidiameter_deg": np.full(len(utc), 0.9),
        })

    def test_last_complete_24_hour_start_is_included_not_overrun(self):
        frame = self.timeline()
        sweep = build_delay_sweep(frame, 24, step_minutes=5)
        self.assertEqual(len(sweep), 289)
        self.assertEqual(sweep.delay_minutes.iloc[-1], 1440)
        self.assertEqual(sweep.sample_count.iloc[-1], 288)
        self.assertEqual(sweep.window_end_ist.iloc[-1], pd.Timestamp("2026-11-16 00:00", tz="Asia/Kolkata"))
        chosen = select_half_open_window(frame, sweep.window_start_ist.iloc[-1], 24)
        self.assertEqual(len(chosen), 288)
        self.assertLess(chosen.timestamp_utc.max(), sweep.window_end_ist.iloc[-1].tz_convert("UTC"))

    def test_2_hour_step_is_two_hours_and_has_22_six_hour_windows(self):
        sweep = build_delay_sweep(self.timeline(), 6, step_hours=2)
        self.assertEqual(len(sweep), 22)
        self.assertEqual(sweep.delay_hours.tolist(), list(range(0, 43, 2)))

    def test_fractional_step_hours_are_rejected_not_truncated(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            build_delay_sweep(self.timeline(), 6, step_hours=1.5)

    def test_step_hours_and_step_minutes_cannot_both_be_supplied(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            build_delay_sweep(self.timeline(), 6, step_hours=1, step_minutes=5)

    def test_half_open_window_rejects_partial_window_past_last_timestamp(self):
        frame = self.timeline()
        last = frame.timestamp_utc.iloc[0] + pd.Timedelta(hours=24, minutes=5)
        with self.assertRaisesRegex(ValueError, "extends beyond"):
            select_half_open_window(frame, last, 24)

    def test_nontrivial_step_function_has_unique_hand_computed_optimum(self):
        frame = self.timeline()
        # In a 6 h / 72-sample window starting at 31 h, 70 rows are visible.
        # A 5-minute shift either way loses at least one additional visible row.
        begin = 31 * 12
        frame.loc[begin:begin + 71, "earth_visible_fraction_approx"] = 1.0
        frame.loc[begin:begin + 71, "earth_visibility_state"] = "full"
        frame.loc[[begin + 10, begin + 40], "earth_visible_fraction_approx"] = 0.0
        frame.loc[[begin + 10, begin + 40], "earth_visibility_state"] = "hidden"
        sweep = build_delay_sweep(frame, 6, step_minutes=5)
        row = sweep.loc[np.isclose(sweep.delay_hours, 31.0)].iloc[0]
        self.assertAlmostEqual(row.earth_any_visible_pct, 100.0 * 70 / 72, places=6)
        opt = classify_objective(sweep, "earth_any_visible_pct", materiality_pct=1.0, tolerance_pp=0.01)
        self.assertEqual(opt["status"], "unique")
        self.assertEqual(float(opt["best_rows"].iloc[0].delay_hours), 31.0)

    def test_half_percent_objective_is_unachieved_at_one_percent_materiality(self):
        sweep = pd.DataFrame({"objective": [0.5, 0.5, 0.25]})
        result = classify_objective(sweep, "objective", materiality_pct=1.0)
        self.assertEqual(result["status"], "unachieved")

    def test_one_percent_is_material_and_can_be_tied(self):
        sweep = pd.DataFrame({"objective": [1.0, 1.0, 0.5]})
        result = classify_objective(sweep, "objective", materiality_pct=1.0)
        self.assertEqual(result["status"], "tied")
        self.assertEqual(len(result["best_rows"]), 2)

    def test_longest_run_elapsed_span_never_exceeds_timeline_span(self):
        frame = pd.DataFrame({"fraction": [0, 0, 1, 1, 1, 1, 0]})
        result = longest_runs(frame, "fraction", threshold=0.5, mode="at_or_above", sample_minutes=5)
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["elapsed_span_minutes"], 15)
        self.assertEqual(result["sample_slot_minutes"], 20)
        self.assertLessEqual(result["elapsed_span_minutes"], (len(frame) - 1) * 5)

    def test_half_open_selector_and_sweep_share_same_rows(self):
        frame = self.timeline()
        sweep = build_delay_sweep(frame, 6, step_minutes=5)
        row = sweep.iloc[100]
        selected = select_half_open_window(frame, row.window_start_ist.tz_convert("UTC"), 6)
        self.assertEqual(len(selected), int(row.sample_count))
        self.assertAlmostEqual(selected.earth_visible_fraction_approx.mean() * 100,
                               row.earth_mean_disk_fraction_pct)


if __name__ == "__main__":
    unittest.main()
