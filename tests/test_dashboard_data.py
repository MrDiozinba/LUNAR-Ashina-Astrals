from __future__ import annotations
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.dashboard_data import load_scientific_bundle, summarize_timeline, snapshot_slider_bounds

ROOT = Path(__file__).resolve().parents[1]


class DashboardDataBehaviourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_scientific_bundle(ROOT)

    def test_scientific_bundle_passes_required_artifact_hashes_and_cross_file_checks(self):
        self.assertTrue(all(self.bundle["hash_checks"].values()))
        self.assertEqual(len(self.bundle["timelines"][1]), 577)
        self.assertEqual(len(self.bundle["horizons"][5]), 3600)

    def test_one_sample_date_has_no_slider_bounds(self):
        self.assertIsNone(snapshot_slider_bounds(1))
        self.assertEqual(snapshot_slider_bounds(577), (0, 576))

    def test_snapshot_bounds_reject_invalid_counts(self):
        with self.assertRaises(ValueError):
            snapshot_slider_bounds(-1)

    def test_fraction_below_one_percent_is_not_any_visible(self):
        frame = self.bundle["timelines"][1].head(10).copy()
        frame["sun_visible_fraction_approx"] = 0.005
        frame["earth_visible_fraction_approx"] = 0.005
        frame["sun_visibility_state"] = "partial"
        frame["earth_visibility_state"] = "partial"
        summary = summarize_timeline(frame)
        self.assertEqual(summary["sun_any_visible_pct"], 0.0)
        self.assertEqual(summary["earth_any_visible_pct"], 0.0)
        self.assertEqual(summary["both_any_visible_pct"], 0.0)

    def test_fraction_at_one_percent_is_any_visible(self):
        frame = self.bundle["timelines"][1].head(10).copy()
        frame["sun_visible_fraction_approx"] = 0.01
        frame["earth_visible_fraction_approx"] = 0.01
        frame["sun_visibility_state"] = "partial"
        frame["earth_visibility_state"] = "partial"
        summary = summarize_timeline(frame)
        self.assertEqual(summary["sun_any_visible_pct"], 100.0)
        self.assertEqual(summary["earth_any_visible_pct"], 100.0)
        self.assertEqual(summary["both_any_visible_pct"], 100.0)

    def test_manifest_detects_timeline_file_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "data", root / "data")
            target = root / "data/spice/site01_spice_timeline_h1m.csv"
            target.write_text(target.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_scientific_bundle(root)

    def test_loader_rejects_timeline_horizon_mismatch_even_if_manifest_hash_is_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "data", root / "data")
            timeline_path = root / "data/spice/site01_spice_timeline_h1m.csv"
            frame = pd.read_csv(timeline_path)
            frame.loc[0, "sun_terrain_horizon_deg"] += 0.1
            frame.to_csv(timeline_path, index=False)
            manifest_path = root / "data/spice/scientific_file_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            import hashlib
            manifest["sha256"]["data/spice/site01_spice_timeline_h1m.csv"] = hashlib.sha256(timeline_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2))
            with self.assertRaisesRegex(ValueError, "does not match circular interpolation"):
                load_scientific_bundle(root)

    def test_loader_requires_regional_coverage_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "data", root / "data")
            hpath = root / "data/horizon_masks/site01_multires_h1m.csv"
            frame = pd.read_csv(hpath).drop(columns=["regional_80m_dominates"])
            frame.to_csv(hpath, index=False)
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_scientific_bundle(root)


if __name__ == "__main__":
    unittest.main()
