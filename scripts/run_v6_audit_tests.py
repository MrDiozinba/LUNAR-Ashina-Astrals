#!/usr/bin/env python3
"""Run the explicit LUNAR v6 audit suite without discovering legacy tests."""
from __future__ import annotations
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEST_FILES = (
    "test_app_regression.py",
    "test_dashboard_data.py",
    "test_dashboard_visual_layout.py",
    "test_decision_support.py",
    "test_spice_disk_geometry.py",
    "test_terrain_azimuth_curvature.py",
    "test_dashboard_runtime_regressions.py",
)

suite = unittest.TestSuite()
loader = unittest.defaultTestLoader
for filename in TEST_FILES:
    path = ROOT / "tests" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required v6 test file is missing: {path}")
    module_name = f"lunar_v6_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite.addTests(loader.loadTestsFromModule(module))

result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
