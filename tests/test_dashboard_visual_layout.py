"""Lightweight Plotly layout regression tests; does not import or launch Streamlit."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
CHART_FUNCTIONS = {"style_chart", "elevation_chart", "fraction_chart", "sky_dome_chart", "availability_band_chart"}


def load_chart_functions():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in CHART_FUNCTIONS]
    namespace = {"np": np, "pd": pd, "go": go}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(APP_PATH), "exec"), namespace)
    return namespace


class DashboardVisualLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = load_chart_functions()
        timestamps = pd.date_range("2026-11-14T00:00:00Z", periods=3, freq="5min")
        cls.frame = pd.DataFrame({
            "timestamp_ist": timestamps.tz_convert("Asia/Kolkata"),
            "utc": timestamps.astype(str),
            "sun_center_elevation_deg": [1.0, 1.1, 1.2],
            "sun_terrain_horizon_deg": [-1.0, -0.9, -0.8],
            "sun_angular_semidiameter_deg": [0.27] * 3,
            "sun_visibility_state": ["full"] * 3,
            "sun_azimuth_deg": [250.0, 251.0, 252.0],
            "sun_visible_fraction_approx": [1.0] * 3,
            "earth_center_elevation_deg": [-4.0, -3.9, -3.8],
            "earth_terrain_horizon_deg": [2.0] * 3,
            "earth_angular_semidiameter_deg": [0.95] * 3,
            "earth_visibility_state": ["hidden"] * 3,
            "earth_azimuth_deg": [135.0, 136.0, 137.0],
            "earth_visible_fraction_approx": [0.0] * 3,
            "sun_visible": [True] * 3,
            "earth_visible": [False] * 3,
            "both_visible": [False] * 3,
        })
        cls.horizon = pd.DataFrame({
            "azimuth_center_deg": np.arange(0.05, 360.0, 0.1),
            "combined_horizon_elevation_deg": np.zeros(3600),
        })

    def test_elevation_chart_has_no_in_plot_legend(self):
        fig = self.functions["elevation_chart"](self.frame, "Sun", "#65D7E8")
        self.assertFalse(fig.layout.showlegend)
        self.assertIsNone(fig.layout.title.text)
        self.assertEqual(len(fig.data), 4)
        self.assertEqual(fig.data[0].name, "Centre elevation")
        self.assertEqual(fig.data[1].name, "Terrain horizon · regional lower bound")

    def test_sky_dome_has_no_overlaid_legend(self):
        fig = self.functions["sky_dome_chart"](self.frame.iloc[0], self.horizon)
        self.assertFalse(fig.layout.showlegend)
        self.assertEqual(len(fig.data), 3)
        self.assertEqual(fig.layout.polar.radialaxis.angle, 24)
        self.assertEqual(list(fig.layout.polar.radialaxis.ticktext), ["Zenith", "60°", "30°", "0°", "−10°"])

    def test_single_series_visibility_chart_hides_legend(self):
        fig = self.functions["fraction_chart"](self.frame, "Sun", "#65D7E8")
        self.assertFalse(fig.layout.showlegend)

    def test_shared_legend_is_reserved_above_plot_area(self):
        fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2], name="Series"))
        fig.update_layout(title="Test")
        fig = self.functions["style_chart"](fig, show_legend=True, top_margin=80)
        self.assertGreaterEqual(fig.layout.margin.t, 90)
        self.assertEqual(fig.layout.title.y, 0.985)
        self.assertGreater(fig.layout.legend.y, 1.0)

    def test_visibility_band_title_remains_in_bounds(self):
        fig = self.functions["availability_band_chart"](self.frame)
        self.assertGreaterEqual(fig.layout.title.y, 0.0)
        self.assertLessEqual(fig.layout.title.y, 1.0)
        self.assertFalse(fig.layout.showlegend)

    def test_app_uses_external_chart_renderers(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("def render_elevation_chart", source)
        self.assertIn("chart-legend", source)
        self.assertIn("sky-key", source)
        self.assertNotIn('st.plotly_chart(elevation_chart(df, "Sun"', source)


if __name__ == "__main__":
    unittest.main()
