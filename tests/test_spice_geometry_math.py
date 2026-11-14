import math
import unittest

import numpy as np

from backend.spice_site_geometry import (
    angular_semidiameter_deg,
    circular_disk_visible_fraction,
    interpolate_circular_horizon,
    local_enu_basis,
    observer_position_fixed_km,
    vector_to_azimuth_elevation,
    visibility_state,
)


class GeometryTests(unittest.TestCase):
    def test_enu_basis_is_orthonormal(self):
        e, n, u = local_enu_basis(-89.463162834037, 222.51044707800085)
        self.assertAlmostEqual(float(np.linalg.norm(e)), 1.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(n)), 1.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(u)), 1.0, places=12)
        self.assertAlmostEqual(float(np.dot(e, n)), 0.0, places=12)
        self.assertAlmostEqual(float(np.dot(e, u)), 0.0, places=12)
        self.assertAlmostEqual(float(np.dot(n, u)), 0.0, places=12)

    def test_direction_conversion_cardinal_directions(self):
        e, n, u = local_enu_basis(0.0, 0.0)
        az, el, _ = vector_to_azimuth_elevation(n, 0.0, 0.0)
        self.assertAlmostEqual(az, 0.0, places=9)
        self.assertAlmostEqual(el, 0.0, places=9)
        az, el, _ = vector_to_azimuth_elevation(e, 0.0, 0.0)
        self.assertAlmostEqual(az, 90.0, places=9)
        self.assertAlmostEqual(el, 0.0, places=9)
        az, el, _ = vector_to_azimuth_elevation(u, 0.0, 0.0)
        self.assertAlmostEqual(el, 90.0, places=9)

    def test_observer_radius(self):
        xyz = observer_position_fixed_km(0.0, 0.0, 100.0, 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(xyz)), (1737400.0 + 101.0) / 1000.0, places=9)

    def test_circular_interpolation_wraparound(self):
        az = np.array([0.05, 90.05, 180.05, 270.05])
        hz = np.array([1.0, 2.0, 3.0, 4.0])
        at_zero = interpolate_circular_horizon(0.0, az, hz)
        self.assertTrue(0.0 < at_zero < 4.0)
        self.assertAlmostEqual(interpolate_circular_horizon(90.05, az, hz), 2.0, places=9)

    def test_disk_fraction_boundaries_and_midpoint(self):
        a = 0.25
        self.assertEqual(circular_disk_visible_fraction(-1.0, 0.0, a), 0.0)
        self.assertEqual(circular_disk_visible_fraction(1.0, 0.0, a), 1.0)
        self.assertAlmostEqual(circular_disk_visible_fraction(0.0, 0.0, a), 0.5, places=12)

    def test_semidiameter_and_states(self):
        self.assertGreater(angular_semidiameter_deg(1.0, 1000.0), 0.0)
        self.assertEqual(visibility_state(0.0), "hidden")
        self.assertEqual(visibility_state(0.5), "partial")
        self.assertEqual(visibility_state(1.0), "full")


if __name__ == "__main__":
    unittest.main()
