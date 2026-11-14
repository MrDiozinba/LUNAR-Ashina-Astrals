import unittest
from backend.spice_site_geometry import circular_disk_visible_fraction

class DiskClippingTests(unittest.TestCase):
    def test_disk_is_half_visible_when_center_is_on_horizon(self):
        self.assertAlmostEqual(circular_disk_visible_fraction(0.0, 0.0, 1.0), 0.5, places=12)

    def test_disk_is_hidden_at_lower_limb_tangent(self):
        self.assertEqual(circular_disk_visible_fraction(-1.0, 0.0, 1.0), 0.0)

    def test_disk_is_full_at_upper_limb_tangent(self):
        self.assertEqual(circular_disk_visible_fraction(1.0, 0.0, 1.0), 1.0)

    def test_disk_partial_fraction_is_symmetric(self):
        below = circular_disk_visible_fraction(-0.5, 0.0, 1.0)
        above = circular_disk_visible_fraction(0.5, 0.0, 1.0)
        self.assertAlmostEqual(below + above, 1.0, places=12)

if __name__ == '__main__': unittest.main()
