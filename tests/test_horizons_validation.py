import unittest
from datetime import datetime, timezone
from scripts.validate_spice_against_horizons import parse_horizons_text, angular_difference_deg

SAMPLE = '''API VERSION: 1.2
Target body name: Sun (10) {source: DE441}
Center body name: Moon (301)
Center geodetic : 222.5104471,-89.4631628,1.94583 {E-lon(deg),Lat(deg),Alt(km)}
Center pole/equ : MEAN_ME (high precision)
 Date__(UT)__HR:MN, , ,Azi_(a-app), Elev_(a-app),
$$SOE
 2026-Nov-13 18:30,*, , 266.886012, 1.527413,
 2026-Nov-13 18:35,*, , 266.843688, 1.527019,
$$EOE
'''

class HorizonsParserTests(unittest.TestCase):
    def test_parse_rows_and_header(self):
        rows, header = parse_horizons_text(SAMPLE)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]['azimuth_deg'], 266.886012)
        self.assertAlmostEqual(rows[1]['elevation_deg'], 1.527019)
        self.assertEqual(header['Center pole/equ'], 'MEAN_ME (high precision)')

    def test_azimuth_wraparound(self):
        self.assertAlmostEqual(angular_difference_deg(0.1, 359.9), 0.2)
        self.assertAlmostEqual(angular_difference_deg(359.9, 0.1), -0.2)

    def test_missing_table_raises(self):
        with self.assertRaises(ValueError):
            parse_horizons_text('API ERROR: bad request')

if __name__ == '__main__':
    unittest.main()
