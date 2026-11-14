import unittest
from pathlib import Path
APP=Path(__file__).resolve().parents[1]/'app.py'
class AppReviewFixTests(unittest.TestCase):
    def test_single_sample_date_does_not_create_zero_range_slider(self):
        s=APP.read_text()
        self.assertIn('if len(df) > 1:',s)
        self.assertIn('snapshot control is disabled',s)
    def test_coverage_warning_says_all_azimuths_unclosed(self):
        s=APP.read_text()
        self.assertIn('All {range_limited_bins:,} of 3,600 azimuth directions are range-limited',s)
        self.assertNotIn('if cap_hits:',s)
    def test_height_classification_is_computed_from_aligned_timelines(self):
        s=APP.read_text()
        self.assertIn('compare_timeline_visibility(timelines[1], timelines[5])',s)
        self.assertNotIn('does not alter the sampled Sun/Earth visibility classifications',s)
    def test_delay_is_not_called_launch_optimizer(self):
        s=APP.read_text()
        self.assertIn('This is not a launch-window optimizer',s)
        self.assertIn('step_minutes=5',s)
    def test_materiality_threshold_used_in_summary(self):
        s=APP.read_text()
        self.assertIn('sun >= threshold',s)
        self.assertIn('earth >= threshold',s)
if __name__=='__main__': unittest.main()
