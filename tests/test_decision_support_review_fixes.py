import unittest
import numpy as np
import pandas as pd
from backend.decision_support import build_delay_sweep, classify_objective, compare_timeline_visibility, summarize_horizon_sectors

class ReviewFixTests(unittest.TestCase):
    def timeline(self, n=577):
        utc=pd.date_range('2026-11-13T18:30:00Z',periods=n,freq='5min')
        return pd.DataFrame({'timestamp_utc':utc,'timestamp_ist':utc.tz_convert('Asia/Kolkata'),
          'sun_visible_fraction_approx':np.ones(n),'earth_visible_fraction_approx':np.zeros(n),
          'sun_visibility_state':['full']*n,'earth_visibility_state':['hidden']*n})

    def test_all_azimuths_are_reported_range_limited_not_only_argmax_at_cap(self):
        az=np.arange(.05,360,.1)
        h=pd.DataFrame({'azimuth_center_deg':az,'combined_horizon_elevation_deg':np.sin(np.deg2rad(az)),
          'combined_distance_at_horizon_m':np.where(az<180,1000,45000), 'regional_80m_dominates':0})
        out=summarize_horizon_sectors(h)
        self.assertEqual(int(out['Range-limited azimuth bins'].sum()),3600)

    def test_five_minute_step_returns_every_feasible_start(self):
        out=build_delay_sweep(self.timeline(),24,step_minutes=5)
        self.assertEqual(len(out),(48-24)*12+1)
        self.assertEqual(out.delay_minutes.iloc[-1],1440)

    def test_step_hours_two_maps_to_two_hour_step(self):
        out=build_delay_sweep(self.timeline(),6,step_hours=2)
        self.assertEqual(out.delay_hours.tolist(),list(range(0,43,2)))

    def test_materiality_threshold_marks_zero_objective_unachieved(self):
        out=build_delay_sweep(self.timeline(),6,step_minutes=5)
        result=classify_objective(out,'earth_mean_disk_fraction_pct')
        self.assertEqual(result['status'],'unachieved')

    def test_nontrivial_window_objective_unique(self):
        out=build_delay_sweep(self.timeline(),6,step_minutes=5)
        out.loc[10,'earth_mean_disk_fraction_pct']=25
        result=classify_objective(out,'earth_mean_disk_fraction_pct')
        self.assertEqual(result['status'],'unique')
        self.assertEqual(int(result['best_rows'].iloc[0].delay_minutes),50)

    def test_observer_timelines_compare_actual_rows(self):
        a=self.timeline(12); b=a.copy(); b.loc[3,'earth_visibility_state']='partial'; b.loc[4,'earth_visible_fraction_approx']=.2
        r=compare_timeline_visibility(a,b)
        self.assertEqual(r['earth_state_different_rows'],1)
        self.assertEqual(r['earth_fraction_different_rows'],1)

    def test_one_sample_ist_day_is_valid_input_to_summary_logic(self):
        one=self.timeline(1)
        self.assertEqual(len(one),1)

if __name__=='__main__': unittest.main()
