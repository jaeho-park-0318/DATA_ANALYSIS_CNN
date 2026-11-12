"""합성 경계 시험. 이 자료는 실제 보고서/그래프에 사용하지 않는다."""
import unittest
import numpy as np
import pandas as pd
from src.common import CFG,SENSORS
from src.labels import make_labels,POS,NEG,NA,EXCLUDE
from src.features import calculate,schema
from src.query import broken_line,read_window

def fixture(times=None):
    t=np.arange(0,1601,4) if times is None else np.asarray(times)
    d=pd.DataFrame({'time':t,'recipe':1.,'stage':1.,'recipe_step':2.,'state_conflict':False,'segment_id':0,'segment_start':0})
    for j,s in enumerate(SENSORS):d[s]=t*(j+1)/100+np.sin(t/30)
    return d

def faults():
    return pd.DataFrame([{'Tool':'01M01','fault_type':'low','fault_time':1000.,'event_id':'low1','tagged':False}, {'Tool':'01M01','fault_type':'high','fault_time':1400.,'event_id':'high1','tagged':False}])

class DiagnosticsTests(unittest.TestCase):
    def test_label_boundaries(self):
        t=np.array([399,400,939,940,941,1000,1001])
        l,*_=make_labels(t,faults(),'01M01',2000)
        np.testing.assert_equal(l['conservative__low__h600'],[NEG,POS,POS,POS,EXCLUDE,EXCLUDE,NA])

    def test_imminent_precedes_other_type_positive(self):
        l,*_=make_labels([970],faults(),'01M01',2000)
        for p in CFG['policies']:
            for k in ['low','high','leak','any']:self.assertEqual(l[f'{p}__{k}__h600'][0],EXCLUDE)

    def test_missing_ttf_not_zero(self):
        l,*_=make_labels([400],faults(),'01M01',2000)
        self.assertEqual(l['conservative__leak__h600'][0],NA)
        self.assertEqual(l['sensitivity__leak__h600'][0],NEG)

    def test_research_needs_both_history_interval_and_future_sensor_span(self):
        l,*_=make_labels([1100,1100],faults(),'01M01',2000,[1500,1200])
        self.assertEqual(l['research__leak__h300'][0],NEG)
        self.assertEqual(l['research__leak__h300'][1],NA)

    def test_no_wafer_quarantine(self):
        e=pd.concat([faults(),pd.DataFrame([{'Tool':'01M01','fault_type':'high','fault_time':800.,'event_id':'tag','tagged':True}])],ignore_index=True)
        l,*_=make_labels([600],e,'01M01',2000)
        self.assertEqual(l['sensitivity__high__h600'][0],NA)

    def test_future_sensor_does_not_change_features(self):
        d=fixture();a=np.array([960,1020]);x=calculate(d,a);d.loc[d.time>1020,SENSORS]=10000
        y=calculate(d,a);np.testing.assert_allclose(x[schema()['C']],y[schema()['C']],equal_nan=True)

    def test_anchor_window_is_exact_and_raw_resolution(self):
        d=fixture();a=np.array([1002]);f=calculate(d,a)
        v=d.loc[(d.time>702)&(d.time<=1002),SENSORS[0]]
        self.assertAlmostEqual(f[SENSORS[0]+'__mean_300'].iloc[0],v.mean(),places=10)
        self.assertAlmostEqual(f[SENSORS[0]+'__std_300'].iloc[0],v.std(ddof=0),places=7)
        self.assertAlmostEqual(f[SENSORS[0]+'__min_300'].iloc[0],v.min(),places=10)

    def test_chunk_invariance(self):
        d=fixture();a=np.array([960,1020,1080,1500]);whole=calculate(d,a)
        parts=pd.concat([calculate(d[d.time<=1080],a[:3]),calculate(d[d.time>580],a[3:])],ignore_index=True)
        np.testing.assert_allclose(whole[schema()['C']],parts[schema()['C']],atol=1e-8,equal_nan=True)

    def test_feature_families_are_independent_masks(self):
        f=calculate(fixture(),[60,360,960])
        np.testing.assert_equal(f.A_available.to_numpy(),[True,True,True]);np.testing.assert_equal(f.B_available.to_numpy(),[False,True,True]);np.testing.assert_equal(f.C_available.to_numpy(),[False,False,True])
        self.assertEqual([len(schema()[x]) for x in 'ABC'],[8,32,44])

    def test_gap_resets_history(self):
        d=fixture(np.r_[np.arange(0,501,4),np.arange(1000,1601,4)])
        d.loc[d.time>=1000,'segment_start']=1000;d.loc[d.time>=1000,'segment_id']=1
        f=calculate(d,[1020]);self.assertTrue(f.A_available.iloc[0]);self.assertFalse(f.B_available.iloc[0])

    def test_plot_break_preserves_real_zero(self):
        x,y=broken_line([0,4,100],[0.,1.,2.])
        self.assertEqual(y,[0.,1.,None,2.]);self.assertEqual(x,[0.,4.,None,100.])

    def test_empty_plot(self):
        self.assertEqual(broken_line([],[]),([],[]))

    def test_oversized_query_rejected_before_io(self):
        with self.assertRaisesRegex(ValueError,'확대'):read_window('01M01',0,1000000)

    def test_context_conflict_disables_all_features(self):
        d=fixture();d.loc[d.time==960,'state_conflict']=True
        f=calculate(d,[960]);self.assertFalse(f.A_available.iloc[0]);self.assertFalse(f.B_available.iloc[0]);self.assertFalse(f.C_available.iloc[0])

    def test_sensor_summary_excludes_later_values_and_handles_empty(self):
        from src.detail import summarize_sensors
        d=fixture();d['Lot']=1;d['runnum']=2
        before=summarize_sensors(d,400,940)
        d.loc[d.time>940,SENSORS]=100000
        self.assertEqual(before,summarize_sensors(d,400,940))
        missing=summarize_sensors(d.iloc[:0],0,100)
        self.assertTrue(all(r['observations']==0 and r['delta'] is None for r in missing))

if __name__=='__main__':unittest.main()
