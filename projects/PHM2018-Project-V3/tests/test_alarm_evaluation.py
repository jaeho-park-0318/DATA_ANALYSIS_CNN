import unittest
import numpy as np
import pandas as pd
from src.alarm_evaluation import alarm_episodes,evaluate,observed_support

def scores(times,values=None,segments=None,labels=None):
    n=len(times)
    return pd.DataFrame({'Tool':['01M01']*n,'time':times,'segment_id':[0]*n if segments is None else segments,'score':[1.]*n if values is None else values,'audit_label':[1]*n if labels is None else labels})
def events(t=1000,tag=False):
    return pd.DataFrame([dict(event_id='e1',Tool='01M01',fault_time=t,fault_type='low',tagged=tag)])

class AlarmTests(unittest.TestCase):
    def test_closed_warning_boundaries(self):
        s,e,a,m=evaluate(scores([400,940]),events(),.5)
        self.assertEqual(s['detected_events'],1);self.assertEqual(e.lead_raw.iloc[0],600);self.assertEqual(len(m),2)
    def test_too_late_is_not_detection(self):
        s,e,a,m=evaluate(scores([941,1000],labels=[3,3]),events(),.5)
        self.assertEqual(s['detected_events'],0);self.assertEqual(e.late_alarm_episodes.iloc[0],2)
    def test_continuous_alarm_not_retroactively_detected(self):
        s,e,a,m=evaluate(scores(np.arange(100,1001,60)),events(),.5)
        self.assertEqual(len(a),1);self.assertEqual(s['detected_events'],0);self.assertTrue(e.available.iloc[0])
    def test_gap_and_segment_boundary_break_episode(self):
        a=alarm_episodes(scores([0,60,180,240],segments=[0,0,0,1]),.5)
        self.assertEqual(len(a),3)
    def test_non_alarm_breaks_episode(self):
        self.assertEqual(len(alarm_episodes(scores([0,60,120],[1,0,1]),.5)),2)
    def test_nan_scores_rejected(self):
        with self.assertRaises(ValueError):alarm_episodes(scores([0],[np.nan]),.5)
    def test_duplicate_anchors_rejected(self):
        with self.assertRaises(ValueError):alarm_episodes(scores([0,0]),.5)
    def test_no_input_distinct_from_missed(self):
        s,e,a,m=evaluate(scores([0],[0]),events(),.5)
        self.assertEqual(e.status.iloc[0],'NO_INPUT');self.assertTrue(np.isnan(s['recall_available']))
    def test_unknown_alarm_not_false_positive(self):
        s,e,a,m=evaluate(scores([0],labels=[2]),events(),.5)
        self.assertEqual(a.category.iloc[0],'UNKNOWN_START')
    def test_tag_event_separated(self):
        s,e,a,m=evaluate(scores([400]),events(tag=True),.5)
        self.assertEqual(s['scorable_events'],0);self.assertEqual(e.status.iloc[0],'TAG_UNVERIFIED')
    def test_many_events_do_not_duplicate_alarm_burden(self):
        ev=pd.concat([events(),events(1060).assign(event_id='e2')])
        s,e,a,m=evaluate(scores([500]),ev,.5)
        self.assertEqual(s['detected_events'],2);self.assertEqual(s['alarm_episodes'],1)
    def test_exposure_does_not_fill_missing_observations(self):
        self.assertEqual(observed_support([0,4,12,100],[0,0,0,1],scores([0])),12)
    def test_exposure_cuts_cross_anchor_support(self):
        self.assertEqual(observed_support([58],[0],scores([60])),2)
        self.assertEqual(observed_support([58],[0],scores([0,60])),4)
    def test_chunk_concatenation_preserves_episode(self):
        a=alarm_episodes(pd.concat([scores([0,60]),scores([120,180])]),.5)
        self.assertEqual(len(a),1)
    def test_exposure_union_and_chunk_edge(self):
        self.assertEqual(observed_support([0,2],[0,0],scores([0])),6)
        parts=observed_support([0],[0],scores([0]),next_time=2)+observed_support([2],[0],scores([0]))
        self.assertEqual(parts,6)
    def test_other_imminent_fault_excludes_credit(self):
        s,e,a,m=evaluate(scores([500],labels=[3]),events(),.5)
        self.assertEqual(s['detected_events'],0)
        self.assertEqual(a.category.iloc[0],'LATE_OR_EXCLUDED_START')

if __name__=='__main__':unittest.main()
