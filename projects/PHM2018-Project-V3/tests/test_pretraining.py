"""합성 시험은 실제 후보 집계에 포함하지 않는다."""
import unittest
import numpy as np
import pandas as pd
from src.pretraining import *

def frame(times,delta=10000):
    d=pd.DataFrame({'time':times})
    for k in FAULTS:
        d['ttf_'+k]=delta
        d[f'label__conservative__{k}__h600']=NEG
    return d

class PretrainingTests(unittest.TestCase):
    def test_both_sides_of_boundary_purged(self):
        s,y=assign_and_label(frame([899,900,3199,3200,5899,5900]),[(0,0,5000),(1,5000,10000)],600)
        np.testing.assert_array_equal(s,[-1,0,0,-1,-1,1])
    def test_future_partition_cannot_justify_negative(self):
        s,y=assign_and_label(frame([1000]),[(0,0,5000)],600)
        self.assertEqual(y[0],NA)
        d=frame([1000],2000);s,y=assign_and_label(d,[(0,0,5000)],600)
        self.assertEqual(y[0],NEG)
    def test_end_is_exclusive(self):
        s,y=assign_and_label(frame([1000],4000),[(0,0,5000)],600)
        self.assertEqual(y[0],NA)
    def test_known_positive_survives_unknown_other_type(self):
        d=frame([1000]);d['ttf_low']=600;d['label__conservative__low__h600']=POS
        s,y=assign_and_label(d,[(0,0,5000)],600);self.assertEqual(y[0],POS)
    def test_imminent_exclusion_has_priority(self):
        d=frame([1000]);d['ttf_low']=600;d['label__conservative__low__h600']=POS
        d['ttf_high']=30;d['label__conservative__high__h600']=EXCLUDE
        s,y=assign_and_label(d,[(0,0,5000)],600);self.assertEqual(y[0],EXCLUDE)
    def test_tool_pairs_do_not_cross_group_split(self):
        for tool in range(1,11):
            self.assertEqual(partitions(f'{tool:02}M01',0,10000,'group'),partitions(f'{tool:02}M02',0,10000,'group'))
    def test_feature_whitelist_has_no_answer_fields(self):
        for f in 'ABC':
            self.assertFalse(any(any(k in c.lower() for k in ['ttf','fault','event','time','tool','label']) for c in schema()[f]))

if __name__=='__main__':unittest.main()
