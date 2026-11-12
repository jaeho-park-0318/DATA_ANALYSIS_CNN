"""경보 부담과 검출률이 단조롭지 않을 때의 후보 선정 계약."""
import unittest
import pandas as pd
from scripts.evaluate_stage2 import pareto_mask

class ParetoTests(unittest.TestCase):
    def test_tradeoffs_and_dominated_points(self):
        d=pd.DataFrame({'alarm_episodes_per_10000_raw':[0,2,3,4,5,7],'recall_available':[0,.2,.1,.4,.4,.3]})
        self.assertEqual(pareto_mask(d).tolist(),[True,True,False,True,False,False])

    def test_identical_points_and_equal_burden(self):
        d=pd.DataFrame({'alarm_episodes_per_10000_raw':[1,1,2,2],'recall_available':[.1,.2,.3,.3]})
        self.assertEqual(pareto_mask(d).tolist(),[False,True,True,True])

if __name__=='__main__':unittest.main()
