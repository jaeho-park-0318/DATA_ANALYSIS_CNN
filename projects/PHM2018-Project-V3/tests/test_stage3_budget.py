import unittest
import pandas as pd
from scripts.diagnose_stage3 import choose_budget

class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.curve=pd.DataFrame({'threshold':[1.1,.8,.6,.4,.2],'alarm_episodes_per_10000_raw':[0,2,3,5,8],'detected_events':[0,5,4,5,6]})

    def test_nonmonotonic_detection_and_least_burden_tie(self):
        row=choose_budget(self.curve,5)
        self.assertEqual(row.threshold,.8)
        self.assertEqual(row.detected_events,5)

    def test_zero_budget_keeps_no_alarm(self):
        row=choose_budget(self.curve,0)
        self.assertEqual(row.detected_events,0)
        self.assertEqual(row.alarm_episodes_per_10000_raw,0)

if __name__=='__main__':unittest.main()
