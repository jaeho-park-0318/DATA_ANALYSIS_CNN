import unittest
import pandas as pd
from scripts.select_stage4 import choose_grid_config

class Stage4SelectionTests(unittest.TestCase):
    def test_burden_precedes_complexity_when_detection_ties(self):
        points=pd.DataFrame([dict(config_id='simple',budget_per_10000_raw=5,detected_events=10,alarm_episodes_per_10000_raw=4,depth=2,positive_weight=1,feature_set='A'),dict(config_id='lower_alarm',budget_per_10000_raw=5,detected_events=10,alarm_episodes_per_10000_raw=3,depth=4,positive_weight=16,feature_set='B')])
        self.assertEqual(choose_grid_config(points,5).config_id,'lower_alarm')

    def test_simpler_configuration_breaks_exact_performance_tie(self):
        points=pd.DataFrame([dict(config_id='weighted',budget_per_10000_raw=10,detected_events=12,alarm_episodes_per_10000_raw=5,depth=2,positive_weight=16,feature_set='A'),dict(config_id='unit',budget_per_10000_raw=10,detected_events=12,alarm_episodes_per_10000_raw=5,depth=2,positive_weight=1,feature_set='B'),dict(config_id='different_budget',budget_per_10000_raw=20,detected_events=50,alarm_episodes_per_10000_raw=18,depth=2,positive_weight=1,feature_set='A')])
        self.assertEqual(choose_grid_config(points,10).config_id,'unit')

if __name__=='__main__':unittest.main()
