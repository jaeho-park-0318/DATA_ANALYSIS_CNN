"""실험 성능과 구분된 지표 경계·경보 병합·특징 금지 목록 검사."""
import unittest
import numpy as np
import pandas as pd
from src.evaluate_models import choose_threshold,sample_metrics,episodes
from src.model_common import configs
from unittest.mock import patch
from src.evaluate_models import operational_metrics


class ModelTests(unittest.TestCase):
    def test_threshold_uses_validation_only(self):
        y=np.array([0,0,1,1]);score=np.array([.1,.2,.7,.9])
        threshold=choose_threshold(y,score)
        self.assertEqual(threshold,.7)
        self.assertEqual(sample_metrics(y,score,threshold)['f1'],1)

    def test_undefined_metrics_are_not_invented(self):
        result=sample_metrics(np.array([0,0]),np.array([.1,.2]),.5)
        self.assertIsNone(result['recall']);self.assertIsNone(result['pr_auc_ap']);self.assertIsNone(result['roc_auc'])
        result=sample_metrics(np.array([0,1]),np.array([.1,.1]),.5)
        self.assertIsNone(result['precision']);self.assertEqual(result['f1'],0)

    def test_alarm_episodes_gap_rearm_and_state(self):
        frame=pd.DataFrame({'Tool':['01M01']*6,'segment_id':[0]*6,'stage':[1,1,1,1,0,0],
                            'time':[0,60,120,660,720,780],'score':[.9]*6})
        e=episodes(frame,.5,600)
        self.assertEqual(e.episode_id.nunique(),3)
        continuous=pd.DataFrame({'Tool':['01M01']*13,'segment_id':[0]*13,'stage':[1]*13,
                                 'time':np.arange(13)*60,'score':[.9]*13})
        self.assertEqual(episodes(continuous,.5,600).episode_id.nunique(),2)

    def test_features_exclude_target_and_identifiers(self):
        self.assertEqual({k:len(v) for k,v in configs().items()},{'A':8,'B':80,'C':83})
        for names in configs().values():
            self.assertFalse(any(any(t in c.lower() for t in ('ttf','fault','label','event','time','lot','usage','tool')) for c in names))

    def test_one_alarm_does_not_detect_two_sequential_events(self):
        times=[0,60,120,180,240,290,330]
        first='01M01:300:f1';second='01M01:360:f1'
        frame=pd.DataFrame({'Tool':['01M01']*7,'time':times,'stage':[1]*7,'segment_id':[0]*7,
                            'score':[.9]*7,'label_any_h600raw':[1]*7,'label_f1_h600raw':[1]*7,
                            'next_event_f1':[first]*6+[second]})
        records=[{'Tool':'01M01','event_key':key,'fault_code':'f1','tagged':False,
                  'inside_sensor_range':True,'split_tool':'evaluation','event_time_split':'evaluation'} for key in (first,second)]
        with patch('src.evaluate_models.event_table',return_value=pd.DataFrame({'tagged':[False]})),patch('src.evaluate_models.read_json',return_value={'fault_events':records}):
            metrics=operational_metrics(frame,.5,600,'split_tool','evaluation')
        self.assertEqual(metrics['evaluable_events'],2)
        self.assertEqual(metrics['detected_events'],1)
        self.assertEqual(metrics['event_recall'],.5)


if __name__=='__main__':unittest.main()
