"""그룹 분리, NA 보존, 경보와 이벤트의 일대시각 대응 검증."""
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.group_validation45 import fold_plan,operational,TARGET
from src.model_common import configs,matrix

class ImprovementTests(unittest.TestCase):
    def test_groups(self):
        plans=fold_plan();seen=[]
        for p in plans:
            roles=[set(p[k]) for k in ['train','early_stop','calibration','evaluation']]
            self.assertEqual(sum(map(len,roles)),len(set.union(*roles)))
            self.assertTrue(all(int(t[:2])<=6 for t in set.union(*roles)))
            seen+=p['evaluation']
        self.assertEqual(len(seen),len(set(seen)))

    def test_future_columns(self):
        for cols in configs().values():
            self.assertFalse(any(any(s in c.lower() for s in ['label','ttf','event','time','tool','lot','usage']) for c in cols))
        with self.assertRaises(AssertionError):matrix(pd.DataFrame({'ttf':[1]}),['ttf'])

    def test_unknown_alarm(self):
        f=pd.DataFrame({'Tool':['01M01']*3,'time':[0.,60.,120.],'segment_id':[0]*3,'stage':[1]*3,TARGET:[np.nan]*3,'score':[.9]*3})
        for c in ['f1','f2','f3']:f[f'label_{c}_h600raw']=np.nan;f['next_event_'+c]=None
        empty=pd.DataFrame(columns=['Tool','time','tagged','event_key','fault_code'])
        empty['tagged']=empty.tagged.astype(bool)
        with patch('src.group_validation45.event_table',return_value=empty):r=operational(f,.5)
        self.assertEqual(r['unknown_episodes'],1);self.assertEqual(r['false_episodes'],0)
        self.assertIsNone(r['false_alarms_per_active_hour']);self.assertIsNone(r['event_recall'])

    def test_alarm_single_event_time(self):
        f=pd.DataFrame({'Tool':['01M01']*3,'time':[0.,60.,120.],'segment_id':[0]*3,'stage':[1]*3,TARGET:[1]*3,'score':[.9]*3})
        for c in ['f1','f2','f3']:f[f'label_{c}_h600raw']=np.nan;f['next_event_'+c]=None
        f['label_f1_h600raw']=1;f['next_event_f1']=['01M01:100:f1','01M01:100:f1','01M01:200:f1']
        ev=pd.DataFrame({'Tool':['01M01']*2,'time':[100.,200.],'tagged':[False]*2,'event_key':['01M01:100:f1','01M01:200:f1'],'fault_code':['f1']*2})
        with patch('src.group_validation45.event_table',return_value=ev):r=operational(f,.5)
        self.assertEqual(r['eligible_events'],2);self.assertEqual(r['detected_events'],1)

if __name__=='__main__':unittest.main()
