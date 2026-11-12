from pathlib import Path
import sys,json,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.phase2 import CACHE,nominal_labels,trajectories,split_runs
from src.features.phase2 import windows,statistical,load
from src.evaluation.phase2 import episodes,threshold

class LeakageTests(unittest.TestCase):
    def test_simulation_families_are_disjoint(self):
        a=set(split_runs('train'));b=set(split_runs('validation'))
        self.assertFalse(a&b);self.assertEqual(a|b,set(range(500)))
        runs=pd.read_csv(ROOT/'reports/tables/b_all_runs.csv')
        train=runs[runs.file.str.contains('Training') & runs.run.sub(1).isin(a)]
        val=runs[runs.file.str.contains('Training') & runs.run.sub(1).isin(b)]
        test=runs[runs.file.str.contains('Testing')]
        self.assertFalse(set(train.full)&set(val.full))
        self.assertFalse(set(train.prefix20)&set(val.prefix20))
        self.assertFalse(set(train.prefix20)&set(test.prefix20))
        self.assertFalse(set(val.prefix20)&set(test.prefix20))

    def test_real_future_changes_do_not_change_past_features(self):
        x=np.array(trajectories('train')[1,split_runs('train')[0]])
        original=statistical(windows(x,[99]))
        x[100:]+=12345
        np.testing.assert_array_equal(original,statistical(windows(x,[99])))

    def test_windows_cannot_cross_run_start_or_end(self):
        x=trajectories('train')[0,0]
        with self.assertRaises(ValueError):windows(x,[18])
        with self.assertRaises(ValueError):windows(x,[500])
        w=windows(x,[19]);np.testing.assert_array_equal(w[0],x[:20])

    def test_boundary_labels(self):
        np.testing.assert_array_equal(nominal_labels(7,[19,20,21,22],'train'),[0,0,7,7])
        np.testing.assert_array_equal(nominal_labels(7,[159,160,161,162],'test'),[0,0,7,7])
        np.testing.assert_array_equal(nominal_labels(0,[1,20,161,960],'test'),[0,0,0,0])

    def test_metadata_and_shared_training_prefix(self):
        x,y,m=load('train')
        self.assertEqual(x.shape[1],312)
        self.assertFalse(np.any((m[:,0]>0)&(m[:,2]<=20)))
        self.assertTrue(set(m[:,1])<=set(split_runs('train')))
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(len(set(y)),21)

    def test_alarm_confirmation_and_episode_merging(self):
        self.assertEqual(episodes([0,1,1,1,1,0,1,1,0,1,1,1]),[(3,5),(11,12)])

    def test_scaler_is_fitted_only_to_training(self):
        import joblib
        scalerfile=ROOT/'models/phase2/temporal_scaler.joblib'
        if not scalerfile.exists():self.skipTest('Temporal training has not created scaler yet')
        scaler=joblib.load(scalerfile);x,_,_=load('train')
        np.testing.assert_allclose(scaler.mean_,x[:,:52].mean(axis=0,dtype=np.float64),rtol=1e-10,atol=1e-10)
        self.assertEqual(scaler.n_samples_seen_,len(x))

    def test_held_out_faults_never_enter_unknown_model_training(self):
        files=list((ROOT/'models/phase2').glob('Unknown*.json'))
        if not files:self.skipTest('Unknown experiment not yet trained')
        for p in files:
            v=json.loads(p.read_text())
            self.assertFalse(set(v['unknown_faults']) & set(v['training_fault_labels']))

if __name__=='__main__':unittest.main()
