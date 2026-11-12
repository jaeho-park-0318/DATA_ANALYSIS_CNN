"""Consequential checks against the delivered real-data audit products."""
from pathlib import Path
import json, unittest
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
T=ROOT/'reports/tables'

class Phase1Validation(unittest.TestCase):
    def test_all_lfs_objects_verified(self):
        logs=json.loads((ROOT/'reports/sources/github_download.json').read_text())
        self.assertEqual(len(logs),214)
        self.assertTrue(all(v['status']=='verified' for v in logs))

    def test_a_schema_time_and_labels(self):
        a=pd.read_csv(T/'a_inventory.csv')
        self.assertEqual(len(a),214)
        self.assertTrue(a['columns'].eq(42).all())
        self.assertTrue(np.allclose(a.dt_hours,1/60))
        self.assertEqual(a.nonmonotonic.sum(),0)
        self.assertEqual(a.nonfinite.sum(),0)
        f=a[a.fault>0]
        self.assertEqual(set(f.fault),set(range(1,22)))
        self.assertTrue(f.groupby('fault').run.nunique().eq(10).all())

    def test_b_official_checksums_and_sample_integrity(self):
        b=pd.read_csv(T/'b_inventory.csv')
        self.assertEqual(len(b),4)
        self.assertTrue(b.md5_verified.all())
        self.assertEqual(b.rows.sum(),15330000)
        runs=pd.read_csv(T/'b_all_runs.csv')
        self.assertEqual(len(runs),21000)
        self.assertEqual(runs.duplicate_keys.sum(),0)
        self.assertEqual(runs.sample_errors.sum(),0)
        for filename,g in runs.groupby('file'):
            expected=500 if 'Training' in filename else 960
            self.assertTrue(g.rows.eq(expected).all())
            self.assertTrue(g.sample_min.eq(1).all())
            self.assertTrue(g.sample_max.eq(expected).all())

    def test_duplicate_and_shared_prefix_risks_are_exposed(self):
        a=pd.read_csv(T/'a_run_hashes.csv')
        self.assertEqual(a.full.nunique(),168)
        b=pd.read_csv(T/'b_all_runs.csv')
        self.assertEqual(b.prefix20.nunique(),1000)
        train=set(b[b.file.str.contains('Training')].prefix20)
        test=set(b[b.file.str.contains('Testing')].prefix20)
        self.assertFalse(train & test)
        for _,g in b.groupby('file'):
            if 'Faulty' in g.iloc[0].file:
                self.assertTrue(g.groupby('run').prefix20.nunique().eq(1).all())

    def test_unknown_causes_and_phase2_gate(self):
        fc=pd.read_csv(T/'fault_catalog.csv')
        self.assertTrue(fc[fc.fault.between(16,20)].cause.eq('UNKNOWN').all())
        self.assertEqual(fc[fc.fault==21].form.iloc[0],'CONFLICT')
        config=json.loads((ROOT/'configs/datasets.json').read_text())
        self.assertIsInstance(config['phase2_training_authorized'],bool)
        if config['phase2_training_authorized']:
            self.assertTrue((ROOT/'configs/phase2.json').exists())
        self.assertEqual(len(list((ROOT/'reports/figures').glob('fault_??_analysis.png'))),21)

    def test_nominal_injection_is_supported_by_model_wiring(self):
        import re
        model=(ROOT/'tennessee-eastman-dataset-main/simulator/MultiLoop_mode1.mdl').read_text()
        self.assertRegex(model,r'BlockType="VariableTransportDelay"[^>]*SID="250"')
        self.assertRegex(model,r'Name="Constant" SID="249">[\s\S]*?<P Name="Value">10</P>')
        for src,dst in [('3#out:1','250#in:1'),('249#out:1','250#in:2'),('250#out:1','31#in:13')]:
            self.assertTrue(any(src in chunk and dst in chunk for chunk in re.findall(r'<Line>[\s\S]*?</Line>',model)))

if __name__=='__main__':unittest.main()
