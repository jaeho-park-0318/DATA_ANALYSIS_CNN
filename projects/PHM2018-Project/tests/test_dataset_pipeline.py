"""분석 결과가 아닌, 라벨 경계·미래 누수·청크 일관성에 대한 단위 테스트."""
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.build_features import INPUT_COLUMNS, make_features
from src.build_labels import classify
from src.dataset_common import DEST, LOOKBACK, GAP, feature_columns, read_batches, temporal_split


class PipelineTests(unittest.TestCase):
    def test_label_evidence_gap_and_fault_boundary(self):
        t=np.zeros(6)
        delta=np.array([100,400,100,400,100,0],dtype=float)
        evidence=np.array([True,True,False,True,True,True])
        end=np.array([100,1000,1000,250,70,1000],dtype=float)
        labels=classify(t,delta,evidence,end,np.zeros(6,dtype=bool),np.array([False]*5+[True]),300)
        self.assertEqual(labels[:2].tolist(),[1,0])
        self.assertTrue(labels[2:].isna().all())

    def test_future_perturbation_and_chunk_equivalence(self):
        # 실제 대표 장비의 첫 연속 청크 일부를 사용한다. 무작위 행 추출은 하지 않는다.
        data=next(read_batches(DEST/"observations/01M01",INPUT_COLUMNS)).head(4000).copy()
        baseline=make_features(data)
        cutoff=float(data.time.iloc[2500])
        altered=data.copy()
        for column in feature_columns():
            if "__current" in column:
                sensor=column.split("__")[0]
                altered.loc[altered.time>cutoff,sensor]+=1000
        changed=make_features(altered)
        cols=["row_id"]+feature_columns()+["feature_ready"]
        assert_frame_equal(baseline.loc[baseline.time<=cutoff,cols].reset_index(drop=True),
                           changed.loc[changed.time<=cutoff,cols].reset_index(drop=True))
        pieces=[];halo=None
        for start in range(0,len(data),137):
            fresh=data.iloc[start:start+137]
            joined=pd.concat([halo,fresh],ignore_index=True) if halo is not None else fresh
            piece=make_features(joined,len(halo) if halo is not None else 0)
            pieces.append(piece)
            halo=joined[joined.time>=joined.time.iloc[-1]-LOOKBACK-GAP].copy()
        chunked=pd.concat(pieces,ignore_index=True)
        # 이력이 충분한 행에 대해 청크 크기가 결과를 바꾸지 않는지 확인한다.
        expected=baseline[baseline.feature_ready].set_index("row_id")[feature_columns()]
        actual=chunked[chunked.feature_ready].set_index("row_id")[feature_columns()]
        self.assertGreater(len(expected),0)
        assert_frame_equal(expected,actual,check_exact=False,atol=2e-6,rtol=2e-6)
        self.assertTrue((baseline.source_max_time<=baseline.time).all())

    def test_feature_input_rejects_target(self):
        data=next(read_batches(DEST/"observations/01M01",INPUT_COLUMNS)).head(200).copy()
        data["TTF_fake"]=0
        with self.assertRaises(ValueError):make_features(data)

    def test_temporal_purge(self):
        times=np.array([600,10100,10300,12000,12500,16000,16600],dtype=float)
        splits=temporal_split(times,0,20000)
        self.assertEqual(splits.tolist(),["train","train","purged","purged","purged","purged","evaluation"])

    def test_model_loader_selects_only_features(self):
        from src.dataset_loader import iter_model_batches
        x,y=next(iter_model_batches("label_any_h300raw","train",batch_size=256))
        self.assertEqual(x.columns.tolist(),feature_columns())
        self.assertEqual(len(x),len(y))
        self.assertFalse(y.isna().any())
        self.assertTrue(y.isin([0,1]).all())


if __name__=="__main__":unittest.main()
