"""Check saved artifact against cached held-out predictions and input edge cases."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.packages'))
import numpy as np
import pandas as pd
import joblib
from wafer_features import extract_features
d=np.load(ROOT/'results/features.npz'); s=np.load(ROOT/'results/split_indices.npz')
b=joblib.load(ROOT/'results/spatial_model.joblib')
record=pd.read_csv(ROOT/'results/test_predictions.csv')
pred=b['model'].predict(d['X'][s['test']])
assert [b['classes'][int(i)] for i in pred]==record.predicted.tolist()
assert not set(d['groups'][s['fit']]) & set(d['groups'][s['validation']])
for shape in [(1,1),(5,9),(52,52),(100,80)]:
    for value in [0,1,2]:
        x=extract_features(np.full(shape,value,dtype=np.uint8))
        assert len(x)==d['X'].shape[1] and np.isfinite(x).all()
(ROOT/'results/verification.txt').write_text('PASS: serialized model reproduces all 7,894 Test predictions.\nPASS: fit/validation lots disjoint.\nPASS: finite, fixed-length features for 12 input edge cases.\n')
print('PASS')
