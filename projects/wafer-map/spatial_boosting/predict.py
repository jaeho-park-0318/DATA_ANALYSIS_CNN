"""Usage: python predict.py map.npy --model results/spatial_model.joblib"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.packages'))
import argparse,json
import numpy as np
import joblib
from wafer_features import extract_features

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('map'); p.add_argument('--model',default=str(ROOT/'results/spatial_model.joblib')); a=p.parse_args()
    wafer=np.load(a.map,allow_pickle=False)
    if wafer.ndim!=2 or wafer.size==0 or not np.isin(wafer,[0,1,2]).all():
        raise ValueError('Expected a 2D wafer map with 0=outside, 1=good die, 2=defect die')
    bundle=joblib.load(a.model)
    probs=bundle['model'].predict_proba(extract_features(wafer)[None])[0]
    print(json.dumps({'prediction':bundle['classes'][int(probs.argmax())],'scores':dict(zip(bundle['classes'],probs.tolist()))},indent=2))
