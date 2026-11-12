"""Past-only features, aligned endpoints, no crossing of simulation runs."""
from pathlib import Path
import sys, json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.data.phase2 import CACHE,CFG,trajectories,split_runs,nominal_labels

def windows(x,endpoints,window=20):
    endpoints=np.asarray(endpoints,dtype=int)
    if np.any(endpoints<window-1) or np.any(endpoints>=len(x)): raise ValueError('Invalid past window endpoint')
    return x[endpoints[:,None]+np.arange(-window+1,1)[None,:]]

def statistical(w):
    return np.concatenate([w[:,-1],w.mean(axis=1),w.std(axis=1),w.min(axis=1),w.max(axis=1),(w[:,-1]-w[:,0])/57.0],axis=1).astype(np.float32)

def names():
    sensors=json.loads((CACHE/'sensor_names.json').read_text())
    return [f'{summary}__{s}' for summary in ['current','mean60min','std60min','min60min','max60min','slope_per_minute'] for s in sensors]

def build(split):
    dest=CACHE/f'{split}_features.npz'
    if dest.exists(): print('Feature cache exists',split,flush=True);return
    x=trajectories(split);ids=split_runs(split)
    stride=CFG['training_stride'] if split=='train' else CFG['comparison_stride']
    endpoint=np.arange(CFG['window']-1,x.shape[2],stride)
    features=[];ys=[];metadata=[]
    for fault in range(21):
        for run in ids:
            e=endpoint
            # A shared normal prefix is represented only by the fault-free family in training.
            if split=='train' and fault>0:e=e[e+1>20]
            w=windows(x[fault,run],e,CFG['window'])
            features.append(statistical(w));ys.append(nominal_labels(fault,e+1,split))
            metadata.append(np.column_stack([np.full(len(e),fault),np.full(len(e),run),e+1]))
        print('Features',split,'fault',fault,flush=True)
    np.savez(dest,X=np.concatenate(features),y=np.concatenate(ys),meta=np.concatenate(metadata).astype(np.int32))

def load(split):
    d=np.load(CACHE/f'{split}_features.npz')
    return d['X'],d['y'],d['meta']

if __name__=='__main__':
    for split in ['train','validation','test']:build(split)
