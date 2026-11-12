"""Read-only reuse of frozen TEP artifacts. All writes stay in experiment output roots."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
CFG=json.loads((HERE/'config.json').read_text())
OUT=ROOT/'reports/sensor_availability'
CACHE=HERE/'cache'
for p in [OUT,OUT/'tables',OUT/'sources',OUT/'figures',CACHE]:p.mkdir(parents=True,exist_ok=True)
from src.evaluation.phase2 import episodes,binary_metrics

def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def protected_snapshot():
    files=[]
    for folder in ['src','tools','tests','configs','models','notebooks']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.extend(p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.md','.txt','.ps1','.RData'])
    for name in ['testing.npy','training.npy','primary_dense_probabilities.npy','split_manifest.json','sensor_names.json','experiment_config.json']:
        files.append(ROOT/'data/processed/phase2'/name)
    return {str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(set(files))}

def raw_mask(first=141,last=960):
    sample=np.arange(1,961)
    return ~((sample>=first)&(sample<=last))

def valid_windows(sensor_present,window=20):
    """Each endpoint requires every time and all52 channels; no future values."""
    present=np.asarray(sensor_present,dtype=bool)
    if present.ndim==2:present=present.all(axis=1)
    if present.ndim!=1:raise ValueError('Expected time or time-by-channel availability')
    counts=np.r_[0,np.cumsum(present,dtype=int)]
    return (counts[window:]-counts[:-window])==window

def selected_runs(rate,seed):
    """Exactly round(rate*10000) selected, per-fault counts within one, nested by rate."""
    total=int(round(rate*10000));base,extra=divmod(total,20)
    allocation=np.full(20,base,dtype=int)
    extra_faults=np.random.default_rng(seed).permutation(20)[:extra]
    allocation[extra_faults]+=1
    mask=np.zeros((20,500),bool)
    for fault in range(20):
        order=np.random.default_rng(np.random.SeedSequence([seed,fault+1])).permutation(500)
        mask[fault,order[:allocation[fault]]]=True
    assert mask.sum()==total and np.ptp(mask.sum(1))<=1
    return mask

def score_array():
    p=np.load(ROOT/'data/processed/phase2/primary_dense_probabilities.npy',mmap_mode='r')
    return (1-p[:,0]).reshape(21,500,941).copy()

def evaluate_event(scores,valid,fault,run,threshold):
    sample=np.arange(20,961);post=sample>160
    available=int(valid[post].sum());fraction=available/800
    # False here means no observed alarm at that grid point, NOT a predicted normal label.
    # Keeping the grid prevents alarms on opposite sides of a gap being concatenated.
    observed_alarm=np.zeros(941,dtype=bool)
    observed_alarm[valid]=scores[valid]>=threshold
    events=episodes(observed_alarm[post])
    first=int(sample[post][events[0][0]]) if events else None
    return dict(fault=fault,run=run,valid_post_windows=available,input_available_fraction=fraction,
        status='UNAVAILABLE' if available==0 else 'PARTIAL' if available<800 else 'FULL',
        evaluable=available>0,detected=bool(events),first_detection_sample=first,
        delay_minutes=(first-160)*3 if first is not None else None)

def normal_burden(scores,threshold):
    # Only Fault0 full normal runs; those runs are never masked in this experiment.
    count=sum(len(episodes(s>=threshold)) for s in scores[0])
    hours=scores[0].size*3/60
    return dict(normal_runs=500,normal_valid_windows=int(scores[0].size),normal_alarm_episodes=count,
        normal_exposure_hours=hours,normal_alarm_episodes_per_hour=count/hours,
        normal_sample_false_alarm_rate=float((scores[0]>=threshold).mean()))
