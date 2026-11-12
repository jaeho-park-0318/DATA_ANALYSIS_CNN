"""Baseline candidates. Test data are never read by this training stage."""
from pathlib import Path
import sys,json,time,warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.features.phase2 import load
from src.data.phase2 import CFG
from src.evaluation.phase2 import threshold,joint_metrics,binary_metrics
M=ROOT/'models/phase2';M.mkdir(parents=True,exist_ok=True)
T=ROOT/'reports/tables/phase2';T.mkdir(parents=True,exist_ok=True)

def make(family,seed,balanced):
    if family=='LogisticRegression':return LogisticRegression(C=1.,solver='lbfgs',max_iter=250,tol=.001,class_weight='balanced' if balanced else None)
    if family=='RandomForest':return RandomForestClassifier(n_estimators=100,max_depth=20,min_samples_leaf=3,max_features='sqrt',max_samples=.7,n_jobs=8,random_state=seed,class_weight='balanced' if balanced else None)
    return XGBClassifier(n_estimators=100,max_depth=4,learning_rate=.1,subsample=.85,colsample_bytree=.9,
        tree_method='hist',device='cpu',n_jobs=8,random_state=seed,eval_metric='mlogloss',max_bin=128)

def fit_one(family,feature,seed,balanced,X,y,V,vy):
    tag=f'{family}_{feature}_{"balanced" if balanced else "natural"}_s{seed}'
    out=M/(tag+'.joblib');summary=M/(tag+'.json')
    if summary.exists(): return json.loads(summary.read_text())
    cols=52 if feature=='Current' else 312
    scaler=StandardScaler().fit(X[:,:cols]) if family=='LogisticRegression' else None
    xx=scaler.transform(X[:,:cols]) if scaler is not None else X[:,:cols]
    vv=scaler.transform(V[:,:cols]) if scaler is not None else V[:,:cols]
    model=make(family,seed,balanced)
    start=time.perf_counter()
    weights=compute_sample_weight('balanced',y) if balanced and family=='XGBoost' else None
    with threadpool_limits(limits=8):
        with warnings.catch_warnings(record=True) as caught:
            model.fit(xx,y,**({'sample_weight':weights} if weights is not None else {}))
        seconds=time.perf_counter()-start
        p=model.predict_proba(vv).astype(np.float32)
    th=threshold(vy,1-p[:,0]);metrics,_=joint_metrics(vy,p,th,auc=False)
    row=dict(model_id=tag,family=family,feature=feature,seed=seed,weight='balanced' if balanced else 'natural',training_seconds=seconds,
        train_rows=len(y),validation_rows=len(vy),warnings=[str(v.message) for v in caught],**metrics)
    row['selection_score']=(row['diagnosis_macro_f1']+row['detection_f1'])/2
    joblib.dump({'model':model,'scaler':scaler,'feature':feature,'threshold':th,'metadata':row},out,compress=3)
    row['model_bytes']=out.stat().st_size
    summary.write_text(json.dumps(row,indent=2),encoding='utf-8')
    print('VALIDATED',tag,'diag',round(row['diagnosis_macro_f1'],4),'detect',round(row['detection_f1'],4),'sec',round(seconds),flush=True)
    return row

def run():
    X,y,_=load('train');V,vy,_=load('validation');rows=[]
    for family in ['LogisticRegression','RandomForest','XGBoost']:
        for feature in ['Current','Statistical']:
            candidates=[fit_one(family,feature,42,b,X,y,V,vy) for b in [False,True]]
            rows+=candidates
            chosen=max(candidates,key=lambda r:r['selection_score'])
            for seed in [7,2026]:rows.append(fit_one(family,feature,seed,chosen['weight']=='balanced',X,y,V,vy))
            pd.DataFrame(rows).to_csv(T/'validation_baselines.csv',index=False)
    # A separate binary detector tests class weighting/resampling where the imbalance is substantial.
    by=(y>0).astype(int);bvy=(vy>0).astype(int)
    br=[]
    for strategy in ['natural','balanced','undersample','oversample']:
        seeds=CFG['seeds'] if strategy in ['natural','balanced'] else [42]
        for seed in seeds:
            tag=f'BinaryXGBoost_Statistical_{strategy}_s{seed}'
            meta=M/(tag+'.json')
            if meta.exists():br.append(json.loads(meta.read_text()));continue
            rng=np.random.default_rng(seed);idx=np.arange(len(y))
            if strategy in ['undersample','oversample']:
                groups=[np.flatnonzero(by==c) for c in [0,1]]
                n=min(map(len,groups)) if strategy=='undersample' else max(map(len,groups))
                idx=np.concatenate([rng.choice(g,n,replace=len(g)<n) for g in groups]);rng.shuffle(idx)
            model=make('XGBoost',seed,False);model.set_params(eval_metric='logloss')
            start=time.perf_counter();model.fit(X[idx],by[idx],sample_weight=compute_sample_weight('balanced',by[idx]) if strategy=='balanced' else None)
            seconds=time.perf_counter()-start;p=model.predict_proba(V)[:,1];th=threshold(bvy,p)
            row=dict(model_id=tag,family='BinaryXGBoost',feature='Statistical',weight=strategy,seed=seed,training_seconds=seconds,train_rows=len(idx),**binary_metrics(bvy,p,th))
            joblib.dump({'model':model,'scaler':None,'feature':'Statistical','threshold':th,'binary':True,'metadata':row},M/(tag+'.joblib'),compress=3)
            meta.write_text(json.dumps(row,indent=2));br.append(row);print('VALIDATED',tag,round(row['detection_f1'],4),flush=True)
    pd.DataFrame(br).to_csv(T/'validation_binary.csv',index=False)
    # Unknown-fault experiment: neither held-out fault's training rows nor validation outcomes are used to fit/select threshold.
    unknown=CFG['unknown_faults'];keep=~np.isin(y,unknown);vk=~np.isin(vy,unknown)
    for seed in CFG['seeds']:
        tag=f'UnknownBinaryXGBoost_Statistical_balanced_s{seed}'
        meta=M/(tag+'.json')
        if meta.exists():continue
        model=make('XGBoost',seed,False);model.set_params(eval_metric='logloss')
        start=time.perf_counter();model.fit(X[keep],by[keep],sample_weight=compute_sample_weight('balanced',by[keep]))
        th=threshold(bvy[vk],model.predict_proba(V[vk])[:,1])
        row=dict(model_id=tag,seed=seed,unknown_faults=unknown,threshold=th,training_seconds=time.perf_counter()-start,training_fault_labels=sorted(np.unique(y[keep]).tolist()))
        joblib.dump({'model':model,'feature':'Statistical','scaler':None,'threshold':th,'binary':True,'metadata':row},M/(tag+'.joblib'),compress=3)
        meta.write_text(json.dumps(row,indent=2));print('Unknown detector trained',seed,flush=True)

if __name__=='__main__':run()
