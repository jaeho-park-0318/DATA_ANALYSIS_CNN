"""SHAP attribution and measured response are kept separate from physical causes."""
from pathlib import Path
import sys,json,time
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
import shap
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.phase2 import trajectories,split_runs,CACHE
from src.features.phase2 import load,statistical,names
from src.evaluation.run_phase2 import Predictor,M,T

class ClassLogit(nn.Module):
    def __init__(self,model,output):super().__init__();self.model=model;self.output=output
    def forward(self,x):return self.model(x)[:,self.output:self.output+1]

def run():
    lock=json.loads((M/'selection_lock.json').read_text());primary=Predictor(lock['primary_model'])
    x,y,meta=load('train');rawtrain=trajectories('train');rawtest=trajectories('test')
    sensors=json.loads((CACHE/'sensor_names.json').read_text())
    # Reference distribution uses only fault-free Training families allocated to Train.
    healthy=np.asarray(rawtrain[0,split_runs('train')]).reshape(-1,52)
    mu=healthy.mean(axis=0,dtype=np.float64);sd=healthy.std(axis=0,dtype=np.float64);sd[sd==0]=1
    np.savez(CACHE/'healthy_train_reference.npz',mean=mu,std=sd)
    responses=[];rows=[]
    rng=np.random.default_rng(42)
    bgix=rng.choice(len(y),64,replace=False)
    bgmeta=meta[bgix]
    background=rawtrain[bgmeta[:,0,None],bgmeta[:,1,None],bgmeta[:,2,None]-20+np.arange(20)[None,:]]
    if primary.feature=='Temporal':
        bgnorm=((background-primary.scaler.mean_.astype(np.float32))/primary.scaler.scale_.astype(np.float32)).astype(np.float32)
        bg=torch.from_numpy(bgnorm).to(primary.device)
    baseline=pd.read_csv(T/'validation_baselines.csv')
    tree_candidates=baseline[baseline.family=='XGBoost'].groupby(['feature','weight']).selection_score.mean().sort_values(ascending=False)
    treefeat,treeweight=tree_candidates.index[0]
    treeid=f'XGBoost_{treefeat}_{treeweight}_s42';tree=Predictor(treeid)
    tree_explainer=shap.TreeExplainer(tree.model,feature_perturbation='tree_path_dependent',model_output='raw')
    for fault in range(1,21):
        data=np.asarray(rawtest[fault])
        pre=data[:,120:160].mean(axis=1,dtype=np.float64);post=data[:,160:].mean(axis=1,dtype=np.float64)
        delta=((post-pre)/sd).mean(axis=0)
        for j,sensor in enumerate(sensors):responses.append(dict(fault=fault,variable=sensor,post_minus_pre_over_healthy_train_sd=float(delta[j]),pre_window='samples121:160',post_window='samples161:960',runs=500))
        # A reproducible explanatory cohort: 5 Test runs and 3 established post-injection observation times.
        mm=np.array([(fault,r,s) for r in range(5) for s in [240,480,960]],dtype=int)
        w=rawtest[mm[:,0,None],mm[:,1,None],mm[:,2,None]-20+np.arange(20)[None,:]]
        f=statistical(w)
        tx=f[:,:52] if tree.feature=='Current' else f
        values=np.asarray(tree_explainer.shap_values(tx))
        att=np.abs(values[:,:,fault]).mean(axis=0)
        grouped=att.reshape(-1,52).sum(axis=0)
        for rank,j in enumerate(np.argsort(grouped)[::-1],1):rows.append(dict(model_id=treeid,method='TreeSHAP raw class logit',fault=fault,rank=rank,variable=sensors[j],mean_absolute_attribution=float(grouped[j]),explained_samples=len(w),known_cause_match='NOT_QUANTIFIED'))
        if primary.feature=='Temporal':
            normalized=((w-primary.scaler.mean_.astype(np.float32))/primary.scaler.scale_.astype(np.float32)).astype(np.float32)
            explained=torch.from_numpy(normalized).to(primary.device)
            wrapper=ClassLogit(primary.model,fault).eval()
            torch.backends.cudnn.enabled=False
            explainer=shap.GradientExplainer(wrapper,bg,batch_size=32)
            vals=np.asarray(explainer.shap_values(explained,nsamples=64,rseed=42))
            torch.backends.cudnn.enabled=True
            if vals.ndim==4:vals=vals[:,:,:,0]
            assert vals.shape==(len(w),20,52),vals.shape
            importance=np.abs(vals).mean(axis=0).sum(axis=0)
            for rank,j in enumerate(np.argsort(importance)[::-1],1):rows.append(dict(model_id=primary.tag,method='GradientSHAP class logit; nsamples64',fault=fault,rank=rank,variable=sensors[j],mean_absolute_attribution=float(importance[j]),explained_samples=len(w),known_cause_match='NOT_QUANTIFIED'))
        print('Explained fault',fault,flush=True)
        pd.DataFrame(rows).to_csv(T/'shap_sensor_rankings.csv',index=False)
        pd.DataFrame(responses).to_csv(T/'b_fault_sensor_response.csv',index=False)
    metadata=dict(primary=primary.tag,tree_model=treeid,background='64 training windows, RNG42',cohort='Test runs1-5; samples240,480,960; 15 windows per fault',
        output='known fault class raw logit, not physical causal effect',gradient_samples=64,units='logit contribution',aggregation='sum absolute contributions across derived summaries or temporal lags',
        cause_accuracy='not computed: direct cause-to-measurement mapping and B simulator fault implementation not fully verified')
    (ROOT/'reports/sources/phase2/explainability.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')

if __name__=='__main__':run()
