"""Causal, paired 20x52 -> 1x52 comparison; outputs confined to this folder."""
from pathlib import Path
import os, sys, json, time, argparse, hashlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
os.environ.setdefault('HF_HOME',str(HERE/'hf_cache'))
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'mpl_cache'))
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib

CACHE=ROOT/'data/processed/phase2'
NAMES=json.loads((CACHE/'sensor_names.json').read_text())
COHORTS=['all','healthy_context','post_fault','faultfree_only']
MODELS=['Persistence','LSTM','TimesFM']

class NextStepLSTM(nn.Module):
    def __init__(self):
        super().__init__();self.lstm=nn.LSTM(52,64,num_layers=2,dropout=.1,batch_first=True);self.delta=nn.Linear(64,52)
    def forward(self,x):return x[:,-1]+self.delta(self.lstm(x)[0][:,-1])

def data_setup():
    scaler=joblib.load(ROOT/'experiments/next_step_forecasting/models/scaler.joblib')
    split=json.loads((CACHE/'split_manifest.json').read_text())
    assert len(split['train'])==400 and len(split['validation'])==100 and len(split['test'])==500
    assert not set(split['train']) & set(split['validation'])
    assert sorted(split['train']+split['validation'])==list(range(500))
    assert NAMES==[f'xmeas_{i}' for i in range(1,42)]+[f'xmv_{i}' for i in range(1,12)]
    train=np.load(CACHE/'training.npy',mmap_mode='r');test=np.load(CACHE/'testing.npy',mmap_mode='r')
    assert train.shape==(21,500,500,52) and test.shape==(21,500,960,52)
    normal=np.asarray(train[0,split['train']]).reshape(-1,52)
    np.testing.assert_allclose(normal.mean(0,dtype=np.float64),scaler.mean_,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(normal.std(0,dtype=np.float64),scaler.scale_,atol=1e-10,rtol=1e-10)
    assert scaler.n_samples_seen_==200000
    return scaler,test

def windows(trajectory,endpoints,mean,scale):
    assert trajectory.shape==(960,52)
    assert endpoints.min()>=19 and endpoints.max()<959
    idx=endpoints[:,None]+np.arange(-19,1)[None,:]
    assert np.all(idx.max(1)<endpoints+1)
    x=((trajectory[idx]-mean)/scale).astype(np.float32)
    y=((trajectory[endpoints+1]-mean)/scale).astype(np.float32)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return x,y

def lstm_model(device):
    model=NextStepLSTM().to(device)
    ck=torch.load(ROOT/'experiments/next_step_forecasting/models/lstm_s42.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['state_dict']);model.eval();return model

def timesfm_model(device,batch):
    from timesfm3 import TimesFM3Evaluator,ModelConfig
    return TimesFM3Evaluator(ModelConfig(checkpoint_path=str(HERE/'checkpoint'),device=device,per_core_batch_size=batch,local_files_only=True))

def tf_predict(model,x,precision):
    # Official evaluator: 52 targets -> first32 + last20 padded to32.
    # Shape of each context is (variates, time), NOT (time, variates).
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
        out=list(model.predict_batch([np.ascontiguousarray(w.T) for w in x],horizon=1,
            return_quantiles=False,use_symmetric_averaging=False,make_positive=False,
            sort_quantiles=True,use_znorm=False,padding_mode='none'))
    p=np.stack([o.forecast[:,0] for o in out]).astype(np.float32)
    assert p.shape==(len(x),52) and np.isfinite(p).all()
    return p

def pilot(batch=32,precision='fp32'):
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    scaler,test=data_setup();mean=scaler.mean_.astype(np.float32);scale=scaler.scale_.astype(np.float32)
    tf=timesfm_model(device,batch);lstm=lstm_model(device)
    ep=np.array([19,158,159,160,170,300,600,958])
    x,y=windows(test[1,0],ep,mean,scale)
    start=time.perf_counter();p=tf_predict(tf,x,precision);elapsed=time.perf_counter()-start
    # Batch size / order must not mix trajectories or change outputs.
    solo=tf_predict(tf,x[:1],precision)
    np.testing.assert_allclose(solo,p[:1],atol=5e-4 if precision=='bf16' else 2e-5,rtol=5e-4 if precision=='bf16' else 2e-5)
    trajectory=np.array(test[1,0]);trajectory[20:]+=10000
    changed,_=windows(trajectory,np.array([19]),mean,scale)
    np.testing.assert_array_equal(changed,x[:1])
    np.testing.assert_allclose(tf_predict(tf,changed,precision),solo,atol=1e-6,rtol=1e-6)
    with torch.inference_mode():lp=lstm(torch.from_numpy(x).to(device)).cpu().numpy()
    np.savez_compressed(HERE/'pilot_predictions.npz',endpoints=ep,actual=y,persistence=x[:,-1],lstm=lp,timesfm=p)
    # Steady-state throughput on genuine observations, same fixed inference config.
    bx,by=windows(test[1,1],np.arange(19,19+batch),mean,scale)
    start=time.perf_counter();bp=tf_predict(tf,bx,precision);seconds=time.perf_counter()-start
    result={'passed':True,'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
        'precision':precision,'batch':batch,'pilot_windows':len(ep),'first_call_seconds':elapsed,
        'steady_windows':len(bx),'steady_seconds':seconds,'windows_per_second':len(bx)/seconds,
        'estimated_dense_hours':9870000*seconds/len(bx)/3600,
        'batch_independence_checked':True,'future_perturbation_checked':True,
        'scaler_train_only_verified':True,'train_validation_family_disjoint':True,
        'official_test_separate_rng_family':True,'input_shape':[20,52],'horizon':1,
        'timesfm_native_multivariate':True,'official_evaluator_chunks':[32,20],
        'checkpoint_sha':json.loads((HERE/'sources/checkpoint_info.json').read_text())['sha']}
    (HERE/'pilot_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

def save_tables(acc,scale,mode,status,n_expected,dest=None):
    summary=[];variables=[]
    for (model,cohort),(n,sse,sae) in sorted(acc.items()):
        summary.append(dict(model=model,cohort=cohort,mode=mode,status=status,target_rows=n,expected_target_rows=n_expected if cohort=='all' else None,
            normalized_rmse=float(np.sqrt(sse.sum()/(n*52))),normalized_mae=float(sae.sum()/(n*52)),unit='normal Training standard deviations'))
        for j,name in enumerate(NAMES):
            variables.append(dict(model=model,cohort=cohort,mode=mode,status=status,variable=name,target_rows=n,
                normalized_rmse=float(np.sqrt(sse[j]/n)),normalized_mae=float(sae[j]/n),
                raw_rmse=float(np.sqrt(sse[j]/n)*scale[j]),raw_mae=float(sae[j]/n*scale[j]),raw_unit='original dataset units (not independently verified)'))
    dest=dest or (HERE if mode=='dense' else HERE/mode)
    dest.mkdir(exist_ok=True)
    pd.DataFrame(summary).to_csv(dest/'model_comparison.csv',index=False)
    pd.DataFrame(variables).to_csv(dest/'variable_metrics.csv',index=False)

def evaluate(mode='dense',batch=32,precision='fp32',max_runs=None,baseline_only=False,runs_per_fault=500,tag=None):
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    scaler,test=data_setup();mean=scaler.mean_.astype(np.float32);scale=scaler.scale_.astype(np.float32)
    endpoints=np.arange(19,959,10 if mode in ['common','representative'] else 1);samples=endpoints+2
    dest=HERE/tag if tag else (HERE if mode in ['dense','representative'] else HERE/mode)
    if baseline_only:dest=dest/'baselines'
    dest.mkdir(parents=True,exist_ok=True)
    checkpoint=dest/('baseline_progress.npz' if baseline_only else 'progress.npz')
    meta_path=dest/('baseline_progress.json' if baseline_only else 'progress.json')
    config=dict(mode=mode,batch=batch,precision=precision,baseline_only=baseline_only,max_runs=max_runs,runs_per_fault=runs_per_fault,
        window=20,horizon=1,seed_lstm=42,inference_flags={'symmetric_averaging':False,'make_positive':False,'sort_quantiles':True,'use_znorm':False,'padding_mode':'none'})
    active=MODELS[:2] if baseline_only else MODELS
    acc={};done=0;prior_seconds=0
    if checkpoint.exists():
        z=np.load(checkpoint);meta=json.loads(str(z['metadata_json']));assert meta['config']==config,'Resume configuration mismatch'
        done=meta['completed_trajectories'];prior_seconds=meta['seconds']
        for key in meta['keys']:
            n,sse,sae=z[key+'__n'],z[key+'__sse'],z[key+'__sae'];acc[tuple(key.split('|'))]=[int(n),sse,sae]
    lstm=lstm_model(device);tf=None if baseline_only else timesfm_model(device,batch)
    assert 1<=runs_per_fault<=500
    run_ids=(np.sort(np.random.default_rng(20261112).choice(500,runs_per_fault,replace=False)).tolist()
        if mode=='representative' else list(range(runs_per_fault)))
    trajectory_ids=[f*500+r for r in run_ids for f in range(21)]
    if max_runs is not None:trajectory_ids=trajectory_ids[:max_runs]
    total=len(trajectory_ids)
    expected=len(endpoints)*total
    sampling={'mode':mode,'seed':20261112,'simulationRuns':[r+1 for r in run_ids],
        'faultNumbers':list(range(21)),'trajectories':total,'targets_per_trajectory':len(endpoints),
        'expected_target_rows':expected,'full_dense_target_rows':9870000,
        'time_sampling':'one uniformly random endpoint within each consecutive 10-endpoint bin, independently per trajectory' if mode=='representative' else 'fixed stride',
        'selection_independent_of_values_and_errors':True}
    (dest/'sampling_manifest.json').write_text(json.dumps(sampling,indent=2),encoding='utf-8')
    start=time.perf_counter()
    examples=[]
    for trajectory_id in range(done,total):
        fault,r=divmod(trajectory_ids[trajectory_id],500)
        if mode=='representative':
            rng=np.random.default_rng(np.random.SeedSequence([20261112,fault,r]))
            endpoints=np.arange(19,959,10)+rng.integers(0,10,size=94);samples=endpoints+2
        x,y=windows(test[fault,r],endpoints,mean,scale)
        with torch.inference_mode():lp=lstm(torch.from_numpy(x).to(device)).cpu().numpy()
        preds={'Persistence':x[:,-1], 'LSTM':lp}
        if tf is not None:
            preds['TimesFM']=tf_predict(tf,x,precision)
        post=(samples>160)&(fault>0)
        masks={'all':np.ones(len(x),bool),'healthy_context':~post,'post_fault':post,'faultfree_only':np.full(len(x),fault==0)}
        for name,p in preds.items():
            error=(p-y).astype(np.float64)
            for cohort,mask in masks.items():
                if not mask.any():continue
                key=(name,cohort)
                if key not in acc:acc[key]=[0,np.zeros(52),np.zeros(52)]
                acc[key][0]+=int(mask.sum());acc[key][1]+=np.square(error[mask]).sum(0);acc[key][2]+=np.abs(error[mask]).sum(0)
        # Fixed illustrative cases, never selected on error magnitude.
        if r==run_ids[0] and fault in [0,1,9,15] and not baseline_only:
            np.savez_compressed(dest/f'example_fault{fault}_run{r+1}.npz',target_sample=samples,actual=y,**preds)
        if not baseline_only:
            pred_dir=dest/'predictions';pred_dir.mkdir(exist_ok=True)
            np.savez_compressed(pred_dir/f'fault{fault:02d}_run{r+1:03d}.npz',target_sample=samples,
                actual=y,**preds)
        if (trajectory_id+1)%10==0 or trajectory_id==total-1:
            seconds=prior_seconds+time.perf_counter()-start
            arrays={}
            for key,(n,sse,sae) in acc.items():
                k='|'.join(key);arrays[k+'__n']=n;arrays[k+'__sse']=sse;arrays[k+'__sae']=sae
            meta={'config':config,'completed_trajectories':trajectory_id+1,'total_trajectories':total,'seconds':seconds,'keys':['|'.join(k) for k in acc],
                'status':('complete' if total==10500 else 'sample_complete') if trajectory_id+1==total else 'running','expected_target_rows':expected}
            arrays['metadata_json']=json.dumps(meta)
            tmp=checkpoint.with_name(checkpoint.stem+'_tmp.npz');np.savez(tmp,**arrays);tmp.replace(checkpoint)
            mt=meta_path.with_suffix('.tmp');mt.write_text(json.dumps(meta,indent=2),encoding='utf-8');mt.replace(meta_path)
            save_tables(acc,scale,mode,meta['status'],expected,dest)
            print(f'{mode} trajectories {trajectory_id+1}/{total}; {seconds:.1f}s; fault={fault}; run={r+1}',flush=True)
    if total==10500 and mode in ['common','dense']:
        df=pd.read_csv(dest/'model_comparison.csv')
        old=pd.read_csv(ROOT/'reports/next_step_forecasting/tables/test_summary.csv')
        for model in MODELS[:2]:
            for cohort in COHORTS:
                a=df[(df.model==model)&(df.cohort==cohort)].iloc[0]
                b=old[(old.seed==42)&(old['mode']==mode)&(old.model==model)&(old.cohort==cohort)].iloc[0]
                assert a.target_rows==b.target_rows
                np.testing.assert_allclose([a.normalized_rmse,a.normalized_mae],[b.normalized_rmse,b.normalized_mae],rtol=2e-6,atol=1e-7)
        (dest/'baseline_reproduction.json').write_text(json.dumps({'passed':True,'rtol':2e-6,'all_cohorts':True,'target_rows':expected},indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--mode',choices=['dense','common','representative'],default='dense')
    p.add_argument('--batch',type=int,default=32);p.add_argument('--precision',choices=['fp32','bf16'],default='fp32');p.add_argument('--max-runs',type=int)
    p.add_argument('--baseline-only',action='store_true');p.add_argument('--runs-per-fault',type=int,default=500);p.add_argument('--tag');args=p.parse_args()
    if args.pilot:pilot(args.batch,args.precision)
    else:evaluate(args.mode,args.batch,args.precision,args.max_runs,args.baseline_only,args.runs_per_fault,args.tag)
