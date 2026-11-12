"""Freeze validation choices before any final Test scoring."""
from pathlib import Path
import sys,json,time,datetime,gc
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.data.phase2 import CACHE,CFG,trajectories
from src.features.phase2 import load,windows,statistical
from src.evaluation.phase2 import joint_metrics,binary_metrics,episodes
M=ROOT/'models/phase2';T=ROOT/'reports/tables/phase2';P=ROOT/'reports/sources/phase2'
P.mkdir(parents=True,exist_ok=True)

def freeze():
    path=M/'selection_lock.json'
    if path.exists():return json.loads(path.read_text())
    baseline=pd.read_csv(T/'validation_baselines.csv');deep=pd.read_csv(T/'validation_temporal.csv')
    allrows=pd.concat([baseline,deep],ignore_index=True)
    grouped=allrows.groupby(['family','feature','weight']).agg(seeds=('seed','nunique'),score=('selection_score','mean')).reset_index()
    eligible=grouped[grouped.seeds==3]
    win=eligible.sort_values('score',ascending=False).iloc[0]
    selected=allrows[(allrows.family==win.family)&(allrows.feature==win.feature)&(allrows.weight==win.weight)&(allrows.seed==42)].iloc[0]
    # Dedicated detector selected on validation F1 alone; no diagnostic outcome is involved.
    binary=pd.read_csv(T/'validation_binary.csv')
    bw=binary.groupby('weight').detection_f1.mean().idxmax()
    bchosen=binary[(binary.weight==bw)&(binary.seed==42)].iloc[0]
    lock=dict(locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),primary_model=selected.model_id,
        binary_model=bchosen.model_id,selection_score=float(win.score),selection_criterion=CFG['selection_metric'],
        evaluated_models=allrows.model_id.tolist(),binary_models=binary.model_id.tolist(),unknown_models=[p.stem for p in M.glob('Unknown*.json')],
        config=CFG)
    path.write_text(json.dumps(lock,indent=2),encoding='utf-8')
    print('LOCKED',lock['primary_model'],'binary',lock['binary_model'],flush=True)
    return lock

class Predictor:
    def __init__(self,tag):
        self.tag=tag;self.device='cpu'
        if (M/(tag+'.pt')).exists():
            import torch
            from src.models.temporal import TemporalNet
            self.torch=torch;self.device='cuda' if torch.cuda.is_available() else 'cpu'
            ck=torch.load(M/(tag+'.pt'),map_location='cpu',weights_only=False)
            self.model=TemporalNet(ck['kind'],ck['hidden']).to(self.device);self.model.load_state_dict(ck['state_dict']);self.model.eval()
            self.scaler=joblib.load(M/'temporal_scaler.joblib');self.feature='Temporal';self.th=ck['threshold'];self.binary=False
        else:
            ck=joblib.load(M/(tag+'.joblib'));self.model=ck['model'];self.scaler=ck['scaler'];self.feature=ck['feature'];self.th=ck['threshold'];self.binary=ck.get('binary',False)
    def proba(self,features=None,w=None):
        if self.feature=='Temporal':
            x=((w-self.scaler.mean_.astype(np.float32))/self.scaler.scale_.astype(np.float32)).astype(np.float32)
            parts=[]
            with self.torch.inference_mode():
                for i in range(0,len(x),1024):parts.append(self.model(self.torch.from_numpy(x[i:i+1024]).to(self.device)).softmax(1).cpu().numpy())
            return np.concatenate(parts)
        cols=52 if self.feature=='Current' else 312
        x=features[:,:cols]
        if self.scaler is not None:x=self.scaler.transform(x)
        return self.model.predict_proba(x).astype(np.float32)

def comparison(lock):
    x,y,meta=load('test');raw=trajectories('test')
    rows=[];per_fault=[]
    completed=T/'test_comparison.csv'
    if completed.exists():rows=pd.read_csv(completed).to_dict('records')
    pfpath=T/'test_comparison_per_fault.csv'
    if pfpath.exists():per_fault=pd.read_csv(pfpath).to_dict('records')
    done={r['model_id'] for r in rows}
    for tag in lock['evaluated_models']+lock['binary_models']+lock['unknown_models']:
        if tag in done:continue
        predictor=Predictor(tag);parts=[];start=time.perf_counter()
        for i in range(0,len(x),10000):
            mm=meta[i:i+10000]
            w=raw[mm[:,0,None],mm[:,1,None],mm[:,2,None]-20+np.arange(20)[None,:]] if predictor.feature=='Temporal' else None
            parts.append(predictor.proba(x[i:i+10000],w))
        prob=np.concatenate(parts);elapsed=time.perf_counter()-start
        if predictor.binary:
            metrics=binary_metrics(y,prob[:,1],predictor.th);pred=(prob[:,1]>=predictor.th);score=prob[:,1]
        else:
            metrics,cm=joint_metrics(y,prob,predictor.th)
            np.save(T/(tag+'_confusion.npy'),cm)
            score=1-prob[:,0];pred=score>=predictor.th
        row=dict(model_id=tag,feature=predictor.feature,evaluation='common_endpoints_every_10_samples',inference_seconds=elapsed,**metrics)
        if tag.startswith('Unknown'):
            for kind,mask in [('unseen_faults',np.isin(y,CFG['unknown_faults'])|(y==0)),('seen_faults',~np.isin(y,CFG['unknown_faults']))]:
                subset=binary_metrics(y[mask],score[mask],predictor.th)
                for k,v in subset.items():row[f'{kind}_{k}']=v
        for fault in range(1,21):
            mask=y==fault
            result=dict(model_id=tag,fault=fault,samples=int(mask.sum()),detection_recall=float(pred[mask].mean()))
            if not predictor.binary:
                diag=prob[mask,1:].argmax(axis=1)+1
                result.update(conditional_diagnosis_accuracy=float((diag==fault).mean()),end_to_end_recall=float((pred[mask]&(diag==fault)).mean()))
            per_fault.append(result)
        rows.append(row)
        pd.DataFrame(rows).to_csv(completed,index=False);pd.DataFrame(per_fault).to_csv(pfpath,index=False)
        print('TEST',tag,'detect',round(metrics['detection_f1'],4),'diag',round(metrics.get('diagnosis_macro_f1',0),4),'sec',round(elapsed),flush=True)
        del prob,parts,predictor;gc.collect()

def dense(lock):
    tag=lock['primary_model'];primary=Predictor(tag)
    detector=Predictor(lock['binary_model'])
    raw=trajectories('test');endpoint=np.arange(19,960);n=21*500*len(endpoint)
    probfile=CACHE/'primary_dense_probabilities.npy';binaryfile=CACHE/'binary_dense_scores.npy'
    done=P/'dense_complete.json'
    if not done.exists():
        prob=np.lib.format.open_memmap(probfile,mode='w+',dtype=np.float32,shape=(n,21))
        bs=np.lib.format.open_memmap(binaryfile,mode='w+',dtype=np.float32,shape=(n,))
        offset=0;start=time.perf_counter()
        for fault in range(21):
            for first in range(0,500,10):
                runs=range(first,min(first+10,500))
                w=np.concatenate([windows(raw[fault,r],endpoint) for r in runs])
                f=statistical(w)
                p=primary.proba(f,w);bp=detector.proba(f)[:,1]
                prob[offset:offset+len(w)]=p;bs[offset:offset+len(w)]=bp;offset+=len(w)
            print('DENSE TEST fault',fault,flush=True)
        prob.flush();bs.flush();done.write_text(json.dumps({'primary':tag,'binary':lock['binary_model'],'rows':n,'inference_seconds':time.perf_counter()-start},indent=2))
    prob=np.load(probfile,mmap_mode='r');bs=np.load(binaryfile,mmap_mode='r')
    faultvec=np.repeat(np.arange(21),500*len(endpoint));runvec=np.tile(np.repeat(np.arange(500),len(endpoint)),21)
    samples=np.tile(endpoint+1,21*500);y=np.where(samples>160,faultvec,0).astype(np.int8)
    metrics,cm=joint_metrics(y,prob,primary.th)
    metrics['normal_faultfree_FAR']=float(((1-prob[faultvec==0,0])>=primary.th).mean())
    premask=(faultvec>0)&(samples<=160)
    metrics['pre_injection_FAR']=float(((1-prob[premask,0])>=primary.th).mean())
    bmetrics=binary_metrics(y,bs,detector.th)
    (P/'dense_metrics.json').write_text(json.dumps({'joint_primary':metrics,'dedicated_binary':bmetrics},indent=2))
    np.save(T/'primary_dense_confusion.npy',cm)
    diagnostics=[];alarmrows=[];counts=np.zeros((500,21,21),dtype=np.int64);diagcounts=np.zeros((500,20,20),dtype=np.int64);bcounts=np.zeros((500,2,2),dtype=np.int64)
    per_fault=[]
    for fault in range(21):
        a=fault*500*len(endpoint);b=(fault+1)*500*len(endpoint)
        p=np.asarray(prob[a:b]);yy=y[a:b];sc=1-p[:,0];dp=p[:,1:].argmax(1)+1;pred=np.where(sc>=primary.th,dp,0)
        if fault>0:
            mask=yy>0
            pr,rc,f1,sup=precision_recall_fscore_support(yy[mask],dp[mask],labels=np.arange(1,21),zero_division=0)
            per_fault.append(dict(fault=fault,post_samples=int(mask.sum()),detection_recall=float((sc[mask]>=primary.th).mean()),conditional_diagnosis_recall=float((dp[mask]==fault).mean()),end_to_end_recall=float((pred[mask]==fault).mean())))
        for run in range(500):
            sl=slice(run*len(endpoint),(run+1)*len(endpoint));thisy=yy[sl];thispred=pred[sl];alarm=sc[sl]>=primary.th
            counts[run]+=confusion_matrix(thisy,thispred,labels=np.arange(21))
            positive=thisy>0
            bcounts[run]+=confusion_matrix(positive,alarm,labels=[False,True])
            if fault>0:diagcounts[run]+=confusion_matrix(thisy[positive]-1,dp[sl][positive]-1,labels=np.arange(20))
            for system,scores,th in [('joint_primary',sc[sl],primary.th),('dedicated_binary',np.asarray(bs[a+run*len(endpoint):a+(run+1)*len(endpoint)]),detector.th)]:
                pre=samples[:len(endpoint)]<=160 if fault>0 else np.ones(len(endpoint),bool)
                pre_episodes=episodes(scores[pre]>=th)
                post=~pre
                events=episodes(scores[post]>=th) if fault>0 else []
                post_samples=(endpoint+1)[post]
                detected=bool(events)
                first_sample=int(post_samples[events[0][0]]) if detected else None
                alarmrows.append(dict(system=system,fault=fault,run=run+1,normal_samples=int(pre.sum()),pre_alarm_episodes=len(pre_episodes),
                    detected=detected if fault>0 else None,first_detection_sample=first_sample,
                    detection_delay_minutes=(first_sample-160)*3 if detected else None,
                    post_alarm_episodes=len(events),normal_exposure_hours=pre.sum()*3/60))
    pd.DataFrame(per_fault).to_csv(T/'primary_dense_per_fault.csv',index=False)
    alarm=pd.DataFrame(alarmrows);alarm.to_csv(T/'dense_alarm_episodes.csv',index=False)
    summary=alarm[alarm.fault>0].groupby(['system','fault']).agg(runs=('run','count'),detected_runs=('detected','sum'),median_delay_minutes=('detection_delay_minutes','median'),mean_delay_minutes=('detection_delay_minutes','mean'),p90_delay_minutes=('detection_delay_minutes',lambda s:s.quantile(.9))).reset_index()
    summary['run_detection_rate']=summary.detected_runs/summary.runs;summary['miss_rate']=1-summary.run_detection_rate
    summary.to_csv(T/'dense_detection_delays.csv',index=False)
    # Cluster bootstrap: an entire Test simulationRun family is resampled together across all faults.
    rng=np.random.default_rng(42);boot=[]
    def fscore(c):
        tp=np.diag(c);den=2*tp+c.sum(0)-tp+c.sum(1)-tp
        return float(np.mean(np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0)))
    for _ in range(500):
        ix=rng.integers(0,500,500);c=bcounts[ix].sum(0);d=diagcounts[ix].sum(0);g=counts[ix].sum(0)
        boot.append(dict(detection_f1=2*c[1,1]/max(1,2*c[1,1]+c[0,1]+c[1,0]),sample_false_alarm_rate=c[0,1]/max(1,c[0].sum()),diagnosis_macro_f1=fscore(d),end_to_end_macro_f1=fscore(g)))
    ci=pd.DataFrame(boot).quantile([.025,.975]).T.reset_index();ci.columns=['metric','lower_95','upper_95'];ci.to_csv(T/'primary_cluster_bootstrap_ci.csv',index=False)
    print('Dense metrics complete',metrics,flush=True)

if __name__=='__main__':
    lock=freeze();comparison(lock);dense(lock)
