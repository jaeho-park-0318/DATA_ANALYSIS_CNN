"""Artifact integrity, portable inference equivalence and measured warmed latency."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.phase2 import trajectories,CACHE
from src.models.inference import TEPModel
from src.evaluation.run_phase2 import Predictor,M,T,P

def run():
    bundle=M/'final_model'
    manifest=json.loads((bundle/'manifest.json').read_text())
    for name,checksum in manifest.items():assert hashlib.sha256((bundle/name).read_bytes()).hexdigest()==checksum
    torch.set_num_threads(4)
    names=json.loads((CACHE/'sensor_names.json').read_text())
    raw=np.asarray(trajectories('test')[1,0,:200])
    frame=pd.DataFrame(raw,columns=names);frame['sample']=np.arange(1,201);frame['simulationRun']=1;frame['faultNumber']=1
    model=TEPModel(bundle,'cpu');pred=model.predict(frame)
    assert pred.eligible.sum()==181 and pred.fault_probability.iloc[:19].isna().all()
    dense=np.load(CACHE/'primary_dense_probabilities.npy',mmap_mode='r')
    expected=1-dense[500*941:500*941+181,0]
    error=float(np.max(np.abs(pred.fault_probability.iloc[19:].to_numpy()-expected)))
    # Ampere cuDNN may use TF32 while portable CPU inference uses float32.
    # Check probabilities at an explicit 0.001 absolute tolerance and decisions separately.
    assert error<1e-3,error
    agreement=float(np.mean((pred.fault_probability.iloc[19:].to_numpy()>=model.threshold)==(expected>=model.threshold)))
    assert agreement==1.,agreement
    altered=frame.copy();altered.loc[150:,names]+=1234
    future=model.predict(altered)
    np.testing.assert_allclose(pred.fault_probability.iloc[19:150],future.fault_probability.iloc[19:150],atol=1e-6)
    for bad in [frame.drop(columns=[names[0]]),pd.concat([frame,frame],ignore_index=True)]:
        try:model.predict(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid schema or nonconsecutive samples accepted')
    lock=json.loads((M/'selection_lock.json').read_text());latency=[]
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        predictor=Predictor(lock['primary_model']);predictor.model=predictor.model.to(device);predictor.device=device
        for batch in [1,256]:
            w=np.repeat(raw[None,:20],batch,axis=0).astype(np.float32)
            for _ in range(20):predictor.proba(w=w)
            timings=[]
            for _ in range(100):
                if device=='cuda':torch.cuda.synchronize()
                start=time.perf_counter();predictor.proba(w=w)
                if device=='cuda':torch.cuda.synchronize()
                timings.append((time.perf_counter()-start)*1000)
            latency.append(dict(model_id=predictor.tag,device=device,batch=batch,repeats=100,median_batch_ms=np.median(timings),p95_batch_ms=np.quantile(timings,.95),median_ms_per_window=np.median(timings)/batch,scope='numpy scaling + transfer + model + softmax; warmed; feature extraction/disk excluded'))
    pd.DataFrame(latency).to_csv(T/'latency_benchmark.csv',index=False)
    (P/'artifact_validation.json').write_text(json.dumps({'manifest_files_verified':len(manifest),'portable_cpu_dense_gpu_max_probability_error':error,'cpu_gpu_probability_absolute_tolerance':.001,'cpu_gpu_threshold_decision_agreement_in_checked_181_windows':agreement,'precision_note':'cuDNN TF32 vs CPU float32; threshold-adjacent decisions may differ on other inputs','no_future_reference':True,'warmup_verified':True,'schema_and_run_boundary_errors_rejected':True},indent=2),encoding='utf-8')
    print('Artifact validation passed',error,flush=True)

if __name__=='__main__':run()
