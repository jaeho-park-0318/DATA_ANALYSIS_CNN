"""Fresh full-Test inference, exact metric/event reproduction, fail-closed gate."""
import sys,time,json
sys.dont_write_bytecode=True
from common import *

def run():
    before=OUT/'sources/protected_files_before.json'
    if not before.exists():
        print('Hashing protected original artifacts',flush=True)
        write_json(before,protected_snapshot())
    from src.evaluation.run_phase2 import Predictor
    import torch
    torch.set_num_threads(4)
    lock=json.loads((ROOT/'models/phase2/selection_lock.json').read_text())
    model=Predictor(lock['primary_model'])
    assert model.feature=='Temporal'
    raw=np.load(ROOT/'data/processed/phase2/testing.npy',mmap_mode='r')
    reference=np.load(ROOT/'data/processed/phase2/primary_dense_probabilities.npy',mmap_mode='r')
    score=np.lib.format.open_memmap(CACHE/'fresh_baseline_scores.npy',mode='w+',dtype=np.float32,shape=(21,500,941))
    endpoint=np.arange(19,960);offset=0;maximum=0.;start=time.perf_counter()
    for fault in range(21):
        for first in range(0,500,10):
            w=np.concatenate([raw[fault,r][endpoint[:,None]+np.arange(-19,1)[None,:]] for r in range(first,first+10)])
            p=model.proba(w=w)
            old=np.asarray(reference[offset:offset+len(p)])
            maximum=max(maximum,float(np.max(np.abs(p-old))))
            score[fault,first:first+10]=(1-p[:,0]).reshape(10,941)
            offset+=len(p)
        print('Fresh baseline inference Fault',fault,flush=True)
    score.flush();inference=time.perf_counter()-start
    flat=np.asarray(score).reshape(-1)
    labels=np.broadcast_to((np.arange(20,961)>160)[None,None,:],(21,500,941)).copy()
    labels[0]=False
    actual=binary_metrics(labels.reshape(-1),flat,model.th)
    expected=json.loads((ROOT/'reports/sources/phase2/dense_metrics.json').read_text())['joint_primary']
    checks=[]
    for key in ['detection_roc_auc','detection_f1','detection_precision','detection_recall','sample_false_alarm_rate']:
        checks.append(dict(metric=key,expected=expected[key],reproduced=actual[key],absolute_error=abs(expected[key]-actual[key]),passed=abs(expected[key]-actual[key])<=CFG['baseline_metric_absolute_tolerance']))
    valid=np.ones(941,dtype=bool)
    events=pd.DataFrame([evaluate_event(score[f,r],valid,f,r+1,model.th) for f in range(1,21) for r in range(500)])
    reference_events=pd.read_csv(ROOT/'reports/tables/phase2/dense_alarm_episodes.csv')
    reference_events=reference_events[(reference_events.system=='joint_primary')&(reference_events.fault>0)]
    reference_events=reference_events.sort_values(['fault','run']).reset_index(drop=True)
    for ours,old in [('detected','detected'),('first_detection_sample','first_detection_sample'),('delay_minutes','detection_delay_minutes')]:
        aa=events[ours].to_numpy(dtype=float);bb=reference_events[old].to_numpy(dtype=float)
        same=np.array_equal(aa,bb,equal_nan=True)
        checks.append(dict(metric=f'per_event_{ours}',expected='10000 original records',reproduced='10000 fresh records',absolute_error=0 if same else None,passed=bool(same)))
    burden=normal_burden(score,model.th)
    normal=reference_events # original fault-free records are read separately
    all_events=pd.read_csv(ROOT/'reports/tables/phase2/dense_alarm_episodes.csv')
    normal=all_events[(all_events.system=='joint_primary')&(all_events.fault==0)]
    for key,refvalue in [('normal_alarm_episodes',normal.pre_alarm_episodes.sum()),('normal_exposure_hours',normal.normal_exposure_hours.sum()),('normal_sample_false_alarm_rate',expected['normal_faultfree_FAR'])]:
        error=abs(float(burden[key])-float(refvalue))
        checks.append(dict(metric=key,expected=float(refvalue),reproduced=burden[key],absolute_error=error,passed=error<1e-9))
    summary=dict(model=lock['primary_model'],threshold=model.th,device=model.device,fresh_full_test_inference_seconds=inference,
        maximum_absolute_probability_difference=maximum,detection_metrics=actual,total_fault_runs=10000,
        detected_fault_runs=int(events.detected.sum()),run_detection_rate=float(events.detected.mean()),
        median_delay_minutes=float(events.delay_minutes.median()),mean_delay_minutes=float(events.delay_minutes.mean()),
        p90_delay_minutes=float(events.delay_minutes.quantile(.9)),**burden,
        all_checks_passed=all(r['passed'] for r in checks))
    pd.DataFrame(checks).to_csv(OUT/'tables/baseline_reproduction.csv',index=False)
    events.to_csv(OUT/'tables/baseline_events.csv',index=False)
    write_json(OUT/'baseline.json',summary)
    if not summary['all_checks_passed']:
        write_json(OUT/'STOP_baseline_mismatch.json',summary)
        raise RuntimeError('Baseline mismatch. Degradation experiments are prohibited; inspect baseline_reproduction.csv.')
    write_json(OUT/'sources/baseline_gate.json',{'passed':True,'metrics_tolerance':CFG['baseline_metric_absolute_tolerance'],'threshold':model.th})
    print('BASELINE PASSED',summary,flush=True)

if __name__=='__main__':run()
