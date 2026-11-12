"""Measured availability experiment. No refitting or threshold selection."""
from common import *

class SampleMetrics:
    """Exact ROC-AUC with tied-score groups, avoiding repeated sorting of10M predictions."""
    def __init__(self,scores,threshold):
        self.scores=scores
        self.positive=np.broadcast_to((np.arange(20,961)>160)[None,None,:],scores.shape).copy();self.positive[0]=False
        self.threshold=threshold
        self.values,ids=np.unique(scores.reshape(-1),return_inverse=True)
        self.ids=ids.reshape(scores.shape)
    def evaluate(self,valid):
        pos=valid&self.positive;neg=valid&~self.positive
        hp=np.bincount(self.ids[pos],minlength=len(self.values));hn=np.bincount(self.ids[neg],minlength=len(self.values))
        npos=int(hp.sum());nneg=int(hn.sum())
        below=np.cumsum(hn,dtype=np.float64)-hn
        auc=float(np.dot(hp.astype(np.float64),below+.5*hn)/(npos*nneg))
        above=self.values>=self.threshold
        tp=int(hp[above].sum());fp=int(hn[above].sum());fn=npos-tp
        return dict(predicted_samples=int(valid.sum()),positive_predicted_samples=npos,negative_predicted_samples=nneg,
            detection_roc_auc=auc,detection_f1=2*tp/(2*tp+fp+fn),sample_precision=tp/(tp+fp),sample_recall=tp/npos)

def scenario(scores,baseline,samplemetrics,rate,seed,kind='full_censor'):
    choice=selected_runs(rate,seed)
    presence=raw_mask(141,960 if kind=='full_censor' else 180)
    masked_valid=valid_windows(presence)
    grid=np.ones_like(scores,dtype=bool)
    for fault in range(1,21):grid[fault,choice[fault-1]]=masked_valid
    events=pd.DataFrame([evaluate_event(scores[f,r],grid[f,r],f,r+1,samplemetrics.threshold) for f in range(1,21) for r in range(500)])
    events['selected']=choice.reshape(-1)
    events['cause']=np.where(~events.evaluable,'A_NO_INPUT',np.where(events.detected,'DETECTED','B_INPUT_BUT_NO_DETECTION'))
    old=baseline.sort_values(['fault','run']).reset_index(drop=True)
    both=events.detected&old.detected
    events['baseline_detected']=old.detected.to_numpy()
    events['baseline_delay_minutes']=old.delay_minutes.to_numpy()
    available=int(events.evaluable.sum());detected=int(events.detected.sum())
    metrics=samplemetrics.evaluate(grid)
    burden=normal_burden(scores,samplemetrics.threshold)
    result=dict(kind=kind,rate=rate,seed=seed,total_fault_runs=10000,selected_fault_runs=int(choice.sum()),
        evaluable_fault_runs=available,unavailable_fault_runs=int((~events.evaluable).sum()),
        partial_fault_runs=int(events.status.eq('PARTIAL').sum()),full_fault_runs=int(events.status.eq('FULL').sum()),
        detected_fault_runs=detected,conditional_detection_rate=detected/available,overall_detection_rate=detected/10000,
        unavailable_fraction=float((~events.evaluable).mean()),
        mean_post_input_availability_all_runs=float(events.input_available_fraction.mean()),
        mean_post_input_availability_evaluable=float(events.loc[events.evaluable,'input_available_fraction'].mean()),
        mean_post_input_availability_partial=float(events.loc[events.status=='PARTIAL','input_available_fraction'].mean()) if events.status.eq('PARTIAL').any() else None,
        cause_A_no_input=int((~events.evaluable).sum()),cause_B_input_but_no_detection=int((events.evaluable&~events.detected).sum()),
        baseline_detections_lost=int((old.detected&~events.detected).sum()),
        lost_previously_detected_due_unavailability=int((old.detected&~events.evaluable).sum()),
        median_delay_minutes=float(events.delay_minutes.median()),mean_delay_minutes=float(events.delay_minutes.mean()),
        p90_delay_minutes=float(events.delay_minutes.quantile(.9)),
        paired_detected_runs=int(both.sum()),paired_median_delay_change_minutes=float((events.loc[both,'delay_minutes']-old.loc[both,'delay_minutes']).median()),
        removed_raw_observation_fraction=float(choice.sum()*(~presence).sum()/(21*500*960)),
        analytical_MCAR_detection_rate=float(old.detected.mean()*(1-rate)) if kind=='full_censor' else None,
        **metrics,**burden)
    assert result['cause_A_no_input']+result['cause_B_input_but_no_detection']+detected==10000
    if kind=='full_censor':
        assert result['unavailable_fault_runs']==int(choice.sum())
        assert result['partial_fault_runs']==0
        assert result['baseline_detections_lost']==result['lost_previously_detected_due_unavailability']
        assert result['paired_median_delay_change_minutes']==0
    else:
        assert result['unavailable_fault_runs']==0 and result['partial_fault_runs']==int(choice.sum())
        assert abs(result['mean_post_input_availability_partial']-761/800)<1e-12
    assert result['normal_alarm_episodes']==705
    per=[]
    for fault,g in events.groupby('fault'):
        av=int(g.evaluable.sum());de=int(g.detected.sum())
        per.append(dict(kind=kind,rate=rate,seed=seed,fault=fault,total=500,available=av,unavailable=500-av,
            partial=int(g.status.eq('PARTIAL').sum()),detected=de,overall_detection_rate=de/500,
            conditional_detection_rate=de/av,median_delay_minutes=float(g.delay_minutes.median())))
    name=f'{kind}_r{rate:.3f}_s{seed}'
    events.to_csv(OUT/'tables'/f'events_{name}.csv',index=False)
    pd.DataFrame(per).to_csv(OUT/'tables'/f'faults_{name}.csv',index=False)
    write_json(OUT/'tables'/f'summary_{name}.json',result)
    print(kind,rate,seed,'available',available,'detected',detected,'overall',round(result['overall_detection_rate'],4),'conditional',round(result['conditional_detection_rate'],4),flush=True)
    return result

def run(stage):
    if not json.loads((OUT/'sources/baseline_gate.json').read_text())['passed']:raise RuntimeError('Baseline gate failed')
    if not json.loads((OUT/'sources/mask_validation.json').read_text())['passed']:raise RuntimeError('Mask validation failed')
    if stage=='full' and not (OUT/'sources/pilot_gate.json').exists():raise RuntimeError('Pilot must pass first')
    scores=np.load(CACHE/'fresh_baseline_scores.npy',mmap_mode='r')
    baseline=pd.read_csv(OUT/'tables/baseline_events.csv')
    b=json.loads((OUT/'baseline.json').read_text());sm=SampleMetrics(scores,b['threshold'])
    # Independent full-Baseline check of exact tied-score ROC implementation.
    original=sm.evaluate(np.ones_like(scores,dtype=bool))
    for metric in ['detection_roc_auc','detection_f1']:assert abs(original[metric]-b['detection_metrics'][metric])<1e-12
    rates=CFG['pilot_rates'] if stage=='pilot' else CFG['rates']
    rows=[]
    for rate in rates:
        for seed in CFG['seeds']:
            name=f'full_censor_r{rate:.3f}_s{seed}'
            existing=OUT/'tables'/f'summary_{name}.json'
            rows.append(json.loads(existing.read_text()) if existing.exists() else scenario(scores,baseline,sm,rate,seed))
    if stage=='pilot':
        control=scenario(scores,baseline,sm,.25,42,'partial_control')
        # Full sample-level sklearn comparison on the25% pilot, not a formula-only assertion.
        from sklearn.metrics import roc_auc_score,f1_score
        chosen=selected_runs(.25,42);valid=np.ones_like(scores,dtype=bool)
        for fault in range(1,21):valid[fault,chosen[fault-1]]=valid_windows(raw_mask())
        actual=next(r for r in rows if r['rate']==.25 and r['seed']==42)
        direct_auc=roc_auc_score(sm.positive[valid],scores[valid]);direct_f1=f1_score(sm.positive[valid],scores[valid]>=b['threshold'])
        assert abs(direct_auc-actual['detection_roc_auc'])<1e-12 and abs(direct_f1-actual['detection_f1'])<1e-12
        write_json(OUT/'sources/pilot_gate.json',{'passed':True,'conditions':rates,'seeds':CFG['seeds'],
            'partial_control_evaluable_runs':control['evaluable_fault_runs'],'partial_control_partial_runs':control['partial_fault_runs'],
            'sklearn_auc_difference':abs(direct_auc-actual['detection_roc_auc']),'sklearn_f1_difference':abs(direct_f1-actual['detection_f1'])})
    pd.DataFrame(rows).to_csv(OUT/'tables'/f'{stage}_summary.csv',index=False)
    if stage=='full':
        faults=pd.concat([pd.read_csv(OUT/'tables'/f'faults_full_censor_r{r:.3f}_s{s}.csv') for r in rates for s in CFG['seeds']],ignore_index=True)
        faults.to_csv(OUT/'tables/fault_results.csv',index=False)
    print(stage.upper(),'PASSED',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','full'],required=True)
    run(p.parse_args().stage)
