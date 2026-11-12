"""Frozen CNN type validation: cached21-class probabilities, no training/selection."""
from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.evaluation.phase2 import episodes
from experiments.sensor_availability.common import selected_runs,valid_windows,raw_mask
OUT=ROOT/'reports/fault_type_validation'
for p in [OUT,OUT/'tables',OUT/'figures',OUT/'sources',HERE/'cache',HERE/'references']:p.mkdir(parents=True,exist_ok=True)

def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def metrics(cm):
    tp=np.diag(cm).astype(float);support=cm.sum(1);pred=cm.sum(0)
    precision=np.divide(tp,pred,out=np.zeros_like(tp),where=pred>0)
    recall=np.divide(tp,support,out=np.zeros_like(tp),where=support>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros_like(tp),where=precision+recall>0)
    return pd.DataFrame({'class':np.arange(len(cm)),'precision':precision,'recall':recall,'f1':f1,'support':support}),dict(
        accuracy=float(tp.sum()/cm.sum()),macro_precision=float(precision.mean()),macro_recall=float(recall.mean()),macro_f1=float(f1.mean()))
def event_result(gated,score,valid,fault,run,threshold):
    samples=np.arange(20,961);post=samples>160;ss=samples[post]
    detection=episodes((score>=threshold)&valid & post)
    correct=episodes((gated==fault)&valid & post)
    # Autonomous diagnosis: require3 consecutive identical nonzero predicted IDs.
    candidates=[]
    for predicted in np.unique(gated[post&valid]):
        if predicted==0:continue
        ev=episodes((gated==predicted)&valid&post)
        if ev:candidates.append((ev[0][0],int(predicted)))
    first_type=min(candidates) if candidates else None
    det_sample=int(samples[detection[0][0]]) if detection else None
    diag_sample=int(samples[correct[0][0]]) if correct else None
    return dict(fault=fault,run=run,valid_post_windows=int(valid[post].sum()),
        detected=bool(detection),correctly_diagnosed=bool(correct),
        first_detection_sample=det_sample,first_correct_diagnosis_sample=diag_sample,
        detection_delay_minutes=(det_sample-160)*3 if det_sample is not None else None,
        correct_diagnosis_delay_minutes=(diag_sample-160)*3 if diag_sample is not None else None,
        additional_diagnosis_delay_minutes=(diag_sample-det_sample)*3 if diag_sample is not None else None,
        first_confirmed_type=first_type[1] if first_type else None,
        first_confirmed_type_correct=(first_type[1]==fault) if first_type else None)

def run():
    protected=[p for p in (ROOT/'notebooks').glob('*.ipynb')]
    protected+=list((ROOT/'models/phase2').glob('*.pt'))+list((ROOT/'models/phase2').glob('*.joblib'))
    protected += [ROOT/'models/phase2/selection_lock.json',ROOT/'configs/phase2.json']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    save('sources/protected_before.json',before)
    lock=json.loads((ROOT/'models/phase2/selection_lock.json').read_text())
    expected=json.loads((ROOT/'reports/sources/phase2/dense_metrics.json').read_text())['joint_primary']
    threshold=expected['threshold']
    probabilities=np.load(ROOT/'data/processed/phase2/primary_dense_probabilities.npy',mmap_mode='r')
    assert probabilities.shape==(9_880_500,21)
    k=3;assert lock['primary_model']=='CNN1D_Temporal_balanced_s42'
    endpoints=np.arange(20,961);post=endpoints>160
    partial_valid=valid_windows(raw_mask(141,180))
    # Mode index0=raw argmax21,1=operational gated21. Type-only confusion is separate.
    cms=np.zeros((2,21,500,21,21),dtype=np.int32)
    pre_cm=np.zeros_like(cms);partial_cm=np.zeros_like(cms)
    type_cm=np.zeros((21,500,20,20),dtype=np.int32);partial_type=np.zeros_like(type_cm)
    events=[];partial_events=[];examples=[]
    for fault in range(21):
        p=np.asarray(probabilities[fault*500*941:(fault+1)*500*941]).reshape(500,941,21)
        score=1-p[:,:,0];candidate=p[:,:,1:].argmax(2)+1;raw=p.argmax(2)
        gated=np.where(score>=threshold,candidate,0)
        truth=np.where(post,fault,0)
        for run in range(500):
            for mode,labels in enumerate([raw[run],gated[run]]):
                ids=21*truth+labels
                cms[mode,fault,run]=np.bincount(ids,minlength=441).reshape(21,21)
                pre_cm[mode,fault,run]=np.bincount(ids[endpoints<141],minlength=441).reshape(21,21)
                partial_cm[mode,fault,run]=np.bincount(ids[partial_valid],minlength=441).reshape(21,21)
            if fault>0:
                type_cm[fault,run,fault-1]=np.bincount(candidate[run,post]-1,minlength=20)
                partial_type[fault,run,fault-1]=np.bincount(candidate[run,post&partial_valid]-1,minlength=20)
                events.append(event_result(gated[run],score[run],np.ones(941,bool),fault,run+1,threshold))
                partial_events.append(event_result(gated[run],score[run],partial_valid,fault,run+1,threshold))
            if fault in [0,1,3,9,15] and run==0:
                for index in [0,140,141,180,380,940]:
                    cp=p[run,index,1:]/max(float(p[run,index,1:].sum()),1e-20)
                    ranked=np.argsort(cp)[::-1][:3]+1
                    examples.append(dict(actual=fault if endpoints[index]>160 else 0,scenario=fault,run=1,sample=int(endpoints[index]),
                        fault_probability=float(score[run,index]),conditional_candidate=int(candidate[run,index]),gated_type=int(gated[run,index]),
                        candidate_probability_given_fault=float(cp[candidate[run,index]-1]),top3_types=','.join(map(str,ranked)),
                        interpretation='conditional candidate is not a confirmed fault when gate fails'))
        print('Type validation Fault',fault,flush=True)
    events=pd.DataFrame(events);pe=pd.DataFrame(partial_events)
    events.to_csv(OUT/'tables/run_type_results.csv',index=False)
    pe.to_csv(OUT/'tables/partial_run_type_results.csv',index=False)
    pd.DataFrame(examples).to_csv(OUT/'tables/prediction_examples.csv',index=False)
    np.savez(HERE/'cache/per_run_confusions.npz',joint=cms,pre=pre_cm,partial=partial_cm,conditional=type_cm,partial_conditional=partial_type)
    totals=[];perclass=[];confusions=[]
    for name,cm in [('argmax21',cms[0].sum((0,1))),('gated21',cms[1].sum((0,1))),('conditional20',type_cm.sum((0,1)))]:
        report,summary=metrics(cm)
        if name=='conditional20':report['class']+=1
        report.insert(0,'mode',name);perclass.append(report);totals.append(dict(mode=name,**summary))
        np.save(OUT/'tables'/f'{name}_confusion.npy',cm)
        pd.DataFrame(cm).to_csv(OUT/'tables'/f'{name}_confusion.csv',index=False)
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i!=j and cm[i,j]>0:confusions.append(dict(mode=name,actual=i+(name=='conditional20'),predicted=j+(name=='conditional20'),count=int(cm[i,j]),actual_class_fraction=float(cm[i,j]/cm[i].sum())))
    pd.DataFrame(perclass[0]).head()
    pd.concat(perclass).to_csv(OUT/'tables/per_class_metrics.csv',index=False)
    pd.DataFrame(totals).to_csv(OUT/'tables/classification_summary.csv',index=False)
    pd.DataFrame(confusions).sort_values(['mode','count'],ascending=[True,False]).to_csv(OUT/'tables/confusion_pairs.csv',index=False)
    got={r['mode']:r for r in totals}
    assert abs(got['conditional20']['macro_f1']-expected['diagnosis_macro_f1'])<1e-12
    assert abs(got['gated21']['macro_f1']-expected['end_to_end_macro_f1'])<1e-12
    old_cm=np.load(ROOT/'reports/tables/phase2/primary_dense_confusion.npy')
    assert np.array_equal(old_cm,cms[1].sum((0,1)))
    original_events=pd.read_csv(ROOT/'reports/sensor_availability/tables/baseline_events.csv')
    assert np.array_equal(events.detected,original_events.detected)
    np.testing.assert_allclose(events.detection_delay_minutes,original_events.delay_minutes,equal_nan=True)
    event_summary=[]
    for fault,g in events.groupby('fault'):
        first=g.first_confirmed_type.notna()
        event_summary.append(dict(fault=fault,runs=500,detected_runs=int(g.detected.sum()),correct_diagnosis_runs=int(g.correctly_diagnosed.sum()),
            run_detection_rate=float(g.detected.mean()),run_correct_diagnosis_rate=float(g.correctly_diagnosed.mean()),
            first_type_accuracy=float(g.loc[first,'first_confirmed_type_correct'].mean()),
            detection_delay_median=g.detection_delay_minutes.median(),correct_diagnosis_delay_median=g.correct_diagnosis_delay_minutes.median(),
            additional_diagnosis_delay_median=g.additional_diagnosis_delay_minutes.median()))
    pd.DataFrame(event_summary).to_csv(OUT/'tables/per_fault_diagnosis_time.csv',index=False)
    early=[]
    for horizon in [30,60,120,240]:
        for fault,g in [('ALL',events),*list(events.groupby('fault'))]:
            early.append(dict(horizon_minutes=horizon,fault=fault,runs=len(g),
                detection_rate_by_horizon=float(g.detection_delay_minutes.le(horizon).mean()),
                correct_diagnosis_rate_by_horizon=float(g.correct_diagnosis_delay_minutes.le(horizon).mean())))
    pd.DataFrame(early).to_csv(OUT/'tables/early_diagnosis_horizons.csv',index=False)
    av_rows=[];av_fault=[];av_classes=[]
    for rate in [0.,.1,.25,.4,.531]:
        for seed in [42,43,44]:
            chosen=selected_runs(rate,seed);keep=~chosen.reshape(-1)
            selected_indices=np.r_[np.zeros(500,bool),chosen.reshape(-1)].reshape(21,500)
            c21=cms[1].sum((0,1))-cms[1][selected_indices].sum(0)+pre_cm[1][selected_indices].sum(0)
            c20=type_cm.sum((0,1))-type_cm[selected_indices].sum(0)
            r21,m21=metrics(c21);r20,m20=metrics(c20)
            av_classes.extend(dict(rate=rate,seed=seed,fault=int(r['class']),precision=r['precision'],recall=r['recall'],f1=r['f1'],support=int(r['support'])) for r in r21.to_dict('records'))
            available=events.loc[keep];de=int(available.detected.sum());di=int(available.correctly_diagnosed.sum())
            av_rows.append(dict(rate=rate,seed=seed,total_runs=10000,available_runs=len(available),unavailable_runs=int(chosen.sum()),
                detected_runs=de,correct_diagnosis_runs=di,overall_detection_rate=de/10000,conditional_detection_rate=de/len(available),
                overall_correct_diagnosis_rate=di/10000,conditional_correct_diagnosis_rate=di/len(available),
                correct_diagnosis_delay_median=available.correct_diagnosis_delay_minutes.median(),
                conditional20_macro_f1=m20['macro_f1'],gated21_macro_f1=m21['macro_f1']))
            for fault in range(1,21):
                g=events[(events.fault==fault)&np.r_[keep]]
                av_fault.append(dict(rate=rate,seed=seed,fault=fault,available=len(g),detected=int(g.detected.sum()),diagnosed=int(g.correctly_diagnosed.sum()),
                    overall_detection_rate=float(g.detected.sum()/500),overall_diagnosis_rate=float(g.correctly_diagnosed.sum()/500),
                    conditional_diagnosis_rate=float(g.correctly_diagnosed.mean()),correct_diagnosis_delay_median=g.correct_diagnosis_delay_minutes.median()))
    av=pd.DataFrame(av_rows);av.to_csv(OUT/'tables/availability_detection_vs_diagnosis.csv',index=False)
    pd.DataFrame(av_fault).to_csv(OUT/'tables/availability_per_fault.csv',index=False)
    pd.DataFrame(av_classes).to_csv(OUT/'tables/availability_per_class_metrics.csv',index=False)
    original_av=pd.read_csv(ROOT/'reports/sensor_availability/tables/full_summary.csv').sort_values(['rate','seed'])
    np.testing.assert_allclose(av.sort_values(['rate','seed']).overall_detection_rate,original_av.overall_detection_rate,atol=0,rtol=0)
    selected=selected_runs(.25,42).reshape(-1)
    partial_events=events.copy();partial_events.loc[selected,pe.columns]=pe.loc[selected].to_numpy()
    partial_events['selected']=selected
    partial_events.to_csv(OUT/'tables/partial_selected_diagnosis.csv',index=False)
    changed=[name for name,value in before.items() if sha(ROOT/name)!=value]
    assert not changed
    summary=dict(model=lock['primary_model'],threshold=threshold,checkpoint_sha256=sha(ROOT/'models/phase2'/f"{lock['primary_model']}.pt"),
        samples=9_880_500,post_fault_samples=8_000_000,detection_run_rate=float(events.detected.mean()),
        correct_diagnosis_run_rate=float(events.correctly_diagnosed.mean()),
        diagnosis_delay_median=float(events.correct_diagnosis_delay_minutes.median()),
        additional_diagnosis_delay_median=float(events.additional_diagnosis_delay_minutes.median()),
        first_confirmed_type_accuracy=float(events.loc[events.first_confirmed_type.notna(),'first_confirmed_type_correct'].mean()),
        all_reference_results_match=True,no_retraining=True,no_threshold_selection=True,unchanged_protected_files=len(before),changed_files=changed,
        rules={'detection':'3 consecutive fixed-threshold positives, post-fault reset',
               'diagnosis':'3 consecutive identical nonzero gated predicted IDs, post-fault reset',
               'correct_diagnosis':'first confirmed predicted ID equal to true Fault; true ID used only in evaluation'},
        classification_metrics=got)
    save('validation.json',summary)
    print('Diagnosis validation completed',summary,flush=True)

if __name__=='__main__':run()
