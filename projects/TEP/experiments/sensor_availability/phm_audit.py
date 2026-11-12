"""Inspect PHM V3 rules/actual event tables without importing or modifying its code."""
from common import *

def run():
    phm=Path(CFG['phm_project'])
    paths=['src/alarm_evaluation.py','src/final_evaluation.py','src/features.py','config/preprocessing.json',
           'outputs/tables/stage6_holdout_summary.csv','outputs/tables/stage6_holdout_events.csv','reports/19_group_and_holdout.md']
    files={name:{'path':str(phm/name),'sha256':digest(phm/name)} for name in paths}
    summary=pd.read_csv(phm/paths[4]);events=pd.read_csv(phm/paths[5])
    chosen=summary[summary.operating_point=='validation_budget_10'].iloc[0]
    own=events[events.operating_point=='validation_budget_10']
    assert len(own)==194 and own.available.sum()==91 and own.detected.sum()==33
    assert np.array_equal(own.available.to_numpy(),own.eligible_anchors.gt(0).to_numpy())
    assert own.loc[~own.available,'status'].eq('NO_INPUT').all()
    settings=json.loads((phm/'config/preprocessing.json').read_text(encoding='utf-8'))
    result=dict(source_project=str(phm),files=files,operating_point='validation_budget_10',
        total_events=len(own),available_events=int(own.available.sum()),unavailable_events=int((~own.available).sum()),
        detected_events=int(own.detected.sum()),conditional_detection_rate=float(own.detected.sum()/own.available.sum()),
        overall_detection_rate=float(own.detected.mean()),unavailable_fraction=float((~own.available).mean()),
        event_available_exactly_eligible_anchors_gt_zero=True,prediction_interval='[fault_time-600, fault_time-60] inclusive',
        unit='raw; not verified as seconds/minutes',feature_family='B: current + 300raw mean/std/min/max',
        feature_requirements={'current_lag_max_raw':settings['gap_threshold'],'nominal_interval_raw':settings['nominal_interval'],
            'history_raw':300,'minimum_coverage':settings['minimum_coverage'],'same_segment_required':True},
        anchor_stride_raw=settings['anchor_stride'],
        event_rule_source='src/alarm_evaluation.py evaluate(): valid=q.time.between(tf-h,tf-lead); available=bool(valid.any())',
        caveat='PHM uses usable pre-fault anchors, not the TEP20x52 rule; reproduce zero-opportunity semantics, not identical physical time/feature rules',
        phm_alarm_burden=float(chosen.alarm_episodes_per_10000_raw),
        phm_alarm_burden_unit='all alarms per10000 observed-support raw; NOT verified false alarm rate',
        primary_public_source='https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/')
    write_json(OUT/'sources/phm_definition_audit.json',result)
    own[['event_id','Tool','fault_type','eligible_anchors','available','detected','status']].to_csv(OUT/'tables/phm_verified_events.csv',index=False)
    print('PHM rule verified: 194 total,91 available,103 NO_INPUT,33 detected',flush=True)

if __name__=='__main__':run()
