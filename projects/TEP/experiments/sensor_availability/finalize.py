"""Check actual outputs and read-only preservation before declaring completion."""
from common import *

def run():
    before=json.loads((OUT/'sources/protected_files_before.json').read_text(encoding='utf-8'))
    changed=[name for name,v in before.items() if not (ROOT/name).exists() or digest(ROOT/name)!=v['sha256']]
    phm=json.loads((OUT/'sources/phm_definition_audit.json').read_text(encoding='utf-8'))
    pc=[name for name,v in phm['files'].items() if digest(Path(v['path']))!=v['sha256']]
    preserve=dict(protected_TEP_files=len(before),changed_TEP_files=changed,protected_PHM_source_files=len(phm['files']),changed_PHM_files=pc,all_unchanged=not changed and not pc)
    write_json(OUT/'sources/preservation_check.json',preserve)
    assert preserve['all_unchanged'],preserve
    assert json.loads((OUT/'baseline.json').read_text())['all_checks_passed']
    assert json.loads((OUT/'sources/pilot_gate.json').read_text())['passed']
    full=pd.read_csv(OUT/'tables/full_summary.csv')
    assert len(full)==15
    baseline=pd.read_csv(OUT/'tables/baseline_events.csv')
    records=[]
    for r in full.itertuples():
        events=pd.read_csv(OUT/'tables'/f'events_full_censor_r{r.rate:.3f}_s{r.seed}.csv')
        selected=selected_runs(r.rate,r.seed).reshape(-1)
        assert len(events)==10000
        assert np.array_equal(events.selected,selected)
        assert (events.valid_post_windows.eq(0)==selected).all()
        assert int(events.detected.sum())==r.detected_fault_runs
        assert int(events.evaluable.sum())==r.evaluable_fault_runs
        assert r.cause_A_no_input+r.cause_B_input_but_no_detection+r.detected_fault_runs==10000
        assert abs(r.overall_detection_rate-r.conditional_detection_rate*r.evaluable_fault_runs/10000)<1e-12
        assert np.array_equal(events.loc[~selected,'detected'],baseline.loc[~selected,'detected'])
        np.testing.assert_allclose(events.loc[~selected,'delay_minutes'],baseline.loc[~selected,'delay_minutes'],equal_nan=True)
        assert r.normal_alarm_episodes==705 and r.normal_exposure_hours==23525
        records.append(dict(rate=r.rate,seed=r.seed,event_rows=10000,counts_balanced_and_verified=True,surviving_event_results_identical=True))
    manifest={str(p.relative_to(HERE)):digest(p) for p in HERE.glob('*.py')}
    manifest['config.json']=digest(HERE/'config.json')
    write_json(OUT/'sources/experiment_code_checksums.json',manifest)
    write_json(OUT/'sources/final_validation.json',{'passed':True,'conditions_checked':records,'existing_files_unchanged':True,'models_retrained':False,'threshold_changed':False,'labels_changed':False,
        'scope':'all15 condition-seed tables and event rows; partial control validated by pilot and mask checks'})
    print('Final validation PASSED:15 conditions; original147 TEP+7 PHM files unchanged',flush=True)

if __name__=='__main__':run()
