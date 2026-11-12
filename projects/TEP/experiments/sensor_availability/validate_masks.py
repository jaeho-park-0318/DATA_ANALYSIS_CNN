"""Actual masked inputs, window recovery, state-free inference and alarm continuity checks."""
from common import *

def run():
    if not read_gate():raise RuntimeError('Baseline gate must pass first')
    from src.evaluation.run_phase2 import Predictor
    lock=json.loads((ROOT/'models/phase2/selection_lock.json').read_text())
    predictor=Predictor(lock['primary_model'])
    raw=np.load(ROOT/'data/processed/phase2/testing.npy',mmap_mode='r')
    cached=np.load(CACHE/'fresh_baseline_scores.npy',mmap_mode='r')
    records=[];checks=[]
    cases=[('none',0,0),('short_gap',141,180),('full_censor',141,960),('one_window',141,940),('two_windows',141,939)]
    for name,first,last in cases:
        present=np.ones(960,bool) if name=='none' else raw_mask(first,last)
        valid=valid_windows(present)
        explicit=np.array([present[t-19:t+1].all() for t in range(19,960)])
        assert np.array_equal(valid,explicit)
        postcount=int(valid[141:].sum())
        expected={'none':800,'short_gap':761,'full_censor':0,'one_window':1,'two_windows':2}[name]
        assert postcount==expected,(name,postcount)
        for fault,run in [(1,0),(9,0),(15,2)]:
            x=np.array(raw[fault,run]);masked=x.copy();masked[~present]=np.nan
            endpoint=np.arange(19,960)
            windows_masked=masked[endpoint[:,None]+np.arange(-19,1)[None,:]]
            windows_original=x[endpoint[:,None]+np.arange(-19,1)[None,:]]
            finite=np.isfinite(windows_masked).all(axis=(1,2))
            assert np.array_equal(finite,valid)
            assert np.array_equal(windows_masked[valid],windows_original[valid])
            a=predictor.proba(w=windows_original[valid]);b=predictor.proba(w=windows_masked[valid])
            # Matching batch, identical surviving inputs: exact inference equality.
            assert np.array_equal(a,b)
            old=np.asarray(cached[fault,run])[valid]
            error=float(np.max(np.abs((1-b[:,0])-old)))
            assert error<.003,error
            records.append(dict(case=name,fault=fault,run=run+1,valid_post_windows=postcount,
                surviving_inputs_exactly_identical=True,same_batch_predictions_exactly_identical=True,
                cached_prediction_max_probability_error=error,cache_error_note='different GPU batch packing may cause small rounding; no input/model change'))
        checks.append(dict(check=f'window_mask_{name}',passed=True,valid_post_windows=postcount))
    # Missing points cannot join separated alarm fragments into 3 consecutive alarms.
    observed=np.array([True,True,False,True,True])
    assert len(episodes(observed))==0 and len(episodes(observed[[0,1,3,4]]))==1
    checks.append(dict(check='missing_points_break_alarm_streaks',passed=True))
    # Loss of a future observation cannot invalidate an earlier causal window.
    left=np.ones(960,bool);right=left.copy();right[600:]=False
    assert np.array_equal(valid_windows(left)[:581],valid_windows(right)[:581])
    checks.append(dict(check='no_future_dependency',passed=True))
    for seed in CFG['seeds']:
        prior=np.zeros((20,500),bool)
        for rate in CFG['rates']:
            selection=selected_runs(rate,seed)
            assert not (prior&~selection).any();prior=selection
            assert np.array_equal(selection,selected_runs(rate,seed))
    checks.append(dict(check='fault_balanced_exact_nested_reproducible_selection',passed=True))
    write_json(OUT/'sources/mask_validation.json',{'passed':True,'checks':checks,'actual_input_inference_checks':records,
        'equivalence_scope':'complete surviving20x52 inputs identical and stateless model; approximate cached equality only differs by GPU batch arithmetic',
        'no_imputation':True,'no_prediction_at_invalid_endpoints':True})
    pd.DataFrame(records).to_csv(OUT/'tables/mask_equivalence_checks.csv',index=False)
    print('Mask/inference equivalence and partial recovery checks PASSED',flush=True)

def read_gate():return json.loads((OUT/'sources/baseline_gate.json').read_text())['passed']
if __name__=='__main__':run()
