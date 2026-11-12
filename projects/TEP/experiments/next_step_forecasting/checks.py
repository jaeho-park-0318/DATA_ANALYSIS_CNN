"""Target alignment, preprocessing scope and causal API checks on real trajectories."""
from predict import *

def run():
    torch.set_num_threads(4)
    model=Forecaster('cpu');raw=np.load(ROOT/'data/processed/phase2/training.npy',mmap_mode='r')
    split=json.loads((ROOT/'data/processed/phase2/split_manifest.json').read_text())
    normal=np.asarray(raw[0,split['train']]).reshape(-1,52)
    np.testing.assert_allclose(model.scaler.mean_,normal.mean(0,dtype=np.float64),atol=1e-10,rtol=1e-10)
    assert model.scaler.n_samples_seen_==200000
    r=split['train'][0];frame=pd.DataFrame(np.asarray(raw[0,r,:60]),columns=model.names);frame['sample']=np.arange(1,61);frame['simulationRun']=r+1
    output=model.predict_next(frame)
    assert len(output)==41 and output.target_sample.iloc[0]==21 and output.target_sample.iloc[-1]==61
    future=frame.copy();future.loc[30:,model.names]+=12345
    altered=model.predict_next(future)
    np.testing.assert_allclose(output.iloc[:11].to_numpy(),altered.iloc[:11].to_numpy(),atol=1e-6,rtol=1e-6)
    for bad in [frame.iloc[:19],frame.drop(columns=[model.names[0]]),pd.concat([frame,frame],ignore_index=True)]:
        try:model.predict_next(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid history/schema/time accepted')
    # Persistence is evaluated on exactly the same common endpoints for all seeds.
    result=pd.read_csv(ROOT/'reports/next_step_forecasting/tables/test_summary.csv')
    naive=result[(result.model=='Persistence')&(result['mode']=='common')]
    assert naive.groupby('cohort').normalized_rmse.nunique().eq(1).all()
    assert result[(result.seed==42)&(result['mode']=='dense')].target_rows.max()==9870000
    from pathlib import Path
    import hashlib
    old=json.loads((ROOT/'reports/fault_type_validation/sources/protected_before.json').read_text(encoding='utf-8'))
    changed=[name for name,h in old.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h]
    assert not changed
    report={'passed':True,'scaler_normal_train_rows':200000,'same_original_split':True,'target_first_sample':21,'single_step_horizon_verified':True,'future_perturbation_leaves_past_forecasts_unchanged':True,'bad_history_schema_nonconsecutive_rejected':True,'persistence_common_results_identical_across_seeds':True,'existing_CNN_models_and_notebooks_unchanged':True}
    (ROOT/'reports/next_step_forecasting/api_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Forecast API/causality/scope checks PASSED',flush=True)

if __name__=='__main__':run()
