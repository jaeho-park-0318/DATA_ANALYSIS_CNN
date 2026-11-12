"""A/B 제한 탐색 또는 사전에 동결한 설정의 seed 민감도 CUDA 학습."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,gc,time,threading,shutil,warnings
import joblib,xgboost as xgb
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from threadpoolctl import threadpool_limits
from src.common import *
from src.pretraining import iter_scoring_batches,PRE_DIR
from scripts.train_stage2 import load_train,monitor

OUT=ROOT/'outputs/stage4';OUT.mkdir(parents=True,exist_ok=True)
CONFIG=json.loads((ROOT/'config/stage4_experiment.json').read_text(encoding='utf-8'))

def config_id(f,depth,weight):return f'{f}_d{depth}_{weight}'

def planned(phase):
    if phase=='grid':return [dict(feature_set=f,depth=d,weight_mode=w,seed=42) for f in CONFIG['grid_families'] for d in CONFIG['depths'] for w in CONFIG['positive_weights']]
    frozen=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    configs={c['config_id']:c for c in frozen['selected_configs']}
    for f in 'ABC':configs.setdefault(config_id(f,3,'unit'),dict(config_id=config_id(f,3,'unit'),feature_set=f,depth=3,weight_mode='unit'))
    return [dict(feature_set=c['feature_set'],depth=c['depth'],weight_mode=c['weight_mode'],seed=seed) for c in configs.values() for seed in CONFIG['sensitivity_seeds']]

def main(phase):
    config_hash=hashlib.sha256((ROOT/'config/stage4_experiment.json').read_bytes()).hexdigest()
    manifest_hash=hashlib.sha256((PRE_DIR/'manifest.json').read_bytes()).hexdigest()
    contract_path=OUT/'source_contract.json'
    if contract_path.exists():
        old=json.loads(contract_path.read_text(encoding='utf-8'));assert old['experiment_sha256']==config_hash and old['pretraining_manifest_sha256']==manifest_hash,'기존 실험 계약 변경'
    compute=json.loads((ROOT/'config/local_compute.json').read_text(encoding='utf-8'));rows=json.loads((OUT/'training_runs.json').read_text(encoding='utf-8')) if (OUT/'training_runs.json').exists() else []
    tasks=[t for t in planned(phase) if config_id(t['feature_set'],t['depth'],t['weight_mode'])+f"_s{t['seed']}" not in {r['run_id'] for r in rows}]
    dump_json(contract_path,{'experiment_sha256':config_hash,'pretraining_manifest_sha256':manifest_hash,'compute_config':compute,'holdout_scored':False,'cohort':'common','seed_selection':'seed42에서 설정 선택, 43/44는 민감도 평가','xgboost_version':xgb.__version__})
    telemetry=[];stop=threading.Event();thread=threading.Thread(target=monitor,args=(stop,telemetry),daemon=True);thread.start()
    train_ledger=pd.read_parquet(ROOT/'outputs/stage3/common_train_ledger.parquet');val_ledger=pd.read_parquet(ROOT/'outputs/stage3/common_validation_ledger.parquet')
    try:
        for family in sorted({t['feature_set'] for t in tasks}):
            train,y,tm=load_train(family,'common');pd.testing.assert_frame_equal(tm.assign(audit_label=y.to_numpy()),train_ledger)
            xs=[];ms=[]
            for x,m in iter_scoring_batches('temporal','validation',family,600,'common'):xs.append(x);ms.append(m)
            vx=pd.concat(xs,ignore_index=True);meta=pd.concat(ms,ignore_index=True);pd.testing.assert_frame_equal(meta,val_ledger)
            prep=joblib.load(ROOT/f'outputs/stage3/{family}_preprocessor.joblib');shutil.copy2(ROOT/f'outputs/stage3/{family}_preprocessor.joblib',OUT/f'{family}_preprocessor.joblib')
            # 동일 train 행의 기존 train 전용 추정기를 그대로 재사용한다. validation fit 없음.
            with threadpool_limits(limits=compute['training_cpu_threads']):
                xt=prep.transform(train).astype('float32');xv=prep.transform(vx).astype('float32');known=meta.audit_label.isin([0,1]).to_numpy();vy=meta.loc[known,'audit_label'].to_numpy()
                for t in [z for z in tasks if z['feature_set']==family]:
                    cid=config_id(family,t['depth'],t['weight_mode']);rid=cid+f"_s{t['seed']}";weight=1. if t['weight_mode']=='unit' else float(np.sqrt((y.eq(0).sum())/y.eq(1).sum()));reuse=t['depth']==3 and t['weight_mode']=='unit' and t['seed']==42
                    started=time.time();print(rid+(' stage3 기준 재사용' if reuse else ' CUDA 학습'),flush=True)
                    if reuse:
                        previous=ROOT/'outputs/stage3';oldid=family+'_xgboost'
                        for suffix in ['.joblib','.ubj','_booster_config.json','_history.json','_validation.parquet']:shutil.copy2(previous/(oldid+suffix),OUT/(rid+suffix))
                        model=joblib.load(OUT/(rid+'.joblib'));pred=pd.read_parquet(OUT/(rid+'_validation.parquet'));pd.testing.assert_frame_equal(pred[meta.columns],meta);prob=pred.score.to_numpy();cfg=json.loads((OUT/(rid+'_booster_config.json')).read_text(encoding='utf-8'));seconds=0.;warn=[]
                    else:
                        model=xgb.XGBClassifier(objective='binary:logistic',device='cuda:0',tree_method='hist',max_depth=t['depth'],scale_pos_weight=weight,learning_rate=.05,n_estimators=1000,subsample=.8,colsample_bytree=.8,min_child_weight=10,reg_lambda=5,random_state=t['seed'],n_jobs=12,max_bin=128,eval_metric='aucpr',early_stopping_rounds=50)
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always');model.fit(xt,y,eval_set=[(xv[known],vy)],verbose=False)
                        seconds=time.time()-started;warn=[str(w.message) for w in caught];cfg=json.loads(model.get_booster().save_config());assert cfg['learner']['generic_param']['device']=='cuda:0','CPU로 전환됨'
                        prob=model.get_booster().predict(xgb.DMatrix(xv),iteration_range=(0,model.best_iteration+1));pred=meta.assign(score=prob)
                        joblib.dump(model,OUT/(rid+'.joblib'));model.save_model(OUT/(rid+'.ubj'));write_parquet(pred,OUT/(rid+'_validation.parquet'));dump_json(OUT/(rid+'_booster_config.json'),cfg);dump_json(OUT/(rid+'_history.json'),model.evals_result())
                    assert np.isfinite(prob).all() and cfg['learner']['generic_param']['device']=='cuda:0'
                    restored=joblib.load(OUT/(rid+'.joblib'));again=restored.get_booster().predict(xgb.DMatrix(xv[:128]),iteration_range=(0,restored.best_iteration+1));np.testing.assert_allclose(prob[:128],again,rtol=1e-6,atol=1e-7)
                    row=dict(run_id=rid,config_id=cid,feature_set=family,model='xgboost',depth=t['depth'],weight_mode=t['weight_mode'],positive_weight=weight,seed=t['seed'],device='cuda:0',reused_stage3=reuse,phase=phase,train_rows=len(y),train_positive=int(y.sum()),validation_all_rows=len(meta),validation_known_rows=int(known.sum()),validation_positive=int(vy.sum()),encoded_features=xt.shape[1],iterations=int(model.best_iteration+1),train_seconds=seconds,AP_known=average_precision_score(vy,prob[known]),ROC_AUC_known=roc_auc_score(vy,prob[known]),log_loss_known=log_loss(vy,prob[known]),artifact_roundtrip_passed=True,warnings=warn)
                    rows.append(row);dump_json(OUT/'training_runs.json',rows);save_table(pd.DataFrame([{k:v for k,v in r.items() if k!='warnings'} for r in rows]),'stage4_training_summary');print(json.dumps(row,ensure_ascii=False),flush=True)
                    del model,restored,pred,prob;gc.collect()
            del train,y,tm,vx,meta,xt,xv,xs,ms;gc.collect()
    finally:
        stop.set();thread.join(timeout=8)
        if telemetry:save_table(pd.DataFrame(telemetry),'stage4_'+phase+'_resource_telemetry')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['grid','sensitivity'],required=True);main(parser.parse_args().phase)
