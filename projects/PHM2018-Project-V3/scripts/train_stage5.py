"""보호된 시간 구간 안에서만 장비 그룹 CUDA 일반화 실험."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import time,gc,threading,warnings
import joblib,xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from threadpoolctl import threadpool_limits
from src.common import *
from src.features import schema
from src.protected_group import PROTOCOL,load
from scripts.train_stage2 import monitor

OUT=ROOT/'outputs/stage5';OUT.mkdir(parents=True,exist_ok=True)

def main():
    assert not (ROOT/'outputs/stage6/holdout_access.json').exists(),'이미 최종 Holdout을 열었음'
    rows=[];counts=[];checks=[];telemetry=[];stop=threading.Event();thread=threading.Thread(target=monitor,args=(stop,telemetry),daemon=True);thread.start();loaded={}
    try:
        for spec in PROTOCOL['group_runs']:
            rid=spec['run_id'];f=spec['family'];cohort=spec['cohort'];print(rid+' 보호된 입력 읽기',flush=True)
            train,tm=load('train',f,cohort,True);val,vm=load('validation',f,cohort,False);y=tm.audit_label.to_numpy();known=vm.audit_label.isin([0,1]).to_numpy();vy=vm.loc[known,'audit_label'].to_numpy()
            assert not set(tm.Tool)&set(vm.Tool) and set(np.unique(y))=={0,1}
            hashes={part:hashlib.sha256(pd.util.hash_pandas_object(m,index=False).to_numpy().tobytes()).hexdigest() for part,m in [('train',tm),('validation',vm)]}
            if cohort=='common':
                if 'common' in loaded:assert hashes==loaded['common']
                else:loaded['common']=hashes
            write_parquet(tm,OUT/(rid+'_train_ledger.parquet'));write_parquet(vm,OUT/(rid+'_validation_ledger.parquet'))
            prep=ColumnTransformer([('numeric',StandardScaler(),[c for c in schema()[f] if c not in ['recipe','stage']]),('categorical',OneHotEncoder(handle_unknown='ignore',dtype=np.float32),['recipe','stage'])],sparse_threshold=1.)
            with threadpool_limits(limits=12):
                xt=prep.fit_transform(train).astype('float32');xv=prep.transform(val).astype('float32');joblib.dump(prep,OUT/(rid+'_preprocessor.joblib'));weight=1. if spec['weight_mode']=='unit' else float(np.sqrt((y==0).sum()/(y==1).sum()))
                model=xgb.XGBClassifier(objective='binary:logistic',device='cuda:0',tree_method='hist',max_depth=spec['depth'],scale_pos_weight=weight,learning_rate=.05,n_estimators=1000,subsample=.8,colsample_bytree=.8,min_child_weight=10,reg_lambda=5,random_state=42,n_jobs=12,max_bin=128,eval_metric='aucpr',early_stopping_rounds=50)
                started=time.time()
                with warnings.catch_warnings(record=True) as caught:warnings.simplefilter('always');model.fit(xt,y,eval_set=[(xv[known],vy)],verbose=False)
                seconds=time.time()-started;cfg=json.loads(model.get_booster().save_config());assert cfg['learner']['generic_param']['device']=='cuda:0'
                score=model.get_booster().predict(xgb.DMatrix(xv),iteration_range=(0,model.best_iteration+1));assert np.isfinite(score).all();pred=vm.assign(score=score)
                joblib.dump(model,OUT/(rid+'.joblib'));model.save_model(OUT/(rid+'.ubj'));dump_json(OUT/(rid+'_booster_config.json'),cfg);dump_json(OUT/(rid+'_history.json'),model.evals_result());write_parquet(pred,OUT/(rid+'_validation.parquet'))
                restored=joblib.load(OUT/(rid+'.joblib'));again=restored.get_booster().predict(xgb.DMatrix(xv[:128]),iteration_range=(0,restored.best_iteration+1));np.testing.assert_allclose(score[:128],again,rtol=1e-6,atol=1e-7)
                rows.append(dict(**spec,device='cuda:0',positive_weight=weight,train_rows=len(y),train_positive=int(y.sum()),validation_all_rows=len(vm),validation_known_rows=int(known.sum()),validation_positive=int(vy.sum()),iterations=model.best_iteration+1,encoded_features=xt.shape[1],train_seconds=seconds,AP_known=average_precision_score(vy,score[known]),ROC_AUC_known=roc_auc_score(vy,score[known]),log_loss_known=log_loss(vy,score[known]),warnings=[str(w.message) for w in caught]))
                dump_json(OUT/'training_runs.json',rows);print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
            for part,m in [('train',tm),('validation',vm)]:
                for tool,q in m.groupby('Tool'):counts.append(dict(run_id=rid,role=part,Tool=tool,rows=len(q),POS=int(q.audit_label.eq(1).sum()),NEG=int(q.audit_label.eq(0).sum()),NA=int(q.audit_label.eq(2).sum()),EXCLUDE=int(q.audit_label.eq(3).sum()),start=q.time.min(),end=q.time.max()))
            checks.append(dict(check=rid+' 그룹 장비 분리·시간 Holdout 제외·공통 행·CUDA·저장 재현',passed=True,row_hashes=hashes));del train,val,tm,vm,xt,xv,model,restored,pred;gc.collect()
    finally:stop.set();thread.join(timeout=8);save_table(pd.DataFrame(telemetry),'stage5_resource_telemetry')
    save_table(pd.DataFrame([{k:v for k,v in r.items() if k!='warnings'} for r in rows]),'stage5_training_summary');save_table(pd.DataFrame(counts),'stage5_input_counts');dump_json(ROOT/'outputs/validation/stage5_training_checks.json',{'checks':checks,'protected_temporal_holdout_used':False});dump_json(OUT/'source_contract.json',{'protocol_sha256':hashlib.sha256((ROOT/'config/stage5_final_protocol.json').read_bytes()).hexdigest(),'pretraining_manifest_sha256':hashlib.sha256((ROOT/'data/pretraining/manifest.json').read_bytes()).hexdigest(),'protected_temporal_holdout_used':False})

if __name__=='__main__':main()
