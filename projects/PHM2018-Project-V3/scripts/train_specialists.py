"""선정 장비 전용 CUDA 모델과 검증 전용 임계값을 학습/동결."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import time,threading,warnings
import joblib,xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from threadpoolctl import threadpool_limits
from src.specialist_experiment import *
from src.features import schema
from scripts.train_stage2 import monitor


def main():
    verify_previous();selection=readjson(OUT/'selection.json')
    assert selection['protocol_sha256']==digest(PROTOCOL_PATH)
    assert not (OUT/'frozen_manifest.json').exists() and selection['selected_tools']
    baseline=pd.read_parquet(ROOT/'outputs/stage2/B_xgboost_validation.parquet')
    prior=readjson(ROOT/'outputs/stage6/frozen_manifest.json')
    exposure=pd.read_parquet(TABLES/'alarm_stage1_summary.parquet')
    exposure=exposure[exposure.scheme.eq('temporal')&exposure.baseline.eq('no_alarm')&exposure.feature_set.eq('B')]
    rows=[];points=[];curves=[];checks=[];telemetry=[];stop=threading.Event();thread=threading.Thread(target=monitor,args=(stop,telemetry),daemon=True);thread.start()
    try:
        for tool in selection['selected_tools']:
            tx,tm=load('train',tool);vx,vm=load('validation',tool)
            known=tm.audit_label.isin([0,1]);tx=tx.loc[known].reset_index(drop=True);tm=tm.loc[known].reset_index(drop=True);y=tm.audit_label
            vk=vm.audit_label.isin([0,1]);vy=vm.loc[vk,'audit_label'];assert y.nunique()==vy.nunique()==2
            assert list(tx)==list(vx)==schema()['B'];assert tm.Tool.eq(tool).all() and vm.Tool.eq(tool).all()
            globalp=baseline[baseline.Tool.eq(tool)].sort_values('time').reset_index(drop=True)
            pd.testing.assert_frame_equal(globalp[vm.columns],vm)
            write_parquet(tm,OUT/(tool+'_train_ledger.parquet'));write_parquet(vm,OUT/(tool+'_validation_ledger.parquet'))
            prep=ColumnTransformer([('numeric',StandardScaler(),[c for c in schema()['B'] if c not in ['recipe','stage']]),('categorical',OneHotEncoder(handle_unknown='ignore',dtype=np.float32),['recipe','stage'])],sparse_threshold=1.)
            with threadpool_limits(limits=12):
                xt=prep.fit_transform(tx).astype('float32');xv=prep.transform(vx).astype('float32')
                np.testing.assert_allclose(prep.named_transformers_['numeric'].mean_,tx[[c for c in schema()['B'] if c not in ['recipe','stage']]].mean(),rtol=1e-6,atol=1e-7)
                model=xgb.XGBClassifier(objective='binary:logistic',device='cuda:0',tree_method='hist',max_depth=3,scale_pos_weight=1,learning_rate=.05,n_estimators=1000,subsample=.8,colsample_bytree=.8,min_child_weight=10,reg_lambda=5,random_state=42,n_jobs=12,max_bin=128,eval_metric='aucpr',early_stopping_rounds=50)
                start=time.time()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');model.fit(xt,y,eval_set=[(xv[vk.to_numpy()],vy)],verbose=False)
                seconds=time.time()-start;cfg=json.loads(model.get_booster().save_config());assert cfg['learner']['generic_param']['device']=='cuda:0'
                score=model.get_booster().predict(xgb.DMatrix(xv),iteration_range=(0,model.best_iteration+1))
                assert np.isfinite(score).all();p=vm.assign(score=score)
                joblib.dump(model,OUT/(tool+'_model.joblib'));joblib.dump(prep,OUT/(tool+'_preprocessor.joblib'));model.save_model(OUT/(tool+'_model.ubj'))
                restored=joblib.load(OUT/(tool+'_model.joblib'));again=restored.get_booster().predict(xgb.DMatrix(xv[:128]),iteration_range=(0,restored.best_iteration+1));np.testing.assert_allclose(score[:128],again,rtol=1e-6,atol=1e-7)
            write_parquet(p,OUT/(tool+'_validation_predictions.parquet'));dump_json(OUT/(tool+'_booster_config.json'),cfg);dump_json(OUT/(tool+'_history.json'),model.evals_result())
            rows.append(dict(Tool=tool,device='cuda:0',train_rows=len(y),train_positive=int(y.sum()),validation_all_rows=len(vm),validation_known_rows=int(vk.sum()),validation_positive=int(vy.sum()),iterations=model.best_iteration+1,train_seconds=seconds,AP_known=float(average_precision_score(vy,score[vk])),ROC_AUC_known=float(roc_auc_score(vy,score[vk])),log_loss_known=float(log_loss(vy,score[vk],labels=[0,1])),warnings=[str(w.message) for w in caught]))
            checks.append(dict(check=tool+' 장비 전용 입력/훈련 전용 전처리/CUDA/저장 모델 재현/공통 점수 행 일치',passed=True))
            ev=events_for(tool,'validation');support=float(exposure[exposure.Tool.eq(tool)].iloc[0].observed_support_raw)
            assert int(exposure[exposure.Tool.eq(tool)].iloc[0].available_anchors)==len(vm)
            for method,frame in [('specialist',p),('global_tool_threshold',globalp)]:
                rr=[]
                for threshold in grid(frame.score):
                    row,_,_,_=alarm_result(frame,ev,threshold,support);row.update(Tool=tool,method=method);rr.append(row)
                curve=pd.DataFrame(rr);curves.append(curve)
                for budget in PROTOCOL['budgets']:
                    row=choose(curve,budget).to_dict();row['budget']=budget;points.append(row)
            for point in prior['operating_points']:
                row,_,_,_=alarm_result(globalp,ev,point['threshold'],support);row.update(Tool=tool,method='global_frozen',budget=point['validation_budget_per_10000_raw']);points.append(row)
            print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
    finally:
        stop.set();thread.join(timeout=8);save_table(pd.DataFrame(telemetry),'stage7_resource_telemetry')
    save_table(pd.DataFrame([{k:v for k,v in r.items() if k!='warnings'} for r in rows]),'stage7_training_summary');dump_json(OUT/'training_runs.json',rows)
    save_table(pd.concat(curves,ignore_index=True),'stage7_validation_curve');save_table(pd.DataFrame(points),'stage7_validation_points');dump_json(OUT/'training_checks.json',checks)
    paths=[PROTOCOL_PATH,OUT/'selection.json',TABLES/'stage7_validation_points.parquet',ROOT/'src/alarm_evaluation.py',ROOT/'src/pretraining.py']
    for tool in selection['selected_tools']:paths.extend([OUT/(tool+s) for s in ['_model.joblib','_model.ubj','_preprocessor.joblib','_train_ledger.parquet','_validation_ledger.parquet','_validation_predictions.parquet']])
    frozen=dict(frozen_at_utc=utc(),tools=selection['selected_tools'],points=[{k:r[k] for k in ['Tool','method','budget','threshold']} for r in points],sha256={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in paths},later_partition_scored=False,independent_final_test=False,seed=42,refit_after_freeze=False)
    with (OUT/'frozen_manifest.json').open('x',encoding='utf-8') as handle:json.dump(frozen,handle,ensure_ascii=False,indent=2)
    verify_previous();print('전용 모델 및 비교 임계값 동결 완료',flush=True)


if __name__=='__main__':main()
