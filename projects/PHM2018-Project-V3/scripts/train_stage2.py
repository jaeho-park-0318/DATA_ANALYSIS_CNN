"""A/B × 로지스틱/XGBoost 첫 비교. train만 fit, validation만 점수화한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gc
import time
import threading
import subprocess
import warnings
import joblib
import psutil
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from src.common import *
from src.pretraining import iter_batches,iter_scoring_batches
from src.features import schema

OUT=ROOT/'outputs/stage2';OUT.mkdir(parents=True,exist_ok=True)

def monitor(stop,records):
    proc=psutil.Process()
    while not stop.is_set():
        r={'unix_time':time.time(),'rss_bytes':proc.memory_info().rss,'cpu_percent':proc.cpu_percent()}
        try:
            p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,creationflags=subprocess.CREATE_NO_WINDOW)
            vals=p.stdout.strip().split(',');r.update(gpu_util_percent=float(vals[0]),gpu_used_mib=float(vals[1]),gpu_free_mib=float(vals[2]))
        except Exception as e:r['probe_error']=str(e)
        records.append(r);stop.wait(2)

def load_train(family,cohort='maximum'):
    xs=[];ys=[];ms=[]
    for x,y,m in iter_batches('temporal','train',family,600,cohort):xs.append(x);ys.append(y);ms.append(m)
    return pd.concat(xs,ignore_index=True),pd.concat(ys,ignore_index=True).astype('int8'),pd.concat(ms,ignore_index=True)

def main(stage='stage2',families='AB',cohort='maximum'):
    global OUT
    OUT=ROOT/'outputs'/stage;OUT.mkdir(parents=True,exist_ok=True)
    from src.pretraining import PRE_DIR
    assert all(c['passed'] for c in json.loads((ROOT/'outputs/validation/alarm_independent_audit.json').read_text(encoding='utf-8'))['checks'])
    config=json.loads((ROOT/'config/local_compute.json').read_text(encoding='utf-8'));rows=[];telemetry=[];stop=threading.Event();thread=threading.Thread(target=monitor,args=(stop,telemetry),daemon=True);thread.start()
    gpu_smoke=False;cohort_hashes={}
    try:
        for family in families:
            print(f'{family} train/validation 읽기',flush=True)
            train,y,train_meta=load_train(family,cohort);xs=[];ms=[]
            for x,m in iter_scoring_batches('temporal','validation',family,600,cohort):xs.append(x);ms.append(m)
            validation=pd.concat(xs,ignore_index=True);metadata=pd.concat(ms,ignore_index=True);known=metadata.audit_label.isin([0,1]).to_numpy();vy=metadata.loc[known,'audit_label'].to_numpy()
            assert len(train)>0 and y.nunique()==2 and set(train)==set(schema()[family])
            keys=train_meta.assign(audit_label=y.to_numpy())
            hashes={split:hashlib.sha256(pd.util.hash_pandas_object(d[['Tool','time','segment_id','audit_label']],index=False).to_numpy().tobytes()).hexdigest() for split,d in [('train',keys),('validation',metadata)]}
            if cohort=='common':
                if cohort_hashes:assert hashes==cohort_hashes,'A/B/C 행·정답·순서 불일치'
                else:
                    cohort_hashes=hashes
                    write_parquet(keys,OUT/'common_train_ledger.parquet');write_parquet(metadata,OUT/'common_validation_ledger.parquet')
            dump_json(OUT/f'{family}_cohort_contract.json',{'cohort':cohort,'row_label_order_sha256':hashes,'train_rows':len(y),'validation_rows':len(metadata)})
            prep=ColumnTransformer([('numeric',StandardScaler(),[c for c in schema()[family] if c not in ['recipe','stage']]),('categorical',OneHotEncoder(handle_unknown='ignore',dtype=np.float32),['recipe','stage'])],sparse_threshold=1.)
            with threadpool_limits(limits=config['training_cpu_threads']):
                xt=prep.fit_transform(train).astype('float32');xv=prep.transform(validation).astype('float32')
                names=prep.get_feature_names_out().tolist();joblib.dump(prep,OUT/f'{family}_preprocessor.joblib')
                dump_json(OUT/f'{family}_input_contract.json',{'features_before_encoding':schema()[family],'features_after_encoding':names,'category_vocabulary':{c:values.tolist() for c,values in zip(['recipe','stage'],prep.named_transformers_['categorical'].categories_)},'fit_partition':'train','train_rows':len(y),'train_positive':int(y.sum()),'validation_all_rows':len(metadata),'validation_known_rows':int(known.sum()),'unknown_categories':'OneHot all-zero; no validation fit','H':600,'L':60})
                if not gpu_smoke:
                    rng=np.random.default_rng(42);pos=np.flatnonzero(y.to_numpy()==1);neg=np.flatnonzero(y.to_numpy()==0)
                    ix=np.r_[rng.choice(pos,min(1000,len(pos)),replace=False),rng.choice(neg,min(49000,len(neg)),replace=False)];rng.shuffle(ix)
                    smoke=xgb.XGBClassifier(device='cuda:0',tree_method='hist',n_estimators=20,max_depth=3,max_bin=128,n_jobs=12,random_state=42)
                    started=time.time();smoke.fit(xt[ix],y.iloc[ix]);cfg=json.loads(smoke.get_booster().save_config());device=cfg['learner']['generic_param']['device']
                    if not device.startswith('cuda'):raise RuntimeError('GPU 학습이 CPU로 전환됨: '+device)
                    dump_json(OUT/'cuda_training_smoke.json',{'success':True,'device':device,'rows':len(ix),'positive':int(y.iloc[ix].sum()),'trees':20,'seconds':time.time()-started,'source':'실제 train 후보에서 seed42로 선택; 검증 점수로 사용하지 않음','booster_config':cfg})
                    print('CUDA 실제 학습 성공: '+device,flush=True);gpu_smoke=True;del smoke
                for model_name in ['logistic','xgboost']:
                    run_id=f'{family}_{model_name}';print(run_id+' 학습 시작',flush=True);started=time.time()
                    if model_name=='logistic':model=LogisticRegression(C=1.,solver='lbfgs',max_iter=1500,tol=1e-5,random_state=42)
                    else:model=xgb.XGBClassifier(objective='binary:logistic',device='cuda:0',tree_method='hist',max_depth=3,learning_rate=.05,n_estimators=1000,subsample=.8,colsample_bytree=.8,min_child_weight=10,reg_lambda=5,random_state=42,n_jobs=12,max_bin=128,eval_metric='aucpr',early_stopping_rounds=50)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        if model_name=='logistic':model.fit(xt,y)
                        else:model.fit(xt,y,eval_set=[(xv[known],vy)],verbose=False)
                    if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise RuntimeError(run_id+' 미수렴: 정규화/solver 검토 필요')
                    train_seconds=time.time()-started
                    if model_name=='xgboost':
                        cfg=json.loads(model.get_booster().save_config());device=cfg['learner']['generic_param']['device']
                        if not device.startswith('cuda'):raise RuntimeError('GPU 학습의 CPU 전환')
                        dump_json(OUT/(run_id+'_booster_config.json'),cfg)
                        model.save_model(OUT/(run_id+'.ubj'));dump_json(OUT/(run_id+'_history.json'),model.evals_result())
                        # CPU scipy 행렬을 명시적인 DMatrix로 전달해 inplace 장치 불일치 경고를 피한다.
                        probability=model.get_booster().predict(xgb.DMatrix(xv),iteration_range=(0,model.best_iteration+1))
                        iterations=model.best_iteration+1
                    else:
                        device='cpu';probability=model.predict_proba(xv)[:,1];iterations=int(model.n_iter_[0])
                    joblib.dump(model,OUT/(run_id+'.joblib'))
                    assert np.isfinite(probability).all() and len(probability)==len(metadata)
                    pred=metadata.copy();pred['score']=probability;write_parquet(pred,OUT/(run_id+'_validation.parquet'))
                    # 저장 후 동일 입력에서 재현되는지 확인한다.
                    restored=joblib.load(OUT/(run_id+'.joblib'))
                    if model_name=='logistic':again=restored.predict_proba(xv[:128])[:,1]
                    else:again=restored.get_booster().predict(xgb.DMatrix(xv[:128]),iteration_range=(0,restored.best_iteration+1))
                    np.testing.assert_allclose(probability[:128],again,rtol=1e-6,atol=1e-7)
                    row={'run_id':run_id,'feature_set':family,'model':model_name,'device':device,'train_rows':len(y),'train_positive':int(y.sum()),'validation_all_rows':len(metadata),'validation_known_rows':int(known.sum()),'validation_positive':int(vy.sum()),'encoded_features':xt.shape[1],'iterations':iterations,'train_seconds':train_seconds,'AP_known':average_precision_score(vy,probability[known]),'ROC_AUC_known':roc_auc_score(vy,probability[known]),'log_loss_known':log_loss(vy,probability[known]),'warnings':[str(w.message) for w in caught],'artifact_roundtrip_passed':True}
                    rows.append(row);dump_json(OUT/'training_runs.json',rows);print(json.dumps(row,ensure_ascii=False),flush=True)
                    del model,restored,pred,probability;gc.collect()
            del train,y,train_meta,keys,validation,metadata,xt,xv,xs,ms;gc.collect()
    except Exception as e:
        dump_json(OUT/'failure.json',{'error':str(e),'gpu_smoke_succeeded':gpu_smoke,'completed_runs':[r['run_id'] for r in rows]});raise
    finally:
        stop.set();thread.join(timeout=8);save_table(pd.DataFrame(telemetry),stage+'_resource_telemetry')
    save_table(pd.DataFrame([{k:v for k,v in r.items() if k!='warnings'} for r in rows]),stage+'_training_summary')
    dump_json(OUT/'source_contract.json',{'pretraining_manifest_sha256':hashlib.sha256((PRE_DIR/'manifest.json').read_bytes()).hexdigest(),'compute_config':config,'xgboost_version':xgb.__version__,'holdout_scored':False,'cohort':cohort,'families':families,'same_row_label_order':cohort_hashes,'model_selection_status':'EXPLORATORY_COMPARISON_NOT_FINAL_SELECTION'})

if __name__=='__main__':
    from src.pretraining import PRE_DIR
    main()
