"""모델 입력의 미등록 공정과 선택된 검증 운영점의 불확실성을 기록한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
from src.pretraining import iter_batches,iter_scoring_batches
from sklearn.metrics import average_precision_score,roc_auc_score
import joblib

OUT=ROOT/'outputs/stage2'

def main():
    diagnostics=[];states=[];models=[]
    for family in 'AB':
        combo=set()
        for x,y,m in iter_batches('temporal','train',family,600):combo.update(x[['recipe','stage']].drop_duplicates().itertuples(index=False,name=None))
        prep=joblib.load(OUT/f'{family}_preprocessor.joblib');vocab=prep.named_transformers_['categorical'].categories_
        dump_json(OUT/f'{family}_training_process_combinations.json',{'fit_partition':'train','recipe_stage_pairs':sorted(combo),'constructed_after_training_from_identical_source_contract':True})
        for x,m in iter_scoring_batches('temporal','validation',family,600):
            z=m[['Tool']].copy();z['rows']=1;z['unknown_recipe']=(~x.recipe.isin(vocab[0])).to_numpy();z['unknown_stage']=(~x.stage.isin(vocab[1])).to_numpy();z['unseen_combination']=[t not in combo for t in x[['recipe','stage']].itertuples(index=False,name=None)]
            q=z.groupby('Tool').sum().reset_index();q['feature_set']=family;diagnostics.append(q)
        print(family+' train 공정 사전 대조 완료',flush=True)
    save_table(pd.concat(diagnostics).groupby(['feature_set','Tool']).sum().reset_index(),'stage2_unseen_process')
    for rid in ['A_logistic','A_xgboost','B_logistic','B_xgboost']:
        model=joblib.load(OUT/(rid+'.joblib'));params={k:v for k,v in model.get_params().items() if v is None or isinstance(v,(str,float,int,bool))};models.append(dict(run_id=rid,parameters=params))
        p=pd.read_parquet(OUT/(rid+'_validation.parquet'))
        for label,q in p.groupby('audit_label'):
            states.append(dict(run_id=rid,label=['NEG','POS','NA','EXCLUDE'][int(label)],rows=len(q),score_min=q.score.min(),score_p50=q.score.median(),score_p90=q.score.quantile(.9),score_p99=q.score.quantile(.99),score_max=q.score.max()))
    save_table(pd.DataFrame(states),'stage2_score_states');dump_json(OUT/'model_parameters.json',models)
    points=pd.read_parquet(TABLES/'stage2_operating_points.parquet');ev=pd.read_parquet(TABLES/'stage2_events.parquet');rng=np.random.default_rng(42);uncertainty=[];toolmetrics=[]
    for point in points.itertuples():
        q=ev[ev.run_id.eq(point.run_id)&ev.threshold.eq(point.threshold)&~ev.tagged]
        # 한 Tool 안의 동시/근접 이벤트 의존성을 유지하는 장비 군집 재표집.
        counts=q.groupby('Tool').agg(detected=('detected','sum'),available=('available','sum'),recorded=('event_id','size'))
        counts=counts.reindex(sorted(ev.Tool.unique()),fill_value=0)
        ix=rng.integers(0,len(counts),size=(2000,len(counts)));sample=counts.to_numpy()[ix].sum(axis=1)
        for den,index in [('available',1),('recorded',2)]:
            valid=sample[:,index]>0;values=sample[valid,0]/sample[valid,index]
            uncertainty.append(dict(run_id=point.run_id,operating_point=point.operating_point,threshold=point.threshold,denominator=den,recall=q.detected.sum()/(q.available.sum() if den=='available' else len(q)),ci_low=np.quantile(values,.025),ci_high=np.quantile(values,.975),bootstrap_samples=2000,valid_bootstrap_samples=int(valid.sum()),lead_p10=q[q.detected].lead_raw.quantile(.1),lead_median=q[q.detected].lead_raw.median()))
        for tool,c in counts.iterrows():toolmetrics.append(dict(run_id=point.run_id,operating_point=point.operating_point,Tool=tool,**c.to_dict()))
    save_table(pd.DataFrame(uncertainty),'stage2_uncertainty');save_table(pd.DataFrame(toolmetrics),'stage2_tool_event_results')
    files={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ['.joblib','.ubj','.parquet','.json']}
    dump_json(ROOT/'outputs/validation/stage2_artifacts.json',{'sha256':files,'python':sys.executable,'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','joblib','threadpoolctl','psutil']},'holdout_scored':False,'bootstrap':'20 Tool 군집 재표집 2000회 seed42; 임계값 고정 조건부 95% 구간, 선택 낙관/이력 불완전/장비쌍 의존성 보정 아님'})

if __name__=='__main__':main()
