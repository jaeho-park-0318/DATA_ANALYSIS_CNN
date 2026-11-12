"""공통 코호트 일치, 동일 경보 상한, 장비쌍 군집의 대응 차이 분석."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
import joblib
from scripts.evaluate_stage2 import pareto_mask

OUT=ROOT/'outputs/stage3'

def choose_budget(curve,budget):
    return curve[curve.alarm_episodes_per_10000_raw<=budget].sort_values(['detected_events','alarm_episodes_per_10000_raw','threshold'],ascending=[False,True,False]).iloc[0]

def main():
    config=json.loads((ROOT/'config/stage3_experiment.json').read_text(encoding='utf-8'));runs=pd.read_parquet(TABLES/'stage3_training_summary.parquet');curve=pd.read_parquet(TABLES/'stage3_alarm_curve.parquet');ev=pd.read_parquet(TABLES/'stage3_events.parquet')
    base=json.loads((OUT/'A_cohort_contract.json').read_text(encoding='utf-8'));checks=[];states=[];parameters=[]
    train=pd.read_parquet(OUT/'common_train_ledger.parquet');validation=pd.read_parquet(OUT/'common_validation_ledger.parquet')
    for f in 'ABC':
        c=json.loads((OUT/f'{f}_cohort_contract.json').read_text(encoding='utf-8'));assert c==base
        prep=joblib.load(OUT/f'{f}_preprocessor.joblib');a=joblib.load(OUT/'A_preprocessor.joblib')
        for ac,fc in zip(a.named_transformers_['categorical'].categories_,prep.named_transformers_['categorical'].categories_):np.testing.assert_array_equal(ac,fc)
        np.testing.assert_allclose(a.named_transformers_['numeric'].mean_,prep.named_transformers_['numeric'].mean_[:6],rtol=0,atol=0)
        np.testing.assert_allclose(a.named_transformers_['numeric'].scale_,prep.named_transformers_['numeric'].scale_[:6],rtol=0,atol=0)
        checks.append(dict(check=f+' train/validation 행·라벨·순서 및 공유 전처리 일치',passed=True))
    for r in runs.itertuples():
        p=pd.read_parquet(OUT/(r.run_id+'_validation.parquet'));pd.testing.assert_frame_equal(p[validation.columns],validation)
        model=joblib.load(OUT/(r.run_id+'.joblib'));parameters.append({'run_id':r.run_id,'parameters':{k:v for k,v in model.get_params().items() if v is None or isinstance(v,(str,int,float,bool))}})
        for label,q in p.groupby('audit_label'):states.append(dict(run_id=r.run_id,label=['NEG','POS','NA','EXCLUDE'][int(label)],rows=len(q),score_p50=q.score.median(),score_p90=q.score.quantile(.9),score_p99=q.score.quantile(.99),unique_scores=q.score.nunique()))
        checks.append(dict(check=r.run_id+' 저장 예측 전체 행·라벨 일치',passed=True))
    chosen=[];chosen_events=[];uncertainty=[];deltas=[]
    for rid,g in curve.groupby('run_id'):
        for budget in config['comparison_budgets_per_10000_raw']:
            row=choose_budget(g,budget).to_dict();row['budget_per_10000_raw']=budget;chosen.append(row)
            e=ev[ev.run_id.eq(rid)&ev.threshold.eq(row['threshold'])].copy();e['budget_per_10000_raw']=budget;chosen_events.append(e)
    points=pd.DataFrame(chosen);pe=pd.concat(chosen_events,ignore_index=True)
    joint=curve[curve.run_id.str.endswith('xgboost')].copy();joint=joint[pareto_mask(joint)&joint.detected_events.gt(0)].drop_duplicates(['run_id','detected_events','alarm_episodes_per_10000_raw']).sort_values('alarm_episodes_per_10000_raw');save_table(joint,'stage3_joint_frontier')
    pair_ids=sorted(pe.Tool.str[:2].unique());rng=np.random.default_rng(42);ix=rng.integers(0,len(pair_ids),(2000,len(pair_ids)))
    for (rid,budget),e in pe.groupby(['run_id','budget_per_10000_raw']):
        z=e[~e.tagged].copy();z['pair']=z.Tool.str[:2];counts=z.groupby('pair').agg(detected=('detected','sum'),available=('available','sum'),recorded=('event_id','size')).reindex(pair_ids,fill_value=0);sample=counts.to_numpy()[ix].sum(axis=1)
        for den,col in [('available',1),('recorded',2)]:
            valid=sample[:,col]>0;v=sample[valid,0]/sample[valid,col];n=z.available.sum() if den=='available' else len(z)
            uncertainty.append(dict(run_id=rid,budget_per_10000_raw=budget,denominator=den,recall=z.detected.sum()/n,ci_low=np.quantile(v,.025),ci_high=np.quantile(v,.975),valid_bootstrap_samples=int(valid.sum()),lead_p10=z.loc[z.detected,'lead_raw'].quantile(.1),lead_median=z.loc[z.detected,'lead_raw'].median()))
    for model in ['logistic','xgboost']:
        for budget in config['comparison_budgets_per_10000_raw']:
            for before,after in [('A','B'),('B','C'),('A','C')]:
                a=pe[pe.run_id.eq(before+'_'+model)&pe.budget_per_10000_raw.eq(budget)];b=pe[pe.run_id.eq(after+'_'+model)&pe.budget_per_10000_raw.eq(budget)]
                z=a[['event_id','Tool','available','tagged','detected']].merge(b[['event_id','Tool','available','tagged','detected']],on='event_id',suffixes=('_before','_after'),validate='one_to_one')
                assert len(z)==len(a)==len(b) and z.available_before.equals(z.available_after) and z.Tool_before.equals(z.Tool_after)
                z=z[~z.tagged_before].copy();z['pair']=z.Tool_before.str[:2];z['delta']=z.detected_after.astype(int)-z.detected_before.astype(int)
                c=z.groupby('pair').agg(delta=('delta','sum'),available=('available_before','sum')).reindex(pair_ids,fill_value=0);s=c.to_numpy()[ix].sum(axis=1);valid=s[:,1]>0;values=s[valid,0]/s[valid,1]
                deltas.append(dict(model=model,budget_per_10000_raw=budget,comparison=after+'-'+before,detected_before=int(z.detected_before.sum()),detected_after=int(z.detected_after.sum()),gained_events=int(z.delta.eq(1).sum()),lost_events=int(z.delta.eq(-1).sum()),recall_delta=z.delta.sum()/z.available_before.sum(),ci_low=np.quantile(values,.025),ci_high=np.quantile(values,.975),valid_bootstrap_samples=int(valid.sum())))
    for name,d in [('score_states',pd.DataFrame(states)),('budget_points',points),('budget_events',pe),('uncertainty',pd.DataFrame(uncertainty)),('paired_deltas',pd.DataFrame(deltas))]:save_table(d,'stage3_'+name)
    dump_json(OUT/'model_parameters.json',parameters)
    coverage=pd.DataFrame([{'partition':'train_known','rows':len(train),'positive':int(train.audit_label.eq(1).sum()),'negative':int(train.audit_label.eq(0).sum()),'NA':int(train.audit_label.eq(2).sum()),'EXCLUDE':int(train.audit_label.eq(3).sum())},{'partition':'validation_all','rows':len(validation),'positive':int(validation.audit_label.eq(1).sum()),'negative':int(validation.audit_label.eq(0).sum()),'NA':int(validation.audit_label.eq(2).sum()),'EXCLUDE':int(validation.audit_label.eq(3).sum())}]);save_table(coverage,'stage3_cohort_counts')
    dump_json(ROOT/'outputs/validation/stage3_common_contract.json',{'checks':checks,'row_label_order_sha256':base['row_label_order_sha256'],'paired_bootstrap':'10 장비 번호 군집(각 M01/M02를 함께) 2000회 seed42. 임계값 고정 조건부이며 선택 낙관/이력 불완전성 미보정.','comparison_budgets':config['comparison_budgets_per_10000_raw'],'holdout_scored':False})
    files={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ['.joblib','.ubj','.parquet','.json']}
    dump_json(ROOT/'outputs/validation/stage3_artifacts.json',{'sha256':files,'python':sys.executable,'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','joblib','threadpoolctl','psutil']},'holdout_scored':False})
    print('COMMON CONTRACT / PAIRED COMPARISONS COMPLETE',flush=True)

if __name__=='__main__':main()
