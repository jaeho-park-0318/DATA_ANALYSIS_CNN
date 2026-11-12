"""seed별 임계값 재선택과 seed42 임계값 고정 이전을 분리한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concurrent.futures import ProcessPoolExecutor,as_completed
from src.common import *
from src.alarm_evaluation import evaluate
from scripts.diagnose_stage3 import choose_budget
from scripts.evaluate_stage2 import pareto_mask

OUT=ROOT/'outputs/stage4'

def fixed_run(task):
    run,refs=task;rid=run['run_id'];p=pd.read_parquet(OUT/(rid+'_validation.parquet'));events=pd.read_parquet(TABLES/'events.parquet');bounds=pd.read_parquet(TABLES/'pretraining_split_boundaries.parquet');summaries=[];details=[];computed={}
    cached=pd.read_parquet(OUT/'evaluation'/(rid+'_curve.parquet'));ce=pd.read_parquet(OUT/'evaluation'/(rid+'_events.parquet'))
    for ref in refs:
        threshold=ref['threshold'];budget=ref['budget_per_10000_raw']
        if threshold not in computed:
            match=cached[cached.threshold.eq(threshold)]
            if len(match):row=match.iloc[0].to_dict();e=ce[ce.threshold.eq(threshold)].copy()
            else:
                ss=[];ee=[]
                for tool,q in p.groupby('Tool',sort=True):
                    b=bounds[bounds.scheme.eq('temporal')&bounds.split.eq('validation')&bounds.Tool.eq(tool)].iloc[0];ev=events[events.Tool.eq(tool)&events.fault_time.ge(b.start)&events.fault_time.lt(b.end_exclusive)];s,e,a,links=evaluate(q,ev,threshold);ss.append(s);ee.append(e)
                    assert links.lead_raw.between(60,600).all() and not a[a.episode_id.isin(links.episode_id)].start_label.eq(3).any()
                s=pd.DataFrame(ss);e=pd.concat(ee,ignore_index=True);cols=['recorded_events','tagged_events','scorable_events','available_events','detected_events','alarm_episodes']+[c for c in s if c.startswith('alarms_')];row=s[cols].sum().to_dict();row.update(threshold=threshold,run_id=rid,feature_set=run['feature_set'],observed_support_raw=float(cached.observed_support_raw.iloc[0]))
                for den in ['recorded','scorable','available']:row['recall_'+den]=row['detected_events']/row[den+'_events']
                row['alarm_episodes_per_10000_raw']=10000*row['alarm_episodes']/row['observed_support_raw'];row['median_lead_raw']=e.lead_raw.median()
            computed[threshold]=(row,e)
        row,e=computed[threshold];row=dict(row);e=e.copy();row.update(config_id=run['config_id'],seed=run['seed'],budget_per_10000_raw=budget,threshold_source_seed=42,exceeds_budget=row['alarm_episodes_per_10000_raw']>budget)
        for k,v in [('run_id',rid),('config_id',run['config_id']),('seed',run['seed']),('budget_per_10000_raw',budget),('threshold',threshold)]:e[k]=v
        summaries.append(row);details.append(e)
    return summaries,pd.concat(details,ignore_index=True)

def main():
    config=json.loads((ROOT/'config/stage4_experiment.json').read_text(encoding='utf-8'));runs=pd.read_parquet(TABLES/'stage4_training_summary.parquet');curve=pd.read_parquet(TABLES/'stage4_alarm_curve.parquet');events=pd.read_parquet(TABLES/'stage4_events.parquet');frozen=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((TABLES/'stage4_grid_budget_points.parquet').read_bytes()).hexdigest()==frozen['source_grid_sha256']
    sensitivity_ids=sorted(runs.groupby('config_id').seed.nunique().loc[lambda s:s.eq(3)].index);rr=runs[runs.config_id.isin(sensitivity_ids)];retuned=[];retuned_events=[]
    for run in rr.itertuples():
        for budget in config['comparison_budgets_per_10000_raw']:
            row=choose_budget(curve[curve.run_id.eq(run.run_id)],budget).to_dict();row.update(config_id=run.config_id,seed=run.seed,budget_per_10000_raw=budget);retuned.append(row)
            e=events[events.run_id.eq(run.run_id)&events.threshold.eq(row['threshold'])].copy();e['config_id']=run.config_id;e['seed']=run.seed;e['budget_per_10000_raw']=budget;retuned_events.append(e)
    retuned=pd.DataFrame(retuned);re=pd.concat(retuned_events,ignore_index=True);tasks=[]
    for run in rr.to_dict('records'):tasks.append((run,retuned[retuned.config_id.eq(run['config_id'])&retuned.seed.eq(42)].to_dict('records')))
    fixed=[];fe=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(fixed_run,t) for t in tasks]):
            rows,e=future.result();fixed.extend(rows);fe.append(e);print(rows[0]['run_id']+' 고정 임계값 이전 완료',flush=True)
    fixed=pd.DataFrame(fixed).sort_values(['config_id','seed','budget_per_10000_raw']).reset_index(drop=True);fe=pd.concat(fe,ignore_index=True)
    # 두 방식 모두 seed42 기준점은 같은 결과여야 한다.
    z=retuned[retuned.seed.eq(42)].merge(fixed[fixed.seed.eq(42)],on=['config_id','seed','budget_per_10000_raw'],suffixes=('_retuned','_fixed'),validate='one_to_one')
    assert np.array_equal(z.detected_events_retuned,z.detected_events_fixed) and np.allclose(z.alarm_episodes_per_10000_raw_retuned,z.alarm_episodes_per_10000_raw_fixed)
    ranges=[]
    for mode,d in [('retuned',retuned),('fixed_seed42_threshold',fixed)]:
        for (cid,budget),g in d.groupby(['config_id','budget_per_10000_raw']):
            assert set(g.seed)=={42,43,44}
            ranges.append(dict(mode=mode,config_id=cid,budget_per_10000_raw=budget,detected_min=g.detected_events.min(),detected_max=g.detected_events.max(),detected_mean=g.detected_events.mean(),burden_min=g.alarm_episodes_per_10000_raw.min(),burden_max=g.alarm_episodes_per_10000_raw.max(),exceeding_seeds=int(g.alarm_episodes_per_10000_raw.gt(budget).sum())))
    comparisons=[];rng=np.random.default_rng(42);pairs=sorted(events.Tool.str[:2].unique());ix=rng.integers(0,len(pairs),(2000,len(pairs)))
    for selection in frozen['selected_by_budget']:
        budget=selection['selection_budget'];cid=selection['config_id'];base=selection['feature_set']+'_d3_unit'
        for seed in [42,43,44]:
            a=re[re.config_id.eq(base)&re.seed.eq(seed)&re.budget_per_10000_raw.eq(budget)];b=re[re.config_id.eq(cid)&re.seed.eq(seed)&re.budget_per_10000_raw.eq(budget)]
            z=a[['event_id','Tool','available','tagged','detected']].merge(b[['event_id','available','detected']],on='event_id',suffixes=('_base','_tuned'),validate='one_to_one');assert len(z)==232 and z.available_base.equals(z.available_tuned)
            z=z[~z.tagged].copy();z['pair']=z.Tool.str[:2];z['delta']=z.detected_tuned.astype(int)-z.detected_base.astype(int);c=z.groupby('pair').agg(delta=('delta','sum'),available=('available_base','sum')).reindex(pairs,fill_value=0);s=c.to_numpy()[ix].sum(axis=1);valid=s[:,1]>0;v=s[valid,0]/s[valid,1]
            comparisons.append(dict(config_id=cid,baseline_config=base,seed=seed,budget_per_10000_raw=budget,detected_baseline=int(z.detected_base.sum()),detected_tuned=int(z.detected_tuned.sum()),gained_events=int(z.delta.eq(1).sum()),lost_events=int(z.delta.eq(-1).sum()),recall_delta=z.delta.sum()/z.available_base.sum(),ci_low=np.quantile(v,.025),ci_high=np.quantile(v,.975)))
    joint=curve.merge(runs[['run_id','seed','config_id']],on='run_id',validate='many_to_one');joint=joint[joint.seed.eq(42)].copy();joint=joint[pareto_mask(joint)&joint.detected_events.gt(0)].drop_duplicates(['run_id','detected_events','alarm_episodes_per_10000_raw']).sort_values('alarm_episodes_per_10000_raw')
    for name,d in [('retuned_points',retuned),('retuned_events',re),('fixed_points',fixed),('fixed_events',fe),('seed_ranges',pd.DataFrame(ranges)),('paired_deltas',pd.DataFrame(comparisons)),('joint_frontier',joint)]:save_table(d,'stage4_'+name)
    dump_json(ROOT/'outputs/validation/stage4_analysis_contract.json',{'sensitivity_configs':sensitivity_ids,'selection_sha256':hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest(),'seeds':[42,43,44],'events_are_not_tripled':True,'holdout_scored':False,'bootstrap':'10 장비 번호 군집 2000회; 고정 선택 조건부이며 validation 선택 낙관/다중 비교 미보정','reference_preprocessor':'stage3의 동일 train 전용 추정기 재사용; validation fit 없음'})
    files={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ['.joblib','.ubj','.json','.parquet']}
    dump_json(ROOT/'outputs/validation/stage4_artifacts.json',{'sha256':files,'python':sys.executable,'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','joblib','threadpoolctl','psutil']},'holdout_scored':False})
    print('STAGE4 SENSITIVITY ANALYSIS COMPLETE',flush=True)

if __name__=='__main__':main()
