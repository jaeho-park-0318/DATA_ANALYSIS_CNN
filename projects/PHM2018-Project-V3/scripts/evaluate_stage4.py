"""모델별 검증 경보 평가. CPU3프로세스, 입력/코드 해시로 검증한 캐시."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concurrent.futures import ProcessPoolExecutor,as_completed
from src.common import *
from src.alarm_evaluation import evaluate
from scripts.evaluate_stage2 import pareto_mask

OUT=ROOT/'outputs/stage4';EVAL=OUT/'evaluation';EVAL.mkdir(parents=True,exist_ok=True)

def fingerprint(run):
    paths=[OUT/(run['run_id']+'_validation.parquet'),Path(__file__),ROOT/'src/alarm_evaluation.py',TABLES/'events.parquet',TABLES/'pretraining_split_boundaries.parquet',TABLES/'alarm_stage1_summary.parquet']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def evaluate_run(run):
    rid=run['run_id'];signature=fingerprint(run);mark=EVAL/(rid+'_contract.json')
    if mark.exists() and json.loads(mark.read_text(encoding='utf-8'))['fingerprint']==signature:return rid,'캐시 검증'
    p=pd.read_parquet(OUT/(rid+'_validation.parquet'));events=pd.read_parquet(TABLES/'events.parquet');bounds=pd.read_parquet(TABLES/'pretraining_split_boundaries.parquet');ex=pd.read_parquet(TABLES/'alarm_stage1_summary.parquet');ex=ex[ex.scheme.eq('temporal')&ex.baseline.eq('no_alarm')&ex.feature_set.eq('C')]
    ledger=pd.read_parquet(ROOT/'outputs/stage3/common_validation_ledger.parquet');pd.testing.assert_frame_equal(p[ledger.columns],ledger);assert np.isfinite(p.score).all() and p.score.between(0,1).all()
    if run['reused_stage3']:
        old=run['feature_set']+'_xgboost';c=pd.read_parquet(TABLES/'stage3_alarm_curve.parquet');c=c[c.run_id.eq(old)].copy();e=pd.read_parquet(TABLES/'stage3_events.parquet');e=e[e.run_id.eq(old)].copy();c['run_id']=rid;e['run_id']=rid
    else:
        grid=np.unique(np.r_[0,.001,.003,.01,.03,.1,.3,.5,.7,.9,1,np.nextafter(1.,2.),np.quantile(p.score,[.001,.005,.01,.025,.05,.1,.25,.5,.75,.9,.95,.99,.995,.999,.9995,.9999])]);groups={};evs={}
        for tool,q in p.groupby('Tool',sort=True):
            b=bounds[bounds.scheme.eq('temporal')&bounds.split.eq('validation')&bounds.Tool.eq(tool)].iloc[0]
            assert q.time.ge(b.start+900).all() and q.time.lt(b.end_exclusive-1800).all()
            assert len(q)==ex[ex.Tool.eq(tool)].available_anchors.iloc[0]
            groups[tool]=q[['Tool','time','segment_id','score','audit_label']];evs[tool]=events[events.Tool.eq(tool)&events.fault_time.ge(b.start)&events.fault_time.lt(b.end_exclusive)]
        rows=[];ee=[]
        for threshold in grid:
            summaries=[];parts=[]
            for tool,q in groups.items():
                s,e,a,links=evaluate(q,evs[tool],float(threshold));summaries.append(s);parts.append(e)
                assert s['detected_events']<=s['available_events']<=s['scorable_events'];assert links.lead_raw.between(60,600).all();assert not a[a.episode_id.isin(links.episode_id)].start_label.eq(3).any();assert len(a)==sum(s[k] for k in s if k.startswith('alarms_'))
            e=pd.concat(parts,ignore_index=True);ss=pd.DataFrame(summaries);cols=['recorded_events','tagged_events','scorable_events','available_events','detected_events','alarm_episodes']+[k for k in ss if k.startswith('alarms_')];row=ss[cols].sum().to_dict();row.update(run_id=rid,feature_set=run['feature_set'],threshold=float(threshold),available_anchors=len(p),observed_support_raw=float(ex.observed_support_raw.sum()))
            for den in ['recorded','scorable','available']:row['recall_'+den]=row['detected_events']/row[den+'_events']
            row['alarm_episodes_per_10000_raw']=10000*row['alarm_episodes']/row['observed_support_raw'];row['median_lead_raw']=e.lead_raw.median();rows.append(row);e['run_id']=rid;e['threshold']=threshold;ee.append(e)
        c=pd.DataFrame(rows);c['pareto']=pareto_mask(c);e=pd.concat(ee,ignore_index=True)
    assert c.recorded_events.eq(232).all() and c.available_events.eq(53).all()
    write_parquet(c,EVAL/(rid+'_curve.parquet'));write_parquet(e,EVAL/(rid+'_events.parquet'));dump_json(mark,{'fingerprint':signature,'run_id':rid,'rows':len(p),'thresholds':len(c),'holdout_scored':False,'reused_stage3':run['reused_stage3']});return rid,f'{len(c)}개 임계값 완료'

def main():
    runs=json.loads((OUT/'training_runs.json').read_text(encoding='utf-8'))
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(evaluate_run,r) for r in runs]
        for f in as_completed(futures):print(' / '.join(f.result()),flush=True)
    for suffix,name in [('curve','alarm_curve'),('events','events')]:save_table(pd.concat([pd.read_parquet(EVAL/(r['run_id']+'_'+suffix+'.parquet')) for r in runs],ignore_index=True),'stage4_'+name)
    print('STAGE4 EVALUATION COMPLETE',flush=True)

if __name__=='__main__':main()
