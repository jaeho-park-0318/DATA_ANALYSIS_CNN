"""학습 모델의 validation 경보 곡선. holdout 접근 없음."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
from src.alarm_evaluation import evaluate

OUT=ROOT/'outputs/stage2'

def pareto_mask(d):
    x=d.alarm_episodes_per_10000_raw.to_numpy();y=d.recall_available.to_numpy()
    return np.array([not np.any((x<=a)&(y>=b)&((x<a)|(y>b))) for a,b in zip(x,y)])

def main(stage='stage2',common=False):
    global OUT
    OUT=ROOT/'outputs'/stage
    events=pd.read_parquet(TABLES/'events.parquet');bounds=pd.read_parquet(TABLES/'pretraining_split_boundaries.parquet')
    exposure=pd.read_parquet(TABLES/'alarm_stage1_summary.parquet')
    exposure=exposure[exposure.scheme.eq('temporal')&exposure.baseline.eq('no_alarm')]
    runs=json.loads((OUT/'training_runs.json').read_text(encoding='utf-8'))
    curves=[];all_events=[];by_tool=[];selected=[];saved_alarms=[];saved_links=[];checks=[]
    for run in runs:
        rid=run['run_id'];family=run['feature_set'];p=pd.read_parquet(OUT/(rid+'_validation.parquet'))
        assert np.isfinite(p.score).all() and p.score.between(0,1).all()
        assert not p.duplicated(['Tool','time']).any()
        grid=np.unique(np.r_[0,.001,.003,.01,.03,.1,.3,.5,.7,.9,1,np.nextafter(1.,2.),np.quantile(p.score,[.001,.005,.01,.025,.05,.1,.25,.5,.75,.9,.95,.99,.995,.999,.9995,.9999])])
        groups={};evs={};supports={}
        for tool,q in p.groupby('Tool',sort=True):
            groups[tool]=q[['Tool','time','segment_id','score','audit_label']].sort_values('time')
            b=bounds[bounds.scheme.eq('temporal')&bounds.split.eq('validation')&bounds.Tool.eq(tool)].iloc[0]
            assert q.time.ge(b.start+900).all() and q.time.lt(b.end_exclusive-1800).all()
            evs[tool]=events[events.Tool.eq(tool)&events.fault_time.ge(b.start)&events.fault_time.lt(b.end_exclusive)]
            er=exposure[exposure.Tool.eq(tool)&exposure.feature_set.eq('C' if common else family)].iloc[0]
            assert len(q)==er.available_anchors
            supports[tool]=float(er.observed_support_raw)
        run_rows=[]
        for threshold in grid:
            summaries=[];event_parts=[]
            for tool,q in groups.items():
                s,e,a,links=evaluate(q,evs[tool],float(threshold))
                assert s['detected_events']<=s['available_events']<=s['scorable_events']
                assert len(a)==sum(s[k] for k in s if k.startswith('alarms_'))
                assert links.lead_raw.between(60,600).all()
                assert not a[a.episode_id.isin(links.episode_id)].start_label.eq(3).any()
                s.update(Tool=tool,observed_support_raw=supports[tool]);summaries.append(s)
                e['run_id']=rid;e['threshold']=threshold;event_parts.append(e)
                by_tool.append(dict(run_id=rid,feature_set=family,threshold=threshold,**s))
            ee=pd.concat(event_parts,ignore_index=True);ss=pd.DataFrame(summaries)
            cols=['recorded_events','tagged_events','scorable_events','available_events','detected_events','alarm_episodes','observed_support_raw']+[c for c in ss if c.startswith('alarms_')]
            row=ss[cols].sum().to_dict();row.update(run_id=rid,feature_set=family,threshold=float(threshold))
            for den in ['recorded','scorable','available']:row['recall_'+den]=row['detected_events']/row[den+'_events']
            row['alarm_episodes_per_10000_raw']=10000*row['alarm_episodes']/row['observed_support_raw']
            row['median_lead_raw']=ee.lead_raw.median();row['available_anchors']=len(p)
            run_rows.append(row);all_events.append(ee)
        rr=pd.DataFrame(run_rows);rr['pareto']=pareto_mask(rr);curves.append(rr)
        front=rr[rr.pareto].copy();maximum=rr.detected_events.max()
        # 운영 허용치 미정: 모델 내 최대 검출의 50/80/100%를 확보하는 최소 부담 후보.
        for fraction,name in [(.5,'검출최대의50%'),(.8,'검출최대의80%'),(1.,'검출최대의100%')]:
            candidates=front[front.detected_events>=fraction*maximum]
            pick=candidates.sort_values(['alarm_episodes_per_10000_raw','detected_events','threshold'],ascending=[True,False,False]).iloc[0].to_dict()
            pick.update(operating_point=name,target_fraction_of_run_max=fraction);selected.append(pick)
        unique={r['threshold'] for r in selected if r['run_id']==rid}
        for threshold in unique:
            for tool,q in groups.items():
                _,_,a,links=evaluate(q,evs[tool],threshold)
                for df,dest in [(a,saved_alarms),(links,saved_links)]:
                    df['run_id']=rid;df['threshold']=threshold;dest.append(df)
        checks.append(dict(check=rid+' 전체 입력·경계·매칭·분모·경보 분류',passed=True,thresholds=len(grid),rows=len(p)))
        print(rid+' 경보 평가 완료: '+str(len(grid))+' thresholds; max detected='+str(int(maximum)),flush=True)
    for name,df in [('alarm_curve',pd.concat(curves,ignore_index=True)),('events',pd.concat(all_events,ignore_index=True)),('by_tool',pd.DataFrame(by_tool)),('operating_points',pd.DataFrame(selected)),('selected_episodes',pd.concat(saved_alarms,ignore_index=True)),('selected_matches',pd.concat(saved_links,ignore_index=True))]:save_table(df,stage+'_'+name)
    assert snapshot()==json.loads((ROOT/'outputs/source_snapshot_before.json').read_text(encoding='utf-8'))
    dump_json(ROOT/'outputs/validation'/f'{stage}_evaluation.json',{'checks':checks,'holdout_scored':False,'raw_metadata_unchanged':True,'cohort':'common' if common else 'maximum','threshold_grid':'고정 확률12개 + validation 전체 점수 분위수16개; 중복 제거. stage2에서 정한 규칙을 유지.','selection_rule':'각 모델의 탐색 격자 내 최대 검출수 50/80/100% 이상을 확보하는 Pareto 후보 중 최소 경보 부담. 최종 운영 임계값 아님.','evaluation_partition':'validation; early stopping과 임계값 탐색에 재사용하므로 독립 일반화 성능 아님'})

if __name__=='__main__':main()
