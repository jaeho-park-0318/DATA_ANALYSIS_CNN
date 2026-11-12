"""검증 분할의 실제 입력으로 경보 평가기를 검증한다. 모델 점수/학습 없음."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
from src.pretraining import *
from src.alarm_evaluation import evaluate,observed_support
from src.query import manifest

def main():
    events=pd.read_parquet(TABLES/'events.parquet');bounds=pd.read_parquet(TABLES/'pretraining_split_boundaries.parquet');expected=pd.read_parquet(TABLES/'pretraining_candidates.parquet')
    summaries=[];event_rows=[];alarm_rows=[];match_rows=[];checks=[];states=[]
    for scheme in ['temporal','group']:
        for tool in sorted(events.Tool.unique()):
            b=bounds[bounds.scheme.eq(scheme)&bounds.split.eq('validation')&bounds.Tool.eq(tool)]
            if b.empty:continue
            b=b.iloc[0];ev=events[events.Tool.eq(tool)&events.fault_time.ge(b.start)&events.fault_time.lt(b.end_exclusive)]
            profiles={}
            for f in 'ABC':
                chunks=[]
                for x,meta in iter_scoring_batches(scheme,'validation',f,600,'maximum',tool=tool):
                    assert list(x.columns)==schema()[f]
                    chunks.append(meta[['Tool','time','segment_id','audit_label']])
                d=pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame(columns=['Tool','time','segment_id','audit_label'])
                row=expected[expected.scheme.eq(scheme)&expected.split.eq('validation')&expected.Tool.eq(tool)&expected.H.eq(600)&expected.feature_set.eq(f)&expected.cohort.eq('maximum')].iloc[0]
                assert len(d)==row.available_anchors
                assert int(d.audit_label.isin([0,1]).sum())==row.positive_anchors+row.negative_anchors
                assert not d.duplicated(['Tool','time']).any()
                assert d.time.ge(b.start+900).all() and d.time.lt(b.end_exclusive-1800).all()
                checks.append(dict(check=f'{scheme}/{tool}/{f} 전체 가용/분류 분모·경계·스키마',passed=True))
                profiles[f]=d
                for code in [0,1,2,3]:states.append(dict(scheme=scheme,Tool=tool,feature_set=f,status=['NEG','POS','NA','EXCLUDE'][code],anchors=int(d.audit_label.eq(code).sum())))
            exposure={f:0. for f in 'ABC'};m=manifest(tool)
            for index,item in enumerate(m['normalized_index']):
                if item['end']<b.start or item['start']>=b.end_exclusive:continue
                n=pd.read_parquet(CACHE/'source'/tool/item['file'],columns=['time','segment_id'])
                following=m['normalized_index'][index+1]['start'] if index+1<len(m['normalized_index']) else None
                for f,d in profiles.items():exposure[f]+=observed_support(n.time.to_numpy(),n.segment_id.to_numpy(),d,next_time=following)
            for f,d in profiles.items():
                for baseline,value in [('no_alarm',0.),('always_alarm',1.)]:
                    summary,e,a,links=evaluate(d.assign(score=value),ev,.5)
                    prefix=dict(scheme=scheme,split='validation',Tool=tool,feature_set=f,baseline=baseline)
                    summaries.append(dict(**prefix,**summary,available_anchors=len(d),observed_support_raw=exposure[f],alarm_episodes_per_10000_raw=10000*summary['alarm_episodes']/exposure[f] if exposure[f] else np.nan))
                    for table,dest in [(e,event_rows),(a,alarm_rows),(links,match_rows)]:
                        q=table.copy()
                        for k,v in prefix.items():q[k]=v
                        dest.append(q)
                    assert summary['detected_events']<=summary['available_events']<=summary['scorable_events']
                    if not value:assert summary['detected_events']==summary['alarm_episodes']==0
            print(scheme+' '+tool+' 전체 가용 입력·경보 평가 완료',flush=True)
    for name,rows in [('alarm_stage1_summary',summaries),('alarm_stage1_label_states',states)]:save_table(pd.DataFrame(rows),name)
    for name,rows in [('alarm_stage1_events',event_rows),('alarm_stage1_episodes',alarm_rows),('alarm_stage1_matches',match_rows)]:save_table(pd.concat(rows,ignore_index=True),name)
    assert snapshot()==json.loads((ROOT/'outputs/source_snapshot_before.json').read_text(encoding='utf-8'))
    dump_json(ROOT/'outputs/validation/alarm_stage1_checks.json',{'checks':checks,'holdout_scored':False,'scores':'고정 기준0/1; 학습 모델 점수 아님','raw_metadata_unchanged':True,'exposure_definition':'가용 Anchor의 [t,t+60) 점수 유지 구간과 같은 segment의 관측 [s,s+4) 지원 구간 교집합. 4는 nominal raw. 실제 가동시간 아님.'})
    print('STAGE1 COMPLETE: no model training; holdout untouched')

if __name__=='__main__':main()
