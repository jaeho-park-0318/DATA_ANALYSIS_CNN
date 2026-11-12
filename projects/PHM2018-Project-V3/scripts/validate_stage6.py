"""Holdout을 다시 추론하지 않고 저장 결과와 동결 계약을 점검한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datetime import datetime
from sklearn.metrics import average_precision_score,roc_auc_score
from src.common import *
from scripts.evaluate_stage6 import verify_frozen

OUT=ROOT/'outputs/stage6'

def main():
    checks=[]
    def check(n,c):
        assert bool(c),n
        checks.append(dict(check=n,passed=True))
    frozen=json.loads((OUT/'frozen_manifest.json').read_text(encoding='utf-8'));access=json.loads((OUT/'holdout_access.json').read_text(encoding='utf-8'));verify_frozen(frozen)
    check('동결 후 단일 Holdout 추론',access['status']=='complete' and access['full_prediction_passes']==1 and not access['threshold_search'] and not access['refit'] and datetime.fromisoformat(frozen['frozen_at_utc'])<datetime.fromisoformat(access['opened_at_utc']))
    check('동결 manifest 보존',hashlib.sha256((OUT/'frozen_manifest.json').read_bytes()).hexdigest()==access['frozen_manifest_sha256'])
    check('단일 예측 결과 보존',hashlib.sha256((OUT/'holdout_predictions.parquet').read_bytes()).hexdigest()==access['prediction_sha256'])
    p=pd.read_parquet(OUT/'holdout_predictions.parquet');bounds=pd.read_parquet(TABLES/'pretraining_split_boundaries.parquet');b=bounds[bounds.scheme.eq('temporal')&bounds.split.eq('holdout')].set_index('Tool')
    check('모든 예측은 최종 시간 Holdout 내부',p.time.ge(p.Tool.map(b.start)+900).all() and p.time.lt(p.Tool.map(b.end_exclusive)-1800).all());check('중복/비유한 예측 없음',not p.duplicated(['Tool','time']).any() and np.isfinite(p.score).all() and p.score.between(0,1).all())
    registry=json.loads((ROOT/'data/pretraining/manifest.json').read_text(encoding='utf-8'));assignments={r['source']:r['assignment'] for r in registry['files'] if r['scheme']=='temporal'}
    for source,g in p.groupby('source_file'):
        a=pd.read_parquet(ROOT/assignments[source],columns=['time','split','y_h600']).iloc[g.source_row.to_numpy()];check(source+' Holdout 시각/라벨/분할 원장',a.split.eq(2).all() and np.array_equal(a.time,g.time) and np.array_equal(a.y_h600,g.audit_label))
    summary=pd.read_parquet(TABLES/'stage6_holdout_summary.parquet');events=pd.read_parquet(TABLES/'stage6_holdout_events.parquet');alarms=pd.read_parquet(TABLES/'stage6_holdout_episodes.parquet');links=pd.read_parquet(TABLES/'stage6_holdout_matches.parquet');raw_events=pd.read_parquet(TABLES/'events.parquet');expected=raw_events[raw_events.fault_time.ge(raw_events.Tool.map(b.start))&raw_events.fault_time.lt(raw_events.Tool.map(b.end_exclusive))]
    check('동결한 두 운영점만 평가',set(summary.operating_point)=={x['operating_point'] for x in frozen['operating_points']} and len(summary)==len(frozen['operating_points']))
    index=pd.MultiIndex.from_frame(p[['Tool','time']])
    for point in frozen['operating_points']:
        name=point['operating_point'];r=summary[summary.operating_point.eq(name)].iloc[0];e=events[events.operating_point.eq(name)];a=alarms[alarms.operating_point.eq(name)];l=links[links.operating_point.eq(name)]
        check(name+' 임계값 불변',r.threshold==point['threshold']);check(name+' 이벤트 분모/검출',set(e.event_id)==set(expected.event_id) and e.detected.sum()==r.detected_events and e[~e.tagged].available.sum()==r.available_events and len(e)==r.recorded_events)
        check(name+' 경보 분류/부담',len(a)==r.alarm_episodes and r.alarm_episodes==sum(r[k] for k in summary if k.startswith('alarms_')) and np.isclose(r.alarm_episodes_per_10000_raw,10000*r.alarm_episodes/r.observed_support_raw))
        check(name+' 발화 Anchor 보존',a.anchors.sum()==p.score.ge(r.threshold).sum())
        if len(a):
            ix=index.get_indexer(pd.MultiIndex.from_arrays([a.Tool,a.start]));check(name+' 실제 경보 시작점',np.all(ix>=0) and np.all(p.score.to_numpy()[ix]>=r.threshold) and np.array_equal(p.audit_label.to_numpy()[ix],a.start_label))
            previous=index.get_indexer(pd.MultiIndex.from_arrays([a.Tool,a.start-60]));safe=np.maximum(previous,0);continued=(previous>=0)&(p.score.to_numpy()[safe]>=r.threshold)&(p.segment_id.to_numpy()[safe]==a.segment_id.to_numpy());check(name+' 시작 이전 연속 발화 없음',not continued.any())
        m=l.merge(a[['episode_id','start','start_label']],on='episode_id',validate='many_to_one').merge(e[['event_id','fault_time']],on='event_id',validate='many_to_one');check(name+' 매칭 원시각 독립 대조',m.start.between(m.fault_time-600,m.fault_time-60).all() and m.start_label.ne(3).all() and np.array_equal(m.fault_time-m.start,m.lead_raw) and l.event_id.nunique()==r.detected_events)
    metrics=json.loads((OUT/'holdout_classification.json').read_text(encoding='utf-8'));k=p.audit_label.isin([0,1]);check('분류 분모',int(k.sum())==metrics['known_anchors'])
    if p.loc[k,'audit_label'].nunique()==2:check('분류 지표 재계산',np.isclose(average_precision_score(p.loc[k,'audit_label'],p.loc[k,'score']),metrics['AP_known']) and np.isclose(roc_auc_score(p.loc[k,'audit_label'],p.loc[k,'score']),metrics['ROC_AUC_known']))
    check('원본 크기/mtime 보존',snapshot()==json.loads((ROOT/'outputs/source_snapshot_before.json').read_text(encoding='utf-8')))
    dump_json(ROOT/'outputs/validation/stage6_checks.json',{'checks':checks,'holdout_scored':True,'full_prediction_passes':1,'frozen_files_unchanged':True,'no_reselection':True});print('STAGE6 CHECKS:',len(checks),'PASS')

if __name__=='__main__':main()
