"""실제 모델 산출물과 경보 집계의 독립 일관성 점검."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
from sklearn.metrics import average_precision_score,roc_auc_score

def main(stage='stage2'):
    checks=[]
    def check(name,condition):
        assert bool(condition),name
        checks.append({'check':name,'passed':True})
    out=ROOT/'outputs'/stage
    def table(name):return pd.read_parquet(TABLES/(stage+'_'+name+'.parquet'))
    runs=table('training_summary');curve=table('alarm_curve');events=table('events');points=table('operating_points');alarms=table('selected_episodes');links=table('selected_matches')
    for r in runs.itertuples():
        p=pd.read_parquet(out/(r.run_id+'_validation.parquet'));k=p.audit_label.isin([0,1])
        check(r.run_id+' 예측 분모·유한 점수',len(p)==r.validation_all_rows and np.isfinite(p.score).all() and k.sum()==r.validation_known_rows)
        check(r.run_id+' 분류 지표 재계산',np.isclose(average_precision_score(p.loc[k,'audit_label'],p.loc[k,'score']),r.AP_known) and np.isclose(roc_auc_score(p.loc[k,'audit_label'],p.loc[k,'score']),r.ROC_AUC_known))
        check(r.run_id+' 저장 모델', (out/(r.run_id+'.joblib')).is_file() and r.artifact_roundtrip_passed)
        if r.model=='xgboost':check(r.run_id+' 실제 CUDA 학습',r.device=='cuda:0')
        f=curve[curve.run_id.eq(r.run_id)].sort_values(['alarm_episodes_per_10000_raw','recall_available'],ascending=[True,False]);best=-1;best_x=None
        for row in f.itertuples():
            expected=row.recall_available>best or (row.recall_available==best and row.alarm_episodes_per_10000_raw==best_x)
            check(r.run_id+f' Pareto {row.threshold:.17g}',row.pareto==expected)
            if row.recall_available>best:best=row.recall_available;best_x=row.alarm_episodes_per_10000_raw
    for r in curve.itertuples():
        e=events[events.run_id.eq(r.run_id)&events.threshold.eq(r.threshold)];key=r.run_id+f'/{r.threshold:.17g}'
        check(key+' 이벤트 분모/검출 재집계',len(e)==r.recorded_events==232 and e.detected.sum()==r.detected_events and e[~e.tagged].available.sum()==r.available_events)
        check(key+' 검출 시간 경계',e.loc[e.detected,'lead_raw'].between(60,600).all())
        check(key+' 부담 계산',np.isclose(r.alarm_episodes_per_10000_raw,10000*r.alarm_episodes/r.observed_support_raw))
    for r in points.drop_duplicates(['run_id','threshold']).itertuples():
        a=alarms[alarms.run_id.eq(r.run_id)&alarms.threshold.eq(r.threshold)];l=links[links.run_id.eq(r.run_id)&links.threshold.eq(r.threshold)];e=events[events.run_id.eq(r.run_id)&events.threshold.eq(r.threshold)]
        key=r.run_id+f'/{r.threshold:.17g}'
        check(key+' 저장 경보 횟수/ID',len(a)==r.alarm_episodes and not a.episode_id.duplicated().any())
        check(key+' 매칭 고유 이벤트',l.event_id.nunique()==r.detected_events)
        m=l.merge(a[['episode_id','start','start_label']],on='episode_id',validate='many_to_one').merge(e[['event_id','fault_time']],on='event_id',validate='many_to_one')
        check(key+' 매칭 원시각 독립 대조',m.start.between(m.fault_time-600,m.fault_time-60).all() and m.start_label.ne(3).all() and np.array_equal(m.fault_time-m.start,m.lead_raw))
        pred=pd.read_parquet(out/(r.run_id+'_validation.parquet'),columns=['Tool','time','score','audit_label'])
        j=a.merge(pred,left_on=['Tool','start'],right_on=['Tool','time'],how='left',validate='one_to_one')
        check(key+' 경보 시작 실제 점수/라벨',j.score.notna().all() and j.score.ge(r.threshold).all() and np.array_equal(j.audit_label,j.start_label))
    for p,h in json.loads((ROOT/'outputs/validation'/f'{stage}_artifacts.json').read_text(encoding='utf-8'))['sha256'].items():check('아티팩트 SHA256 '+p,hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h)
    check('원본 크기/mtime 보존',snapshot()==json.loads((ROOT/'outputs/source_snapshot_before.json').read_text(encoding='utf-8')))
    check('holdout 미점수화 계약',json.loads((out/'source_contract.json').read_text(encoding='utf-8'))['holdout_scored'] is False)
    dump_json(ROOT/'outputs/validation'/f'{stage}_checks.json',{'checks':checks,'training_performed':True,'holdout_scored':False});print(stage.upper()+' CHECKS: '+str(len(checks))+' PASS')

if __name__=='__main__':main()
