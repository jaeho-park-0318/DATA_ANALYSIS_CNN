"""학습·선정·동결·저장 예측의 실제 원장/경보 독립 대조."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datetime import datetime
import joblib,xgboost as xgb
from sklearn.metrics import average_precision_score,roc_auc_score
from src.specialist_experiment import *
from src.pretraining import SPLITS


def main():
    checks=[]
    def check(name,value):
        assert bool(value),name
        checks.append(dict(check=name,passed=True))
    verify_previous();check('이전 최종 모델/예측/표/접근 기록 바이트 보존',True)
    selection=readjson(OUT/'selection.json');frozen=readjson(OUT/'frozen_manifest.json');access=readjson(OUT/'evaluation_access.json');eligible=pd.read_parquet(TABLES/'stage7_eligibility.parquet')
    check('기준 변경 없음',selection['protocol_sha256']==digest(PROTOCOL_PATH)==readjson(OUT/'audit_started.json')['protocol_sha256'])
    check('선정 표 보존',selection['eligibility_table_sha256']==digest(TABLES/'stage7_eligibility.parquet'))
    for r in eligible.to_dict('records'):
        check(r['Tool']+' 최소 데이터 기준',r['eligible']==all(r[k]>=v for k,v in PROTOCOL['eligibility'].items()))
    rank=eligible[eligible.eligible].sort_values(['train_positive_clusters','train_positive_rows','Tool'],ascending=[False,False,True]).head(PROTOCOL['maximum_tools'])
    check('장비 수 제한 및 학습 데이터 기준 순위',rank.Tool.tolist()==selection['selected_tools']==frozen['tools'] and len(rank)<=3)
    check('후반 예측 이전 동결',datetime.fromisoformat(selection['selected_at_utc'])<datetime.fromisoformat(frozen['frozen_at_utc'])<datetime.fromisoformat(access['opened_at_utc']))
    check('후속 연구임을 명시',access['status']=='complete' and not access['independent_final_test'] and not access['threshold_search_on_later_partition'] and not access['refit'] and access['specialist_prediction_passes_per_tool']==1)
    check('동결 manifest 보존',digest(OUT/'frozen_manifest.json')==access['frozen_manifest_sha256'])
    for path,h in frozen['sha256'].items():check(path+' 동결 해시',digest(ROOT/path)==h)
    summary=pd.read_parquet(TABLES/'stage7_later_summary.parquet');ee=pd.read_parquet(TABLES/'stage7_later_events.parquet');aa=pd.read_parquet(TABLES/'stage7_later_episodes.parquet');ll=pd.read_parquet(TABLES/'stage7_later_matches.parquet')
    curve=pd.read_parquet(TABLES/'stage7_validation_curve.parquet');vpoints=pd.read_parquet(TABLES/'stage7_validation_points.parquet')
    common=pd.read_parquet(ROOT/'outputs/stage6/holdout_predictions.parquet');registry=readjson(ROOT/'data/pretraining/manifest.json');assign={r['source']:r['assignment'] for r in registry['files'] if r['scheme']=='temporal'}
    bounds=pd.read_parquet(TABLES/'pretraining_split_boundaries.parquet');classif=pd.read_parquet(TABLES/'stage7_later_classification.parquet')
    check('평가 결과 표 해시',digest(TABLES/'stage7_later_summary.parquet')==access['summary_sha256'])
    for tool in frozen['tools']:
        p=pd.read_parquet(OUT/(tool+'_later_predictions.parquet'));g=common[common.Tool.eq(tool)].sort_values('time').reset_index(drop=True)
        check(tool+' 예측 보존/유한',digest(OUT/(tool+'_later_predictions.parquet'))==access['prediction_sha256'][tool] and p.score.between(0,1).all() and not p.duplicated(['Tool','time']).any())
        pd.testing.assert_frame_equal(p.drop(columns='score'),g.drop(columns='score'));check(tool+' 세 방식의 평가 입력/정답 동일',True)
        for split,name in [('train','_train_ledger'),('validation','_validation_ledger'),('holdout','_later_predictions')]:
            m=pd.read_parquet(OUT/(tool+name+'.parquet'));b=bounds[bounds.scheme.eq('temporal')&bounds.Tool.eq(tool)&bounds.split.eq(split)].iloc[0]
            check(tool+split+' 장비/시간 경계',m.Tool.eq(tool).all() and m.time.ge(b.start+900).all() and m.time.lt(b.end_exclusive-1800).all())
            if split=='train':check(tool+' 학습은 알려진 라벨만',m.audit_label.isin([0,1]).all())
            for source,z in m.groupby('source_file'):
                a=pd.read_parquet(ROOT/assign[source],columns=['time','split','y_h600']).iloc[z.source_row.to_numpy()]
                check(tool+split+source+' 원장 시각/라벨/분할',np.array_equal(a.time,z.time) and np.array_equal(a.y_h600,z.audit_label) and a.split.eq(SPLITS.index(split)).all())
        # 재현 검사는 이미 본 validation128행만 사용. 후반 재추론 없음.
        vx,vm=load('validation',tool);prep=joblib.load(OUT/(tool+'_preprocessor.joblib'));model=joblib.load(OUT/(tool+'_model.joblib'));pred=model.get_booster().predict(xgb.DMatrix(prep.transform(vx.iloc[:128]).astype('float32')),iteration_range=(0,model.best_iteration+1));saved=pd.read_parquet(OUT/(tool+'_validation_predictions.parquet'));np.testing.assert_allclose(pred,saved.score.iloc[:128],rtol=1e-6,atol=1e-7)
        check(tool+' 저장 모델 실제 validation 재현',True);check(tool+' 학습 CUDA',readjson(OUT/(tool+'_booster_config.json'))['learner']['generic_param']['device']=='cuda:0')
        ev=events_for(tool,'holdout')
        for point in [r for r in frozen['points'] if r['Tool']==tool]:
            method=point['method'];budget=point['budget'];key=f'{tool}/{method}/{budget}'
            mask=lambda d:d.Tool.eq(tool)&d.method.eq(method)&d.budget.eq(budget)
            r=summary[mask(summary)].iloc[0];e=ee[mask(ee)];a=aa[mask(aa)];l=ll[mask(ll)];frame=p if method=='specialist' else g
            check(key+' 숫자 임계값 보존',r.threshold==point['threshold']==vpoints[mask(vpoints)].iloc[0].threshold)
            if method!='global_frozen':check(key+' validation 선택 재계산',choose(curve[curve.Tool.eq(tool)&curve.method.eq(method)],budget).threshold==r.threshold)
            check(key+' 이벤트 분모와 검출',set(e.event_id)==set(ev.event_id) and len(e)==r.recorded_events and e.detected.sum()==r.detected_events and e.available.sum()==r.available_events)
            times=frame.time.to_numpy()
            available=np.searchsorted(times,e.fault_time.to_numpy()-60,side='right')>np.searchsorted(times,e.fault_time.to_numpy()-600)
            check(key+' 입력 가용성 독립 계산',np.array_equal(available,e.available.to_numpy()))
            check(key+' 에피소드 분류/점수 발화 합계',len(a)==r.alarm_episodes and sum(r[k] for k in summary if k.startswith('alarms_'))==len(a) and a.anchors.sum()==frame.score.ge(r.threshold).sum())
            check(key+' 부담 재계산',np.isclose(r.alarm_episodes_per_10000_raw,10000*len(a)/r.observed_support_raw))
            index=pd.Index(frame.time)
            if len(a):
                ix=index.get_indexer(a.start);check(key+' 경보 시작점',np.all(ix>=0) and np.all(frame.score.to_numpy()[ix]>=r.threshold))
                previous=index.get_indexer(a.start-60);safe=np.maximum(previous,0);continued=(previous>=0)&(frame.score.to_numpy()[safe]>=r.threshold)&(frame.segment_id.to_numpy()[safe]==a.segment_id.to_numpy());check(key+' 연속 발화 중복 시작 없음',not continued.any())
            match=l.merge(a[['episode_id','start','start_label']],on='episode_id',validate='many_to_one').merge(e[['event_id','fault_time']],on='event_id',validate='many_to_one')
            check(key+' 고장 매칭 독립 대조',match.start.between(match.fault_time-600,match.fault_time-60).all() and match.start_label.ne(3).all() and np.array_equal(match.fault_time-match.start,match.lead_raw) and l.event_id.nunique()==r.detected_events)
        for method,frame in [('global',g),('specialist',p)]:
            r=classif[classif.Tool.eq(tool)&classif.method.eq(method)].iloc[0];k=frame.audit_label.isin([0,1]);y=frame.loc[k,'audit_label'];score=frame.loc[k,'score'];check(tool+method+' 분류 지표 분모',k.sum()==r.known_rows)
            if y.nunique()==2:check(tool+method+' AP/AUC',np.isclose(average_precision_score(y,score),r.AP_known) and np.isclose(roc_auc_score(y,score),r.ROC_AUC_known))
    check('원본 크기/수정시각 보존',snapshot()==readjson(ROOT/'outputs/source_snapshot_before.json'))
    dump_json(OUT/'checks.json',dict(checks=checks,checked_at_utc=utc(),additional_later_inference_during_validation=0,independent_final_test=False))
    print('SPECIALIST CHECKS:',len(checks),'PASS')


if __name__=='__main__':main()
