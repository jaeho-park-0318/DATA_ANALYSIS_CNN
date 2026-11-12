"""이미 공개된 후반 구간의 연구 비교. 기존 공통 예측 재사용."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,xgboost as xgb
from sklearn.metrics import average_precision_score,roc_auc_score
from src.specialist_experiment import *


def main():
    verify_previous();frozen=readjson(OUT/'frozen_manifest.json')
    for path,h in frozen['sha256'].items():assert digest(ROOT/path)==h,path
    assert not (OUT/'evaluation_access.json').exists(),'재예측 금지: 완료/실패 기록 먼저 확인'
    state=dict(status='opened',opened_at_utc=utc(),independent_final_test=False,reference_predictions_reused=True,specialist_prediction_passes_per_tool=1,threshold_search_on_later_partition=False,refit=False,frozen_manifest_sha256=digest(OUT/'frozen_manifest.json'))
    with (OUT/'evaluation_access.json').open('x',encoding='utf-8') as handle:json.dump(state,handle,ensure_ascii=False,indent=2)
    common=pd.read_parquet(ROOT/'outputs/stage6/holdout_predictions.parquet');supports=readjson(ROOT/'outputs/stage6/holdout_exposure.json')
    rows=[];eventparts=[];alarmparts=[];linkparts=[];classification=[];pairs=[]
    for tool in frozen['tools']:
        x,m=load('holdout',tool);globalp=common[common.Tool.eq(tool)].sort_values('time').reset_index(drop=True);pd.testing.assert_frame_equal(m,globalp[m.columns])
        prep=joblib.load(OUT/(tool+'_preprocessor.joblib'));model=joblib.load(OUT/(tool+'_model.joblib'))
        score=model.get_booster().predict(xgb.DMatrix(prep.transform(x).astype('float32')),iteration_range=(0,model.best_iteration+1));assert np.isfinite(score).all();p=m.assign(score=score);write_parquet(p,OUT/(tool+'_later_predictions.parquet'))
        ev=events_for(tool,'holdout');support=supports[tool];eventmap={}
        for point in [r for r in frozen['points'] if r['Tool']==tool]:
            frame=p if point['method']=='specialist' else globalp
            row,e,a,l=alarm_result(frame,ev,point['threshold'],support);row.update(point);rows.append(row)
            for df,dest in [(e,eventparts),(a,alarmparts),(l,linkparts)]:
                df['method']=point['method'];df['budget']=point['budget'];df['Tool']=tool;dest.append(df)
            eventmap[(point['method'],point['budget'])]=e
        for method,frame in [('specialist',p),('global',globalp)]:
            k=frame.audit_label.isin([0,1]);y=frame.loc[k,'audit_label'];s=frame.loc[k,'score'];classification.append(dict(Tool=tool,method=method,all_rows=len(frame),known_rows=int(k.sum()),positive=int(y.eq(1).sum()),AP_known=average_precision_score(y,s) if y.nunique()==2 else np.nan,ROC_AUC_known=roc_auc_score(y,s) if y.nunique()==2 else np.nan))
        for budget in PROTOCOL['budgets']:
            special=eventmap[('specialist',budget)]
            for control in ['global_frozen','global_tool_threshold']:
                ref=eventmap[(control,budget)];z=special[['event_id','available','detected']].merge(ref[['event_id','available','detected']],on='event_id',suffixes=('_specialist','_control'),validate='one_to_one')
                assert z.available_specialist.equals(z.available_control)
                pairs.append(dict(Tool=tool,budget=budget,control=control,recorded=len(z),available=int(z.available_specialist.sum()),both_detected=int((z.detected_specialist&z.detected_control).sum()),specialist_only=int((z.detected_specialist&~z.detected_control).sum()),control_only=int((~z.detected_specialist&z.detected_control).sum()),neither=int((~z.detected_specialist&~z.detected_control).sum())))
        print(tool+' 후반 구간 전용 예측 '+str(len(p))+'행 완료; 공통 점수는 재사용',flush=True)
    for name,d in [('later_summary',pd.DataFrame(rows)),('later_events',pd.concat(eventparts,ignore_index=True)),('later_episodes',pd.concat(alarmparts,ignore_index=True)),('later_matches',pd.concat(linkparts,ignore_index=True)),('later_classification',pd.DataFrame(classification)),('paired_events',pd.DataFrame(pairs))]:save_table(d,'stage7_'+name)
    verify_previous()
    for path,h in frozen['sha256'].items():assert digest(ROOT/path)==h,path
    state.update(status='complete',completed_at_utc=utc(),prediction_sha256={tool:digest(OUT/(tool+'_later_predictions.parquet')) for tool in frozen['tools']},summary_sha256=digest(TABLES/'stage7_later_summary.parquet'))
    dump_json(OUT/'evaluation_access.json',state);print('후속 연구 평가 완료; 기존 최종 결과 보존',flush=True)


if __name__=='__main__':main()
