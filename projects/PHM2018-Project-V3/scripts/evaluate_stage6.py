"""동결된 한 모델로 최종 시간 Holdout을 한 번 추론·평가한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,time
from datetime import datetime,timezone
import joblib,xgboost as xgb
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
from src.common import *
from src.pretraining import iter_scoring_batches
from src.final_evaluation import at_threshold,exposure_by_tool

OUT=ROOT/'outputs/stage6'

def verify_frozen(frozen):
    for name,h in frozen['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,'동결 파일 변경: '+name

def main(preflight=False):
    frozen=json.loads((OUT/'frozen_manifest.json').read_text(encoding='utf-8'));verify_frozen(frozen);family=frozen['family'];model=joblib.load(OUT/'frozen_model.joblib');prep=joblib.load(OUT/'frozen_preprocessor.joblib')
    if preflight:
        x,m=next(iter_scoring_batches('temporal','validation',family,600,'maximum'));x=x.iloc[:128];m=m.iloc[:128];score=model.get_booster().predict(xgb.DMatrix(prep.transform(x).astype('float32')),iteration_range=(0,model.best_iteration+1));saved=pd.read_parquet(ROOT/f'outputs/stage2/{family}_xgboost_validation.parquet',columns=['Tool','time','score']);expected=m[['Tool','time']].merge(saved,on=['Tool','time'],how='left',validate='one_to_one');np.testing.assert_allclose(score,expected.score,rtol=1e-6,atol=1e-7);print('FROZEN MODEL VALIDATION PREFLIGHT PASS');return
    access=OUT/'holdout_access.json'
    if access.exists():
        state=json.loads(access.read_text(encoding='utf-8'))
        if state['status']=='complete':print('이미 완료: 저장된 단일 평가 결과를 사용합니다. 재추론하지 않습니다.');return
        raise RuntimeError('평가 시작 기록이 이미 존재함. 자동 재평가 금지; 완료 여부 점검 필요')
    state=dict(status='opened',opened_at_utc=datetime.now(timezone.utc).isoformat(),frozen_manifest_sha256=hashlib.sha256((OUT/'frozen_manifest.json').read_bytes()).hexdigest(),full_prediction_passes=1,threshold_search=False,refit=False)
    with access.open('x',encoding='utf-8') as handle:json.dump(state,handle,ensure_ascii=False,indent=2)
    started=time.time();parts=[]
    for x,m in iter_scoring_batches('temporal','holdout',family,600,'maximum'):
        score=model.get_booster().predict(xgb.DMatrix(prep.transform(x).astype('float32')),iteration_range=(0,model.best_iteration+1));assert np.isfinite(score).all();parts.append(m.assign(score=score))
    p=pd.concat(parts,ignore_index=True).sort_values(['Tool','time']).reset_index(drop=True);write_parquet(p,OUT/'holdout_predictions.parquet');print('Holdout 단일 추론 완료: '+str(len(p))+'행',flush=True)
    bounds=pd.read_parquet(TABLES/'pretraining_split_boundaries.parquet');bounds=bounds[bounds.scheme.eq('temporal')&bounds.split.eq('holdout')];events=pd.read_parquet(TABLES/'events.parquet');b=bounds.set_index('Tool');events=events[events.fault_time.ge(events.Tool.map(b.start))&events.fault_time.lt(events.Tool.map(b.end_exclusive))].copy();supports=exposure_by_tool(p);dump_json(OUT/'holdout_exposure.json',supports)
    summaries=[];evs=[];alarms=[];links=[];tools=[];uncertainty=[];pair_ids=sorted(events.Tool.str[:2].unique());rng=np.random.default_rng(42);ix=rng.integers(0,len(pair_ids),(2000,len(pair_ids)))
    for point in frozen['operating_points']:
        row,e,a,l,t=at_threshold(p,events,point['threshold'],supports);row.update(operating_point=point['operating_point'],validation_budget_per_10000_raw=point['validation_budget_per_10000_raw'],family=family);summaries.append(row)
        for df,dest in [(e,evs),(a,alarms),(l,links),(t,tools)]:df['operating_point']=point['operating_point'];dest.append(df)
        z=e[~e.tagged].copy();z['pair']=z.Tool.str[:2];c=z.groupby('pair').agg(detected=('detected','sum'),available=('available','sum'),recorded=('event_id','size')).reindex(pair_ids,fill_value=0);sample=c.to_numpy()[ix].sum(axis=1)
        for den,j in [('available',1),('recorded',2)]:
            good=sample[:,j]>0;v=sample[good,0]/sample[good,j];uncertainty.append(dict(operating_point=point['operating_point'],denominator=den,ci_low=np.quantile(v,.025),ci_high=np.quantile(v,.975),valid_samples=int(good.sum()),lead_p10=z.loc[z.detected,'lead_raw'].quantile(.1),lead_median=z.loc[z.detected,'lead_raw'].median()))
        print(json.dumps(row,ensure_ascii=False),flush=True)
    for name,d in [('summary',pd.DataFrame(summaries)),('events',pd.concat(evs,ignore_index=True)),('episodes',pd.concat(alarms,ignore_index=True)),('matches',pd.concat(links,ignore_index=True)),('by_tool',pd.concat(tools,ignore_index=True)),('uncertainty',pd.DataFrame(uncertainty))]:save_table(d,'stage6_holdout_'+name)
    known=p.audit_label.isin([0,1]);y=p.loc[known,'audit_label'];score=p.loc[known,'score'];metrics=dict(all_anchors=len(p),known_anchors=int(known.sum()),positive=int(y.eq(1).sum()),AP_known=float(average_precision_score(y,score)) if y.nunique()==2 else None,ROC_AUC_known=float(roc_auc_score(y,score)) if y.nunique()==2 else None,log_loss_known=float(log_loss(y,score,labels=[0,1])),label_counts={str(k):int(v) for k,v in p.audit_label.value_counts().items()});dump_json(OUT/'holdout_classification.json',metrics)
    verify_frozen(frozen);assert snapshot()==json.loads((ROOT/'outputs/source_snapshot_before.json').read_text(encoding='utf-8'))
    state.update(status='complete',completed_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-started,prediction_sha256=hashlib.sha256((OUT/'holdout_predictions.parquet').read_bytes()).hexdigest(),summary_sha256=hashlib.sha256((TABLES/'stage6_holdout_summary.parquet').read_bytes()).hexdigest());dump_json(access,state)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');main(parser.parse_args().preflight)
