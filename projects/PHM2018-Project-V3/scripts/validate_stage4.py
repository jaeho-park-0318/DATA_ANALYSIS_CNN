"""동결 설정·동일 행·CUDA·점수·경보·seed 범위·임계값 이전 결과 점검."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
from sklearn.metrics import average_precision_score,roc_auc_score
from scripts.select_stage4 import choose_grid_config
from scripts.evaluate_stage4 import fingerprint

OUT=ROOT/'outputs/stage4'

def main():
    checks=[]
    def check(name,condition):
        assert bool(condition),name
        checks.append(dict(check=name,passed=True))
    def table(n):return pd.read_parquet(TABLES/('stage4_'+n+'.parquet'))
    runs=table('training_summary');curve=table('alarm_curve');ev=table('events');retuned=table('retuned_points');fixed=table('fixed_points');ledger=pd.read_parquet(ROOT/'outputs/stage3/common_validation_ledger.parquet');frozen=json.loads((OUT/'selection.json').read_text(encoding='utf-8'));raw_runs=json.loads((OUT/'training_runs.json').read_text(encoding='utf-8'))
    check('설정 동결 전 seed42 grid만 사용',frozen['seed42_only'] and hashlib.sha256((TABLES/'stage4_grid_budget_points.parquet').read_bytes()).hexdigest()==frozen['source_grid_sha256'])
    for selected in frozen['selected_by_budget']:
        p=choose_grid_config(table('grid_budget_points'),selected['selection_budget']);check('동결 설정 규칙 '+str(selected['selection_budget']),p.config_id==selected['config_id'] and p.threshold==selected['threshold'])
    for r in raw_runs:
        rid=r['run_id'];p=pd.read_parquet(OUT/(rid+'_validation.parquet'));pd.testing.assert_frame_equal(p[ledger.columns],ledger);known=p.audit_label.isin([0,1]);cfg=json.loads((OUT/(rid+'_booster_config.json')).read_text(encoding='utf-8'))
        check(rid+' 동일 행·유한 위험 점수',len(p)==309203 and p.score.between(0,1).all() and np.isfinite(p.score).all())
        check(rid+' 분류 지표 재계산',np.isclose(average_precision_score(p.loc[known,'audit_label'],p.loc[known,'score']),r['AP_known']) and np.isclose(roc_auc_score(p.loc[known,'audit_label'],p.loc[known,'score']),r['ROC_AUC_known']))
        check(rid+' 실제 CUDA 장치',r['device']=='cuda:0' and cfg['learner']['generic_param']['device']=='cuda:0')
        mark=json.loads((OUT/'evaluation'/(rid+'_contract.json')).read_text(encoding='utf-8'));check(rid+' 경보 캐시 입력/코드 해시',mark['fingerprint']==fingerprint(r))
        check(rid+' 양성 가중치 train 근거',np.isclose(r['positive_weight'],1 if r['weight_mode']=='unit' else np.sqrt((r['train_rows']-r['train_positive'])/r['train_positive'])))
        if r['reused_stage3']:
            old=pd.read_parquet(ROOT/f"outputs/stage3/{r['feature_set']}_xgboost_validation.parquet");pd.testing.assert_frame_equal(p,old);check(rid+' 기준 모델 재사용 점수 일치',True)
    for r in curve.itertuples():
        e=ev[ev.run_id.eq(r.run_id)&ev.threshold.eq(r.threshold)];check(r.run_id+f'/{r.threshold:.17g} 이벤트 집계',len(e)==232 and e.available.sum()==53 and e.detected.sum()==r.detected_events and e.loc[e.detected,'lead_raw'].between(60,600).all());check(r.run_id+f'/{r.threshold:.17g} 경보 분류/부담',r.alarm_episodes==sum(getattr(r,c) for c in curve if c.startswith('alarms_')) and np.isclose(r.alarm_episodes_per_10000_raw,10000*r.alarm_episodes/r.observed_support_raw))
    for mode,points,events in [('retuned',retuned,table('retuned_events')),('fixed',fixed,table('fixed_events'))]:
        for r in points.itertuples():
            e=events[events.run_id.eq(r.run_id)&events.budget_per_10000_raw.eq(r.budget_per_10000_raw)];check(mode+':'+r.run_id+':'+str(r.budget_per_10000_raw),len(e)==232 and e.detected.sum()==r.detected_events and e.loc[e.detected,'lead_raw'].between(60,600).all())
            if mode=='retuned':
                q=curve[curve.run_id.eq(r.run_id)&curve.alarm_episodes_per_10000_raw.le(r.budget_per_10000_raw)];check('상한내 최대 검출 '+r.run_id+str(r.budget_per_10000_raw),r.detected_events==q.detected_events.max())
            else:
                ref=retuned[retuned.config_id.eq(r.config_id)&retuned.seed.eq(42)&retuned.budget_per_10000_raw.eq(r.budget_per_10000_raw)].iloc[0];check('seed42 임계값 불변 '+r.run_id+str(r.budget_per_10000_raw),r.threshold==ref.threshold and r.exceeds_budget==(r.alarm_episodes_per_10000_raw>r.budget_per_10000_raw))
    ranges=table('seed_ranges')
    for r in ranges.itertuples():
        d=retuned if r.mode=='retuned' else fixed;q=d[d.config_id.eq(r.config_id)&d.budget_per_10000_raw.eq(r.budget_per_10000_raw)];check('3seed 범위 '+r.config_id+r.mode+str(r.budget_per_10000_raw),set(q.seed)=={42,43,44} and r.detected_min==q.detected_events.min() and r.detected_max==q.detected_events.max() and r.burden_max==q.alarm_episodes_per_10000_raw.max())
    for path,h in json.loads((ROOT/'outputs/validation/stage4_artifacts.json').read_text(encoding='utf-8'))['sha256'].items():check('아티팩트 SHA256 '+path,hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h)
    check('원본 크기/mtime 보존',snapshot()==json.loads((ROOT/'outputs/source_snapshot_before.json').read_text(encoding='utf-8')));check('holdout 미사용 계약',json.loads((OUT/'source_contract.json').read_text(encoding='utf-8'))['holdout_scored'] is False)
    dump_json(ROOT/'outputs/validation/stage4_checks.json',{'checks':checks,'training_performed':True,'holdout_scored':False});print('STAGE4 CHECKS:',len(checks),'PASS')

if __name__=='__main__':main()
