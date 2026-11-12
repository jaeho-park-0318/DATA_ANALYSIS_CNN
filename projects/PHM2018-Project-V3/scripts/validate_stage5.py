"""장비 그룹 실험이 보호된 시간/라벨 계약을 지켰는지 동결 전 확인."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
from sklearn.metrics import average_precision_score,roc_auc_score
from src.common import *
from src.features import schema
from src.protected_group import PROTOCOL,protected_mask,validation_events

def main():
    checks=[]
    def check(n,c):
        assert bool(c),n
        checks.append(dict(check=n,passed=True))
    out=ROOT/'outputs/stage5';bounds=pd.read_parquet(TABLES/'pretraining_split_boundaries.parquet');registry=json.loads((ROOT/'data/pretraining/manifest.json').read_text(encoding='utf-8'));assignments={i['source']:i['assignment'] for i in registry['files'] if i['scheme']=='temporal'};runs=json.loads((out/'training_runs.json').read_text(encoding='utf-8'))
    for r in runs:
        rid=r['run_id'];ledgers={}
        for role in ['train','validation']:
            m=pd.read_parquet(out/(rid+'_'+role+'_ledger.parquet'));ledgers[role]=m
            check(rid+'/'+role+' 시간Holdout 제외',protected_mask(m,bounds).all());check(rid+'/'+role+' 장비 역할',m.Tool.str[:2].isin(PROTOCOL['group_'+role+'_tool_prefixes']).all())
            for source,g in m.groupby('source_file'):
                a=pd.read_parquet(ROOT/assignments[source],columns=['time','split','y_h600']).iloc[g.source_row.to_numpy()];check(rid+'/'+role+'/'+source+' temporal 라벨 근거',np.array_equal(a.time,g.time) and np.array_equal(a.y_h600,g.audit_label) and np.array_equal(a.split,g.temporal_partition.map({'train':0,'validation':1})))
        check(rid+' 학습/검증 장비 분리',not set(ledgers['train'].Tool)&set(ledgers['validation'].Tool));p=pd.read_parquet(out/(rid+'_validation.parquet'));pd.testing.assert_frame_equal(p[ledgers['validation'].columns],ledgers['validation']);known=p.audit_label.isin([0,1]);check(rid+' AP/ROC 재계산',np.isclose(average_precision_score(p.loc[known,'audit_label'],p.loc[known,'score']),r['AP_known']) and np.isclose(roc_auc_score(p.loc[known,'audit_label'],p.loc[known,'score']),r['ROC_AUC_known']))
        prep=joblib.load(out/(rid+'_preprocessor.joblib'));check(rid+' 특징 화이트리스트',list(prep.feature_names_in_)==schema()[r['family']]);cfg=json.loads((out/(rid+'_booster_config.json')).read_text(encoding='utf-8'));check(rid+' 실제CUDA',cfg['learner']['generic_param']['device']=='cuda:0')
    curves=pd.read_parquet(TABLES/'stage5_alarm_curve.parquet');events=pd.read_parquet(TABLES/'stage5_events.parquet');expected=validation_events()
    for r in curves.itertuples():
        e=events[events.run_id.eq(r.run_id)&events.threshold.eq(r.threshold)];check(r.run_id+f'/{r.threshold:.17g} 이벤트/경보 집계',set(e.event_id)==set(expected.event_id) and e.detected.sum()==r.detected_events and e.loc[e.detected,'lead_raw'].between(60,600).all() and np.isclose(r.alarm_episodes_per_10000_raw,10000*r.alarm_episodes/r.observed_support_raw))
    check('현재 최종Holdout 미개봉',not (ROOT/'outputs/stage6/holdout_access.json').exists());check('원본 크기/mtime 보존',snapshot()==json.loads((ROOT/'outputs/source_snapshot_before.json').read_text(encoding='utf-8')))
    dump_json(ROOT/'outputs/validation/stage5_checks.json',{'checks':checks,'protected_temporal_holdout_used':False});print('STAGE5 CHECKS:',len(checks),'PASS')

if __name__=='__main__':main()
