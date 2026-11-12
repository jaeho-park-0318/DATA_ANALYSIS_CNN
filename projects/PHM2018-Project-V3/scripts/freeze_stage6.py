"""보호된 검증만으로 하나의 모델과 두 임계값을 동결한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import shutil
from datetime import datetime,timezone
from src.common import *
from src.protected_group import PROTOCOL
from scripts.diagnose_stage3 import choose_budget

OUT=ROOT/'outputs/stage6'

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'holdout_access.json').exists()
    if (OUT/'frozen_manifest.json').exists():raise RuntimeError('이미 동결됨; 재선택 금지')
    checks=json.loads((ROOT/'outputs/validation/stage5_checks.json').read_text(encoding='utf-8'));assert all(c['passed'] for c in checks['checks'])
    temporal=pd.read_parquet(TABLES/'stage2_alarm_curve.parquet');group=pd.read_parquet(TABLES/'stage5_alarm_curve.parquet');candidates=[];details=[]
    for family in PROTOCOL['final_candidates']:
        pairs=[];scores=[];burdens=[]
        for budget in PROTOCOL['final_budgets']:
            t=choose_budget(temporal[temporal.run_id.eq(family+'_xgboost')],budget);g=choose_budget(group[group.run_id.eq(family+'_base_maximum')],budget);pairs.append(min(t.recall_recorded,g.recall_recorded));scores.extend([t.recall_recorded,g.recall_recorded]);burdens.extend([t.alarm_episodes_per_10000_raw,g.alarm_episodes_per_10000_raw])
            for name,r in [('temporal_validation',t),('protected_group_validation',g)]:details.append(dict(family=family,budget_per_10000_raw=budget,validation=name,detected_events=int(r.detected_events),recorded_events=int(r.recorded_events),recall_recorded=r.recall_recorded,actual_burden=r.alarm_episodes_per_10000_raw,threshold=r.threshold))
        candidates.append(dict(family=family,mean_worst_recall=float(np.mean(pairs)),mean_recall=float(np.mean(scores)),mean_burden=float(np.mean(burdens))))
    ranking=pd.DataFrame(candidates).sort_values(['mean_worst_recall','mean_recall','mean_burden','family'],ascending=[False,False,True,True]);family=ranking.iloc[0].family;points=[]
    for budget in PROTOCOL['final_budgets']:
        r=choose_budget(temporal[temporal.run_id.eq(family+'_xgboost')],budget)
        points.append(dict(operating_point=f'validation_budget_{budget}',validation_budget_per_10000_raw=budget,threshold=float(r.threshold),validation_detected=int(r.detected_events),validation_recorded=int(r.recorded_events),validation_available=int(r.available_events),validation_actual_burden=float(r.alarm_episodes_per_10000_raw)))
    save_table(ranking,'stage6_selection_ranking');save_table(pd.DataFrame(details),'stage6_selection_evidence')
    old=ROOT/'outputs/stage2';shutil.copy2(old/(family+'_xgboost.joblib'),OUT/'frozen_model.joblib');shutil.copy2(old/(family+'_preprocessor.joblib'),OUT/'frozen_preprocessor.joblib');shutil.copy2(old/(family+'_xgboost.ubj'),OUT/'frozen_model.ubj')
    paths=[OUT/'frozen_model.joblib',OUT/'frozen_model.ubj',OUT/'frozen_preprocessor.joblib',ROOT/'config/stage5_final_protocol.json',ROOT/'config/pretraining.json',ROOT/'config/preprocessing.json',ROOT/'scripts/evaluate_stage6.py',ROOT/'src/alarm_evaluation.py',ROOT/'src/final_evaluation.py',ROOT/'src/pretraining.py',ROOT/'src/features.py',ROOT/'data/pretraining/manifest.json',TABLES/'stage6_selection_evidence.parquet']
    frozen=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),family=family,model_source=f'outputs/stage2/{family}_xgboost.joblib',cohort='maximum',seed=42,H=600,L=60,operating_points=points,selection_rule=PROTOCOL['family_selection_rule'],refit_after_selection=False,holdout_scored_at_freeze=False,sha256={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},independence_scope=PROTOCOL['independence_scope'])
    with (OUT/'frozen_manifest.json').open('x',encoding='utf-8') as handle:json.dump(frozen,handle,ensure_ascii=False,indent=2)
    print(json.dumps(frozen,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
