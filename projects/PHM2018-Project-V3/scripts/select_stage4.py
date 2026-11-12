"""seed42 탐색에서 두 비교 상한별 설정을 선택한 후 동결한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
from scripts.diagnose_stage3 import choose_budget

OUT=ROOT/'outputs/stage4'

def choose_grid_config(points,budget):
    return points[points.budget_per_10000_raw.eq(budget)].sort_values(['detected_events','alarm_episodes_per_10000_raw','depth','positive_weight','feature_set'],ascending=[False,True,True,True,True]).iloc[0]

def main():
    config=json.loads((ROOT/'config/stage4_experiment.json').read_text(encoding='utf-8'));runs=pd.read_parquet(TABLES/'stage4_training_summary.parquet');curve=pd.read_parquet(TABLES/'stage4_alarm_curve.parquet');grid=runs[runs.seed.eq(42)&runs.feature_set.isin(['A','B'])];assert len(grid)==12
    rows=[]
    for run in grid.itertuples():
        for budget in config['comparison_budgets_per_10000_raw']:
            pick=choose_budget(curve[curve.run_id.eq(run.run_id)],budget).to_dict();pick.update(budget_per_10000_raw=budget,config_id=run.config_id,depth=run.depth,weight_mode=run.weight_mode,positive_weight=run.positive_weight);rows.append(pick)
    points=pd.DataFrame(rows);save_table(points,'stage4_grid_budget_points');chosen=[]
    for budget in config['selection_budgets']:
        r=choose_grid_config(points,budget)
        chosen.append(dict(selection_budget=budget,config_id=r.config_id,feature_set=r.feature_set,depth=int(r.depth),weight_mode=r.weight_mode,run_id=r.run_id,threshold=float(r.threshold),detected_events=int(r.detected_events),actual_burden=float(r.alarm_episodes_per_10000_raw)))
    frozen={'selected_by_budget':chosen,'selected_configs':list({r['config_id']:{k:r[k] for k in ['config_id','feature_set','depth','weight_mode']} for r in chosen}.values()),'rule':config['selection_rule'],'source_grid_sha256':hashlib.sha256((TABLES/'stage4_grid_budget_points.parquet').read_bytes()).hexdigest(),'seed42_only':True,'holdout_scored':False}
    path=OUT/'selection.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==frozen,'이미 동결한 설정과 다름'
    else:dump_json(path,frozen)
    print(json.dumps(frozen,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
