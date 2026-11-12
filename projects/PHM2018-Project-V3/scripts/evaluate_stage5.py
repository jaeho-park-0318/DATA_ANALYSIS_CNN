"""최종 시간 Holdout을 배제한 장비 검증의 경보 곡선."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concurrent.futures import ProcessPoolExecutor,as_completed
from src.common import *
from src.protected_group import PROTOCOL,validation_events
from src.final_evaluation import exposure_by_tool,at_threshold
from scripts.evaluate_stage2 import pareto_mask
from scripts.diagnose_stage3 import choose_budget

OUT=ROOT/'outputs/stage5'

def run(spec):
    rid=spec['run_id'];p=pd.read_parquet(OUT/(rid+'_validation.parquet'));events=validation_events();supports=exposure_by_tool(p);grid=np.unique(np.r_[0,.001,.003,.01,.03,.1,.3,.5,.7,.9,1,np.nextafter(1.,2.),np.quantile(p.score,[.001,.005,.01,.025,.05,.1,.25,.5,.75,.9,.95,.99,.995,.999,.9995,.9999])]);rows=[];evs=[]
    for threshold in grid:
        row,e,a,l,s=at_threshold(p,events,threshold,supports);row.update(run_id=rid,family=spec['family'],cohort=spec['cohort']);rows.append(row);e['run_id']=rid;e['threshold']=threshold;evs.append(e)
    d=pd.DataFrame(rows);d['pareto']=pareto_mask(d);write_parquet(d,OUT/(rid+'_curve.parquet'));write_parquet(pd.concat(evs,ignore_index=True),OUT/(rid+'_events.parquet'));dump_json(OUT/(rid+'_exposure.json'),supports);return rid

def main():
    assert not (ROOT/'outputs/stage6/holdout_access.json').exists()
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(run,s) for s in PROTOCOL['group_runs']]):print(future.result()+' 장비 검증 경보 완료',flush=True)
    curves=pd.concat([pd.read_parquet(OUT/(s['run_id']+'_curve.parquet')) for s in PROTOCOL['group_runs']],ignore_index=True);events=pd.concat([pd.read_parquet(OUT/(s['run_id']+'_events.parquet')) for s in PROTOCOL['group_runs']],ignore_index=True);points=[]
    for rid,g in curves.groupby('run_id'):
        for budget in PROTOCOL['diagnostic_budgets']:
            r=choose_budget(g,budget).to_dict();r['budget_per_10000_raw']=budget;points.append(r)
    save_table(curves,'stage5_alarm_curve');save_table(events,'stage5_events');save_table(pd.DataFrame(points),'stage5_budget_points')

if __name__=='__main__':main()
