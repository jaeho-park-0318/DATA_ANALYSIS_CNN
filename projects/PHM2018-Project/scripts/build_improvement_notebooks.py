"""09~11 Notebook을 새로 생성하고 같은 로컬 phm2018 커널로 실행한다."""
from pathlib import Path
import os,sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.improvement_common import OUT,save_json
for key,folder in {'IPYTHONDIR':'ipython','JUPYTER_RUNTIME_DIR':'jupyter_runtime','JUPYTER_PATH':'jupyter','MPLCONFIGDIR':'matplotlib'}.items():os.environ[key]=str(OUT/folder)
import nbformat as nbf
from nbclient import NotebookClient

SETUP='''from pathlib import Path
import sys
ROOT=Path.cwd() if (Path.cwd()/"src").is_dir() else Path.cwd().parent
sys.path.insert(0,str(ROOT))
assert Path(sys.prefix).name.lower()=="phm2018"
import pandas as pd
from IPython.display import display,Markdown,Image
from src.improvement_common import OUT,read_json,configs
pd.set_option("display.max_columns",30)
print("실행 Python:",sys.executable)
print("저장된 실제 실험 결과를 읽습니다. 이 Notebook은 모델을 재학습하지 않습니다.")
def show(name):display(Image(filename=str(OUT/"figures"/(name+".png"))))
'''
def md(s):return nbf.v4.new_markdown_cell(s)
def code(s):return nbf.v4.new_code_cell(s)

def books():
    audit=[md('''# 09. 데이터셋 신뢰성과 NA 감사

전체 82,071,907개 정규화 시각을 DuckDB로 컬럼 선택 집계했습니다. 원본은 보존했으며 NA를 0으로 치환하지 않았습니다. 실제 실행 코드는 `src/dataset_audit45.py`, TTF 재검증은 `src/ttf_audit45.py`입니다. 기존 라벨을 완화할 이력 완전성 근거가 없어 ml_v2는 생성하지 않았습니다.'''),code(SETUP),
        md('## 장비·Stage·유형별 미확인 라벨'),code('''audit=pd.read_parquet(OUT/"na_audit.parquet")
for keys in [["fault_code"],["Tool","fault_code"],["state","fault_code"]]:
    stats=audit.groupby(keys)[["rows","positive","negative","unknown"]].sum()
    stats["NA 비율"]=stats.unknown/stats.rows
    display(stats)
show("na_reasons");show("na_tools")'''),
        md('''후속 같은 유형 이벤트가 없는 구간은 고장 이력이 끝났다는 의미입니다. 고장이 없었다는 충분한 증거는 아니므로 정상 라벨로 확대하지 않았습니다. 통합 NA 원인은 배타적 우선순위 분류이며 공백과 이력 종료가 동시에 있을 수 있습니다.'''),
        code('''display(pd.DataFrame(read_json(OUT/"ttf_recheck.json")["rows"]))
display(read_json(OUT/"label_policy.json"))
counts=pd.read_parquet(OUT/"cache_counts.parquet")
display(counts)
display(Markdown(f"확인 음성이 없는 장비는 **{int((counts.negative==0).sum())}대**입니다. 원래 training 포함 조건을 추론에서는 해제하여 {int(counts.rows.sum()):,}개의 준비된 anchor를 보존했습니다. NA는 학습 제외, 경보량에는 포함합니다."))'''),
        md('## 실제 로컬 하드웨어'),code('''hardware=read_json(OUT/"hardware.json")
display({k:hardware[k] for k in ["python","prefix","physical_cores","logical_cores","build","median_seconds","selected_device","cuda_warnings"]})
display(read_json(OUT/"hardware_supplement.json"));show("hardware")'''),
        md('''실제 Booster의 cuda:0과 CPU/GPU 시간 차이를 확인했습니다. GPU 6GB 한계를 고려해 학습을 순차 실행하고 4 CPU 스레드를 사용합니다. 같은 설정의 속도 비교이며 CPU/GPU 모델 성능이 같다는 검증은 아닙니다. 기본 python3 커널의 이름만 믿지 않고 아래 실행 사양에 phm2018 절대 경로를 지정했습니다.''')]
    cv=[md('''# 10. 장비 그룹 교차검증

기존 개발 장비 01~06 계열 12대를 4개 외부 Fold로 분리했습니다. 09/10 기존 평가 장비를 튜닝에 재사용하지 않았습니다. 각 Fold에서 5대 학습, 2대 Early Stopping, 2대 임계값 보정, 3대 외부 평가입니다. 과거 시간 분할 실험과 자료가 겹치는 후향적 분석이므로 새로운 독립 평가라고 부르지 않습니다.

재실행: `python -m src.group_validation45 --pilot` 후 `python scripts/run_improvement45.py`. 완료 실험은 재사용하고 새 설정은 새 경로에 저장해야 합니다.'''),code(SETUP),
        code('''plans=read_json(OUT/"fold_plan.json");display(pd.DataFrame(plans))
counts=pd.concat([pd.DataFrame(read_json(OUT/f"fold_{i}_counts.json")).assign(fold=i) for i in range(4)])
display(counts.groupby(["fold","role"])[["rows","positive","negative","unknown","f1","f2","f3"]].sum())
for p in plans:
    sets=[set(p[k]) for k in ["train","early_stop","calibration","evaluation"]]
    assert sum(map(len,sets))==len(set.union(*sets))==12
display(pd.DataFrame({"구성":configs().keys(),"특징 수":[len(x) for x in configs().values()]}))
display(configs())'''),
        md('''입력은 현재/과거 센서·공정 상태만 허용했습니다. TTF, 정답, next_event, Tool, time, Lot, 사용량은 입력에서 제외됩니다. 장비가 분리되어 동일 이벤트나 중첩 과거 윈도가 역할을 넘지 않습니다. 대치·스케일링은 수행하지 않았습니다.'''),
        code('''metrics=pd.read_parquet(OUT/"cv_metrics.parquet")
display(pd.read_parquet(OUT/"cv_summary.parquet"))
display(metrics[["fold","model","policy","precision","recall","f1","pr_auc_ap","event_recall","false_alarms_per_active_hour"]])
show("cv_comparison")
display(metrics[metrics.policy=="f1"][["fold","model","train_ap","pr_auc_ap","fit_seconds","peak_rss_gib","best_iteration"]])'''),
        md('''특징을 늘려도 성능은 일관되게 좋아지지 않습니다. A의 평균 AP와 B/C의 F1·이벤트 탐지를 분리해서 해석해야 합니다. 정규화 강화 C도 외부 장비 성능을 보장하지 못했습니다. 학습 AP와 평가 AP 차이, 클래스 분포와 장비 분포의 결합은 과적합·분포 이동 가능성을 보여주지만 하나의 원인으로 확정할 수 없습니다.'''),
        code('''show("tool_shift")
display(pd.read_parquet(OUT/"tool_state_distribution.parquet"))
contrasts=pd.read_parquet(OUT/"fault_sensor_contrasts.parquet")
display(contrasts.groupby(["fault_code","sensor"])[["positive_median","negative_median","median_difference_over_negative_iqr"]].median())
display(pd.read_parquet(OUT/"cv_fault_metrics.parquet").groupby(["model","policy","fault_code"])[["eligible","detected"]].sum())'''),
        md('''센서 비교는 유형별 기술 통계이며 장비·Recipe·Stage 교란을 제거한 고장 원인 분석은 아닙니다. Stage=0에서도 고장이 있어 제외하지 않았습니다. 기존 시간 분할 A/C의 순위 역전은 검증과 평가 분포가 다른 상황에서 관찰되었으며, 평가 결과를 보고 C 대신 A를 선택하면 선택 편향이 생깁니다. 이번에는 모든 사전 지정 구성을 보고합니다.''')]
    threshold=[md('''# 11. 고장 이벤트 경보와 임계값 분석

F1 임계값과 Event Recall/FAR 제약 임계값은 별도 내부 보정 장비에서만 정했습니다. FAR≤0.1/h는 실험용 사전 제약이며 전체 운영 오경보율을 의미하지 않습니다. 이미 확인한 4단계 평가 결과는 원인 진단에만 사용했고 새 임계값을 적용해 개선 성능으로 보고하지 않았습니다.'''),code(SETUP),
        code('''display(pd.DataFrame(read_json(OUT/"diagnostics.json")["old_tool_xgb_fault_scores"]))
show("old_fault_scores")
old=pd.DataFrame(read_json(OUT/"old_validation_threshold_grid.json"))
display(old[["threshold","precision","recall","f1","pr_auc_ap","event_recall","false_alarms_per_active_hour","lead_time_median_raw","unknown_episodes"]])'''),
        md('''압력 저하 f1의 평가 양성 점수 최대 0.62040이 검증 임계값 0.65273보다 낮아 30건을 탐지하지 못했습니다. 이것은 임계값 미통과의 직접 증거이며 센서 전조 부재의 증거는 아닙니다. 임계값을 낮출 때 알려진 음성 경보와 미확인 구간 경보가 함께 늘 수 있습니다.'''),
        code('''grid=pd.read_parquet(OUT/"threshold_metrics.parquet")
display(grid[["fold","model","threshold","precision","recall","f1","pr_auc_ap","event_recall","false_alarms_per_active_hour","lead_time_median_raw","unknown_episodes"]])
show("thresholds")
metrics=pd.read_parquet(OUT/"cv_metrics.parquet")
display(metrics[["fold","model","policy","threshold","eligible_events","detected_events","active_negative_hours","active_false_episodes","false_alarms_per_active_hour","unknown_episodes","unknown_rows","unknown_alarm_rows","lead_time_median_raw"]])'''),
        md('''연속 경보는 같은 장비·세그먼트·가동 상태에서 묶고, 72초 초과 공백 또는 음성 예측에서 끊으며 600초마다 재발행합니다. 한 경보는 가장 이른 이벤트 시각에만 매칭합니다. NA가 포함된 미매칭 에피소드는 false 대신 unknown입니다. FAR 분모는 인접 확인 음성 anchor 사이의 Stage>0 시간이며 미확인 시간은 분모에 넣지 않습니다. Lead Time은 탐지된 이벤트에만 존재합니다.'''),
        code('''summary=metrics.groupby(["model","policy"])[["pr_auc_ap","f1","event_recall","false_alarms_per_active_hour"]].agg(["mean","std","count"])
display(summary)
display(Markdown("보정 장비의 FAR 제약이 외부 장비에 그대로 유지되지 않습니다. 기존 4단계와 분할이 달라 직접 개선률을 계산할 수 없습니다. 먼저 고장 이력 완전성과 장비별 확인 음성 부족을 해결하고 새 장비·기간에서 고정 정책을 검증해야 합니다. **LSTM은 아직 진행하지 않는 것을 권고합니다.**"))''')]
    return [('09_dataset_audit.ipynb',audit),('10_group_cross_validation.ipynb',cv),('11_threshold_analysis.ipynb',threshold)]

def run():
    kernel=OUT/'jupyter/kernels/phm2018_stage45';kernel.mkdir(parents=True,exist_ok=True)
    save_json(kernel/'kernel.json',{'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Python (phm2018 stage4.5)','language':'python'})
    results=[]
    for name,cells in books():
        nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'phm2018_stage45','display_name':'Python (phm2018 stage4.5)','language':'python'}})
        path=ROOT/'notebooks'/name;start=time.monotonic()
        NotebookClient(nb,timeout=180,kernel_name='phm2018_stage45',resources={'metadata':{'path':str(ROOT)}}).execute()
        nbf.write(nb,path)
        results.append({'notebook':name,'python':sys.executable,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'errors':sum(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs),'seconds':time.monotonic()-start})
        save_json(OUT/'notebook_execution.json',results);print('실행 완료:',name,flush=True)

if __name__=='__main__':run()
