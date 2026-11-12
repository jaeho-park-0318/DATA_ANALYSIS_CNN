"""06~08 Notebook을 새로 작성하고 phm2018 커널에서 실제 실행한다."""
from pathlib import Path
import json
import argparse
import os
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
LOCAL=ROOT/'outputs/stage4'
for key,value in {'IPYTHONDIR':LOCAL/'ipython','JUPYTER_RUNTIME_DIR':LOCAL/'jupyter_runtime',
                  'MPLCONFIGDIR':LOCAL/'matplotlib','JUPYTER_PATH':LOCAL/'jupyter'}.items():os.environ[key]=str(value)
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
import nbformat as nbf
from nbclient import NotebookClient

SETUP='''from pathlib import Path
import sys
ROOT=Path.cwd() if (Path.cwd()/"src").is_dir() else Path.cwd().parent
sys.path.insert(0,str(ROOT))
assert Path(sys.prefix).name.lower()=="phm2018"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
from src.model_common import OUT,configs,read_json
from src.model_summary import collect,selected_per_family
from src import model_plots as plots
plots.setup()
objects,metrics=collect()
audit=read_json(OUT/"preflight.json")
pd.set_option("display.max_columns",40)
print("실행 환경:",sys.executable)
print("실제 학습 완료 모델을 읽음. 기존 점수와 임계값을 재사용하며 평가 세트로 재조정하지 않음.")
'''


def md(s):return nbf.v4.new_markdown_cell(s.strip())
def code(s):return nbf.v4.new_code_cell(s.strip())


def books():
    six=[md('''# 06. 학습 사전 검증과 Baseline

기존 3단계 코드·데이터·Notebook을 유지하고 새 모델을 `outputs/stage4/`에 저장했습니다. 600 원시시간 구간을 먼저 실행했으며 소규모 점검 후 전체 처리된 학습 세트로 확장했습니다. 아래 코드는 실제 학습 산출물과 예측 점수를 읽습니다. 원본 전체를 한 번에 읽지 않았고, 해당 타깃의 준비된 처리 데이터만 학습 행렬로 만들었습니다.'''),code(SETUP),
        md('''## 시간 해석과 데이터 대표성

[공식 설명](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/)의 초 예시와 [Huang 등(2018) 5.3절](https://papers.phmsociety.org/index.php/phmconf/article/download/590/phmc_18_590)의 약 4초 기록 설명을 실제 원본 차분·TTF 감소와 대조했습니다. 이 근거들은 초 해석을 지지합니다. 외부 시계 보정은 검증하지 않았으며 기존 raw 컬럼명과 데이터 정책 파일은 변경하지 않았습니다. **h600raw는 이 해석에서 약 10분**입니다.

전체 82,071,907개 시각에서 60 원시단위 bin의 첫 시각을 고르고, 과거 600 이력·관측 밀도·유한 특징을 요구한 뒤 12개 타깃 중 하나라도 알려진 행을 남겨 1,466,770행이 되었습니다. 이것은 무작위 대표 표본이 아닙니다. 미래 관측의 지속 여부와 TTF 결측으로 조건화되어 공백 근처 고장·후기 정상 상태가 과소대표됩니다. 클래스별 재표본화는 하지 않았지만 필터의 실제 보존율은 유형/상태별로 다릅니다.'''),
        code('''time_check=pd.DataFrame(audit["time_evidence"])
display(time_check)
assert (time_check.same_event_pairs==time_check.ttf_decrease_matches_dt).all()
old=read_json(ROOT/"reports/local/dataset_v1/dataset_summary.json")
full=pd.DataFrame(old["full_counts"]).query("fault_code=='any' and horizon_raw==600")
ready=pd.DataFrame(old["ready_counts"]).query("fault_code=='any' and horizon_raw==600")
retention=pd.DataFrame({"전체 알려진 관측":full[["positive","negative"]].sum(),"학습 후보 보존":ready[["positive","negative"]].sum()})
retention["보존율"]=retention["학습 후보 보존"]/retention["전체 알려진 관측"]
display(retention)
display(pd.DataFrame(audit["distributions"]).query("horizon==600"))
e=pd.DataFrame(audit["events"])
display(e[e.horizon==600].groupby(["scheme","split","fault_code"]).event_key.nunique().rename("평가 가능한 고유 유형 이벤트 수").to_frame())
display(Markdown("장비 평가의 누출 이벤트는 4건으로 매우 적습니다. 고정 분할을 섞거나 바꾸지 않았습니다. 다음 단계에서는 장비 그룹 교차검증으로 변동성을 확인할 수 있으나 이번 실험에는 적용하지 않았습니다."))'''),
        md('''## 입력 구성 및 전처리

A는 현재 센서 6개와 recipe/stage로 8개, B는 A에 과거 센서 통계를 추가한 80개, C는 공정 이력 3개를 더한 83개입니다. 모두 현재 또는 과거 기록으로 계산할 수 있습니다. 실제 현장 수집 지연은 미검증입니다. TTF·라벨·미래 기록·Tool/Lot·절대 time·사용량은 주 실험 입력에 없습니다. 유한 특징만 사용하므로 대치·스케일링을 하지 않았습니다. recipe/stage 코드는 숫자로 처리한 트리 기준이며 범주 처리의 최적성은 미검증입니다.'''),
        code('''display(pd.DataFrame([(name,len(cols)) for name,cols in configs().items()],columns=["입력 구성","특징 수"]))
for name,cols in configs().items():display(pd.DataFrame({name+" 입력":cols}))
print("소규모 점검:",read_json(OUT/"smoke.json"))
display(Markdown("소규모 점검은 학습 장비 01M01/01M02만 사용해 RF 8개 트리·XGBoost 10회가 실행되는지 확인했습니다. 평가 데이터는 사용하지 않았습니다. 이후 모든 준비된 학습 행으로 고정 설정의 전체 실험을 실행했습니다."))'''),
        md('''## Dummy·압력 임계값·Random Forest

Dummy는 학습 양성 비율을 출력하며 임계값은 0.5입니다. 압력 모델은 학습 AP로 방향을 정하고 학습 경험적 CDF로 점수를 변환한 뒤 검증 F1로 임계값을 골랐습니다. CDF 점수는 보정 확률이 아닙니다.

RF는 64개 트리, 깊이 10, 최소 리프 20, balanced_subsample을 사용했습니다. 내부 bootstrap은 학습 세트에만 적용되고 데이터셋의 시간/이벤트 분할은 바꾸지 않습니다. 검증/평가 행은 재표본화하지 않았습니다. RF 임계값도 검증 F1로 정했습니다. 설정은 평가 결과를 보기 전에 고정했고 평가 데이터로 수정하지 않았습니다.'''),
        code('''from src.train_baseline import train
for scheme in ("split_tool","split_time"):
    cached=train(scheme,600)
    table=metrics[(metrics.scheme==scheme)&(metrics.horizon==600)&(metrics.split=="evaluation")&(~metrics.identity_diagnostic)&(~metrics.name.str.startswith("XGB"))]
    display(table[["name","n","positive","precision","recall","f1","pr_auc_ap","roc_auc","tn","fp","fn","tp"]])
display(Markdown("정의 불가능한 지표는 NaN/None으로 표시합니다. 예를 들어 경보가 없는 Dummy의 Precision은 정의되지 않지만, 양성 정답이 있는 평가에서 F1=0은 실제 계산 결과입니다. Accuracy는 보조 지표로만 저장했습니다."))
fig,axes=plt.subplots(1,2,figsize=(14,5),layout="constrained")
for ax,scheme in zip(axes,("split_tool","split_time")):
    table=metrics[(metrics.scheme==scheme)&(metrics.horizon==600)&(metrics.split=="evaluation")&(~metrics.identity_diagnostic)&(~metrics.name.str.startswith("XGB"))].set_index("name")
    table[["f1","pr_auc_ap"]].rename(columns={"f1":"F1-score","pr_auc_ap":"PR-AUC(AP)"}).plot.bar(ax=ax)
    ax.set(title=f"{scheme} / h600raw / 정답 확인 평가",xlabel="Baseline 모델",ylabel="점수 (0~1)");ax.tick_params(axis="x",rotation=30);ax.legend();ax.grid(axis="y",alpha=.2)
plots.save(fig,"baseline_comparison")
display(Markdown("압력 순위의 AP와 검증 임계값의 F1은 다르게 움직일 수 있습니다. 특히 시간 평가에서 압력 AP가 높아도 고정 임계값의 경보는 없어 F1=0입니다. 평가 결과에 맞춰 임계값을 바꾸지 않았습니다."))''')]
    seven=[md('''# 07. XGBoost 학습 및 식별자 영향

통합 h600raw의 A/B/C를 장비·시간 분할에 독립적으로 학습했습니다. 기존 모델과 데이터를 변경하지 않았고, XGBoost는 설치된 phm2018 3.2.0에서 실제 실행했습니다. LSTM과 SHAP 상세 분석은 수행하지 않았습니다.'''),code(SETUP),
        md('''## 불균형·Early Stopping·임계값

scale_pos_weight는 학습 음성/양성 수로만 계산합니다. 고정 설정은 max_depth=4, learning_rate=0.05, 최대 500회, subsample/colsample=0.8, L1=0.1, L2=5입니다. 검증 aucpr의 30회 무개선으로 Early Stopping하고 검증 F1 최대 임계값을 사용합니다. 평가 세트는 조기 종료·가중치·임계값·특징 선택에 사용하지 않았습니다.

같은 검증 데이터를 조기 종료와 임계값·입력 구성 비교에 반복 사용했으므로 검증 과적합 가능성이 있습니다. 별도 내부 교차검증이나 확률 보정은 다음 단계의 과제입니다. 불균형 가중치를 준 출력은 보정된 실제 고장 확률이라고 간주하지 않습니다.'''),
        code('''from src.train_xgboost import train
for scheme in ("split_tool","split_time"):
    train(scheme,600)
    rows=[r for r in objects if r["scheme"]==scheme and r["horizon"]==600 and r["name"] in ("XGB_A","XGB_B","XGB_C")]
    display(pd.DataFrame([{ "모델":r["name"],"학습 가중치":r["details"]["parameters"]["scale_pos_weight"],"최적 반복":r["details"]["best_iteration"],"검증 임계값":r["threshold"],"검증 AP":r["metrics"]["validation"]["pr_auc_ap"],"평가 AP":r["metrics"]["evaluation"]["pr_auc_ap"],"평가 F1":r["metrics"]["evaluation"]["f1"]} for r in rows]))
fig,axes=plt.subplots(1,2,figsize=(14,5),layout="constrained")
for ax,scheme in zip(axes,("split_tool","split_time")):
    for config in ("A","B","C"):
        curve=read_json(OUT/"experiments"/f"{scheme}_h600"/("XGB_"+config)/"learning_curve.json")
        values=curve["validation_0"]["aucpr"]
        ax.plot(values,label=config)
    ax.set(title=f"{scheme} 검증 aucpr 학습 곡선",xlabel="부스팅 반복",ylabel="검증 aucpr (XGBoost 계산)");ax.legend(title="입력 구성");ax.grid(alpha=.2)
plots.save(fig,"xgboost_learning_curves")
display(Markdown("학습 곡선의 XGBoost aucpr와 최종 표의 sklearn Average Precision은 계산 정의가 달라 정확히 같은 숫자를 기대하지 않습니다. 보고서의 PR-AUC는 Average Precision(AP)로 통일했습니다."))'''),
        md('''## Tool·Lot·절대 시각·사용량의 별도 진단

주 실험 C에 Tool, time, Lot, 사용량(runnum과 사용량 카운터 3개), 전체를 각각 추가한 XGBoost 10개를 별도로 실행했습니다. 원본을 청크로 읽고 동일 시각의 현재 정보만 연결했습니다. 식별자 인코더는 학습에서만 적합하고 새 범주는 -1로 처리합니다. 수치 결측은 XGBoost 기본 처리입니다. 이 진단 모델은 주 실험의 최종 구성 선정에서 제외했습니다.

장비 홀드아웃의 Tool 범주는 평가에서 모두 새로운 값이므로 식별자 효과를 일반화 가능한 장비 정보로 해석할 수 없습니다. 단일 시드의 별도 적합이므로 효과 차이에 모델 변동성도 포함됩니다. 절대 시각·Lot·사용량의 개선은 시간 위치/생산 이력 기억일 수 있으며 고장 전조의 증거가 아닙니다.'''),
        code('''diagnostics=metrics[(metrics.horizon==600)&metrics.identity_diagnostic&(metrics.split=="evaluation")]
display(diagnostics[["scheme","name","precision","recall","f1","pr_auc_ap","roc_auc","event_recall"]])
display(pd.DataFrame([{ "분할":r["scheme"],"모델":r["name"],**r["details"].get("unknown_category_fraction",{})} for r in objects if r["details"].get("identity_diagnostic_only")]))
display(Markdown("랭킹 AP가 높아져도 검증 임계값이 평가 구간으로 전달되지 않아 F1이 낮을 수 있습니다. 식별자 진단 결과를 보고 평가 임계값을 다시 조정하지 않았습니다. 주 실험의 특징 목록에는 이 변수들이 들어가지 않습니다."))'''),
        md('''## 300·1800 확장 실험

600 본 실험이 실행·분할 검사를 통과한 뒤 같은 설정으로 확장했습니다. 예측 구간마다 학습 클래스 비율과 검증 임계값·Early Stopping을 따로 계산했으며, 평가 결과에 맞춰 설정을 조정하지 않았습니다.'''),
        code('''print("본 실험 통과 기록:",read_json(OUT/"primary_gate.json"))
print("실행 상태:",read_json(OUT/"run_status.json"))
display(metrics[(metrics.split=="evaluation")&(~metrics.identity_diagnostic)][["scheme","horizon","name","precision","recall","f1","pr_auc_ap","event_recall"]].sort_values(["horizon","scheme","name"]))''')]
    eight=[md('''# 08. 모델 성능·이벤트 경보·신뢰성 비교

모든 수치는 실제 저장된 예측 점수와 고정된 검증 임계값에서 계산했습니다. 모델 구성의 대표 비교는 **각 RF/XGBoost 계열 내 검증 AP 최대**를 사용합니다. 평가 점수가 가장 높은 구성을 사후 선정하지 않습니다. 별도 정답 미확인 구간 추론은 RF/XGBoost A/B/C 6개 중 검증 AP 최대 모델을 사용했습니다.'''),code(SETUP),
        md('''## 샘플 지표와 곡선

Precision, Recall, F1, PR-AUC(AP), ROC-AUC, 혼동행렬을 제공합니다. 주 곡선은 정답이 확인된 평가 행에 한정됩니다. 전체 현장의 무편향 성능이 아닙니다. 아래 그래프는 600 구간이며 확장 수치는 별도 표에 있습니다.'''),
        code('''for scheme in ("split_tool","split_time"):
    plots.comparison(scheme);plots.pr_curves(scheme);plots.confusion(scheme);plots.score_histograms(scheme)
selected=selected_per_family(objects)
display(pd.DataFrame([{ "분할":r["scheme"],"검증 선정 모델":r["name"],"Precision":r["metrics"]["evaluation"]["precision"],"Recall":r["metrics"]["evaluation"]["recall"],"F1":r["metrics"]["evaluation"]["f1"],"AP":r["metrics"]["evaluation"]["pr_auc_ap"],"ROC-AUC":r["metrics"]["evaluation"]["roc_auc"]} for r in selected]))'''),
        md('''## 이벤트·경보·노출시간 정의

이벤트 키는 Tool·고장 시각·유형입니다. 평가 가능한 이벤트는 자기 next_event 키와 일치하는 양성 anchor가 있는 경우입니다. 경보는 같은 Tool·관측 세그먼트·Stage=0/>0 상태에서 연속 양성인 점들을 병합합니다. 음성, 72 초과 간격, 상태 변경에서 끊고 H마다 재발행합니다.

한 경보는 H 이내의 가장 이른 미래 이벤트 시각에만 대응합니다. 동시각의 여러 유형은 함께 기록할 수 있으며, 한 이벤트의 여러 경보는 탐지 한 번으로 셉니다. Lead Time은 매칭된 실제 경보 시각에서 이벤트까지의 차이이며 탐지된 이벤트에만 정의됩니다.

오경보는 관련 이벤트에 매칭되지 않고 에피소드 전체가 확인된 음성일 때만 셉니다. NA를 포함하면 미확인 경보로 남깁니다. 노출시간은 같은 세그먼트의 인접 확인 음성 anchor 사이에서 간격 72 이하인 실제 시간만 합칩니다. 가동 시간당 비율은 Stage>0인 두 끝점으로 제한한 노출을 분모로 합니다. 초 해석에 따라 3600 원시단위를 시간으로 환산합니다. 전체 운전 노출이 아니므로 현장 전체 FAR로 일반화하지 않습니다.'''),
        code('''display(pd.DataFrame([{ "분할":r["scheme"],"모델":r["name"],"평가 가능 이벤트":r["metrics"]["evaluation"]["evaluable_events"],"탐지 이벤트":r["metrics"]["evaluation"]["detected_events"],"Event Recall":r["metrics"]["evaluation"]["event_recall"],"가동시간당 오경보":r["metrics"]["evaluation"]["false_alarms_per_active_hour_seconds_interpretation"],"Lead Time 중앙값 raw":r["metrics"]["evaluation"]["lead_time_median_raw"]} for r in selected]))
for r in selected:
    display(Markdown(f"**{r['scheme']} / {r['name']} 유형별 이벤트**"))
    display(pd.DataFrame(r["metrics"]["evaluation"]["per_type"]).T)
display(Markdown("장비 평가의 누출 4건 같은 작은 분모에서 수치가 크게 변할 수 있습니다. 양성 이벤트가 없으면 Recall·Lead Time을 임의 생성하지 않습니다. 한두 이벤트의 성패로 안정적 성능을 주장하지 않습니다."))'''),
        md('''## 정답 미확인 구간을 포함한 전향적 추론

학습 후보 선정의 미래 정답 조건이 현장 성능을 가리지 않도록, 평가 분할의 모든 feature_ready 행에도 고정 모델을 적용했습니다. 이때 NA는 예측은 하되 정상으로 바꾸지 않습니다. 확인 가능한 이벤트 Recall과 전체 기록 대비 도달률을 구분하고, 미확인 경보 수를 함께 제공합니다.'''),
        code('''prospective=[read_json(OUT/"prospective"/s/"result.json") for s in ("split_tool","split_time")]
display(pd.DataFrame([{ "분할":r["scheme"],"RF/XGB 검증 선정":r["selected_name"],**{k:v for k,v in r["metrics"].items() if not isinstance(v,(dict,list))}} for r in prospective]))
display(Markdown("정답 미확인 구간에서 나온 경보는 확인된 오경보율의 분자에 넣을 수 없습니다. 다수의 미확인 경보가 존재하므로 현재 낮아 보이는 조건부 FAR만으로 현장 도입을 판단할 수 없습니다. 전체 이력 대비 도달률에는 관측/이력 부족도 반영됩니다."))
for scheme in ("split_tool","split_time"):plots.case_plots(scheme)'''),
        md('''## 특징 중요도와 과적합

시간 순서로 선정한 실제 첫 사례에서 장비 평가 RF A의 점수는 낮은 수준으로 일정하고 임계값을 넘지 못했습니다. 시간 평가 XGBoost C의 점수는 앞쪽에서 일시적으로 높아졌지만 임계값보다 낮았으며, 마지막 600 구간에서도 경보가 발생하지 않았습니다. 두 사례에서 일관된 고장 직전 상승 패턴은 확인되지 않았습니다. 탐지 성공 사례를 골라 보여주지 않았으며, 이 미탐과 전체 이벤트 Recall을 함께 해석해야 합니다.

RF의 불순도 감소와 XGBoost gain은 다른 척도입니다. 공정 범주 수, 상관 특징 및 학습 분포에 영향을 받으므로 인과적 고장 원인으로 해석하지 않습니다. Tool/Lot/time/사용량은 별도 진단에만 있습니다. SHAP 상세 분석은 하지 않았습니다.'''),
        code('''for scheme in ("split_tool","split_time"):plots.importance(scheme)
comparison=metrics[(metrics.horizon==600)&(~metrics.identity_diagnostic)&metrics.name.str.startswith(("RF_","XGB_"))]
display(comparison.pivot_table(index=["scheme","name"],columns="split",values=["f1","pr_auc_ap"]))
display(Markdown("학습 AP와 검증/평가 AP의 큰 차이, 장비 분할과 시간 분할의 차이는 과적합과 분포 이동 가능성을 보여줍니다. 이는 누수가 없다는 검사 결과와 양립합니다. 83개 특징이 현재값 중심 구성보다 항상 좋은 것은 아닙니다. 다음 단계는 관측/라벨 검열 정책의 전향 검증, 장비 그룹 교차검증, 검증 분리와 확률 보정입니다."))'''),
        code('''import unittest
suite=unittest.TestLoader().discover(str(ROOT/"tests"),pattern="test_models_phase4.py")
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful()
print("최종 검사:",read_json(OUT/"final_checks.json"))
display(Markdown("총 48개 주·확장 실험과 10개 식별자 진단 모델을 실제 실행했습니다. 모델·그래프·예측·설정은 outputs/stage4에 저장했습니다. 기존 원본·코드·Notebook·보고서는 보존했고 외부 업로드는 하지 않았습니다. LSTM·SHAP·현장 확률 보정·장비 교차검증은 미실행입니다."))''')]
    return [('06_baseline_models.ipynb',six),('07_xgboost_training.ipynb',seven),('08_model_comparison.ipynb',eight)]


def main():
    if Path(sys.prefix).name.lower()!='phm2018':raise RuntimeError('phm2018 환경에서 실행하세요.')
    kernel=LOCAL/'jupyter/kernels/phm2018';kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Python (phm2018)','language':'python'}),encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--only');args=parser.parse_args()
    status_path=LOCAL/'notebook_execution.json'
    statuses=json.loads(status_path.read_text(encoding='utf-8')) if args.only and status_path.exists() else []
    for name,cells in books():
        if args.only and name!=args.only:continue
        path=ROOT/'notebooks'/name;book=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'phm2018','display_name':'Python (phm2018)','language':'python'}})
        nbf.write(book,path);start=time.monotonic();print('Notebook 실행:',name,flush=True)
        try:NotebookClient(book,kernel_name='phm2018',timeout=1200,resources={'metadata':{'path':str(ROOT)}},allow_errors=False).execute()
        finally:nbf.write(book,path)
        codes=[c for c in book.cells if c.cell_type=='code']
        row={'notebook':name,'code_cells':len(codes),'executed':sum(c.execution_count is not None for c in codes),'errors':sum(o.output_type=='error' for c in codes for o in c.outputs),'elapsed_seconds':time.monotonic()-start,'python':sys.executable}
        statuses=[r for r in statuses if r['notebook']!=name]+[row];statuses.sort(key=lambda r:r['notebook']);print(row,flush=True)
        (LOCAL/'notebook_execution.json').write_text(json.dumps(statuses,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
