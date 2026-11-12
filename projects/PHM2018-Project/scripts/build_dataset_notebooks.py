"""기존 Notebook을 변경하지 않고 04/05 Notebook을 생성·실행한다."""
from pathlib import Path
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
LOCAL=ROOT/"reports/local/dataset_v1"
for key,value in {"IPYTHONDIR":LOCAL/"ipython","JUPYTER_RUNTIME_DIR":LOCAL/"jupyter_runtime",
                  "MPLCONFIGDIR":LOCAL/"matplotlib","JUPYTER_PATH":LOCAL/"jupyter"}.items():os.environ[key]=str(value)
if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
import nbformat as nbf
from nbclient import NotebookClient


SETUP='''from pathlib import Path
import sys,json
ROOT=Path.cwd() if (Path.cwd()/"src").is_dir() else Path.cwd().parent
sys.path.insert(0,str(ROOT))
assert Path(sys.prefix).name.lower()=="phm2018"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
from src.dataset_common import DEST,REPORT,POLICY,FAULTS,SENSORS,HORIZONS,feature_columns,label_columns
summary=json.loads((REPORT/"dataset_summary.json").read_text(encoding="utf-8"))
assert summary["all_checks_passed"]
assert len(summary["tools"])==20,"전체 장비 구축·검증을 먼저 완료하세요."
full=pd.DataFrame(summary["full_counts"])
ready=pd.DataFrame(summary["ready_counts"])
events=pd.DataFrame(summary["fault_events"])
plt.rcParams.update({"font.family":"Malgun Gothic","axes.unicode_minus":False,"figure.dpi":110})
pd.set_option("display.max_columns",30)
def save_figure(fig,name):
    folder=REPORT/"figures";folder.mkdir(exist_ok=True)
    fig.savefig(folder/(name+".png"),dpi=130,bbox_inches="tight")
    plt.show()
print("실행 Python:",sys.executable)
print("데이터 경로:",DEST)
'''


def md(text):return nbf.v4.new_markdown_cell(text.strip())
def code(text):return nbf.v4.new_code_cell(text.strip())


def build():
    four=[
        md('''# 04. 고유 이벤트 정책과 고장 예측 라벨

기존 03_fault_analysis.ipynb와 `eda_common.py`, `eda_detail.py`의 검증 결과를 활용하여 **새 경로 `data/processed/ml_v1/`**에 구축했습니다. 원본과 기존 EDA 파일·결과는 변경하지 않았습니다. 전체 센서/TTF는 200,000행 청크로 함께 읽고 시각열과 다음 고장 차이를 다시 대조했습니다. 아래 표는 실제 Parquet의 집계이며 모델 학습 결과가 아닙니다.'''),
        code(SETUP),
        md('''## 시간 단위 및 정합성

[PHM Society 공식 설명](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/)은 익명화 단위 미공개와 시간의 초 예시를 제시하지만 물리 단위를 확정하지 않습니다. 실제 기록의 최빈 간격은 4이고 TTF는 fault와 같은 좌표 차이입니다. 이것만으로 물리 초를 증명할 수 없습니다.

따라서 **라벨 이름은 h300raw/h600raw/h1800raw**입니다. 각각 5/10/30분으로 부를 수 있는 것은 `time=초`가 확인될 때뿐입니다. 현재 결과를 검증된 물리 분 예측 데이터라고 해석하지 않습니다. 공식 test의 정답은 로컬 파일에 없으며 내부 평가에 넣지 않았습니다.'''),
        code('''display(pd.Series(POLICY,name="정책").to_frame())
checks=pd.DataFrame(summary["checks"])
display(checks)
print("전체 센서/TTF 재대조 불일치:",sum(json.loads((REPORT/f"labels_{t}.json").read_text(encoding="utf-8"))["ttf_mismatches"] for t in summary["tools"]))
display(Markdown("원본 fault 1,236행에서 동일 Tool·time·원문 유형의 중복 25행을 제거한 고유 이벤트는 1,211건입니다. 기본 타깃에서는 `[NoWaferID]` 1건을 추가 제외하여 1,210건을 사용합니다. 원본 이력은 그대로 보존했습니다. 센서 동시각 행은 같은 시각 배치의 센서 평균으로 정리했고 Recipe/Stage가 모순된 시각은 격리했습니다."))'''),
        md('''## 라벨 정의와 음성의 근거

각 유형 f1=저압, f2=고압, f3=누출에 대해 다음을 적용합니다.

- 양성 1: 검증된 유한 TTF와 다음 동일 유형 이벤트가 일치하고 `0<TTF<=H`이며, 이벤트 직전까지 연속 관측이 유지된 경우입니다. 이벤트가 마지막 센서 관측보다 최대 12 원시단위 뒤에 있는 경우까지 허용합니다.
- 음성 0: 검증된 유한 `TTF>H`이고 현재부터 H까지 같은 연속 관측 세그먼트인 경우만 부여합니다.
- NA: TTF 결측·모순, 미래 관측 공백·종료, 공정 상태 모순, 모든 고장 시각 자체 등입니다. 누락된 고장 이력의 완전성을 보증하는 정책은 아니므로 제공 이력에 대한 음성임을 명시합니다.
- 연속 세그먼트는 인접 관측 간격 12 원시단위 초과와 모순 시각에서 끊습니다. 음성은 구간 전체, 양성은 확인된 이벤트까지의 연속성을 요구합니다. 고장 시각에는 예측하지 않습니다.
- 통합 any는 3종 중 하나라도 1이면 1, 세 유형 모두 0일 때만 0, 나머지는 NA입니다.

라벨·TTF·이벤트 키·관측 세그먼트 종료 시각은 타깃/품질/분할 메타데이터이며 입력 특징에 포함하지 않습니다.'''),
        code('''global_counts=full.groupby(["fault_code","horizon_raw"])[["rows","positive","negative","unknown"]].sum().reset_index()
display(global_counts)
assert (global_counts.positive+global_counts.negative+global_counts.unknown==global_counts.rows).all()
fig,axes=plt.subplots(1,2,figsize=(15,6),layout="constrained")
for fault,g in global_counts.groupby("fault_code"):
    axes[0].plot(g.horizon_raw,g.positive,marker="o",label=fault)
    axes[1].plot(g.horizon_raw,g.unknown/g.rows,marker="o",label=fault)
axes[0].set(title="유형·예측 구간별 양성 관측 수",xlabel="예측 구간 (원시시간 단위)",ylabel="양성 수 (관측 시각)",yscale="log")
axes[1].set(title="유형·예측 구간별 미확인 비율",xlabel="예측 구간 (원시시간 단위)",ylabel="NA 비율 (0~1)")
for ax in axes:ax.legend(title="f1 저압 / f2 고압 / f3 누출 / any 통합");ax.grid(alpha=.2)
save_figure(fig,"label_distribution")
display(Markdown(f"동시각 집계 후 **{summary['label_rows']:,}개 관측 시각**에 12개 기본 타깃을 생성했습니다. 표의 수치는 원본 행 수나 고장 이벤트 수가 아닌 정규화 관측 시각 수입니다. 예측 구간이 길수록 연속 관측 요구가 엄격해져 음성 후보가 NA로 바뀔 수 있습니다."))'''),
        md('''## `[NoWaferID]` 포함/제외 영향

기본 정책은 태그 제외이며 제공 TTF와 일치합니다. 대안은 태그를 포함한 fault 이력을 우선하여 후보 라벨을 다시 계산합니다. 이 대안은 03M02의 제공 TTF와 충돌하므로 **훈련 타깃에 사용하지 않습니다**. 충돌 구간을 격리하는 경우 추가 NA 수를 함께 제시합니다. 원문 3,489행의 불일치와 동시각 집계 후 충돌 관측 수는 분모가 다를 수 있습니다.'''),
        code('''impact=pd.DataFrame(summary["policy_impact"])
display(impact)
event_policy=events.groupby(["fault_code","tagged"]).size().rename("고유 이벤트 수").reset_index()
display(event_policy)
display(Markdown("고유 이벤트 수는 태그 포함 1,211건, 제외 1,210건입니다. 고압은 333건에서 332건으로 바뀝니다. 제공 TTF와의 일치가 확보되는 태그 제외 정책을 기본으로 채택했으며, 태그 포함 대안의 라벨은 별도 `alt_` 컬럼에만 보존합니다."))'''),
        md('''## 공정 상태 분포와 고장 시점의 상태

관측 시각의 Stage=0, Stage>0, 모순 상태를 구분합니다. 고장 상태는 동일 Tool의 고장 시각 이하 마지막 센서 기록을 사용하되 간격이 12 원시단위보다 크면 미확인으로 둡니다. 고장 시각 이후 기록으로 Stage를 채우지 않습니다. Recipe/Stage 범주 크기는 연속 강도가 아닙니다.'''),
        code('''process=pd.DataFrame(summary["process_distribution"])
display(full.query("fault_code=='any' and horizon_raw==300").groupby("state_group")["rows"].sum().rename("관측 시각 수").to_frame())
display(process.groupby("recipe")["rows"].sum().sort_values(ascending=False).rename("관측 시각 수").to_frame())
event_stage=pd.crosstab(events.loc[~events.tagged,"fault_code"],events.loc[~events.tagged,"state_group"])
display(event_stage)
fig,axes=plt.subplots(1,2,figsize=(15,6),layout="constrained")
event_stage.plot.bar(stacked=True,ax=axes[0]);axes[0].set(title="고장 유형별 관측된 공정 상태",xlabel="고장 유형 코드",ylabel="고유 이벤트 수 (건)");axes[0].legend(title="공정 상태")
top=process.groupby("recipe").rows.sum().nlargest(15)
axes[1].bar(top.index.astype(str),top.values,label="관측 수 상위 15 Recipe")
axes[1].set(title="Recipe별 관측 분포",xlabel="Recipe 범주 ID (단위 없음)",ylabel="관측 시각 수");axes[1].tick_params(axis="x",rotation=60);axes[1].legend()
save_figure(fig,"stage_recipe_distribution")
display(Markdown("Stage=0을 자동 정상이나 자동 이상으로 분류하지 않았습니다. 전체 상태 데이터와 Stage>0 필터를 적용한 가동 상태 데이터를 같은 원본 분할에서 비교할 수 있습니다. 고장 시각과 센서 기록 사이의 긴 공백 때문에 많은 이벤트의 Stage는 미확인일 수 있습니다."))''')
    ]
    five=[
        md('''# 05. 과거 시계열 특징과 누수 없는 학습 데이터

원시 관측을 한 번에 적재하지 않고 장비별 Parquet 배치를 읽었습니다. 전체 라벨은 모든 정규화 관측 시각에 보존하고, 특징은 **60 원시시간 단위 bin에서 첫 실제 관측 시각**에 생성합니다. 마지막 행이나 bin 평균을 미래 시점에서 되돌려 붙이지 않습니다. 선택 주기는 라벨과 무관하고 학습·검증·평가 모두 동일합니다. 원래 모든 시점의 라벨은 별도 Parquet에 유지됩니다.'''),
        code(SETUP),
        md('''## 특징 정의 및 입력 허용 목록

6개 센서의 현재값과 각 w=60/300 원시시간 구간의 평균·표준편차(ddof=0)·최솟값·최댓값을 사용합니다. 현재 구간은 `(t-w,t]`, 이전 구간은 `(t-2w,t-w]`입니다. 평균 변화량은 두 구간 평균의 차이, 변화율은 그 차이를 w로 나눈 값입니다. 물리 분 또는 백분율 변화로 해석하지 않습니다.

현재 Recipe/Stage와 과거 300 원시단위의 Stage=0 비율, Recipe/Stage 전환 횟수를 추가합니다. 입력은 센서·현재/과거 공정 정보의 명시적 목록만 허용합니다. TTF·fault·라벨·다음 이벤트·분할·미래 세그먼트 종료 시각은 입력에서 제외합니다.'''),
        code('''features=feature_columns()
display(pd.DataFrame({"입력 특징":features}))
print("특징 수:",len(features))
from src.dataset_loader import iter_model_batches
X_example,y_example=next(iter_model_batches("label_any_h300raw","train",batch_size=256))
print("안전한 로더 입력/타깃 크기:",X_example.shape,y_example.shape)
assert X_example.columns.tolist()==features and not y_example.isna().any()
display(pd.DataFrame(summary["checks"]))
build_rows=[json.loads((REPORT/f"features_{t}.json").read_text(encoding="utf-8")) for t in summary["tools"]]
display(pd.DataFrame(build_rows))
display(Markdown(f"총 anchor **{summary['anchor_rows']:,}개** 중 특징 이력이 충분한 행은 **{summary['feature_ready_rows']:,}개**입니다. 앞선 600 원시단위의 연속 이력과 각 현재·이전 윈도의 기대 관측 수 90% 이상을 요구했습니다. 이력 부족을 보간하거나 0으로 채우지 않았습니다."))'''),
        md('''## 실제 특징 값과 과거 정보 범위

아래 예시는 대표 장비의 이력이 충분한 첫 연속 구간입니다. 원시 센서와 과거 평균을 비교하며, 비어 있는 시간은 선으로 이어붙이지 않습니다.'''),
        code('''import duckdb
conn=duckdb.connect();conn.execute("SET memory_limit='1GB'")
path=str(DEST/"training/01M01/dataset.parquet")
sample=conn.execute("SELECT time,source_max_time,history_min_time,FLOWCOOLPRESSURE__current,FLOWCOOLPRESSURE__mean_60raw,FLOWCOOLPRESSURE__mean_300raw FROM read_parquet(?) ORDER BY time LIMIT 120",[path]).df()
display(sample.head(8))
fig,ax=plt.subplots(figsize=(13,5),layout="constrained")
for col in ("FLOWCOOLPRESSURE__current","FLOWCOOLPRESSURE__mean_60raw","FLOWCOOLPRESSURE__mean_300raw"):
    y=sample[col].copy();y.loc[sample.time.diff()>72]=np.nan
    ax.plot(sample.time-sample.time.min(),y,label=col)
ax.set(title="대표 장비 실제 특징: 현재 센서와 과거 평균",xlabel="첫 표시 시각 이후 (원시시간 단위)",ylabel="센서값 (익명화 수치)")
ax.legend(fontsize=8);ax.grid(alpha=.2);save_figure(fig,"causal_feature_example")
assert (sample.source_max_time<=sample.time).all()
conn.close()'''),
        md('''## 장비 평가와 시간 순서 검증

장비 분할은 번호 접두어 01~06 학습, 07~08 검증, 09~10 평가로 고정했습니다. 같은 접두어의 M01/M02도 같은 쪽에 둡니다. 분할을 고장 분포나 성능에 맞춰 재선정하지 않았습니다.

별도 시간 분할은 각 장비 원시 측정 범위의 60/20/20% 경계입니다. 전체 과거 특징 이력과 최대 1800 원시단위의 미래 타깃 구간이 경계를 넘으면 `purged`로 표시합니다. 같은 이벤트의 양성 anchor와 중첩 이력이 서로 다른 분할에 걸치지 않는지 검증했습니다. 두 분할 방식은 별도 평가 설계이며 섞어서 사용하지 않습니다.'''),
        code('''base=ready.query("fault_code=='any' and horizon_raw==1800")
display(base.groupby(["Tool","split_tool","split_time"])[["rows","positive","negative","unknown"]].sum())
for scheme in ("split_tool","split_time"):
    display(Markdown(f"**{scheme}별 클래스 분포 — 통합 타깃**"))
    display(ready.query("fault_code=='any'").groupby([scheme,"horizon_raw"])[["rows","positive","negative","unknown"]].sum())
event_support=pd.DataFrame(summary["observed_positive_events"])
display(event_support.groupby(["scheme","split","fault_code","horizon_raw"]).event_key.nunique().rename("양성 anchor로 관측된 고유 이벤트 수").to_frame())
display(pd.DataFrame(summary["temporal_intervals"]))
display(Markdown("고장 이력의 이벤트 수와 실제 이력·라벨이 갖춰진 양성 anchor의 이벤트 수는 다릅니다. 데이터 공백과 엄격한 충분성 조건으로 누락되는 이벤트를 성능 평가 시 숨기지 않아야 합니다. 검증·평가의 양성/음성 비율은 재표본화하지 않았습니다."))'''),
        md('''## 전체 상태 모델과 가동 구간 모델의 학습 가능 크기

`training/`에는 특징이 준비되고 12개 타깃 중 하나 이상 알려진 행만 있습니다. 각 모델은 해당 **target IS NOT NULL**을 추가 필터링해야 합니다. 전체 상태 모델은 모든 Stage, 가동 모델은 **Stage>0**으로 같은 분할을 필터링합니다. 학습 불균형 처리는 아직 하지 않았으며 검증·평가는 동일 주기의 실제 분포를 유지합니다.'''),
        code('''sizes=ready.groupby(["fault_code","horizon_raw"])[["positive","negative","unknown"]].sum()
sizes["타깃별 학습 가능 수"]=sizes.positive+sizes.negative
display(sizes)
active=ready[ready.state_group=="stage_positive"].groupby(["fault_code","horizon_raw"])[["positive","negative","unknown"]].sum()
active["가동 타깃별 학습 가능 수"]=active.positive+active.negative
display(active)
fig,axes=plt.subplots(1,2,figsize=(15,6),layout="constrained")
for ax,scheme in zip(axes,("split_tool","split_time")):
    x=ready.query("fault_code=='any' and horizon_raw==1800").groupby(scheme)[["positive","negative","unknown"]].sum()
    x.plot.bar(stacked=True,ax=ax)
    ax.set(title=f"통합 h1800raw / {scheme}",xlabel="분할",ylabel="학습 후보 행 수");ax.legend(title="라벨 상태");ax.tick_params(axis="x",rotation=30)
save_figure(fig,"training_split_distribution")
display(pd.Series(summary["sizes_bytes"],name="Parquet bytes").to_frame())
display(Markdown(f"적어도 하나의 타깃이 알려진 물리 학습 파일은 **{summary['training_rows']:,}행**, **{summary['sizes_bytes']['training']/1024**3:.3f} GiB**입니다. 각 타깃과 분할에 사용 가능한 실제 수는 위 표를 기준으로 합니다. 물리 분 단위가 미확정이므로 현재 사용 가능한 것은 원시시간 구간 예측 데이터셋입니다."))'''),
        md('''## 누수 검증과 남은 제한

라벨 근거, 시각 키, 키 유일성, 유한 특징, 통합 라벨의 3값 논리, 원본 메타데이터, 이벤트 공유, 시간 범위 중첩 검사를 통과해야 결과 파일을 사용합니다. 아래 단위 테스트는 실제 첫 연속 관측 일부에서 미래 값 변경과 청크 크기 변경을 적용한 검증이며 실험 성능 수치가 아닙니다.'''),
        code('''import unittest
suite=unittest.TestLoader().discover(str(ROOT/"tests"),pattern="test_dataset_pipeline.py")
test_result=unittest.TextTestRunner(verbosity=2).run(suite)
assert test_result.wasSuccessful()
display(pd.Series({"전체 검사 통과":summary["all_checks_passed"],"미래 관측 입력":False,
                   "공유 양성 이벤트":len(summary["shared_positive_events"]),"시간 분할 중첩":len(summary["overlapping_time_splits"]),
                   "원본 메타데이터 유지":summary["source_metadata_unchanged"],"공식 test 정답 사용":False}).to_frame("검증 결과"))
display(Markdown("물리 시간 단위, 고장 이력의 절대적 완전성, Stage>0의 실제 가동/정상 의미, 동시각 배치의 현장 도착 순서는 미확정입니다. 긴 공백·TTF 결측·부족한 이력 때문에 사용 가능한 이벤트가 감소합니다. 공식 test는 로컬 정답이 없으므로 현재 모델 평가 계획은 내부 장비/시간 분할입니다. 모델 학습·스케일링·대치·클래스 재표본화는 수행하지 않았습니다."))''')
    ]
    return [("04_label_generation.ipynb",four),("05_feature_engineering.ipynb",five)]


def main():
    if Path(sys.prefix).name.lower()!="phm2018":raise RuntimeError("phm2018 환경에서 실행하세요.")
    kernel=LOCAL/"jupyter/kernels/phm2018";kernel.mkdir(parents=True,exist_ok=True)
    (kernel/"kernel.json").write_text(json.dumps({"argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],"display_name":"Python (phm2018)","language":"python"}),encoding="utf-8")
    status=[]
    for name,cells in build():
        path=ROOT/"notebooks"/name
        book=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python (phm2018)","name":"phm2018","language":"python"}})
        nbf.write(book,path);start=time.monotonic();print("Notebook 실행:",name,flush=True)
        try:NotebookClient(book,kernel_name="phm2018",timeout=1200,resources={"metadata":{"path":str(ROOT)}},allow_errors=False).execute()
        finally:nbf.write(book,path)
        codes=[c for c in book.cells if c.cell_type=="code"]
        row={"notebook":name,"code_cells":len(codes),"executed":sum(c.execution_count is not None for c in codes),
             "errors":sum(o.output_type=="error" for c in codes for o in c.outputs),"elapsed_seconds":time.monotonic()-start,"python":sys.executable}
        status.append(row);print(row,flush=True)
        (LOCAL/"notebook_execution.json").write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":main()
