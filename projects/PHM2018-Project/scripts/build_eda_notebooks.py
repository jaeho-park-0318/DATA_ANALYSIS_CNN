"""한국어 EDA Notebook 3개를 생성하고 phm2018 커널에서 실제 실행한다."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/"reports/local/eda"
LOCAL.mkdir(parents=True,exist_ok=True)
# 환경 디렉터리 밖의 전역 Jupyter/Matplotlib 설정을 변경하지 않는다.
os.environ["IPYTHONDIR"]=str(LOCAL/"ipython")
os.environ["JUPYTER_RUNTIME_DIR"]=str(LOCAL/"jupyter_runtime")
os.environ["MPLCONFIGDIR"]=str(LOCAL/"matplotlib")
os.environ["JUPYTER_PATH"]=str(LOCAL/"jupyter")
os.environ["PYTHONIOENCODING"]="utf-8"

import nbformat as nbf
from nbclient import NotebookClient


SETUP='''from pathlib import Path
import sys
ROOT = Path.cwd() if (Path.cwd()/"src").is_dir() else Path.cwd().parent
sys.path.insert(0,str(ROOT))
assert Path(sys.prefix).name.lower()=="phm2018", sys.executable
assert sys.version_info[:2]==(3,11), sys.version
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from src.eda_common import DATA, OUT, SENSORS, FAULTS, SOURCE_URL, overview, read_json, load_faults, snapshot
from src.eda_detail import representative, audit_time_keys, audit_tagged_ttf, horizon_labels
from src import eda_plots as plots
plots.setup()
pd.set_option("display.max_columns",30)
pd.set_option("display.max_rows",80)
print("실행 Python:",sys.executable)
print("Python 버전:",sys.version.split()[0])
'''


def md(text): return nbf.v4.new_markdown_cell(text.strip())
def code(text): return nbf.v4.new_code_cell(text.strip())


def notebooks():
    one=[
        md('''# 01. 데이터 구성과 시간·공정 품질

원본은 읽기 전용이며, 전체 센서 파일은 최대 200,000행씩 필요한 열만 읽습니다. 전체 원본을 하나의 Pandas DataFrame으로 결합하지 않습니다. 소형 fault 이력만 전체 읽기합니다. 입력 경로·크기·수정 시각과 코드 해시가 같은 경우 청크 집계 캐시를 재사용합니다.

**범위:** train 20개는 주요 센서 6개와 공정·시간을 집계하고, test 5개는 구조·시간·공정만 확인합니다. 테스트 센서 분포로 특징이나 임계값을 선택하지 않습니다. 모델 학습은 없습니다.'''),
        code(SETUP),
        md('''## 이전 단계 확인과 실제 파일 구조

`src/inspect_data.py`, 초기 `inspection.json` 및 파일별 헤더를 다시 확인합니다. 초기 5행의 dtype 추정과 이번 전체 청크의 엄격한 수치 파싱을 구분합니다.'''),
        code('''initial_path=ROOT/"reports/local/inspection.json"
initial=pd.read_json(initial_path,typ="series")
print("초기 점검:",initial["file_count"],"개 파일 / 대표",initial["sample"]["rows_read"],"행")
print((ROOT/"src/inspect_data.py").read_text(encoding="utf-8")[:350])
result=overview()
schemas=pd.DataFrame(result["schemas"])
schemas["컬럼 수"]=schemas["columns"].map(len)
display(schemas[["path","bytes","컬럼 수"]])
for kind, sub in schemas.groupby(schemas.path.str.contains("train_faults").map({True:"fault",False:"기타"})):
    print(kind,"파일 수:",len(sub))
sensor_schema=initial["sample"]["columns"]
checks=[]
for row in result["schemas"]:
    kind="fault" if "train_faults/" in row["path"] else "ttf" if "train_ttf/" in row["path"] else "sensor"
    expected=["time","fault_name","Tool"] if kind=="fault" else ["time"]+["TTF_"+f for f in FAULTS] if kind=="ttf" else sensor_schema
    checks.append({"파일":row["path"],"유형":kind,"헤더 일치":row["columns"]==expected})
display(pd.DataFrame(checks).groupby("유형")["헤더 일치"].agg(["count","all"]))
display(pd.DataFrame({"컬럼":sensor_schema,"초기 5행 dtype":[initial["sample"]["dtypes_inferred_from_five_rows"][c] for c in sensor_schema]}))'''),
        md('''## Tool별 관측 수, 측정 범위와 실제 간격

`time`은 절대 날짜로 변환하지 않습니다. 원시 좌표의 최소·최대와 차이를 제시합니다. 간격 집계는 원본 순서와 청크 경계를 유지하고 Tool별로 계산합니다. 양의 간격의 최빈값과 0 간격(동시각 행), 역전, 긴 공백을 분리합니다.'''),
        code('''tools=pd.DataFrame(result["tools"])
display(tools.rename(columns={"rows":"관측 행 수","min_time":"시작 원시시각","max_time":"종료 원시시각","duration_raw":"측정 범위 원시단위","mode_dt":"양의 간격 최빈값","duplicate_adjacent_time":"동시각 인접행","time_reversals":"시간 역전","gaps_gt_mode":"최빈간격 초과 수"}))
intervals=pd.DataFrame(result["intervals"])
display(intervals.groupby(["split","delta_raw"])["count"].sum().reset_index().sort_values("count",ascending=False).head(15))
fig,ax=plt.subplots(figsize=(12,6),layout="constrained")
for split,group in tools.groupby("split"):
    ax.bar(group.Tool+"/"+split,group.rows,label=split)
ax.set(title="Tool별 센서 기록 수",xlabel="Tool / 분할",ylabel="관측 수 (행)")
ax.tick_params(axis="x",rotation=70);ax.legend();plots.save(fig,"tool_rows")
train=tools[tools.split=="train"].set_index("Tool"); test=tools[tools.split=="test"].set_index("Tool")
continuity=test[["min_time"]].join(train[["max_time"]],how="left")
continuity["train_test_경계간격"]=continuity.min_time-continuity.max_time
display(continuity)
display(Markdown(f"학습 센서는 **{int(train.rows.sum()):,}행**, 테스트 센서는 **{int(test.rows.sum()):,}행**입니다. 모든 파일의 양의 간격 최빈값은 {sorted(tools.mode_dt.unique())} 원시시간 단위입니다. 학습 데이터에 동시각 인접 기록 **{int(train.duplicate_adjacent_time.sum()):,}개**가 있으므로 `(Tool,time)`은 고유 행 키가 아닙니다. 테스트는 기존 학습 장비 5개의 후속 구간이며 장비 독립 평가로 해석할 수 없습니다."))'''),
        md('''## 주요 센서의 자료형, 결측·무한대, 범위

6개 주요 센서는 전체 학습 행을 `float64`로 엄격하게 파싱했습니다. 아래 값은 전체 학습 데이터 집계이며 테스트 센서값은 포함하지 않습니다. 익명화 값의 음수를 물리적 오류로 간주하지 않습니다.'''),
        code('''sensor_summary=pd.DataFrame(result["sensor_global"])
display(sensor_summary)
display(pd.DataFrame(result["sensor_by_tool"]).pivot(index="Tool",columns="sensor",values="missing_ratio"))
display(Markdown(f"주요 센서 6개의 전체 결측값은 **{int(sensor_summary.missing.sum()):,}개**, 무한대는 **{int(sensor_summary.infinite.sum()):,}개**입니다. 이는 이번에 읽은 6개 센서에 한정되며 나머지 센서의 전체 결측 여부는 미검증입니다. IONGAUGEPRESSURE의 최대값은 **{sensor_summary.set_index('sensor').loc['IONGAUGEPRESSURE','max']:.3f}**로, 분포 꼬리를 별도로 살펴볼 필요가 있습니다. 물리적 고장이나 오기록이라고 단정하지 않습니다."))'''),
        md('''## Recipe·Stage 분포

전체 학습/테스트의 조합별 행 수를 구합니다. 범주 ID의 숫자 크기를 연속적 공정 강도로 해석하지 않습니다.'''),
        code('''groups=pd.DataFrame(result["groups"])
distribution=groups.groupby(["split","recipe","stage"],dropna=False)["rows"].sum().reset_index()
display(distribution)
fig,axes=plt.subplots(1,2,figsize=(15,6),layout="constrained")
for ax,col in zip(axes,["recipe","stage"]):
    counts=groups[groups.split=="train"].groupby(col).rows.sum().sort_values(ascending=False)
    ax.bar(counts.index,counts.values,label="학습 데이터 전체")
    ax.set(title=f"{col}별 관측 분포",xlabel=f"{col} 범주 ID (단위 없음)",ylabel="관측 수 (행)")
    ax.tick_params(axis="x",rotation=60);ax.legend()
plots.save(fig,"recipe_stage_distribution")
print("원본 크기·수정 시각 유지:",result["source_metadata_unchanged"])
display(Markdown("공정 범주와 장비에 따른 관측 수 차이가 있으므로 전체 센서 평균만으로 고장 전조를 설명하지 않습니다. 동일 시각 중복과 공백을 먼저 보존·표시하고, 동일 고장 이벤트의 행들이 학습/검증 경계를 넘지 않게 설계해야 합니다."))'''),
        md('''## 해석의 근거와 시간 단위 제한

[PHM Society 공식 데이터 설명](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/)을 확인했습니다. 수치 단위는 익명화되어 있고, fault 시각은 정비를 위해 정지한 시점입니다. 실제 이상 시작 시각은 제공되지 않습니다. 문서의 초 표기는 예시이므로 분 단위는 확정하지 않습니다. 뒤의 300·600·1800·3600 원시시간 구간을 **time=초 가정의 5·10·30·60분**으로 병기합니다. TTF와 fault의 일치는 동일 시간 좌표를 검증할 뿐 물리 단위 자체를 증명하지 않습니다.''')
    ]
    two=[
        md('''# 02. 센서 분포·추이·공정 조건 EDA

전체 장비의 범위·결측은 01 Notebook에서 집계했습니다. 이 Notebook은 대용량 추가 처리를 줄이기 위해 **사전 고정한 01M01 전체 3,124,704행**을 대상으로 분포와 상관을 집계합니다. 무작위 표본은 사용하지 않습니다. 히스토그램과 장기 추이만 고정 크기 집계로 축약하며, 고장 직전 시계열은 03에서 원래 순서로 분석합니다.'''),
        code(SETUP),
        code('''result=overview()
detail=representative()
print("대표 파일:",detail["file"],"/ 장비:",detail["Tool"],"/ 관측 수:",detail["rows"])
display(pd.DataFrame(result["sensor_by_tool"]).query("Tool == @detail['Tool']") if False else pd.DataFrame([x for x in result["sensor_by_tool"] if x["Tool"]==detail["Tool"]]))'''),
        md('''## 센서별 히스토그램

대표 장비 전체 유한값을 센서마다 120개 동일 폭 bin으로 집계했습니다. 긴 꼬리의 빈도를 보기 위해 y축을 로그로 표시합니다. 특정 물리 단위를 임의로 붙이지 않습니다.'''),
        code('''plots.histograms(detail)
display(Markdown("분포의 여러 봉우리와 경계 부근 집중은 공정 상태 혼합의 영향을 받을 수 있습니다. 익명화 수치가 음수라는 이유만으로 이상값으로 처리하지 않습니다. 아래 공정 조건별 평균과 함께 해석합니다."))'''),
        md('''## 시간 추이

대표 장비의 전체 측정 기간을 약 2,000개 시간 bin으로 집계한 평균입니다. 원래 관측이 없는 bin은 연결하지 않습니다. bin 내부의 짧은 공백과 급격한 변화는 축약될 수 있으므로 전조 판단에는 아래 장기 평균을 쓰지 않습니다.'''),
        code('''plots.timeline(detail)
print("시간 bin 폭:",detail["timeline_bin_raw"],"원시시간 단위")
display(Markdown("장기 평균은 운전 조건과 가동 비율의 변화도 반영합니다. 압력이나 전류의 상승·하락을 장기 평균 그래프만으로 고장 진행으로 단정하지 않습니다."))'''),
        md('''## 결측 및 이상값 후보

히스토그램 내 선형 보간으로 Q1/Q3를 근사하고 `Q1-1.5×IQR`, `Q3+1.5×IQR` 경계를 정했습니다. 그 경계 밖의 행 수는 대표 파일을 다시 청크 순회하여 정확히 셉니다. 분위수 경계 자체는 근사이며, 통계적 꼬리 후보는 고장 라벨이나 물리 오류 판정이 아닙니다. 삭제·클리핑은 하지 않습니다.'''),
        code('''outliers=pd.DataFrame(detail["outlier_fences"]).T
outliers["flagged_ratio"]=outliers.flagged/detail["rows"]
display(outliers)
display(Markdown("IQR 경계는 공정 상태를 섞으면 정상 운전의 드문 상태도 표시할 수 있습니다. 모델링 전에는 장비·Recipe·Stage별 기준과 데이터 사전을 확인해야 합니다. 전체 장비의 IQR 이상값 수는 이번에 계산하지 않았습니다."))'''),
        md('''## 주요 센서 상관관계

대표 장비에서 6개 센서가 모두 유한한 행의 합·제곱합·교차곱을 청크별로 누적한 Pearson 상관입니다. 장비 전체에 대한 결과이며 장비 전체를 메모리에 적재하지 않습니다. 공정 조건 혼합과 시계열 자기상관의 영향이 있습니다.'''),
        code('''plots.correlation(detail)
c=np.array(detail["correlation"])
display(Markdown(f"대표 01M01에서 ETCHBEAMCURRENT–ETCHBEAMVOLTAGE 상관은 **{c[3,4]:.3f}**, ETCHBEAMCURRENT–ETCHSUPPRESSORCURRENT는 **{c[3,5]:.3f}**입니다. 높은 공정 동조성을 시사하지만 고장 인과관계나 다른 장비에서의 동일 상관을 입증하지 않습니다."))'''),
        md('''## Recipe·Stage별 센서 값 차이

대표 장비에서 관측 수가 많은 10개 Recipe/Stage 조합을 사용합니다. 조건별 평균은 전체 청크에서 집계하며, 표에는 모든 조합의 행 수와 센서 평균을 제공합니다.'''),
        code('''g=pd.DataFrame(result["groups"])
g=g[(g.split=="train")&(g.Tool==detail["Tool"])].sort_values("rows",ascending=False)
display(g[["recipe","stage","rows"]+SENSORS])
plots.process_means(result,detail)
spread=pd.DataFrame({"조건별 평균 최소":g[SENSORS].min(),"조건별 평균 최대":g[SENSORS].max()})
display(spread)
display(Markdown("동일 센서도 Recipe/Stage 조합별 평균이 다릅니다. 전체 구간과 고장 전 구간의 평균 차이를 비교할 때 공정 구성 차이를 함께 확인해야 합니다. 03 Notebook에서 실제 구간의 조건별 관측 수와 동일 조건 평균 차이를 대조합니다."))''')
    ]
    three=[
        md('''# 03. 고장 이벤트·고장 전 시계열·TTF 검증

모든 fault 이력과 TTF는 분석용 정답으로만 사용합니다. 고장 이후 데이터나 TTF를 예측 특징으로 만들지 않으며 모델 학습은 수행하지 않습니다. 대표 장비는 01M01로 고정하고, 이벤트는 센서 변화 크기가 아닌 **관측 충분성**으로 선정합니다.

시간 단위는 미확정이므로 5·10·30·60분 표시는 **time=초 가정**입니다. 분석 구간은 각각 300·600·1800·3600 원시시간 단위이며 고장 시각 자체는 제외합니다.'''),
        code(SETUP),
        code('''result=overview();detail=representative();faults=load_faults()
unique_faults=faults.drop_duplicates(["Tool","time","fault_raw"]).copy()
keys=audit_time_keys()
tag_audit=audit_tagged_ttf()
alignment=pd.DataFrame(result["fault_alignment"])
ttf=pd.DataFrame(result["ttf"])
display(pd.DataFrame(keys["files"])[["file","sensor_rows","ttf_rows","ordered_time_keys_equal"]])
assert all(x["ordered_time_keys_equal"] for x in keys["files"])
print("fault 중복 이벤트 행:",result["fault_duplicates"])'''),
        md('''## 고장 유형과 장비별 발생 횟수

원문 이름과 정규화 이름을 함께 제공합니다. 동일 `(Tool,time,fault_raw)` 중복 25행을 제외한 고유 이벤트 수와 원본 이력 행 수를 구분합니다. `[NoWaferID]`는 표기 태그로 분리하여 3종 분포에 포함하되, TTF 의미 검증에서는 포함/제외를 모두 비교합니다. 단위 기간당 고장률은 아닙니다.'''),
        code('''display(pd.Series(result["fault_raw_counts"],name="원문 이벤트 수").to_frame())
display(pd.Series(result["fault_counts"],name="3종 집계 이벤트 수").to_frame())
count_comparison=pd.DataFrame({"원본 이력 행 수":faults.fault_name.value_counts(),"고유 이벤트 수":unique_faults.fault_name.value_counts()})
display(count_comparison)
display(pd.crosstab(unique_faults.Tool,unique_faults.fault_name))
plots.fault_distribution(unique_faults)
display(Markdown(f"원본 고장 이력은 **{len(faults):,}행**, 동일 장비·시각·원문 고장 유형을 중복 제거한 이벤트는 **{len(unique_faults):,}건**입니다. 고유 이벤트는 저압 741건, 고압 333건(태그 1건 포함), 누출 137건입니다. 원본 파일은 변경하지 않았습니다. 장비별 노출 시간과 관측 공백이 다르므로 이 건수만으로 신뢰성을 비교하지 않습니다."))'''),
        md('''## Fault와 센서의 연결 기준 및 시간 정합성

파일 접두어로 후보 장비를 찾은 뒤 실제 `Tool`을 확인하고 시각을 비교합니다. `(Tool,time)`은 센서에서 비고유하므로 센서 행 식별자와 고장 이벤트 ID를 구분합니다. 행 번호로 fault를 붙이지 않습니다. 정확한 시각 일치뿐 아니라 가장 가까운 엄격한 이전/이후 관측을 전 청크에 걸쳐 추적했습니다.'''),
        code('''tool_table=pd.DataFrame(result["tools"])
train_info=tool_table[tool_table.split=="train"].set_index("Tool")
alignment["inside_sensor_range"]=[train_info.loc[x.Tool,"min_time"]<=x.fault_time<=train_info.loc[x.Tool,"max_time"] for x in alignment.itertuples()]
alignment=alignment[alignment.event_id.isin(unique_faults.event_id)].copy()
display(alignment.groupby("Tool").agg(고장수=("event_id","size"),정확일치수=("exact_rows",lambda x:int((x>0).sum())),측정범위내=("inside_sensor_range","sum")))
display(alignment[["previous_gap_raw","next_gap_raw"]].replace([np.inf,-np.inf],np.nan).describe())
finite_previous=alignment.previous_gap_raw[np.isfinite(alignment.previous_gap_raw)]
display(Markdown(f"정확히 센서 시각과 일치하는 이벤트는 **{int((alignment.exact_rows>0).sum())}/{len(alignment)}건**입니다. 측정 범위 밖 이벤트는 **{int((~alignment.inside_sensor_range).sum())}건**입니다. 직전 관측까지의 간격은 결측·무한대를 제외한 중앙값 **{finite_previous.median():g} 원시시간 단위**입니다. 정확 일치가 없는 이벤트를 임의로 가장 가까운 센서 행의 고장으로 덮어쓰지 않습니다. 아래 윈도별 관측 충분성으로 분석 가능 여부를 판정합니다."))'''),
        md('''## 고장 전 5·10·30·60분: 관측 충분성 우선

최빈 양의 간격(4 원시단위)을 기준으로 고유 시각 수/기대 관측 수가 90% 이상이고, 양 끝 공백이 2간격 이하, 내부 최대 공백이 3간격 이하인 경우만 충분하다고 표시합니다. 동일 시각의 중복 행은 원본대로 유지하되 충분성 분자에는 고유 시각 수만 사용합니다. 01M01의 모든 이벤트를 확인한 뒤 충분한 윈도 수가 가장 많은 이벤트, 동률이면 더 이른 이벤트를 선택합니다. 이는 초기 사례 분석이며 고장 유형 전체의 대표성은 보장하지 않습니다.'''),
        code('''candidate_rows=[]
for e in detail["event_candidates"]:
    for w in e["windows"]:candidate_rows.append({"event_id":e["event_id"],"fault_name":e["fault_name"],"time":e["time"],"stage>0 비율":e["stage_nonzero_ratio"],**w})
display(pd.DataFrame(candidate_rows))
event=detail["selected_event"]
pre=pd.read_json(OUT/"pre_event.json");control=pd.read_json(OUT/"control.json")
display(pd.DataFrame(event["windows"]))
display(pd.DataFrame([detail["control"]]))
print("선택 이벤트:",event["event_id"],event["fault_name"],"time=",event["time"])
plots.event_plot(pre,event,control,note="4개 윈도 관측 충분 / 고장 전후를 연결하지 않음")
plots.process_timeline(pre,control,event)'''),
        md('''## 비교 구간과 공정 조건 통제

비교 후보는 같은 장비에서 선택 이벤트보다 과거이며 모든 알려진 fault에서 3600 원시시간 단위 이상 떨어진 구간입니다. **무고장 상태가 보장된 정상 정답이 아닙니다.** 공정이 진행 중인지도 Stage/Recipe와 함께 검토합니다. 두 구간의 동일 Recipe/Stage 조합에 각각 10개 이상 유효 관측이 있을 때만 조건별 비교 가능으로 표시합니다.'''),
        code('''display(pd.DataFrame(detail["comparisons"]))
display(pre.groupby(["recipe","stage"]).size().rename("고장 전 행 수").to_frame())
display(control.groupby(["recipe","stage"]).size().rename("비교 구간 행 수").to_frame())
display(pd.DataFrame(detail["matched_comparisons"]))
hour=pd.DataFrame(detail["comparisons"]).query("minutes_assuming_seconds==60").set_index("sensor")
first=pre[pre.time<event["time"]-1800];last=pre[pre.time>=event["time"]-1800]
within=pd.DataFrame({"앞 30분 평균":first[SENSORS].mean(),"뒤 30분 평균":last[SENSORS].mean()})
within["뒤-앞"]=within["뒤 30분 평균"]-within["앞 30분 평균"]
display(within)
display(Markdown(f"선택 이벤트와 비교 구간은 모두 **Recipe=0, Stage=0**입니다. 가동 중 정상 운전이라고 확인할 수 없는 상태입니다. IONGAUGEPRESSURE의 60분 평균 차이는 **{hour.loc['IONGAUGEPRESSURE','difference']:.4f}**이고, 고장 전 구간 내부 앞/뒤 30분 차이는 **{within.loc['IONGAUGEPRESSURE','뒤-앞']:.4f}**입니다. FLOWCOOLPRESSURE는 고장 전 구간에서 일정하고 비교 구간과의 평균 차이는 **{hour.loc['FLOWCOOLPRESSURE','difference']:.4f}**입니다. 이 한 사례에서는 고압 고장으로 가는 압력 상승 전조를 확인하지 못했습니다. 구간 간 차이와 고장 직전의 진행성 변화를 구분해야 합니다."))'''),
        md('''## 가동 공정의 보조 사례와 정상 운전 비교 후보

Stage>0 관측 비율이 50% 이상이고 최소 2개 윈도가 충분한 이벤트를 보조로 선정합니다. Stage>0은 가동 공정 후보일 뿐 정상 상태를 증명하지 않습니다. 관측이 충분하고 알려진 고장에 인접하지 않은 가동 구간을 추가 제시하되, 시간·Recipe·Stage가 다르면 직접적인 고장 효과로 해석하지 않습니다.'''),
        code('''secondary=detail["secondary_event"]
active=pd.read_json(OUT/"active_control.json")
if secondary is not None:
    secondary_pre=pd.read_json(OUT/"secondary_pre.json")
    display(pd.DataFrame(secondary["windows"]))
    display(secondary_pre.groupby(["recipe","stage"]).size().rename("보조 고장 전 행 수").to_frame())
    display(pd.DataFrame([detail["active_control"]]))
    if len(active):
        display(active.groupby(["recipe","stage"]).size().rename("가동 비교 후보 행 수").to_frame())
        display(pd.DataFrame({"보조 고장 전 평균":secondary_pre[SENSORS].mean(),"가동 비교 후보 평균":active[SENSORS].mean()}))
    plots.event_plot(secondary_pre,secondary,active,name="secondary_pre_event",note="5·10·30분 충분 / 60분 관측 부족 — 보간하지 않음")
    early=secondary_pre[secondary_pre.time<secondary["time"]-300]
    late=secondary_pre[secondary_pre.time>=secondary["time"]-300]
    display(pd.DataFrame({"앞쪽 관측 평균":early[SENSORS].mean(),"마지막 5분 평균":late[SENSORS].mean()}))
    display(Markdown("누출 보조 사례의 60분 윈도는 약 63.3%만 관측되어 충분하지 않습니다. 가동 비교 후보의 평균 차이는 공정·시간 차이도 포함합니다. 이 표와 시계열만으로 정상/고장 구분 능력이나 전조의 일반성을 주장하지 않습니다."))
else:
    display(Markdown("조건을 만족하는 가동 보조 사례가 없어 보조 비교는 미실행입니다."))'''),
        md('''## TTF 구조·결측 의미·Fault 대조

TTF 컬럼 3개를 전 파일에서 청크로 읽고, 각 행의 시각에서 다음 동일 유형 이벤트까지의 차이와 비교합니다. 센서/TTF 시간열은 중복 및 순서를 포함한 digest가 전 파일에서 동일한지 별도로 검증합니다. digest 일치는 행 단위 읽기가 가능하다는 품질 증거이지만 조인 기준은 검증한 장비·시각입니다.

공식 설명은 TTF의 null을 특정 예측 범위 내 이벤트가 없는 경우로 설명하지만, 그 범위와 관측 종료 검열을 숫자로 명시하지 않습니다. 아래 실제 기록에서 **제공된 fault 이력 내 미래 동일 유형 이벤트가 없는 경우**와 결측이 일치하는지 확인합니다. 정상으로 자동 치환하지 않습니다.'''),
        code('''cols=["rows","missing","finite","matches_next_fault","mismatch","missing_with_future_fault","missing_without_future_fault","no_future_fault_finite","positive_300","positive_600","positive_1800"]
display(ttf.groupby("column")[cols].sum())
print("음수 TTF 수:",ttf.negative.sum())
display(ttf[ttf.mismatch>0])
display(pd.DataFrame([{k:v for k,v in a.items() if k!="examples"} for a in tag_audit]))
for a in tag_audit:display(pd.DataFrame(a["examples"]))
display(Markdown("실제 기록에서 결측 TTF는 모두 제공된 동일 유형의 미래 fault가 없는 구간에 있습니다. 관측 종료 이후의 상태는 알 수 없으므로 영구 정상이라는 뜻이 아닙니다. 03M02의 고압 TTF는 `[NoWaferID]` 이벤트를 포함하면 3,489행이 불일치하고, 해당 태그 이벤트를 제외하면 불일치가 0행입니다. 고압의 원본 341행/고유 333건과 태그 제외 원본 340행/고유 332건을 구분해야 합니다. 태그 이력은 삭제하지 않습니다."))'''),
        md('''## 5·10·30분 이내 라벨 정의 가능성

조건부 타깃은 `0 < 다음 고장 시각 - t <= H`, H=300/600/1800 원시단위입니다. 고장 시각 자체와 이력/TTF가 모순되는 행은 제외합니다. `[NoWaferID]` 제외 정책을 먼저 명시해야 합니다. 물리 분 단위는 time=초 가정에 한정됩니다.

양성은 확인된 미래 이벤트를 이용해 정의할 수 있습니다. 음성은 예측 구간 전체의 추적 관측 및 이벤트 기록 완전성이 검증되어야 합니다. `horizon_labels`는 그 검증을 전달하지 않으면 음성을 임의 생성하지 않고 NA를 유지합니다. 아래는 실제 대표 고장 전 구간에 적용한 **라벨 정의 점검**이며 학습 데이터셋이나 예측 결과가 아닙니다.'''),
        code('''target_fault=event["fault_name"]
target_times=faults.loc[(faults.Tool==detail["Tool"])&(faults.fault_raw==target_fault),"time"].to_numpy()
label_counts=[]
for h in (300,600,1800):
    labels=horizon_labels(pre.time.to_numpy(),target_times,pre.time.max(),h)
    label_counts.append({"원시시간 구간":h,"분 (초 가정)":h//60,"양성":int((labels==1).sum()),"음성":int((labels==0).sum()),"미확인":int(labels.isna().sum())})
display(pd.DataFrame(label_counts))
display(Markdown("완전성 미검증 행은 NA로 남겼으므로 음성 수 0은 정상 구간이 없다는 뜻이 아닙니다. TTF가 유한하고 H보다 큰 경우에도 시간 단위·이벤트 정의를 검증한 뒤 음성 후보로 쓸 수 있습니다. 결측 TTF, 관측 종료, 내부 공백 및 태그 관련 불일치 구간은 별도 검열/제외 정책이 필요합니다."))'''),
        md('''## 모델 구축 전에 필요한 처리

1. 시간 단위와 고장·태그 포함 정책을 확정하고, 관측 공백·정지 상태·검열을 구분합니다.
2. 동시각 기록을 원본 행 ID와 함께 보존하고 공정 차이를 확인한 뒤 집계 정책을 정합니다. 임의 중복 삭제와 긴 공백 보간은 하지 않습니다.
3. 장비/Recipe/Stage별 기준을 검토하고 과거 방향 윈도로 특징을 만듭니다. TTF/fault는 정답에만 사용합니다. ACTUALSTEPDURATION 등은 관측 가능 시점 확인 전 특징으로 쓰지 않습니다.
4. 장비 단위 외부 검증과 시간 순서 내부 검증을 설계합니다. 중첩 윈도·동일 이벤트가 경계를 넘지 않도록 윈도/예측 구간에 맞는 간격을 둡니다. 기존 test는 학습 장비의 후속 구간입니다.
5. 이벤트 매칭·경보 병합·최소 Lead Time·정상 운전 노출시간을 정의하고 F1-score, Event Recall, 시간당 False Alarm Rate, Lead Time 분포를 함께 평가합니다.

**미검증:** 물리 단위, 실제 이상 시작 시각, 고장 이력의 완전성, 전체 센서 24컬럼의 품질, 대표 장비 이외의 상세 분포·고장 전 패턴, 통계적 일반성. 실제 모델 학습 및 성능 평가는 수행하지 않았습니다.''')
    ]
    return [("01_data_overview.ipynb",one),("02_EDA.ipynb",two),("03_fault_analysis.ipynb",three)]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--no-execute",action="store_true");args=parser.parse_args()
    if Path(sys.prefix).name.lower()!="phm2018":raise RuntimeError("phm2018 Python으로 실행하세요.")
    kernel=LOCAL/"jupyter/kernels/phm2018";kernel.mkdir(parents=True,exist_ok=True)
    (kernel/"kernel.json").write_text(json.dumps({"argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],
                                                "display_name":"Python (phm2018)","language":"python"}),encoding="utf-8")
    destination=ROOT/"notebooks";destination.mkdir(exist_ok=True)
    statuses=[]
    for name,cells in notebooks():
        book=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python (phm2018)","language":"python","name":"phm2018"},
                                                    "language_info":{"name":"python","version":"3.11"}})
        path=destination/name;nbf.write(book,path)
        start=time.monotonic();print("Notebook 실행:",name,flush=True)
        if not args.no_execute:
            try:
                NotebookClient(book,kernel_name="phm2018",timeout=1200,resources={"metadata":{"path":str(ROOT)}},allow_errors=False).execute()
            except Exception:
                nbf.write(book,path)
                raise
            nbf.write(book,path)
        codes=[c for c in book.cells if c.cell_type=="code"]
        status={"notebook":name,"code_cells":len(codes),"executed":sum(c.execution_count is not None for c in codes),
                "errors":sum(o.output_type=="error" for c in codes for o in c.outputs),"elapsed_seconds":round(time.monotonic()-start,2),
                "python":sys.executable}
        statuses.append(status);print(status,flush=True)
        (LOCAL/"notebook_execution.json").write_text(json.dumps(statuses,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":main()
