"""검증된 로컬 집계에서 한국어 최종 EDA 보고서를 작성한다."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.eda_common import OUT, SENSORS, FAULTS, SOURCE_URL, read_json, load_faults


def table(frame):
    def render(value):
        if isinstance(value,float):return f"{value:.6g}"
        return str(value).replace("|","/")
    return "\n".join(["| "+" | ".join(map(str,frame.columns))+" |",
                      "| "+" | ".join(["---"]*len(frame.columns))+" |"]+
                     ["| "+" | ".join(render(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None)])


def main():
    r=read_json("overview.json");d=read_json("detail.json");validation=read_json("validation.json")
    tools=pd.DataFrame(r["tools"]);faults=load_faults();unique=faults.drop_duplicates(["Tool","time","fault_raw"])
    stats=pd.DataFrame(r["sensor_global"]);ttf=pd.DataFrame(r["ttf"])
    counts=pd.DataFrame({"고장 유형":FAULTS,"원본 행 수":[int((faults.fault_name==f).sum()) for f in FAULTS],
                         "고유 이벤트 수":[int((unique.fault_name==f).sum()) for f in FAULTS]})
    event=d["selected_event"];pre=pd.read_json(OUT/"pre_event.json")
    a=pd.DataFrame(r["fault_alignment"]);a=a[a.event_id.isin(unique.event_id)]
    train=tools[tools.split=="train"].set_index("Tool")
    inside=np.array([train.loc[x.Tool,"min_time"]<=x.fault_time<=train.loc[x.Tool,"max_time"] for x in a.itertuples()])
    ttf_summary=ttf.groupby("column")[["rows","missing","finite"]].sum().reset_index()
    ttf_summary["결측 비율 (%)"]=100*ttf_summary.missing/ttf_summary.rows
    sensor_table=stats[["sensor","read_dtype","missing_ratio","min","max","mean","std"]]
    window_table=pd.DataFrame(event["windows"])[["minutes_assuming_seconds","rows","unique_times","coverage_ratio","max_gap","sufficient"]]
    tool_table=tools[["split","Tool","rows","min_time","max_time","duration_raw","mode_dt","duplicate_adjacent_time"]]
    bytool=pd.crosstab(unique.Tool,unique.fault_name).reset_index()
    text=f'''# 2단계 EDA 결과 — PHM 2018 Ion Mill Etching

2026-11-05 / Windows / Conda `phm2018`, Python 3.11.18.

## 실행 범위와 검증

- 기존 `src/inspect_data.py`, 초기 점검 JSON, 실제 65개 파일의 헤더를 확인했다. 유형별 헤더는 전체 파일에서 일치했다.
- 원본 CSV를 읽기 전용으로 열고 필요한 열만 최대 200,000행 단위로 읽었다. 전체 원본을 Pandas에 한 번에 적재하지 않았다.
- 전체 train 20개: 센서 6종·시간·공정 집계. 전체 test 5개: 구조·시간·공정 집계만 수행했다.
- 전체 TTF 20개: 결측, 범위, 동일 유형 다음 fault와의 차이를 검증했다. 상세 분포·상관·고장 전 분석은 사전 고정한 `01M01`에 한정했다.
- Notebook 3개, 코드 셀 총 {sum(x['실행 코드 셀'] for x in validation['Notebook'])}개를 실제 phm2018 커널에서 실행했다. 미실행 코드 셀 0개, 오류 0개이며 그래프 10개를 Notebook에 포함했다.
- 경계 라벨·관측 충분성·집계 불변식 검증을 통과했다. 초기 점검과 비교해 원본 65개 파일의 크기·수정 시각이 유지되었다. 전체 파일 내용 해시는 계산하지 않았다.
- 모델 학습, 원본 수정, GitHub push, 원격 업로드, 외부 서비스로 원본 전송은 수행하지 않았다.

## 1. 실제 데이터 구성과 주요 컬럼

`data/raw/`는 없고 원본은 `phm_data_challenge_2018/`에 있다. 총 65개 CSV, 34,191,555,540 bytes다.

| 경로 | 파일 수 | 실제 행 수 | 구성 |
| --- | ---: | ---: | --- |
| train/*.csv | 20 | {int(tools[tools.split=='train'].rows.sum()):,} | 24컬럼 센서·공정 시계열 |
| test/*.csv | 5 | {int(tools[tools.split=='test'].rows.sum()):,} | 동일 24컬럼의 후속 시계열 |
| train/train_faults/*.csv | 20 | 1,236 | time, fault_name, Tool |
| train/train_ttf/*.csv | 20 | 82,189,440 | time 및 고장 유형별 TTF 3개 |

센서 식별/공정 컬럼은 `time`, `Tool`, `stage`, `Lot`, `runnum`, `recipe`, `recipe_step`이다. 이번 주요 센서는 `FLOWCOOLPRESSURE`, `FLOWCOOLFLOWRATE`, `IONGAUGEPRESSURE`, `ETCHBEAMCURRENT`, `ETCHBEAMVOLTAGE`, `ETCHSUPPRESSORCURRENT`다. 전체 컬럼은 01 Notebook에 기록했다.

각 train 파일은 별도 Tool 1개로 총 20개 장비다. test는 새로운 장비가 아니라 train에 있는 5개 장비의 후속 기록이며 train 종료와 test 시작은 모두 4 원시시간 단위 차이다. 기존 test를 장비 독립 성능 검증으로 해석하면 안 된다.

Tool별 관측 범위는 절대 날짜가 아닌 원시 시간 좌표다.

{table(tool_table)}

## 2. 고장 유형별·장비별 발생 횟수와 연결 기준

고장 이력 1,236행 중 동일 `(Tool,time,fault_raw)` 중복이 25행 있어 고유 이벤트는 1,211건이다. 원본을 삭제·수정하지 않고 집계에서만 구분했다. 서로 다른 유형의 동시 이벤트는 합치지 않았다.

{table(counts)}

고압 원문 340행과 `[NoWaferID]` 고압 1행을 3종 기술 집계에서 합쳤다. 고유 고압 이벤트는 태그 포함 333건, 태그 제외 332건이다. TTF 검증에서는 태그 이벤트를 제외한 기준이 일치한다.

{table(bytool)}

센서와 fault는 파일 접두어·실제 Tool·시간 좌표를 대조한다. 센서의 `(Tool,time)`은 비고유하므로 원본 행 ID와 이벤트 ID를 별도로 관리해야 한다. 고유 이벤트 중 정확한 센서 시각 일치는 {int((a.exact_rows>0).sum())}/{len(a)}건이다. 원본 이력 행 기준으로는 215/1,236행이다. 센서 측정 범위 밖 고유 이벤트는 {int((~inside).sum())}건이다. 정확 일치만으로 고장 라벨을 붙이면 관측 공백 내 이벤트를 놓치므로 윈도 관측 충분성과 이벤트 시간을 별도로 다룬다.

## 3. 주요 센서의 특징

다음은 전체 학습 82,189,440행의 집계다. 모든 주요 센서가 float64로 파싱되었고 결측·무한대는 0개다. 물리 단위는 미공개이며 음수는 원본의 익명화 수치다.

{table(sensor_table)}

- 대표 01M01의 분포에는 여러 봉우리와 경계값 집중이 있다. Recipe/Stage별 평균이 달라 공정 상태 혼합을 검토해야 한다.
- IONGAUGEPRESSURE의 전체 최대 603.663, ETCHSUPPRESSORCURRENT의 최대 32.096 등 꼬리 값은 점검 후보이며 물리 오류나 고장이라고 단정하지 않는다.
- 대표 장비의 ETCHBEAMCURRENT–ETCHBEAMVOLTAGE Pearson 상관은 {d['correlation'][3][4]:.4f}, ETCHBEAMCURRENT–ETCHSUPPRESSORCURRENT는 {d['correlation'][3][5]:.4f}다. 공정 동조성이 포함된 상관이며 인과관계가 아니다.
- 대표 장비 전체 히스토그램·시간 bin 평균·상관·공정 조건별 평균을 그렸다. IQR 경계는 히스토그램 보간 분위수로 근사하고, 경계 밖 행 수는 추가 청크 순회로 집계했다. 이상값 삭제·클리핑은 하지 않았다. 상세 결과는 02 Notebook에 있다.

## 4. 고장 이전 센서 변화에서 발견한 사항

01M01의 19개 고유 이벤트를 모두 대상으로 고장 이전 300/600/1800/3600 원시시간 단위의 관측 충분성을 조사했다. 무작위 행 표본은 사용하지 않았다. 충분성은 고유 시각/기대 관측 수 90% 이상, 양 끝 공백 2최빈간격 이하, 내부 공백 3최빈간격 이하로 정의했다.

**시간 단위 제한:** 아래 분은 `time=초`를 가정한 조건부 표기다. 실제 물리 분 단위는 미확정이다.

가장 많은 구간이 충분한 첫 이벤트는 고압 고장 time={event['time']:.0f}이다.

{table(window_table)}

관측 수는 각각 75/150/450/900행이며 네 구간 모두 충분하다. 마지막 관측은 고장 2 원시시간 단위 전이고 고장 시각 자체는 제외했다. 비교 구간도 같은 장비의 과거 구간이며 알려진 고장 주변에서 떨어져 있다.

- 해당 고장 전·비교 구간은 모두 Recipe=0, Stage=0이다. 이를 정상 가동 상태라고 확인할 수 없다.
- FLOWCOOLPRESSURE와 FLOWCOOLFLOWRATE는 고장 전 60분 동안 일정했다. 이 사례에서 고압 고장으로 가는 압력 상승 전조는 확인하지 못했다.
- IONGAUGEPRESSURE의 고장 전 60분 평균은 비교 구간보다 약 0.3263 낮았다. 같은 고장 전 구간 내부의 변화와 구간 간 수준 차이를 구분해야 하며, 이 차이를 예측 신호로 일반화하지 않는다.
- 가동 공정 보조 사례는 누출 time=20174938, Recipe=19, Stage=1이다. 5/10/30분 구간은 각각 75/150/450행으로 충분하지만, 60분은 570/900개 고유 시각(63.3%)으로 부족하다. 부족한 앞부분은 보간하지 않았다.
- 보조 사례의 마지막 5분에도 FlowCool 압력·유량·빔 전압은 앞쪽 관측 평균과 같은 수준이다. 작은 전류/게이지 차이를 고장 전조로 단정하지 않았다.
- 별도 가동 비교 후보는 알려진 고장에 인접하지 않은 과거 900행이며 Recipe=2, Stage=2다. 보조 사례와 공정이 달라 평균 차이를 순수 고장 효과로 볼 수 없다. 알려진 fault가 없다는 이유만으로 정상 정답을 부여하지 않았다.

따라서 현재 두 사례만으로 일관된 고장 예고 패턴을 주장할 근거는 없다. 다른 장비·이벤트와 공정이 일치하는 비교 구간으로 확대 검증해야 한다.

## 5. 데이터 품질 및 결측 문제, TTF 정합성

- 전체 파일에 시간 역전과 시간 결측은 없었다. 양의 시간 간격 최빈값은 모두 4 원시단위다.
- train에서 동시각 인접 행은 {int(train.duplicate_adjacent_time.sum()):,}개, 최빈간격을 초과하는 간격은 {int(train.gaps_gt_mode.sum()):,}개다. 가장 큰 간격은 {max(x['delta_raw'] for x in r['intervals'] if x['split']=='train'):g} 원시단위다. 4라는 최빈값만으로 연속 시계열을 가정하면 안 된다.
- 센서와 TTF의 전체 시간열은 20개 대응 파일 모두 행 수·순서·중복을 포함한 digest가 동일했다.
- 유한 TTF는 대부분 `다음 동일 유형 fault 시각 - time`과 정확히 같다. 03M02 고압에서 `[NoWaferID]`를 포함하면 3,489행이 어긋나며, 이를 제외하면 불일치가 0행이다. 다른 유형/장비의 불일치는 0행이다.
- 실제 결측 TTF는 모두 제공된 동일 유형의 미래 fault가 없는 구간이다. 남은 fault 이력 밖의 검열과 실제 무고장을 구분할 수 없으므로 결측을 0 또는 정상으로 바꾸지 않는다. 음수 TTF는 0개다.

{table(ttf_summary)}

공식 설명은 익명화와 단위 미공개, 정비 정지 시각 및 TTF null의 제한을 설명한다. 현지 파일의 fault 세 번째 컬럼은 문서 표의 `stage`와 달리 실제 `Tool`이며, TTF 폴더도 실제 경로를 기준으로 처리했다. [PHM Society 공식 설명]({SOURCE_URL})

## 6. 모델 학습 전 필요한 전처리

1. time의 물리 단위와 fault 정의, `[NoWaferID]` 포함 정책, 이력 완전성을 확정한다. 고장 정지 시점과 실제 이상 시작 시점을 혼동하지 않는다.
2. `(파일,원본 행 순서)` 등 고유 ID를 보존하고 동시각 공정 차이를 확인한 뒤 집계 정책을 정한다. 긴 공백을 무조건 보간하지 않고 연속 세그먼트/관측 마스크를 만든다.
3. 5/10/30분 타깃 후보는 `0 < 다음 고장까지 시간 <= 300/600/1800` 원시단위다. 분 환산은 조건부다. 고장 시각 자체, TTF 모순, 추적 부족·이력 불완전 구간은 제외 또는 NA로 둔다. 양성은 확인된 미래 이벤트에서 정의할 수 있지만 음성은 전체 예측 구간의 추적·라벨 완전성을 검증한 뒤 부여한다.
4. 장비/Recipe/Stage 조건을 반영해 과거 방향 rolling 특징을 만든다. TTF/fault는 타깃·평가용이며 특징에 넣지 않는다. ACTUALSTEPDURATION은 실제 제공 시점을 확인한다. 스케일러·결측 대치·특징 선택은 학습 분할에만 적합한다.
5. 장비 단위 외부 검증과 시간 순서 내부 검증을 설계한다. 동일 이벤트와 중첩 윈도가 경계를 넘지 않게 예측 구간/윈도에 맞는 간격을 둔다. test로 임계값·특징을 조정하지 않는다.
6. 경보 병합·이벤트 매칭·최소 Lead Time·정상 가동 노출시간을 정의하고 F1-score, Event Recall, False Alarm Rate, Lead Time을 함께 평가한다.

## 7. 생성 파일과 재실행

| 파일 | 용도 |
| --- | --- |
| notebooks/01_data_overview.ipynb | 전체 구조, Tool별 행 수·시간, 센서 품질, Recipe/Stage |
| notebooks/02_EDA.ipynb | 대표 장비 전체 분포·추이·IQR 후보·상관·공정 평균 |
| notebooks/03_fault_analysis.ipynb | 고장 건수, 관측 충분성, 고장 전 비교, TTF/라벨 검토 |
| src/eda_common.py | 읽기 전용 청크 집계, fault/TTF 검증, 메타데이터 캐시 |
| src/eda_detail.py | 대표 분포, 이벤트·비교 구간, 시간 키 검사, 보수적 라벨 함수 |
| src/eda_plots.py | 한국어 제목·축·단위·범례의 공통 그래프 |
| scripts/build_eda_notebooks.py | Notebook 생성 및 phm2018 커널 실행 |
| scripts/validate_eda.py | 집계 불변식·라벨 경계·Notebook·원본 메타데이터 검증 |
| scripts/write_eda_report.py | 검증된 수치로 본 보고서 재생성 |
| requirements-eda.txt | 실제 실행 환경의 직접 의존성 버전 |
| reports/local/eda/ | Git 제외: 집계 JSON, 구간 자료, 그래프, 실행·검증 로그 |

```powershell
conda activate phm2018
python scripts/build_eda_notebooks.py
python scripts/validate_eda.py
python scripts/write_eda_report.py
```

Notebook은 직접 순서대로 열어 실행할 수도 있다. 첫 전체 집계가 없거나 입력/공통 코드가 바뀌면 청크 순회를 수행하고, 일치하는 캐시가 있으면 재사용한다. `overview(force=True)`와 `representative(force=True)`로 강제 재집계할 수 있다. 원본을 담은 구간 자료와 로컬 결과는 `.gitignore`의 `reports/local/` 제외 규칙을 따른다.

## 미실행·미검증 범위

미실행 코드 셀은 없다. 전체 장비의 상세 분포·상관·고장 전 패턴, 나머지 센서의 전체 결측/이상값, 물리 단위, 실제 이상 시작 시점, 고장 이력 완전성, 패턴의 통계적 일반성은 미검증이다. IQR 분위수는 근사이며 실제 모델 학습 및 성능 평가를 하지 않았다.
'''
    path=ROOT/"reports/02_EDA_summary.md";path.write_text(text,encoding="utf-8")
    print("보고서 생성:",path)


if __name__=="__main__":main()
