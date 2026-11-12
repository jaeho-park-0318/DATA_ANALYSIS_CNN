# PHM 2018 Ion Mill Etching Fault Prediction

PHM Society 2018 Data Challenge의 Ion Mill Etching 장비 데이터로 이상 탐지와 고장 조기 예측을 준비하는 Python 프로젝트다. 현재 **초기화와 2단계 EDA**를 완료했으며 모델 학습은 수행하지 않았다.

## 2단계 EDA 결과와 실행

[최종 EDA 보고서](reports/02_EDA_summary.md)에 분석 범위, 실제 수치, 관측 충분성, TTF 정합성 및 미검증 항목을 정리했다.

- [01_data_overview.ipynb](notebooks/01_data_overview.ipynb): 전체 파일 구조, Tool별 행 수·측정 범위·시간 간격, 6개 센서 품질, 공정 분포
- [02_EDA.ipynb](notebooks/02_EDA.ipynb): 대표 01M01의 전체 분포·추이·상관·IQR 후보·공정 조건별 차이
- [03_fault_analysis.ipynb](notebooks/03_fault_analysis.ipynb): 고장 이벤트·중복, 고장 전 구간, 비교 후보, TTF·라벨 검증

세 Notebook 모두 `phm2018` 커널로 실행했으며 코드 셀 21개에 미실행/오류가 없다. 원본은 최대 200,000행 청크와 필요한 열로 처리했다. 전체 train은 82,189,440행, test는 7,198,948행이다. 원본 fault 1,236행 중 동일 장비·시각·유형의 중복 25행을 제외하면 고유 이벤트 1,211건이다.

```powershell
conda activate phm2018
python scripts/build_eda_notebooks.py
python scripts/validate_eda.py
python scripts/write_eda_report.py
```

실제 의존성은 `requirements-eda.txt`, 공통 함수는 `src/eda_common.py`, `src/eda_detail.py`, `src/eda_plots.py`에 있다. 결과·그림·구간 자료는 Git에서 제외한 `reports/local/eda/`에 저장한다. 일치하는 입력 메타데이터와 코드의 로컬 캐시를 재사용하며 강제 재집계는 `overview(force=True)`, `representative(force=True)`를 사용한다.

**해석 제한:** 물리 시간 단위가 확정되지 않아 5/10/30/60분은 `time=초` 가정으로 표시했다. 고장 정지 시각과 실제 이상 시작 시각은 다르다. 결측 TTF를 정상으로 처리하지 않았으며 `[NoWaferID]` 태그와 중복 이벤트의 타깃 정책을 별도로 검토했다. 상세 고장 전 분석은 대표 장비의 두 사례에 한정한다.

아래는 1단계 초기 점검 당시의 기록이며, 전체 데이터 통계와 후속 확인 결과는 위 2단계 보고서를 기준으로 한다.

## 현재 구조

초기에는 `.git/`과 `phm_data_challenge_2018/`만 있었다. `data/raw/`는 존재하지 않는다. 원본은 이동하지 않고 현재 경로에서 읽는다.

```text
PHM2018-Project/
├── .git/
├── .gitignore
├── AGENTS.md
├── README.md
├── requirements-inspection.txt
├── scripts/
│   └── check_environment.py
├── src/
│   └── inspect_data.py
├── reports/
│   ├── data_inventory.md       # 전체 파일명·확장자·크기·용도
│   └── local/                  # Git 제외, 로컬 실행 결과
│       ├── environment.json
│       └── inspection.json     # 대표 5행, dtype, 결측값 포함
└── phm_data_challenge_2018/     # 원본, Git 제외
    ├── test/                   # CSV 5개
    └── train/                  # 센서 CSV 20개
        ├── train_faults/       # fault CSV 20개
        └── train_ttf/          # TTF CSV 20개
```

## 실행 환경과 재실행

2026-11-05 확인: Anaconda `phm2018`, Python **3.11.18**, pandas **3.1.1**, NumPy **2.4.6**. 환경 확인 스크립트의 필수 조건을 모두 통과했다. SciPy 1.17.1, scikit-learn 1.9.1, matplotlib 3.11.2가 설치되어 있다. `jupyter` 메타 패키지는 확인되지 않았으며 현재 점검에 필요하지 않다. 패키지 설치나 환경 변경은 하지 않았다.

프로젝트 루트의 Anaconda Prompt 또는 conda 초기화된 PowerShell에서:

```powershell
conda activate phm2018
python scripts/check_environment.py
python src/inspect_data.py

# 데이터 위치를 직접 지정할 때
python src/inspect_data.py --data-root "C:\path\to\dataset"

# 로컬 결과 저장
New-Item -ItemType Directory -Path reports/local -Force | Out-Null
python scripts/check_environment.py | Set-Content reports/local/environment.json -Encoding utf8
python src/inspect_data.py | Set-Content reports/local/inspection.json -Encoding utf8
```

이번 실행은 활성화 셸에 의존하지 않도록 `C:\Users\fl27\anaconda3\envs\phm2018\python.exe`를 직접 사용했다. 환경 스크립트는 실행 Python, 환경 경로, 패키지 버전과 필수 점검 결과를 JSON으로 출력하고 필수 조건 미충족 시 종료 코드 1을 반환한다. `requirements-inspection.txt`는 이번 점검에 사용한 직접 의존성만 기록한다.

`inspect_data.py`는 `data/raw/`, 기존 데이터 폴더 순으로 CSV가 있는 위치를 선택한다. 전체 파일에는 경로·확장자·파일 크기 등 메타데이터만 조회한다. 유형별 대표 파일의 헤더를 읽고 **센서 CSV 하나만 `pd.read_csv(..., nrows=5)`로 읽는다**. 전체 행 수, 전체 통계나 전체 파일 해시는 계산하지 않는다. 압축 파일은 목록에만 표시하고 자동 해제하지 않는다. 데이터가 직접 지정한 경로에 있어도 읽기 전용으로 다룬다.

## 데이터 점검 결과

모두 압축 해제된 `.csv`이며 총 **65개, 34,191,555,540 bytes (약 31.843 GiB)**다. 전체 파일별 크기는 [파일 목록](reports/data_inventory.md)에 기록했다.

| 위치 / 파일 패턴 | 개수 | 용도와 확인 근거 |
| --- | ---: | --- |
| `train/NN_MXX_DC_train.csv` | 20 | 학습용 센서·공정 시계열. 대표 헤더에 시간, Tool, 공정 식별자와 센서값이 있음 |
| `test/NN_MXX_DC_test.csv` | 5 | 테스트용 센서·공정 시계열. 대표 헤더는 학습 센서 파일과 동일 |
| `train/train_faults/NN_MXX_train_fault_data.csv` | 20 | 고장 이벤트 라벨 후보. 대표 헤더 `time,fault_name,Tool` |
| `train/train_ttf/NN_MXX_DC_train.csv` | 20 | 고장 유형별 Time To Failure 타깃 후보. 센서 파일과 이름이 같으므로 상위 경로로 구분 |

센서 학습·fault·TTF의 파일명 접두어는 각각 `01`~`10`과 `M01`, `M02` 조합이다. 대표 센서 표본의 `Tool` 값은 `01M01`이다. **`M01/M02` 토큰만을 전체 장비 ID로 간주하지 않는다.** 접두어 의미와 실제 장비 경계는 후속 검증이 필요하다. 파일명에 대응하는 라벨이 있어도 행 순서로 연결하지 않는다.

대표 TTF 헤더:

```text
time
TTF_FlowCool Pressure Dropped Below Limit
TTF_Flowcool Pressure Too High Check Flowcool Pump
TTF_Flowcool leak
```

이번에는 fault와 TTF의 데이터 행을 읽지 않았다. 고장 유형 의미, 시간/TTF 단위, 결측 TTF의 의미, 센서와 라벨의 시간 정합성, 전체 파일 스키마 일치 여부는 미확인이다. 분류는 폴더·파일명과 유형별 대표 헤더를 근거로 한다.

## 대표 센서 컬럼과 처음 5행

파일: `phm_data_challenge_2018/train/01_M01_DC_train.csv` (1,107,918,027 bytes). **처음 5행에서 추론한 dtype**은 다음과 같고, 아래 24개 컬럼 모두 해당 5행의 결측값 수는 0이다.

| 컬럼 | 표본 dtype |
| --- | --- |
| time | int64 |
| Tool | str |
| stage | int64 |
| Lot | int64 |
| runnum | int64 |
| recipe | int64 |
| recipe_step | float64 |
| IONGAUGEPRESSURE | float64 |
| ETCHBEAMVOLTAGE | float64 |
| ETCHBEAMCURRENT | float64 |
| ETCHSUPPRESSORVOLTAGE | float64 |
| ETCHSUPPRESSORCURRENT | float64 |
| FLOWCOOLFLOWRATE | float64 |
| FLOWCOOLPRESSURE | float64 |
| ETCHGASCHANNEL1READBACK | float64 |
| ETCHPBNGASREADBACK | float64 |
| FIXTURETILTANGLE | float64 |
| ROTATIONSPEED | float64 |
| ACTUALROTATIONANGLE | float64 |
| FIXTURESHUTTERPOSITION | float64 |
| ETCHSOURCEUSAGE | float64 |
| ETCHAUXSOURCETIMER | float64 |
| ETCHAUX2SOURCETIMER | float64 |
| ACTUALSTEPDURATION | float64 |

표본 `time`은 3283106, 3283110, 3283114, 3283118, 3283122로 증가한다. 이 5행만으로 전체 샘플링 주기나 시간 단위를 확정하지 않는다. 전체 파일에는 다른 dtype이나 결측값이 있을 수 있다. 센서값의 스케일링 여부도 미확인이다. 원본 값이 포함된 표본 결과는 `reports/local/inspection.json`에만 저장하고 Git에서 제외한다.

## 1단계 당시 EDA 계획 (2단계 결과는 위 보고서 참조)

1. 모든 파일의 헤더와 데이터 사전을 확인하고 `Tool`·파일 접두어·시간 및 TTF 단위·고장 유형 의미를 정리한다.
2. 필요한 열과 검증한 dtype을 지정해 파일별 청크로 결측·중복·시간 역전·관측 간격·범위·공정 전환을 집계한다. 청크 경계의 이전 상태를 유지한다. 중간 결과는 원본과 분리한다.
3. fault·TTF와 센서의 시간 키를 검증한다. 중복 시각과 장비 경계를 확인하고 고장 이후 행, 정비 구간, 관측 종료 시 검열을 정의한다. 미래 fault/TTF는 타깃 생성에만 쓴다.
4. 예측 시점에 관측 가능한 특징을 검토한다. 특히 `ACTUALSTEPDURATION` 등은 실제 제공 시점을 확인한 후 사용 여부를 정한다. 과거 방향 rolling window만 사용한다.
5. 장비 ID와 시간 순서에 따른 학습/검증 분할을 설계한다. 동일 이벤트·중첩 윈도가 경계를 넘지 않게 간격을 두고, 전처리는 학습 구간에서만 적합한다. 테스트 세트는 최종 평가까지 보류한다.
6. 예측 구간, 최소 유효 Lead Time, 경보 지속·병합 규칙과 일대일 이벤트 매칭을 정한다. F1-score, Event Recall, 정상 운전 시간당 False Alarm Rate, 탐지 이벤트의 Lead Time 분포를 함께 평가한다. 미탐 이벤트는 Event Recall로 반영한다.

원본 수정, 전체 데이터 메모리 적재, 모델 학습, GitHub push, 원격 업로드 및 외부 서비스로의 원본 전송은 수행하지 않았다. 작업 규칙은 [AGENTS.md](AGENTS.md)에 있다.
