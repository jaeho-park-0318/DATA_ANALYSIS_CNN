# TimesFM–TEP 독립 시계열 예측 실험

**대표 표본 비교를 우선 수행한다. 전체 987만 Test 시점의 TimesFM 평가는 미실행이다.**
기존 CNN/LSTM/GRU/XGBoost, 원본 데이터, 기존 보고서는 수정하지 않는다. TimesFM은 추가 학습 없이 공식 사전학습 가중치를 사용한다.

## 결과 확인

- `TimesFM_TEP_실험.ipynb`: 실제 실행 출력과 내장 그래프. 열람 시 별도 환경 없이 기존 출력 확인 가능.
- `experiment_summary.md`: 한국어 실험 설계·수치·해석·미실행 단계.
- `model_comparison.csv`: 세 모델의 동일 표본 전체/정상/고장 이후 RMSE·MAE. 값은 정상 Train 표준편차 단위.
- `variable_metrics.csv`: 52변수별 RMSE·MAE, 정규화/원 스케일 모두 제공. 물리 단위는 원본 B에서 독립 검증되지 않음.
- `figures/`: 300dpi PNG 6종. 소규모 표본임을 제목에 표시.
- `predictions/`: 모든 표본의 동일 `(fault,run,target_sample)` 정답과 세 모델 예측을 NPZ로 보존.
- `pilot_*validation.json`, `result_validation.json`, `preservation_check.json`: 실제 실행·인과성·CSV 재계산·기존 파일 보존 검사.
- `sample/`: 최초 run1×21종, 고정 stride10 진단 실행(주 결과 아님).
- `throughput_benchmark.csv`: 실제 GPU 배치 크기 탐색. BF16은 속도 탐색만, 주 비교는 FP32.

## 비교 조건

Harvard TEP 3분 간격, XMEAS41+XMV11=52변수. 과거20개 관측으로 다음1개 관측을 예측한다. 정상 Training400실행으로 학습한 기존 StandardScaler와 LSTM seed42를 그대로 사용한다. 정상 Validation100실행, 공식 Test500실행 family를 유지한다.

대표 표본은 seed20261112로 Test run50/214/435/489/497을 선택하고 fault0~20 모두 포함한다. trajectory별940개 유효 target을 길이10의94개 시간 bin으로 나누어 각 bin 내부에서 한 시점을 무작위 선택한다. 총105trajectory·9,870개 target(전체 dense Test의0.1%). 고장 시나리오에서 sample>160을 고장 이후로 정의한다. 정상 전용 시나리오는 전 구간 정상이다.

TimesFM3 공식 입력은 `(52,20)`이며 공식 Evaluator가32+20변수 chunk(마지막 chunk 반복 패딩)를 처리한다. 출력52변수를 모두 평가하되 한 번에52개 전체 공동 attention은 아니다. 미래 공변량·추가 학습·XReg는 사용하지 않는다. 입력20개 외에 실제 문맥을 늘리지 않는다. TimesFM 내부 context 기반 정규화·mask 패딩은 공식 모델 구조다.

## 독립 환경에서 재실행

PowerShell:

```powershell
cd C:\Users\fl27\Documents\ChatGPT\TEP\timesfm_experiment
.\run_experiment.ps1

# 다른 컴퓨터에서 설치가 필요할 때 (Python 3.12 실행 파일 경로)
.\run_experiment.ps1 -Install -PythonExe 'C:\path\to\python.exe'

# 전체 Test 평가: 별도 full_evaluation 폴더에 출력, 수일 소요 가능
.\run_experiment.ps1 -Full
```

환경은 `.venv`에만 설치하며 기존 Anaconda·`.deps`를 참조하지 않는다. `requirements.txt`는 이 별도 환경의 고정 버전 목록이다. CUDA12.1 PyTorch2.5.1 wheel용 추가 index URL을 포함한다. 패키지 설치·가중치 다운로드에 인터넷이 필요하지만 공정 데이터는 외부로 전송하지 않는다. 이미 다운로드된 모델은 offline mode로 실행한다.

직접 실행:

```powershell
.\.venv\Scripts\python.exe experiment.py --mode representative --batch 128 --precision fp32 --runs-per-fault 5
.\.venv\Scripts\python.exe finalize.py
```

진행 checkpoint는 `progress.npz`(집계 및 설정을 원자적 저장), `progress.json`(사람이 읽는 상태)이다. 동일 설정으로 재실행하면 완료된 trajectory를 건너뛴다. 설정이 다르면 섞지 않고 오류를 낸다. 다른 구성은 `--tag new_folder`를 지정한다. 전체 평가도 동일하게 재개할 수 있다. 부분 진행 CSV는 status=running, 표본 완료는 sample_complete, 전체 완료는 complete로 구분한다.

## 버전과 출처

- [공식 TimesFM](https://github.com/google-research/timesfm), PyPI timesfm3.0.2 / 모델세대3.0.
- [공식 가중치](https://huggingface.co/google/timesfm-3.0-pytorch), revision `43046b85ec22d584a13f8098c2ed39c889e129c2`.
- 패키지·모델 카드·라이선스는 `sources/`, `checkpoint/`에 보존한다. 3.0 가중치의 비상업·비생산 라이선스는 소스 Apache-2.0와 별도다.
- `environment.json`, `requirements.txt`, `artifact_manifest.json`에 실행 환경과 결과 해시를 기록한다.

기존 전체 Test LSTM 지표와 표본 TimesFM 지표를 섞어 순위를 만들지 않는다. CNN 분류 Accuracy와 회귀 RMSE/MAE는 서로 다른 과제다. 표본5family로 전체500family의 우열이나 통계적 유의성을 확정하지 않는다.
