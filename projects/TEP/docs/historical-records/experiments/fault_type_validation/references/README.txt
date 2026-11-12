# TEP 공정 이상 예측 및 고장 유형 분류

> 시계열 센서 데이터로 **다음 시점의 공정 상태를 예측**하고, 예측값과 직전 이력을 결합해 **21개 정상·고장 유형을 분류**하는 딥러닝 프로젝트

Tennessee Eastman Process(TEP)의 52개 공정 변수로 설비 이상을 진단합니다. 단순히 현재 상태를 분류하는 데 그치지 않고, LSTM으로 다음 시점의 센서값을 먼저 예측한 뒤 실제 이력과 함께 분류기에 입력하는 2단계 파이프라인을 설계했습니다. 학습부터 평가, 모델 저장, Streamlit 시각화, Docker 실행까지 하나의 흐름으로 구성했습니다.

## 핵심 성과

| 지표 | Test 성능 |
|---|---:|
| Accuracy | **0.78** |
| Macro Precision | **0.89** |
| Macro Recall | **0.85** |
| Macro F1-score | **0.86** |

- 52개 센서·조작변수의 시간 의존성을 반영한 **LSTM 기반 다음 시점 예측**
- 최근 실제 데이터 9개와 예측 데이터 1개를 결합한 **21-class 고장 분류**
- 단일 시점 기반 입력보다 시계열 문맥을 확장해 분류 성능 개선
- 클래스 불균형을 `class_weight`로 완화하고 Macro 지표로 모델을 평가
- 90회 규모의 하이퍼파라미터 탐색 결과를 최종 PyTorch 모델에 반영
- 추론, 클래스별 성능, 센서 연관성, 튜닝 결과를 확인하는 Streamlit 대시보드 구현

![분류 모델 성능](report/ppt_graphs/02_classifier_accuracy_precision_recall_f1.png)

## 문제 정의

TEP 데이터에는 정상 상태와 20개 고장 유형(`faultNumber 0~20`)이 포함되어 있습니다. 이 프로젝트는 최근 10개 시점의 공정 데이터를 바탕으로 다음 시점의 상태를 예측하고, 해당 시점에 발생할 정상·고장 유형을 분류하는 것을 목표로 합니다.

입력 변수는 측정 센서 `xmeas_1~xmeas_41`과 조작 변수 `xmv_1~xmv_11`, 총 52개입니다.

## 모델 파이프라인

```mermaid
flowchart LR
    A["최근 10개 시점<br/>10 × 52"] --> B["LSTM Forecaster"]
    B --> C["다음 시점의<br/>52개 변수 예측"]
    D["최근 실제 데이터<br/>9 × 52"] --> E["결합 및 평탄화<br/>520차원"]
    C --> E
    E --> F["Dense Classifier"]
    F --> G["faultNumber 0~20<br/>Softmax 확률"]
```

### 1. Next-step Forecaster

최근 10개 시점의 52개 변수를 입력받아 다음 시점의 전체 변수를 예측합니다. 이상치의 큰 오차에 지나치게 끌리지 않도록 Huber Loss를 사용했습니다.

| 항목 | 설정 |
|---|---|
| Architecture | 2-layer LSTM + Dense |
| LSTM units | 64, 64 |
| Dense units | 192 |
| Dropout | 0.1 |
| Loss | Huber Loss (`delta=1.0`) |
| Optimizer | AdamW |
| Learning rate | 0.002089 |

### 2. Fault Classifier

Forecaster가 만든 다음 시점의 예측값만 사용하면 과거 문맥이 충분히 보존되지 않는 문제가 있었습니다. 이를 개선하기 위해 직전 9개 실제 행과 예측 행 1개를 결합해 총 520차원의 입력을 구성했습니다.

| 항목 | 설정 |
|---|---|
| Input | 실제 9개 행 + 예측 1개 행 |
| Input dimension | `10 × 52 = 520` |
| Hidden units | 448 → 128 → 448 |
| Dropout | 0.4 → 0.1 → 0.2 |
| Regularization | Batch Normalization, class weights |
| Loss | Weighted Cross Entropy |
| Optimizer | AdamW |
| Learning rate | 0.001537 |

## 실험과 개선 과정

1. **베이스라인 구성**  
   현재/예측 시점 위주의 입력으로 Forecaster와 Classifier를 분리해 학습했습니다.

2. **시계열 문맥 확장**  
   분류 입력을 `직전 1개 실제 행 + 예측 행`에서 `직전 9개 실제 행 + 예측 행`으로 확장했습니다. 예측값의 오차를 과거 실제 관측치가 보완하면서 최종 구조의 성능이 더 안정적으로 나타났습니다.

3. **불균형 대응**  
   정상 및 고장 유형별 데이터 편차를 보완하기 위해 가중 Cross Entropy를 적용하고 Accuracy와 함께 Macro Precision, Recall, F1을 확인했습니다.

4. **프레임워크 전환**  
   초기 TensorFlow/Keras 및 KerasTuner 실험을 보존하면서, 최신 NVIDIA GPU의 Windows 실행 환경을 고려해 최종 학습 파이프라인을 PyTorch로 통합했습니다.

5. **서비스 형태 검증**  
   scaler, label encoder, 모델 가중치, 메타데이터를 분리 저장하고 Streamlit 앱에서 동일한 전처리와 추론 흐름을 재현했습니다.

![Forecaster 예측 결과](report/ppt_graphs/01_forecaster_loss_actual_vs_predicted.png)

## Streamlit 대시보드

대시보드에서는 다음 기능을 제공합니다.

- **Predict**: 선택한 10개 시점으로 다음 상태와 faultNumber별 확률 추론
- **Performance**: 전체 및 클래스별 Precision, Recall, F1 확인
- **Sensor Association**: 고장 유형별 센서 평균 변화와 전체 히트맵 분석
- **Hyperparameters**: 최종 설정과 KerasTuner trial 비교
- **PPT Graphs / Metadata**: 발표용 결과 그래프와 모델 실행 정보 확인

센서 연관성은 각 고장 그룹의 센서 평균이 전체 평균에서 표준편차 단위로 얼마나 벗어났는지를 나타내는 탐색 지표입니다. 모델 설명력 기법인 SHAP이나 attention score와는 구분됩니다.

## 기술 스택

| 구분 | 기술 |
|---|---|
| Language | Python |
| Modeling | PyTorch, TensorFlow/Keras |
| Tuning | KerasTuner Hyperband |
| Data / ML | Pandas, NumPy, scikit-learn |
| Visualization | Matplotlib, Streamlit |
| Deployment | Docker, Docker Compose |
| Hardware | CUDA, NVIDIA GPU |

## 프로젝트 구조

```text
.
├── DL_TEP_Sequence_Prediction_Classification.ipynb  # 실험·학습·평가 노트북
├── streamlit_app.py                                 # 추론 및 분석 대시보드
├── model/                                           # 모델·전처리기·메타데이터
│   ├── best_next_forecaster.pt
│   ├── best_fault_classifier.pt
│   ├── tep_feature_scaler.pkl
│   ├── fault_label_encoder.pkl
│   └── dl_model_metadata_pytorch.json
├── tuner/                                           # 하이퍼파라미터 탐색 결과
├── report/                                          # 발표자료 및 결과 그래프
├── requirements.txt
├── Dockerfile.streamlit
└── docker-compose.streamlit.yml
```

## 실행 방법

### 1. 환경 구성

Python 3.10 환경을 권장합니다.

```bash
git clone <repository-url>
cd dlproject
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 데이터 준비

TEP CSV 파일은 용량 및 배포 조건을 고려해 저장소에 포함하지 않았습니다. 아래 구조로 배치합니다.

```text
dataset/
├── TEP_Faulty_Training.csv
└── TEP_Faulty_Testing.csv
```

애플리케이션은 프로젝트 내부의 `dataset/` 또는 프로젝트와 같은 상위 경로의 `dataset/`을 탐색합니다.

### 3. 대시보드 실행

```bash
streamlit run streamlit_app.py
```

브라우저에서 기본 주소 `http://localhost:8501`로 접속합니다.

### Docker Compose

```bash
docker compose -f docker-compose.streamlit.yml up -d --build
```

Docker 실행 시 대시보드는 `http://localhost:8502`에서 확인할 수 있습니다. `dataset/`과 `model/` 디렉터리는 읽기 전용 볼륨으로 연결됩니다.

## 모델 산출물

| 파일 | 역할 |
|---|---|
| `best_next_forecaster.pt` | 다음 시점의 52개 공정 변수 예측 |
| `best_fault_classifier.pt` | 정상 및 20개 고장 유형 분류 |
| `tep_feature_scaler.pkl` | 학습·추론에 동일하게 적용하는 StandardScaler |
| `fault_label_encoder.pkl` | faultNumber 인코딩/디코딩 |
| `dl_model_metadata_pytorch.json` | 입력 구조, feature 목록, 실행 환경 정보 |

## 한계와 다음 단계

- `faultNumber 0`, `9`, `15`는 다른 클래스보다 상대적으로 분류 성능이 낮아 추가적인 특징 분석이 필요합니다.
- 현재 센서 연관성은 분포 기반 탐색 지표이므로, 추후 SHAP 등 예측 기여도 기반 설명 기능을 추가할 수 있습니다.
- 운영 환경에서는 고장 확률뿐 아니라 오탐 비용, 연속 감지 조건, 임계값을 공정 요구사항에 맞게 검증해야 합니다.
- 실제 스트리밍 입력과 알림 시스템을 연결해 배치 추론 앱을 실시간 모니터링 구조로 확장할 수 있습니다.

## 참고 문서

- [`NOTEBOOK_LOGIC_EXPLAINED.md`](NOTEBOOK_LOGIC_EXPLAINED.md): 노트북의 처리 흐름과 코드 설명
- [`DL_PROJECT_NOTES.md`](DL_PROJECT_NOTES.md): 실험 및 개발 과정 기록
- [`WINDOWS_GPU_PYTORCH_COMMANDS.md`](WINDOWS_GPU_PYTORCH_COMMANDS.md): Windows CUDA 실행 참고

---

이 프로젝트는 **시계열 예측 → 고장 분류 → 결과 해석 → 대시보드 제공**까지 산업 공정 이상 진단의 전체 흐름을 구현하는 데 초점을 맞췄습니다.
