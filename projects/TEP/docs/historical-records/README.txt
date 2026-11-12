# TEP 공정 이상 탐지 및 고장 원인 진단

현재 단계는 **Phase 2 완료**다. Phase 1 이후 사용자 승인을 받아 Harvard 데이터 기반 모델을 학습하고 최종 평가했다.

전체 과정과 결과를 한 번에 읽으려면 [종합 Jupyter Notebook](notebooks/00_TEP_전체_연구_과정과_결과.ipynb)을 연다. 데이터 감사·공정·실험 설계·모델 비교·고장별 성능·지연·SHAP·미지 고장·한계를20개 절로 구성했다. 실행된 표26개와 시각 자료15개가 내장되어 있다. `Run All`은 기존 결과에서 표·그림을 재구성하며 모델을 재학습하지 않는다. 14절의 Fault/run 표시 값을 바꿔 실제 실행 사례를 확인할 수 있다.

최종 결과는 [모델 성능·특징 요약](reports/markdown/12_model_card.md), [전체 연구 보고서](reports/markdown/11_final_research_report.md), [연구 질문별 결론](reports/markdown/13_research_questions.md)에 정리한다. 모델 번들은 `models/phase2/TEP_final_model.zip`, 추론 API는 `src/models/inference.py`다.

## Phase 2 재현 및 추론

현재 PC 환경에서 `./run_phase2.ps1`로 재실행한다. 이미 저장된 캐시와 학습 모델은 재사용하며, 전체 계산을 새로 재현하려면 새 프로젝트 출력 디렉터리에서 원본 파일과 코드를 복사하여 실행한다. 기존 결과를 임의로 지우지 않는다. `./run_phase2.ps1 -ReportsOnly`는 저장된 평가표로 보고서·그림·노트북을 재생성한다.

다른 Python 환경에서는 새 가상환경에 requirements를 설치하고 다음 순서로 실행한다.

```powershell
python src/data/phase2.py
python src/features/phase2.py
python src/models/phase2.py
python src/models/temporal.py
python src/evaluation/run_phase2.py
python tools/phase2_explain.py
python tools/package_phase2.py
python tools/validate_phase2.py
python tools/build_phase2_report.py
python -m unittest discover -s tests -v
python tools/execute_notebooks.py --phase2-only
```

`requirements-lock.txt`는 실제 버전을 기록한다. CUDA 버전 PyTorch `2.5.1+cu121`을 동일하게 설치하려면 공식 CUDA12.1 wheel index가 필요하다. CPU 재현은 CPU용 PyTorch를 설치해 실행 가능하며 GPU/CPU 간 확률과 연산 시간에 작은 차이가 있을 수 있다. 기존 환경의 업그레이드를 요구하지 않는다.

```powershell
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements-lock.txt
python src/models/inference.py --bundle models/phase2/final_model --input one_run.csv --output predictions.csv
```

CSV 입력은 sensor_names.json의52개 열과 연속3분 관측을 사용한다. 원본 metadata 열은 특징에서 제외한다. `simulationRun`이 있으면 한 실행만 입력해야 하며, 서로 다른 실행을 붙이면 오류를 낸다. 첫19개 관측은 warm-up으로 결과를 비워 둔다. 추론 번들에는 주 시계열 모델, 별도 이진 XGBoost, 정규화, 입력 schema, 고정 threshold와 체크섬이 포함된다.

전체 Test 모델 비교는 동일997,500개 시점, 선택 모델의 dense 평가는9,880,500개 시점이다. 학습은 모든 학습 실행에서 고정 간격으로 추출한101,200개 창을 사용했고 모든 행을 fit한 것은 아니다. 모델 선택/임계값은 Validation에서 확정했다. 주 모델은 CNN 단일 seed42이며3개 seed ensemble이 아니다. 별도 이진 탐지기와 결합한 새 cascade 성능은 주장하지 않는다.

Phase 2 보고서07~13과 노트북06~10은 성능·고장별 지연·SHAP·미지 고장 결과 및 한계를 포함한다. Phase 1 보고서는 당시 상태의 기록으로 보존했다. 설정을 바꾸면 캐시/모델 재사용을 거부하도록 구성해, 과거 실험 산출물을 새 설정의 결과로 오인하지 않게 했다.

- Dataset A: `tennessee-eastman-dataset-main/` 사용자 제공 checkout. 실제 Excel 대신 LFS 포인터가 있어 원본 214개를 `data/raw/github/`에 복원했다. SHA256 전부 일치.
- Dataset B: 프로젝트 루트의 `TEP_*.RData` 4개. Harvard V1.0 공식 MD5 전부 일치. 원본을 이동하거나 변경하지 않았다.

먼저 [종합 보고서](reports/markdown/06_phase1_summary.md)를 읽는다. 상세 감사는 `reports/markdown/01~05`와 `reports/tables/`, 고장별 그림은 `reports/figures/`에 있다. 노트북 01~05는 생성된 감사 결과를 읽고 분석 표·그림을 보여준다. 원자료 계산은 `src/data/audit.py`, 보고서·그림 재계산은 `tools/build_phase1.py`가 담당한다.

## 실행

기존 Conda 환경을 변경하지 않았다. 이번 실행은 Codex 번들 Python 3.12 및 프로젝트 `.deps/`의 추가 패키지를 사용했다. 버전은 `reports/sources/environment.json`과 `requirements-lock.txt`에 기록했다. 다른 환경에서는 새 Python 가상환경을 사용한다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# 아래의 python은 새 가상환경의 실행 파일로 바꿀 수 있다.
.\.venv\Scripts\python.exe src/data/audit.py
Get-ChildItem -Filter '*.RData' | ForEach-Object {
    .\.venv\Scripts\python.exe src/data/audit.py $_.Name
}
.\.venv\Scripts\python.exe tools/build_phase1.py
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe tools/execute_notebooks.py
```

현재 PC의 번들 Python 및 `.deps`를 그대로 쓸 경우:

```powershell
.\run_phase1.ps1
# 원자료 감사 재실행 없이 표를 이용해 보고서/그림만 재생성
.\run_phase1.ps1 -SkipAudit
```

전체 RData를 한 파일씩 읽으므로 대형 파일을 읽을 수 있는 메모리가 필요하다. 원본 RData 합계 약 1.40 GB지만 압축 해제된 데이터프레임과 그룹 연산은 더 많은 메모리를 사용한다. 모델용 데이터셋 전환이나 학습은 이 실행에 포함되지 않는다.

## 주요 주의점

A 214개 파일은 168개 고유 측정 시계열로 줄어든다. 고유 시계열 수는 독립 난수 실험 수를 보장하지 않는다. A의 `xmv-*` 헤더는 생성 코드가 `simout`에 붙인 이름으로 측정값을 뜻한다. B의 진짜 조작변수 `xmv_*`와 혼동하지 않는다. Fault 21은 README와 소스 정의가 충돌한다. A 배포 모델은 10h 지연 후 교란을 적용하지만 파일별 생성 로그는 없어 10h 기준 전후/편차 시각을 명목 분석으로 표시했다.

B는 21,000개 실행 궤적이 있지만 같은 simulationRun의 정상/여러 고장은 주입 전 구간을 공유한다. Train/Validation 분할에서 같은 run 번호의 Fault들을 한 그룹에 묶는다. 공식 Train/Test는 seed가 다르다고 설명하며, 관측 선두 20샘플의 분할 간 정확 중복도 없었다. 원래 난수 seed 값은 제공되지 않았다.

A와 B를 하나의 학습 테이블로 병합하지 않았고 Phase 2는 B 단독으로 수행했다. 테스트는 Phase 1 감사와 Phase 2 실행 단위 분할·미래 참조·라벨 경계·전처리 학습 범위를 검증한다. 모델 설명은 실제 물리적 원인 적중률과 구분했다.
