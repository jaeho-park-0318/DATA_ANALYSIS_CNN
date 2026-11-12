# PHM 2018 Project V2

반도체 Ion Mill Etching 장비의 기록된 FlowCool 고장 정지 시점을 조기 예측하기 위한 원본 감사·전처리 프로젝트다. 모든 설명/결과는 한국어로 제공한다. 모델은 학습하지 않는다.

**현재 판단: 모델 학습 시작 불가.** 전체20대 원본 감사 및 대표2대 진단 후보만 완료했다. 시간 단위, 고장 이력의 유효범위, 정비 종료가 미확정이므로 전체20대 확정 데이터 생성은 중단했다. 진단 Parquet는 학습용으로 승인되지 않았다.

- [전처리 명세](reports/00_preprocessing_spec.md)
- [원본 감사](reports/01_raw_data_audit.md)
- [라벨·음성 없는 7대 검증](reports/02_label_validation.md)
- [데이터셋 검증·최종 판단](reports/03_dataset_validation.md)

## 경로와 보호 정책

작업 위치는 이 V2 폴더다. V1은 읽기 전용이며 파일 수정·삭제·원본 복사·외부 업로드·GitHub push를 수행하지 않는다. V1의 ml_v1 데이터는 입력으로 사용하지 않는다.
요청한 V1/data/raw가 없어 실제 원본인 V1/phm_data_challenge_2018을 config/preprocessing.json에서 참조한다. 설정을 바꾸면 감사 캐시는 무효화된다.

```text
config/                    원본 경로·진단 설정
src/                       감사·특징·후보 라벨·분할·검증·보고서 코드
tests/                     unittest 경계/누수/배치 검증
notebooks/                 한국어 설명과 실행 결과를 담은 Notebook 4개
reports/                   명세·감사·라벨·검증 보고서 및 figures/
outputs/                   감사 통계, SHA256, 환경/메모리, 분할/검증 JSON
data/processed/
  diagnostic_pilot/        대표2대 후보 Parquet 및 manifest/schema
```

## 로컬 실행

기존 Conda phm2018을 재사용한다. 환경·드라이버 설치/변경은 필요 없다. PowerShell에서 V2 루트를 작업 디렉터리로 지정한다.

```powershell
Set-Location -LiteralPath 'C:\Users\fl27\Documents\ChatGPT\PHM2018-Project-V2'
$py = 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
& $py -m unittest discover -s tests -v
& $py -u -m src.run_pipeline
& $py -u -m src.check_reproducibility
& $py -m src.report
& $py -u -m src.execute_notebooks
& $py -u -m src.final_checks
& $py -m src.report
```

`--audit-only`는 20대 감사만 수행하고, `--pilot-only`는 2대 감사/진단 특징/검증만 수행한다. 기본 실행은 2대 검증 후 20대 원본 감사까지 수행하며 전체20대 특징은 만들지 않는다.
`python -m src.run_pipeline --release`는 확정 학습 데이터 생성이 차단되어 비정상 종료한다. config의 승인 bool을 변경해도 정식 배포 경로는 열리지 않는다. 실제 단위변환·이력범위·정비표 입력과 검증 구현이 먼저 필요하다.

장비별 원본 감사는 코드/설정 해시 및 원본 크기/mtime가 같을 때 재사용한다. 변경되지 않은 크기/mtime로 내용만 바뀐 경우는 캐시가 탐지하지 못한다. 원본을 새로 전수 감사하려면 해당 outputs/audit/<tool>.json을 별도 보관한 후 다시 실행한다.
재현성 검사는 대표2대를 다시 생성하여 전후 Parquet SHA256을 비교하고 정식 배포 명령의 차단도 검사한다. 마지막 보고서 갱신은 실행 결과를 반영하며 코드/설명이 같은 Notebook의 저장된 출력을 유지한다.
대표2대 진단 특징은 항상 재생성하며 완성 후 기존 V2 파생 Parquet를 교체한다. V1 경로에는 쓰지 않는다. 중단 시 완료된 감사 장비는 재사용하며 진단 특징은 장비 단위 재실행이다.

## 결과를 읽을 때

현재 모든 숫자 창은 원시시간 단위다. 초/분이 검증되었다고 해석하지 않는다. `candidate_*`는 0/1/null이며 60미만 제외는 별도 bool이다. `approved_for_training`은 모두 false다. TTF/Fault/절대시각 등 메타데이터를 모델 입력에 넣지 않는다.
A/B/C 특징 화이트리스트는 data/processed/diagnostic_pilot/feature_schema.json에 있으며 8/32/44개다. 빈 격자, Stage=0, 과거이력 부족을 조용히 삭제하지 않는다. 그래프의 평활화나 음성 후보 증가를 고장 예측 성능으로 해석하지 않는다.
V1에서 이미 확인한 장비/기간은 새 독립 미사용 평가 세트가 아니다. 향후 모델 학습에서도 Early Stopping, 임계값 선정, 평가 역할을 분리한다.

## 자원

Polars 4스레드, 장비 순차 처리, 600000 원시단위 시간 블록+900 과거 겹침으로 Rolling을 계산한다. RAM/VRAM 감시는 2초 간격이며 순간 피크는 놓칠 수 있다. GPU는 학습 없이 현황만 조회한다. 미래 XGBoost는 CUDA 검증 후 더 빠를 때 hist/cuda를 선택한다.
