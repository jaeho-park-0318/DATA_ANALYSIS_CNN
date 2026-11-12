# PHM 2018 V3 — 시각화 중심 전처리 진단

## 전체 수행 결과를 한 파일로 보기

[PHM2018 종합 수행 결과 Notebook](notebooks/00_PHM2018_종합_수행결과.ipynb)은 V1/V2/V3의 원본·전처리 진단, 기존 모델 연구, V3 공통 모델/장비 그룹/최종 시간 평가와 장비 전용 파일럿을 통합합니다. 표104개, 그림123개, 원문 보고서24개를 내장했으며 코드18셀 전체를 오류 없이 실행했습니다. 파일 크기는 약50.5MB입니다. 저장 출력으로 바로 열람할 수 있고, 프로젝트 자료가 없는 빈 폴더에서도 내장 결과만으로 재실행했습니다. 새 학습·모델 추론은 수행하지 않습니다.

버전별 특징/라벨/분모와 기존 평가·후속 연구의 차이를 명시합니다. 원본 CSV와 모델 바이너리는 포함하지 않습니다. [실행·무결성 검증](outputs/consolidated/execution_validation.json), [내장 자료 출처](outputs/consolidated/source_manifest.json)를 함께 기록했습니다. 아래 단계별 파일도 계속 보존합니다.

## 추가 완료: 장비 전용 모델 파일럿

최대3대의20분 파일럿으로 설계해 최소 학습/검증 조건을 통과한03M02 한 대를 CUDA 학습했습니다. 공통 모델·장비별 임계값 조정·전용 모델을 비교했고, 전용 모델은 후반 입력 가용 고장3개를 검출하지 못했습니다. 전체 기록27개 중24개는 예측창 입력이 없었습니다. 이번 전용 학습의 개선 근거는 없으며 기존 최종 모델은 그대로 보존합니다. 후반20%는 이미 본 자료를 재사용한 연구 비교로, 독립 Holdout 검증이 아닙니다.

- [장비 전용 추가 프로젝트](experiments/equipment_specialists/README.md)
- [전용 모델 결과 보고서](reports/20_equipment_specialists.md)
- [실행 완료 Notebook](notebooks/16_equipment_specialists.ipynb)
- [세 방식 비교표](outputs/tables/stage7_comparison_display.csv)

## 기존 공통 모델 프로젝트

20대 장비 원본에서 관측 공백·중복/공정 충돌·고장 이벤트·TTF Null·라벨 정책·A/B/C 특징 가용성을 추적하는 로컬 분석 시스템입니다. 전처리, 경보 평가, CUDA 비교 학습, seed 민감도, 보호된 장비 그룹 검증을 완료했습니다. B 공통 모델과 두 임계값을 동결한 뒤 최종 시간 Holdout을 한 번 평가했습니다. 아래 이전 단계의 Holdout 미사용 문구는 당시 상태를 기록한 것입니다.

- [최신 결과: 장비 그룹 검증과 최종 Holdout](reports/19_group_and_holdout.md)
- [실행 완료 Notebook: 최종 결과·동결 검증](notebooks/15_final_holdout.ipynb)
- [최종 결과 표](outputs/tables/stage6_holdout_summary.csv)
- [모델·임계값 동결 기록](outputs/stage6/frozen_manifest.json)
- [Holdout 단일 추론 및 복구 기록](outputs/stage6/holdout_access.json)

## 현재 완료: 그룹 검증 → 동결 → 최종 시간 평가

장비별 개별 모델이 아니라20대의 학습 구간을 합친 공통 B 모델입니다. 별도 그룹 실험은01–06 학습/07–08 검증으로 장비 일반화를 확인했으며 모든 장비의 마지막20%는 제외했습니다. 사전 선택 규칙에 따라 기존 stage2 B 모델을 그대로 동결했습니다. 최종 검출은 낮은/높은 경보 부담 설정에서7/194개,33/194개(입력 가용91개 기준7.7%/36.3%)입니다. 추가 튜닝이나 자동 배포를 하지 않았습니다.

최종 추론736,646행 저장 후 bool 자료형 집계 오류가 발생해 저장 예측으로 집계만 복구했습니다. 전체 추론1회, 추가 추론0회이며 모델·임계값·동결 소스는 그대로입니다. 결과를 재확인하려면 아래의 읽기·검증 명령을 사용하세요. 이전 단계 학습 명령은 과거 실행 기록이며 최종 선택 후 다시 실행할 필요가 없습니다.

```powershell
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/validate_stage6.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/build_final_notebook.py
```

후속 연구에서는 입력 공백과 라벨 이력 완전성을 우선 확인하고 새 평가 데이터를 확보해야 합니다. 이미 확인한 이번 Holdout을 반복 튜닝 후 독립 검증으로 재사용하지 않습니다.

## 이전 단계 산출물

- [최종 결과 보고서](reports/12_visualization_summary.md)
- [전처리·라벨·특징 기준](reports/00_preprocessing_spec.md)
- [시각화 진단](reports/09_visual_diagnostics.md)
- [대표 이벤트 상세 분석](reports/10_event_level_analysis.md)
- [시간 구간 품질 검토](reports/11_segment_quality_review.md)
- [대시보드 실행과 사용법](dashboard/README.md)
- [모델 학습 전 분할·후보·평가 준비](reports/13_pretraining_readiness.md)
- [첫 모델 학습 방향과 조사 근거](reports/14_model_training_direction.md)
- [첫 단계 경보 평가기 결과](reports/15_alarm_evaluation_stage1.md)
- [경보 평가기 실행 Notebook](notebooks/11_alarm_evaluation.ipynb)
- [두 번째 단계 실제 모델 학습 결과](reports/16_stage2_training_results.md)
- [실제 학습 결과·저장 모델 재검증 Notebook](notebooks/12_initial_model_training.ipynb)
- [세 번째 단계 A/B/C 공통 코호트 비교](reports/17_common_cohort_results.md)
- [공통 코호트 6개 모델 실행 Notebook](notebooks/13_common_cohort_training.ipynb)
- [네 번째 단계 제한 튜닝·seed 민감도](reports/18_tuning_seed_results.md)
- [튜닝·임계값 이전 실행 Notebook](notebooks/14_tuning_seed_sensitivity.ipynb)

## 네 번째 단계 재현: 제한 튜닝과 seed 민감도

4단계 당시 기록입니다. seed42에서 A/B 깊이·양성 가중치12설정을 비교하고 설정을 동결한 뒤43/44를 확인했습니다. 새 CUDA 학습18개와 stage3 기준3개 재사용으로 총21개 실행을 보존합니다. 조정 A의 seed42 개선이 다른 seed에 유지되지 않아 기존 A/B를 주 기준으로 유지합니다. 고정 임계값 이전과 seed별 재선택을 분리해 기록했습니다.

```powershell
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/train_stage4.py --phase grid
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/evaluate_stage4.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/select_stage4.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/train_stage4.py --phase sensitivity
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/evaluate_stage4.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/diagnose_stage4.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/report_stage4.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/validate_stage4.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/build_stage4_notebook.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/finalize_delivery.py
```

같은 설정/후보 계약의 완료 학습은 재사용하며, 경보 계산은 입력·평가 코드 해시를 확인한 모델별 캐시를 사용합니다. 설정을 바꾼 새 실험은 별도 디렉터리/버전으로 실행하세요. 기존 stage2/3 결과를 덮어쓰지 않았고 holdout은 계속 사용하지 않습니다.

## 세 번째 단계 재현: A/B/C 동일 코호트 재학습

3단계 기록입니다. 같은 train/validation 행과 라벨로 6개 모델을 새로 학습했습니다. stage2 결과는 별도 보존합니다. 아래 학습 명령은 stage3 모델을 다시 생성합니다.

```powershell
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/train_stage3.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/evaluate_stage3.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/diagnose_stage3.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/report_stage3.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/build_stage3_notebook.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/validate_stage3.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/finalize_delivery.py
```

설정은 `config/stage3_experiment.json`, 실제 모델·행 원장은 `outputs/stage3`, 동일 경보 부담 비교와 장비쌍 대응 차이는 `outputs/tables/stage3_*`에 있습니다. 비교용 경보 상한5에서는 B, 상한10/20에서는 A XGBoost를 후속 주 후보로 유지합니다. C는 이 세 상한에서 앞서지 않았지만 매우 낮은 경보 부담의 비지배 참조 후보로 보존합니다. holdout은 계속 점수화하지 않습니다.

## 두 번째 단계 재현: CUDA 모델 학습과 경보 비교

기존 로컬 환경을 그대로 사용합니다. 아래 학습 명령은 모델 파일을 다시 생성합니다. 결과만 살펴보려면 위 Notebook을 실행하세요.

```powershell
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/audit_alarm_before_training.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/train_stage2.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/evaluate_stage2.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/diagnose_stage2.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/report_stage2.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/build_stage2_notebook.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/validate_stage2.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/finalize_delivery.py
```

학습/검증의 최대 가용 코호트를 사용하며 NA/EXCLUDE까지 전체 가용 입력을 점수화합니다. 검증 구간을 조기 종료와 임계값 비교에 사용하므로 독립 일반화 성능이 아닙니다. GPU smoke와 A/B XGBoost 본학습의 `cuda:0` 실행을 확인했고, 모델·전처리·실제 예측·자원 측정값을 `outputs/stage2`와 `outputs/tables/stage2_*`에 보존했습니다.

## 첫 단계 재현: 전체 입력과 경보 평가

```powershell
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' -m unittest discover -s tests -v
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/profile_local_compute.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/run_alarm_stage1.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/report_alarm_stage1.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/build_alarm_notebook.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/validate_alarm_outputs.py
```

이 단계는 고정 무경보/항상 경보 기준으로 validation의 집계를 검증하며 모델을 학습하지 않습니다. 학습 단계의 CUDA 우선 설정은 `config/local_compute.json`, 탐지 결과는 `outputs/validation/local_compute_profile.json`에 있습니다. GPU 학습 자체는 다음 단계에서 검증합니다.

## 모델 학습 전 추가 준비

기존 진단 후 아래 순서로 실행합니다. 모델과 전처리 추정기는 학습하지 않습니다.

```powershell
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/prepare_pretraining.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/validate_pretraining.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/report_pretraining.py
```

시간 순서 및 장비 그룹의 두 분할, 경계 격리, 분할 밖 정답 근거 제거, X/y/출처 분리를 적용합니다. 기존 전체 이력 진단 Anchor를 직접 모델 입력으로 사용하지 마세요. `src.pretraining.iter_batches`의 명시적 특징 목록을 사용합니다. 상세 계약과 남은 도메인 확인 항목은 위 보고서에 있습니다.

## 실행

PowerShell에서 이 V3 폴더를 작업 경로로 사용합니다. Python은 기존 Conda phm2018 환경입니다.

```powershell
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' -m unittest discover -s tests -v
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' 'scripts/run_pipeline.py'
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' 'scripts/enrich_diagnostics.py'
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' 'scripts/validate_actual.py'
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' -m src.visuals
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' -m src.reporting
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' 'scripts/build_notebooks.py'
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' 'scripts/validate_visuals.py'
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' 'scripts/finalize_delivery.py'
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' 'dashboard/app.py'
```

완료한 장비는 manifest를 확인해 재사용합니다. 원본·설정·알고리즘 의미가 바뀌면 기존 캐시를 조용히 재사용하지 말고 새 버전 디렉터리와 버전을 사용해야 합니다. 미완료 장비는 같은 실행의 파생 부분을 재작성합니다. 원본 CSV는 수정하지 않습니다.

## 산출물과 분모

- `outputs/tables`: 요청된6개 CSV/Parquet, 구간/이벤트 보존/공정/상대시간/제외 사유 보조 표.
- `outputs/figures`: 전체20대 집계 그림과 대표01M01/03M02의 실제 상세 그림.
- `outputs/cases`: 사례별 Anchor 라벨 근거, 공정 상태, 제외 이유, 조회 출처.
- `notebooks/06~10`: 실제 실행 결과가 포함된5개 Notebook. 변수값을 바꿔 구간을 재조회할 수 있습니다.
- `data/cache/source`: 원본 선택 열·정규화 관측·시간 인덱스. 원본 센서는float64, 시간은int64.
- `data/cache/analysis`: 전체 고정 격자 Anchor의44개 특징과 가용성·진단 라벨. 특징만float32.
- `outputs/validation`: 합성 경계 시험과 별도로 실제 집계·그림·Notebook 검증 결과.
- `requirements-local.txt`: 실제 사용한 기존 환경의 버전 기록. 패키지 설치나 업데이트는 하지 않았습니다.
- `outputs/environment.json`, `code_provenance.json`, `source_snapshot_*.json`: 재현 근거. 원본 내용 전체 해시는 계산하지 않았습니다.

원본 행, 고유 시각, Anchor, 고유 이벤트는 다른 분모입니다. 공백 중 Anchor도 보존합니다. A/B/C 최대 코호트와 공통 코호트를 분리합니다. 테스트용 합성 자료는 실제 분석에 섞지 않습니다.

## 제한

시간 단위는raw입니다. 고장 기록은 정비 정지이며 실제 이상 시작과 정비 종료는 알 수 없습니다. 고장 이력 완전성도 확인되지 않았습니다. 연구용·민감도 음성을 검증 정상으로 해석하지 않습니다. 원본의 공식 test는 정답이 없어20대 학습 원본 진단에서 제외했습니다. V1/V2와 원본은 읽기 전용이며 모든 새 파일은V3 안에 있습니다.
