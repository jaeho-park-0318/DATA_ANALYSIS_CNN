# 장비 전용 모델 후속 프로젝트

사용자가 요청한20분 파일럿으로 최대3대까지 선정하되, 사전 최소 조건을 통과한03M02 한 대의 모델을 학습했습니다. 프로젝트의 공통 전처리 캐시를 읽고 모든 추가 산출물은 stage7에 분리 저장했습니다. 원본/V1/V2와 이전 최종 결과는 보존합니다.

- [결과 보고서](../../reports/20_equipment_specialists.md)
- [실행 완료 Notebook](../../notebooks/16_equipment_specialists.ipynb)
- [선정 기준](../../config/stage7_specialists.json), [장비별 선정/제외 표](../../outputs/tables/stage7_eligibility.csv)
- [03M02 모델](../../outputs/stage7/03M02_model.joblib), [전처리](../../outputs/stage7/03M02_preprocessor.joblib)
- [후반 예측](../../outputs/stage7/03M02_later_predictions.parquet), [비교표](../../outputs/tables/stage7_comparison_display.csv)

전용 모델 / 공통 모델+기존 임계값 / 공통 모델+장비별 임계값을 같은 B 입력에서 비교했습니다. 후반은 이전 최종 평가에서 이미 확인한 자료이므로 독립 Holdout 성능이 아닌 후속 연구 결과입니다. 전용 모델이 공통 모델보다 개선됐다는 근거는 확인되지 않았습니다.

결과 확인은 루트에서 아래 명령을 사용합니다. 저장된 후반 예측을 읽고 validation 일부만 재현합니다.

```powershell
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/validate_specialists.py
& 'C:\Users\fl27\anaconda3\envs\phm2018\python.exe' scripts/build_specialist_notebook.py
```

실제 수행 순서는 audit_specialists → train_specialists → evaluate_specialists → validate_specialists → report_specialists → build_specialist_notebook입니다. audit/train/evaluate는 이미 생성된 선정/동결/접근 기록이 있으면 자동 재실행을 막습니다. 추가 연구는 새 실험 버전으로 진행해야 합니다. 이 모델은 원시 CSV 직접 입력이 아니라 인과적으로 계산한 B32 특징과 저장 전처리를 요구하며, 예측 시 선택 반복까지만 사용합니다.
