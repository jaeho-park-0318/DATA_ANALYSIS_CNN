# 고정 CNN의 고장 유형 분류 검증

기존 CNN의21클래스 출력·가중치·입력20×52·경보threshold를 보존했다. 재학습하지 않고 저장된 Test 확률로 유형별 성능과 진단 시각을 검증했다.

- 결과 노트북: `15_fault_type_validation.ipynb`
- 보고서: `reports/fault_type_validation/final_report.md`
- 유형별 Precision/Recall/F1: `reports/fault_type_validation/tables/per_class_metrics.csv`
- 검출과 올바른 유형 진단 시간: `per_fault_diagnosis_time.csv`, `run_type_results.csv`
- 입력 가용성별 두 종류 성공률: `availability_detection_vs_diagnosis.csv`

정상 포함21클래스 argmax, 기존경보threshold 적용21상태, 실제고장구간 조건부20종 분류를 구분했다. 진단확정은 같은양의예측ID가3개연속일 때로 정의했고, 첫진단오류와 나중의올바른진단을 별도로 기록했다. 테스트자료를추가분석한연구이며 신규독립Test나 물리적원인적중률로 주장하지 않는다.

프로젝트 Python 환경에서:

```powershell
python experiments/fault_type_validation/evaluate.py
python experiments/fault_type_validation/checks.py
python experiments/fault_type_validation/report.py
python experiments/fault_type_validation/execute_notebooks.py
```

기존 `.deps`를 쓰려면 프로젝트 루트에서 `$env:PYTHONPATH = (Join-Path (Get-Location) '.deps')`를 설정한다. `PYTHONDONTWRITEBYTECODE=1`을 유지해 기존소스의캐시도 쓰지 않는다.

참고 GitHub 공개문서는 `references/`에 저장했다. 내려받은 코드는 실행하지 않았다. 별도 LSTM 회귀 예측기는 `experiments/next_step_forecasting/`에 있고, 그 예측값을 이 CNN에 넣지 않는다.
