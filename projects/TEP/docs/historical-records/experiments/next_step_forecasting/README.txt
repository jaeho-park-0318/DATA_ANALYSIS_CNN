# 별도 LSTM 다음 시점 공정 변수 예측

입력은 과거20개 관측×52개 변수, 출력은 다음관측(3분후)의52개 변수다. 정상 Train400개 실행 family만 학습하고, 정상 Validation100개 family로 epoch를 선택했다. 원래 CNN의 분류모델·입력·가중치·threshold는 바꾸지 않았다.

- 결과 노트북: `16_next_step_forecasting.ipynb`
- 보고서: `reports/next_step_forecasting/final_report.md`
- 주 모델: `models/lstm_s42.pt`
- Train 전용 scaler: `models/scaler.joblib`
- 비교 seed: 42,7,2026. 대표42는 Test 이전 고정이다.

```powershell
python experiments/next_step_forecasting/run.py
python experiments/next_step_forecasting/checks.py
python experiments/next_step_forecasting/report.py
python experiments/fault_type_validation/execute_notebooks.py

# 한 실행의 실제 센서 관측에서 다음값 예측
python experiments/next_step_forecasting/predict.py --input one_run.csv --output next_step_predictions.csv
```

52개 원 센서 열이 필요하다. `sample`이 있으면 연속값이어야 하고, 한 번에 하나의 `simulationRun`과 고장 시나리오만 전달한다. 20개보다 적거나 결측·비유한값이 있으면 입력을 거부한다. 미래값 보간이나 실행 경계 연결은 하지 않는다.

정규화 RMSE와 MAE를 Persistence(다음값=현재값)와 함께 읽는다. 이번 전체 Test에서는 RMSE는 낮아졌으나 MAE는 높아졌다. 정상만 학습했으므로 고장구간 예측은 외삽 성격이며, 기존 CNN의 분류성능이 개선됐다는 의미가 아니다. 기존에 이미 사용한 Test의 후속연구 평가다.
