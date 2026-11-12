# TEP 학습 모델

`model.pt`는 검증 기준으로 선택한 21클래스 시계열 모델이며 `binary_detector.joblib`는 별도의 이진 XGBoost 탐지기다. 52개 원 변수, 3분 간격의 연속 샘플을 입력한다. 첫 19개 관측은 warm-up으로 예측을 제공하지 않는다. 같은 실행의 데이터만 한 번에 전달한다. 20개 관측점의 처음과 끝 사이 간격은 57분이다.

```powershell
python inference.py --bundle . --input one_run.csv --output predictions.csv
```

Python API: `from inference import TEPModel; predictor = TEPModel(bundle_path); predictor.predict(dataframe)`.

열 이름은 sensor_names.json을 따른다. 메타데이터 faultNumber/simulationRun/sample은 특징으로 사용하지 않는다. 출력 conditional_fault_id는 고장이라고 가정한 분류이므로 정상 상태에서 물리 원인으로 해석하지 않는다. gated_fault_id의 0은 정상 판정이다. alarm_episode_confirmed는 3개 연속 임계값 초과 이후 경보이며 중간 정상 판정에서 초기화한다. 실제 공정 원인이나 안전 제어 조치로 직접 해석하면 안 된다.

학습 출처: Harvard Dataverse https://doi.org/10.7910/DVN/6C3JR1 V1.0. 화학공정 시뮬레이션으로 검증했다. 다른 운전모드·실제 설비 일반화는 검증하지 않았다. 프로젝트의 최종 연구 보고서에서 고장별 약점·탐지 지연·오경보를 확인한다. 전처리/구조/임계값/모델 버전은 이 번들에 고정되어 있다.
