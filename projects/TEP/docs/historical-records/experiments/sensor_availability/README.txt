# 센서 입력 가용성 후속 실험

기존 TEP 고정 CNN을 재학습하지 않고 Test 입력 가용성만 조작했다. 원본·모델·분할·threshold·기존00~10번 노트북은 보존했다.

먼저 `reports/sensor_availability/final_report.md`를 읽는다. 새 노트북은 순서대로 다음과 같다.

1. `11_availability_baseline.ipynb`: 새 전체 Test 추론의 Baseline 일치, PHM V3 판정 근거, 입력 마스크 검증.
2. `12_availability_degradation.ipynb`: 전체 조건·seed의 실측 결과, 두 분모의 검출률, A/B 원인 분해, 고장별 변화·지연·부분 가용 대조.
3. `13_phm_comparison.ipynb`: PHM의194/91/103/33 원표와 TEP의 가용성 비교, 과제 차이·한계.

표와 그림을 노트북에 내장했으므로 열어서 바로 읽을 수 있다. 노트북의 전체 셀 실행은 저장된 결과를 읽는 작업이며 재학습이나 새 전체 Test 추론을 하지 않는다.

프로젝트 루트 PowerShell에서:

```powershell
# 저장된 실측 결과 검증·보고서·새 노트북만 재구성
.\experiments\sensor_availability\run.ps1 -ResultsOnly

# 전체 절차: Baseline 새 추론 → 감사·마스크 검증 → 파일럿 통과 → 전체 조건 → 보고
.\experiments\sensor_availability\run.ps1
```

실행은 기존 Codex 번들 Python과 프로젝트 `.deps`를 사용한다. 각 `.py`를 기존 환경의 Python으로 순서대로 실행해도 된다. 원본 Test는 약정된20×52 창과 기존 시각 기준을 그대로 사용한다.

주 실험에서는 선택 실행의 sample141~960을 누락시켜 고장 후 유효 입력창을0개로 만든다. 단순 짧은 공백은 이후 입력이 복구되므로 별도 부분 가용 대조로 처리했다. 공백을 입력한 원본 사본을 디스크에 덮어쓰지 않고 논리적 마스크를 사용한다. 남은 실제 입력창·예측의 동일성을 검증했다.

53.1%는 전체 고장 실행 중 평가 불가능 비율이다. 센서 샘플 삭제율이 아니다. 조건부 ROC-AUC/F1이 유지돼도 전체 검출 범위는 줄 수 있다. PHM은 사전 예측, TEP는 사후 탐지여서 두 검출률을 같은 과제의 모델 성능으로 비교하지 않는다.
