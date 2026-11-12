"""실제 학습·예측·검증 결과에서 4단계 한국어 보고서를 작성한다."""
from pathlib import Path
import math
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from src.model_common import OUT,REPORT,configs,read_json,save_json
from src.model_summary import collect,selected_per_family


def table(frame):
    def fmt(v):
        if v is None or (isinstance(v,float) and not np.isfinite(v)):return 'NA'
        if isinstance(v,float):return f'{v:.5g}'
        return str(v).replace('|','/')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def wilson(hits,n):
    if not n:return (None,None)
    z=1.959963984540054;p=hits/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return center-half,center+half


def main():
    objects,metrics=collect();audit=read_json(OUT/'preflight.json');checks=read_json(OUT/'final_checks.json')
    assert checks['passed'] and len(objects)==58
    primary=metrics[(metrics.horizon==600)&(~metrics.identity_diagnostic)]
    evaluated=primary[primary.split=='evaluation'].sort_values(['scheme','name'])
    dist=pd.DataFrame(audit['distributions']);dist=dist[(dist.horizon==600)&(dist.split!='purged')]
    support=pd.DataFrame(audit['events']);support=support[support.horizon==600].groupby(['scheme','split','fault_code']).event_key.nunique().rename('events').reset_index()
    choices=selected_per_family(objects)
    comparison=[];event_rows=[];per_type=[]
    for r in choices:
        m=r['metrics']['evaluation'];lo,hi=wilson(m['detected_events'],m['evaluable_events'])
        comparison.append({'scheme':r['scheme'],'selected_by_validation_AP':r['name'],'validation_AP':r['metrics']['validation']['pr_auc_ap'],
                           **{k:m[k] for k in ('precision','recall','f1','pr_auc_ap','roc_auc','tn','fp','fn','tp')}})
        event_rows.append({'scheme':r['scheme'],'model':r['name'],'events':m['evaluable_events'],'detected':m['detected_events'],
                           'event_recall':m['event_recall'],'Wilson95_low':lo,'Wilson95_high':hi,
                           'active_negative_hours':m['active_negative_exposure_raw']/3600,
                           'false_alarm_episodes':m['active_false_alarm_episodes'],'false_per_active_hour':m['false_alarms_per_active_hour_seconds_interpretation'],
                           'median_lead_raw':m['lead_time_median_raw']})
        for c,v in m['per_type'].items():per_type.append({'scheme':r['scheme'],'model':r['name'],'type':c,**v})
    p_results=[read_json(p) for p in sorted((OUT/'prospective').glob('*/result.json'))]
    case_rows=[read_json(p) for p in sorted((OUT/'figures').glob('*_cases.json'))]
    important=[]
    for result in choices:
        values=read_json(OUT/'experiments'/f"{result['scheme']}_h600"/result['name']/'importance.json')
        for rank,(feature,value) in enumerate(sorted(values.items(),key=lambda pair:pair[1],reverse=True)[:5],1):
            important.append({'scheme':result['scheme'],'model':result['name'],'rank':rank,'feature':feature,'importance':value})
    p_table=[]
    for r in p_results:
        m=r['metrics']
        p_table.append({'scheme':r['scheme'],'selected_RF_XGB':r['selected_name'],**{k:m[k] for k in
            ('all_feature_ready_rows','unknown_label_rows','unknown_row_alarm_fraction','recorded_events','evaluable_events','event_coverage',
             'recorded_events_hit','recorded_recall_with_observation_limit','unknown_alarm_episodes','false_alarm_episodes')}})
    old=read_json(REPORT/'dataset_summary.json');f=pd.DataFrame(old['full_counts']);ready=pd.DataFrame(old['ready_counts'])
    before=f[(f.fault_code=='any')&(f.horizon_raw==600)][['positive','negative']].sum()
    after=ready[(ready.fault_code=='any')&(ready.horizon_raw==600)][['positive','negative']].sum()
    retention=pd.DataFrame({'class':before.index,'all_known_observations':before.values,'retained_candidates':after.values,'retention':after.values/before.values})
    stage=ready[(ready.fault_code=='any')&(ready.horizon_raw==600)].groupby('state_group')[['positive','negative','unknown']].sum().reset_index()
    diagnostics=metrics[(metrics.split=='evaluation')&metrics.identity_diagnostic].sort_values(['scheme','name'])
    extensions=metrics[(metrics.split=='evaluation')&(~metrics.identity_diagnostic)&(metrics.horizon!=600)].sort_values(['horizon','scheme','name'])
    notebooks=read_json(OUT/'notebook_execution.json') if (OUT/'notebook_execution.json').exists() else []
    summary={'checks':checks,'selected_per_family':comparison,'event_metrics':event_rows,'per_type':per_type,
             'prospective':p_table,'notebooks':notebooks,'interpretation':'장비 일반화와 임계값 전달이 불안정하며 조건부 성능을 현장 전체 성능으로 일반화하지 않음'}
    save_json(OUT/'summary.json',summary)
    text=f'''# 4단계 머신러닝 모델 구축·비교 결과

2026-11-05 / 로컬 Conda `phm2018`, Python 3.11.18.

## 실행 결과와 핵심 판단

Dummy·압력 임계값·Random Forest·XGBoost를 실제 학습하고 비교했다. 600 본 실험 16개를 먼저 확인한 뒤 고정 설정으로 300/1800의 32개 확장 실험을 수행했다. Tool/time/Lot/사용량 진단 XGBoost 10개를 추가하여 **58개 실험**을 완료했다. LSTM과 SHAP 상세 분석은 하지 않았다.

기존 원본·코드·Notebook·보고서는 변경하지 않았다. 이전 파일 {checks['prior_files_checked']}개의 내용 해시와 원본 65개 파일의 크기·수정 시각 유지 검사를 통과했다. 새 산출물은 `outputs/stage4/`와 06~08 Notebook, 본 보고서에 분리했다. 외부 업로드·GitHub push는 하지 않았다.

**현재 결과는 현장 적용 근거로 충분하지 않다.** 장비 분할과 시간 분할의 성능 차이가 크고, 더 많은 과거 특징이 일관되게 좋아지지 않는다. 정답/관측이 충분한 구간에 한정한 성능이며, 정답 미확인 구간까지 예측하면 대량의 판정 보류 경보가 발생한다. 낮은 조건부 오경보율을 전체 현장 FAR로 해석하지 않는다.

## 1. 시간 단위 검증

- [PHM Society 공식 설명](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/)은 fault time의 초 예시와 수치 단위 익명화를 제시한다.
- 동일 데이터에 관한 [Huang 등(2018), Remaining Useful Life Estimation for Systems with Abrupt Failures, 5.3절](https://papers.phmsociety.org/index.php/phmconf/article/download/590/phmc_18_590)은 장비 가동 중 대략 4초 기록 간격을 설명한다.
- train 원본 20개 각각 첫 10,000행에서 센서/TTF 시각열을 다시 대조했다. 모두 양의 간격 최빈값이 4다. 동일 다음 이벤트를 가리키는 모든 유한 TTF 인접 쌍에서 TTF 감소량과 실제 time 차분이 일치했다. 전체 기간의 최빈값 4는 기존 전체 청크 집계와도 일치한다.
- 이 근거를 종합하여 **초 단위 해석이 지지된다**고 판단했다. 300/600/1800은 각각 약 5/10/30분으로 해석할 수 있다. 외부 기준 시계 보정은 미검증이고, 기존 raw 컬럼명·3단계 정책 파일은 변경하지 않았다. 본 보고서도 h600raw 등 기존 명칭을 유지한다.
- 센서값의 물리 단위는 복원하지 않았다. 고장 시각은 실제 이상 시작이 아니라 정비를 위한 장비 정지 기록이다.

실제 시간 대조 상세는 `outputs/stage4/preflight.json`의 time_evidence에 있다. 모델 학습 전까지는 300/600/1800을 원시시간 단위로 취급했다.

## 2. 학습 후보 선정과 대표성

전체 82,071,907개 정규화 시각 → 라벨 독립 60 원시단위 bin의 첫 실제 시각 5,777,042개 → 충분한 과거 600 이력·관측 밀도·유한 특징을 갖춘 2,331,977개 → 12개 타깃 중 하나라도 알려진 1,466,770개다. 한 구간의 마지막 값을 과거 시점에 되돌려 붙이지 않았다. 시간 순서와 원본 공백은 보존했고 무작위 행 추출은 하지 않았다. 다만 필터 후 모델 표본은 연속 전체 기록이 아니다.

통합 h600raw에서 양성·음성 보존율:

{table(retention)}

클래스별 의도적 과/소표본화는 없지만, 이력·미래 관측 지속성·TTF 정책의 필터 때문에 보존율이 같지 않다. 따라서 전체 고장률을 대표하는 무편향 표본이 아니다. 라벨 생성의 미래 정보는 입력 특징에 들어가지 않았으나 **표본 포함 여부를 조건화하는 선택 편향**은 남는다. 정답을 임의로 완화하거나 NA를 정상으로 바꾸지 않고, 별도 전향적 추론으로 이 한계를 드러냈다.

양성은 검증된 유한 TTF·이벤트와 예측 구간 내 연속 관측 근거가 있는 경우, 음성은 유한 TTF>H 및 H 전체 추적이 가능한 경우다. 결측·모순·공백·종료·고장 시각은 NA이며 학습 타깃에서 제외했다. 25개 중복 고장과 `[NoWaferID]` 제외 정책은 유지했다.

실제 h600raw 분할 크기:

{table(dist[['scheme','split','positive','negative','unknown']])}

위 unknown은 공통 후보 파일에 있지만 해당 실험에서는 제외한 수다. 주 실험은 확인된 positive+negative 전체를 사용했다. 최대 학습 행렬은 해당 타깃 약 48만 행×83 float32로 만들었으며 원본 82백만 행을 메모리에 올리지 않았다. 원본 보조 변수 조사도 필요한 열을 200,000행 청크로 읽었다.

공정 상태별 h600raw 공통 후보 분포:

{table(stage)}

Stage=0을 정상이라고 간주하지 않았다. 이번 주 비교는 전체 상태를 포함한다. Stage>0 전용 모델 재학습은 아직 미실행이며, 가동 시간 노출 지표는 별도로 계산했다.

## 3. 입력 구성과 실제 선택 특징

| 구성 | 수 | 정의 |
| --- | ---: | --- |
| A | 8 | 현재 센서 6개 + 현재 recipe/stage |
| B | 80 | A + 센서별 과거 60/300 구간 mean/std/min/max/delta_mean/rate_mean |
| C | 83 | B + 과거 Stage=0 비율, Stage 전환 수, Recipe 전환 수 |

전체 후보 83개 중 통계적 특징 선택을 수행한 것은 아니다. 사전 정의한 8/80/83개 입력 구성을 비교했다. 정확한 컬럼명은 각 `result.json`의 details.columns, `preflight.json`의 feature_configs와 06 Notebook에 있다. 현재 및 과거 기록으로만 계산 가능하지만 현장 수집 지연은 별도 검증이 필요하다.

검증으로 선정한 대표 모델의 학습 내장 중요도 상위 5개:

{table(pd.DataFrame(important))}

RF 불순도 감소와 XGBoost gain은 다른 척도이며, 높은 중요도가 고장 원인이나 안정적인 장비 일반화를 뜻하지 않는다.

TTF·Fault label·next_event·미래 sensor·미래 segment_end·Tool/Lot·절대 time·사용량은 주 입력에 없다. 대치·스케일링은 하지 않았다. 유한 특징을 그대로 트리에 입력했으며 recipe/stage ID를 숫자 코드로 다룬 한계가 있다. 원본 익명화/정규화의 적합 범위는 공개 정보만으로 검증할 수 없으며 이번 코드가 새로 전체 데이터 스케일러를 적합하지는 않았다.

## 4. 분할과 이벤트 수·누수 검사

장비 실험은 접두어 01~06 학습(12개), 07~08 검증(4개), 09~10 평가(4개)다. 같은 번호의 M01/M02를 같은 쪽에 두었다. 시간 실험은 장비별 원시 측정 범위의 60/20/20%이며 과거 이력과 최대 1800 미래 타깃 구간이 경계를 넘으면 purged다. 두 실험을 독립 실행했고 행을 섞지 않았다.

양성 anchor가 자기 next_event 키를 가리키는 유형별 평가 가능 이벤트 수:

{table(support)}

장비 평가의 누출은 4건에 불과하여 통계적 불확실성이 크다. 평가 결과를 높이기 위해 분할을 재선정하지 않았다. 향후 장비 그룹 교차검증과 이벤트별 신뢰구간을 확대할 수 있으나 이번에는 고정 홀드아웃을 유지했다.

검사 범위에서 동일 이벤트 분할 공유 0건, 특징 이력/최대 타깃 구간의 시간 범위 중첩 0건이다. 입력 화이트리스트로 미래·정답 컬럼을 차단했다. 임계값·조기 종료는 검증에서만 결정했고 평가에서 바꾸지 않았다. 원본 전처리의 생성 방식과 라벨 이력의 절대적 완전성까지 증명한 것은 아니다.

## 5. 모델 설정

- Dummy: 학습 양성 비율을 출력하는 prior 전략, 고정 임계값 0.5.
- 압력 임계값: 학습 AP로 높은/낮은 방향 선택, 학습 경험적 CDF 점수, 검증 F1 최대 임계값. CDF 점수는 보정 확률이 아니다.
- RF: 64개 트리, 깊이 10, min_samples_leaf=20, max_features=sqrt, max_samples=0.5, balanced_subsample, 시드 42, CPU 4개. 내부 bootstrap만 학습에서 사용했다.
- XGBoost: hist, max_depth=4, learning_rate=0.05, 최대 500회, min_child_weight=10, subsample/colsample=0.8, L1=0.1, L2=5, max_bin=128, 시드 42, CPU 4개. scale_pos_weight=학습 음성/양성. 검증 aucpr에서 30회 Early Stopping, 검증 F1 최대 임계값.
- 광범위한 하이퍼파라미터 탐색·확률 보정은 하지 않았다. 동률 F1에서는 더 높은 임계값을 골랐다. 학습/검증/평가의 클래스 비율을 재표본화하지 않았다.
- 조기 종료·임계값·구성 비교에 같은 검증 세트를 반복 사용한 과적합 가능성이 남는다. 평가 세트는 이 결정에 사용하지 않았다.

소규모 검증은 학습 장비 01M01/01M02의 197,439행(양성 560)에서 RF 8개 트리·XGBoost 10회 실행만 확인했고 평가 데이터는 사용하지 않았다. 이후 위 전체 학습 크기로 확장했다.

## 6. 실제 모델 성능 — 600 구간

PR-AUC는 sklearn Average Precision(AP)로 정의했다. XGBoost 학습 곡선의 aucpr와 계산이 완전히 같지는 않다. Accuracy는 보조 지표로 파일에 저장했다. 정의 불가능한 지표는 NA다.

{table(evaluated[['scheme','name','threshold','precision','recall','f1','pr_auc_ap','roc_auc','tn','fp','fn','tp']])}

RF와 XGBoost의 대표 비교는 **각 계열 내 검증 AP 최대 구성**이다. 평가 F1이 가장 높은 구성을 사후 선택하지 않았다.

{table(pd.DataFrame(comparison))}

현재값 중심 A가 평가에서 더 나은 경우가 있지만, 이를 보고 평가 최적 구성을 확정하지 않았다. 학습 대비 검증/평가 AP 하락과 분할 간 차이는 과적합·분포 이동 가능성을 보여준다. 83개 특징을 모두 쓰는 것이 개선을 보장하지 않았다.

학습·검증·평가 AP 비교:

{table(primary[primary.name.str.startswith(('RF_','XGB_'))].pivot_table(index=['scheme','name'],columns='split',values='pr_auc_ap').reset_index())}

## 7. 고장 이벤트·오경보·Lead Time

한 이벤트는 Tool·시각·유형으로 정의한다. 자기 next_event와 일치하는 양성 anchor가 있어야 평가 가능 이벤트다. 동일 Tool·관측 세그먼트·Stage=0/>0 상태의 연속 양성 경보를 병합하고, 음성·72 초과 간격·상태 변경에서 끊으며 H마다 재발행한다.

한 경보는 H 이내 가장 이른 미래 이벤트 시각에만 대응한다. 동시각의 여러 유형은 함께 기록할 수 있지만 가까운 서로 다른 시각의 고장을 한 경보로 중복 탐지하지 않는다. 같은 이벤트의 여러 경보는 Recall에서 한 번만 센다. 이 규칙을 모든 저장된 점수에 재적용했으며 점수·모델·임계값은 바꾸지 않았다.

오경보는 미매칭 에피소드의 모든 관측이 확인된 음성일 때만 센다. NA가 있으면 미확인 경보다. 정상 노출은 같은 세그먼트의 인접한 확인 음성 anchor 사이 중 72 이하 간격만 합산한다. Stage>0인 양 끝점의 노출로 가동 시간당 비율을 계산한다. 초 해석에 따른 시간 환산이며 전체 운전 시간의 무편향 FAR가 아니다. Lead Time은 탐지된 이벤트에 대한 가장 이른 유효 경보에서 정지 시각까지의 차이다.

{table(pd.DataFrame(event_rows))}

Wilson 구간은 단순 이항 근사이며 장비 내 이벤트 상관을 충분히 반영하지 않는다. 아래 유형별 작은 분모도 함께 확인해야 한다.

{table(pd.DataFrame(per_type))}

## 8. 정답 미확인 구간과 실제 시계열 사례

RF/XGBoost A/B/C 6개 중 검증 AP 최대 모델을 골라, 평가 분할의 모든 feature_ready 행에 추론했다. 미래 정답을 알 수 없어 학습 후보에서 빠진 행도 예측하되 NA를 정상으로 바꾸지 않았다.

{table(pd.DataFrame(p_table))}

미확인 구간의 경보 비율과 에피소드 수가 크다. 이들은 정답이 없어 오경보인지 확정할 수 없다. 전체 기록 대비 도달률에는 센서 공백·이력 부족도 포함되므로 조건부 Event Recall과 분리했다. 현재 결과로 현장 경보 부담이 작다고 주장할 수 없다.

고장 전 예측 그래프는 충분한 관측을 갖춘 **시간상 첫 이벤트**를 선택했다. 정상 오경보 그래프도 시간상 첫 확인 가능 오경보를 골랐다. 탐지 성공률이나 최고 점수로 사례를 고르지 않았다. 특정 사례의 상승/하락을 전체 고장 전조로 일반화하지 않는다. 실제 곡선은 08 Notebook 및 아래 파일에 있다.

{table(pd.DataFrame(case_rows))}

실제 첫 장비 평가 사례에서는 RF A의 점수가 낮은 수준으로 일정하여 임계값을 넘지 못했다. 시간 평가 첫 사례에서는 앞쪽 구간에서 점수가 일시적으로 높아졌으나 임계값보다 낮았고, 고장에 가까운 마지막 600 구간에서도 경보가 발생하지 않았다. 이 두 사례에서 고장 직전의 일관된 점수 상승 패턴은 확인되지 않았다. 전체 탐지된 이벤트의 Lead Time과 이 미탐 사례를 함께 해석해야 한다.

- `outputs/stage4/figures/split_tool_pre_fault_scores.png`
- `outputs/stage4/figures/split_time_pre_fault_scores.png`
- `outputs/stage4/figures/split_tool_false_alarm_case.png`
- `outputs/stage4/figures/split_time_false_alarm_case.png`

## 9. 식별자 영향 진단

C에 Tool, 절대 time, Lot, 사용량(runnum/ETCHSOURCEUSAGE/ETCHAUXSOURCETIMER/ETCHAUX2SOURCETIMER), 전체를 각각 추가했다. 현재 시각의 원본 보조 값만 청크로 연결했고 동시각 Lot 모순은 결측으로 두었다. 식별자 인코더는 학습에만 적합했고 새 범주는 -1이다. 수치 결측은 XGBoost가 처리했다. 주 실험의 선정 대상에서는 제외했다.

{table(diagnostics[['scheme','name','precision','recall','f1','pr_auc_ap','roc_auc','event_recall']])}

랭킹 AP가 좋아져도 고정 검증 임계값의 F1은 매우 낮아질 수 있다. 식별자·생산 순서·시간 위치 기억과 확률 분포 이동 가능성을 구분해야 한다. 장비 홀드아웃의 Tool은 새 범주다. 단일 시드 재적합 변동도 있어 식별자 추가 효과를 인과적으로 단정하지 않는다. 이 결과를 보고 평가 임계값이나 주 특징을 바꾸지 않았다.

## 10. 확장 실험과 다음 단계

600 구간의 실행·분할·양성 존재 검사를 통과한 뒤 같은 설정으로 300/1800을 실행했다. 각 구간의 학습 클래스 비율과 검증 임계값·조기 종료만 다시 계산했다.

{table(extensions[['scheme','horizon','name','precision','recall','f1','pr_auc_ap','roc_auc','event_recall']])}

다음 단계 권장 작업:

1. NA 구간·관측 공백·정비/정지 상태의 정답과 노출 시간을 확보해 전향적 성능을 검증한다. 결측을 정상으로 치환하지 않는다.
2. 장비 그룹 교차검증으로 소수 고장 이벤트의 불확실성을 평가한다. 현재 홀드아웃의 결과를 재선정 기준으로 쓰지 않는다.
3. 학습 내부 검증과 임계값/확률 보정용 검증을 분리하고 이벤트 기준 비용·경보 예산을 함께 최적화한다. 보정·기준은 평가에서 조정하지 않는다.
4. Stage/Recipe의 현장 의미와 수집 지연을 확인하고 공정 조건별 분포 이동·학습에서 보지 못한 상태를 점검한다.
5. 현재값 중심 모델을 강한 비교 기준으로 유지하고, 과거 특징의 이득을 장비별로 반복 검증한다. 식별자·절대 시각·사용량은 별도 진단에 유지한다.
6. 모델이 관측 불가 이벤트를 탐지할 수 없다는 상한을 함께 보고하고, 조건부 F1 외 Event Recall·FAR·Lead Time을 지속 평가한다.

LSTM, SHAP 상세 분석, 현장 확률 보정, Stage>0 전용 모델 재학습, 장비 그룹 교차검증은 미실행이다. 성능 수치는 실행된 저장 점수에서만 생성했다.

## 파일·재실행·Notebook 검증

- `notebooks/06_baseline_models.ipynb`, `07_xgboost_training.ipynb`, `08_model_comparison.ipynb`
- `src/train_baseline.py`, `src/train_xgboost.py`, `src/evaluate_models.py`
- `src/model_common.py`, `model_audit.py`, `model_plots.py`, `model_summary.py`
- `scripts/run_model_experiments.py`, `refresh_model_metrics.py`, `build_model_notebooks.py`, `write_model_report.py`
- `tests/test_models_phase4.py`: 임계값·정의 불가 지표·경보 병합/재발행·연속 이벤트 중복 방지·특징 금지 목록 5개 검사
- `outputs/stage4/experiments/`: 모델 joblib/ubj, 검증·평가 예측 Parquet, 학습 설정·중요도·지표
- `outputs/stage4/figures/`: 요청한 7종 그래프를 장비/시간 분할별로 생성
- `outputs/stage4/prospective/`, `identity_aux/`, `preflight.json`, `metrics.parquet`, `final_checks.json`

```powershell
conda activate phm2018
python scripts/run_model_experiments.py
python scripts/run_model_experiments.py --extensions
python -c "from src.model_audit import identity_experiments, prospective_evaluation; identity_experiments(); prospective_evaluation()"
python scripts/refresh_model_metrics.py
python -c "from src.model_summary import final_checks; final_checks()"
python scripts/build_model_notebooks.py
python scripts/write_model_report.py
```

완료된 실험 결과는 캐시로 재사용한다. 설정·입력·코드 변경 시 같은 결과를 조용히 재사용하지 않도록 새 출력 버전 경로에서 수행해야 한다. 새 모델과 예측값은 outputs 하위 Git 제외 규칙으로 보호했다.

Notebook 실제 실행 상태:

{table(pd.DataFrame(notebooks)) if notebooks else 'Notebook 실행 완료 전 — 미검증'}

실행 환경의 패키지 버전은 `preflight.json`에 기록했다. 초기 결과는 원시시간 명칭으로 유지하고, 시간 해석 보강은 이번 새 보고서에만 반영했다.
'''
    path=ROOT/'reports/04_model_training_summary.md';path.write_text(text,encoding='utf-8')
    print('모델 학습 보고서 저장:',path)


if __name__=='__main__':main()
