"""감사 JSON/진단 Parquet를 한국어 보고서와 Notebook으로 재현한다."""
from __future__ import annotations

import json
import os

from src.common import *
from src.features import feature_names
from src.splits import tool_roles

SOURCE = "https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows]) + "\n"


def write(name, text):
    (ROOT / "reports" / name).write_text(text.strip() + "\n", encoding="utf-8")


def main():
    audits = read("outputs/audit_summary.json")
    if len(audits) != 20:
        raise RuntimeError("20대 원본 감사 완료 후 보고서를 생성하세요.")
    pilots = read("outputs/pilot_summary.json")
    checks = read("outputs/validation.json")
    env = read("outputs/environment.json")
    resource = read("outputs/resources.json")
    run_status = read("outputs/run_status.json")
    reproducibility = read("outputs/reproducibility.json") if (OUT / "reproducibility.json").exists() else {}
    notebook_checks = read("outputs/notebook_validation.json") if (OUT / "notebook_validation.json").exists() else []
    final_integrity = read("outputs/final_integrity.json") if (OUT / "final_integrity.json").exists() else {}
    save(OUT / "provenance.json", {
        "config_sha256": digest(ROOT / "config/preprocessing.json"),
        "source_code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "src").glob("*.py"))},
        "raw_sha256": {path: sha for a in audits for path, sha in a["sha256"].items()},
        "diagnostic_sha256": {p['tool']: digest(DEST / f"{p['tool']}.parquet") for p in pilots},
        "training_release": "BLOCKED",
    })
    total = lambda key: sum(a[key] for a in audits)
    sums = {}
    for target in audits[0]["distributions_raw"]:
        sums[target] = {key: sum(a["distributions_raw"][target][key] for a in audits) for key in ["positive", "negative", "NA", "excluded_lt60"]}
    save(OUT / "label_distribution_raw.json", sums)
    split_plan = {a["tool"]: tool_roles(a["tool"]) for a in audits}
    save(OUT / "split_plan.json", split_plan)
    write("00_preprocessing_spec.md", f"""
# V2 전처리 명세 — 2026-11-05

## 목표와 현재 승인 상태
목표는 기록된 FlowCool 정지까지 60~600초의 통합 이진 예측이며 300/1800초도 보존한다. 모델 학습은 수행하지 않는다.
**현재 상태: DIAGNOSTIC_COMPLETE_RELEASE_BLOCKED. 확정 학습 데이터 0행.**
대표 장비 01M01(기본 사례), 03M02(태그/TTF 충돌 사례)를 진단용으로 처리한다. 20대는 원본 감사만 수행하며 전체 특징·확정 라벨 생성을 중단한다.

## 원본 및 환경
요청한 V1/data/raw는 존재하지 않는다. 설정 파일의 raw_root는 실재하는 `{RAW.as_posix()}`이다. 원본을 복사하거나 링크하지 않았다.
V1의 ml_v1 파일은 데이터 입력으로 사용하지 않는다. V1의 코드/보고서 정책만 참고하고, 수치는 원본에서 다시 계산한다.
`phm2018` Conda의 절대 실행 경로, Windows 로컬, Polars 4스레드, 장비 순차 처리, 최대 RSS 12GiB/가용 RAM 최소 4GiB 정책이다.
GPU는 감시만 한다. XGBoost 학습·CUDA 벤치마크는 수행하지 않는다. 이후 학습에서 CUDA 검증 후 더 빠를 때 hist/cuda를 선택한다.

## 단위 및 정답 근거
[공식 PHM Society 설명]({SOURCE})을 2026-11-05 확인했다. 익명화로 단위가 공개되지 않았으며 time의 초 표기는 예시다.
고장 기록은 작업자가 정비를 위해 정지한 시각이고 실제 이상 시작은 미제공이다. TTF Null의 특정 horizon 값과 정비 종료 시각은 미제공이다.
따라서 현 결과의 60/300/600/900/1800은 **원시시간 단위**다. 초/분 성능을 주장하지 않는다.
물리 초 변환, Fault 유효 이력 범위, 정비 종료를 검증해야 요청한 학습 데이터가 된다.

## 후보 라벨 정책
- 개별 유형: 다음 고장과 전체 원본 TTF가 일치하는 이력에서 60≤d≤H는 1, d>H는 0, 미래 이력이 없으면 NA.
- 통합: 유형 중 하나라도 1이면 1, 세 유형 모두 0이면 0, 나머지는 NA. 단 어떤 고장이든 d<60이면 별도 제외가 우선한다.
- 고장 시점 d=0도 제외한다. 고장 후 정비 종료는 확정할 수 없어 maintenance_status에 남기고 **전역 배포 차단**한다. 정비 구간 제거가 완료되었다고 주장하지 않는다.
- Fault 중복은 Tool/time/원문 유형 기준 집계에서만 제거하며 원문 이벤트와 원본 행 수를 보존한다.
- [NoWaferID]는 포함/제외 두 이력이 달라지는 범위를 NA로 격리한다. 태그 제외가 정답이라고 확정하지 않는다.
- TTF Null은 자동으로 0/NA로 확정하지 않는다. 보수적 진단에서는 근거 부족으로 NA, 별도 민감도 분석에서는 센서 종료까지 Fault 이력이 완전하다는 가설을 적용한 결과를 나란히 기록한다.
- 입력의 공백/이력 부족과 정답 관측 가능성을 분리한다. 미래 이벤트가 알려져 있으면 센서 공백 때문에 그 이벤트 자체를 지우지 않는다.
- raw 분포는 원본 행 가중, pilot 분포는 고정 격자 Anchor 기준으로 서로 분모가 다르다.

## 특징 및 Anchor
동시각 센서는 진단용 평균/동시각 원본 행 수를 남기며 상태 상충은 표시한다. 센서가 다른 중복이 있어 평균의 물리적 타당성은 배포 전 검증해야 한다.
과거 직전 관측과 같은 세그먼트에서 Stage/Recipe 변화 여부를 표시한다. Stage0/비0/관측된 공정 전환/미관측은 구분하되 가동·정지·정비의 물리적 의미는 확정하지 않는다.
공백>12 원시단위, 상태 상충에서 세그먼트를 끊는다. 12는 확인된 물리 임계값이 아니라 기존 정책과 비교하는 진단 설정이다.
A={len(feature_names()['A'])}개 현재 센서6+recipe/stage, B={len(feature_names()['B'])}개(과거300 평균/모표준편차/최소/최대 추가), C={len(feature_names()['C'])}개(과거900 OLS기울기/관측 처음-마지막 변화량 추가).
기울기 단위는 센서값/원시시간이며 변화율은 백분율이 아니다. 창은 (t-w,t], 유효 관측만 사용한다. 보간·미래값·밀도 필터에 의한 삭제는 없다.
현재값은 grid 시점 이전 최신 센서로 연결하고 지연>12이면 결측 처리한다. Rolling은 **Anchor 다운샘플링 전 전체 원본 관측**을 사용해 정확한 grid 창에서 계산한다.
600000 원시단위 블록에 900 과거 단위와 직전 관측을 겹쳐 메모리를 제한한다. 배치/일괄 동등성 테스트를 수행한다.
Anchor는 0 원점의 60 간격 고정 격자이며 고장/라벨에 의존하지 않는다. 공백의 Anchor도 삭제하지 않고 current_observed=false를 남긴다.
관측 수, 각 창 출처 최소/최대 시각, 세그먼트 시작, 이력 길이, 상태/공백 사유를 메타데이터에 남긴다. 이력 길이900 충족은 센서값의 밀도/완전성 보장이 아니다.
학습 입력은 feature_schema.json의 A/B/C 화이트리스트만 허용한다. Tool/Lot/절대시간/TTF/Fault/라벨/미래 메타데이터는 입력에서 제외한다.

## 분할
5개 장비 그룹: 번호01~02,03~04,05~06,07~08,09~10. 같은 번호의 M01/M02는 항상 함께 배치한다.
각 Fold는 평가1그룹, 임계값1그룹, Early Stopping1그룹, 학습2그룹이다. outputs/split_plan.json에 20대 배치가 있다.
별도 시간 분할은 장비 시간 범위 50/20/15/15%; t-900~t+1800 전체가 역할 내부에 있을 때만 사용하고 경계는 purged로 보존한다.
평가/검증의 언더샘플링은 없다. V1에서 이미 본 장비·기간을 미사용 독립 평가로 주장하지 않는다.

## 캐시·재현·차단
원본 감사 캐시는 코드/설정 SHA256 및 원본 크기/mtime로 무효화한다. 최초 감사의 원본 SHA256은 장비별 JSON에 남긴다.
메타데이터가 동일한 내용 변조까지 감지하려면 캐시를 재사용하지 않고 다시 감사해야 한다. 이는 보안 검증이 아니다.
특징은 원본에서 재생성하고 임시 파일 완성 후 Parquet를 원자적으로 교체한다. 장비별 manifest를 남긴다.
정식 배포 명령은 항상 차단한다. config의 bool만 바꾸어 승인할 수 없고 단위 변환/정비표/완전성 입력과 검증 구현이 필요하다.
""")
    rows = [[a["tool"], f"{a['raw_rows']:,}", f"{a['time_min']}~{a['time_max']}", a["duplicate_extra_rows"],
             a["duplicate_sensor_or_state_conflict_times"], a["long_gap_count"], sum(a["null_counts_all_columns"].values())] for a in audits]
    write("01_raw_data_audit.md", f"""
# 원본 데이터 감사

## 범위·무결성
학습 장비20대 Sensor/Fault/TTF **60개 파일**을 원본에서 전수 감사했다. 공식 test 5개 파일은 정답 없는 평가용이므로 이번 내용 감사에 포함하지 않았다.
센서 {total('raw_rows'):,}행, 고유 시각 {total('unique_times'):,}개, 중복 추가 행 {total('duplicate_extra_rows'):,}개다. 시간 역전 {total('time_reversals')}건이다.
센서/TTF의 행 수·원래 시각열이 장비별로 일치했다. 행순서만 가정한 병합은 하지 않았다.
전체 24개 열의 Null 수, 대상6센서 비유한 수, Stage/Recipe 분포, 간격 빈도 및 상위20개 공백은 outputs/audit/에 저장했다.
원본을 임의 삭제하지 않았다. 실제 단위 미확정으로 표의 범위/공백은 원시시간이다. 모든 장비의 양수 간격 중앙값은 {sorted(set(a['interval_quantiles'][1] for a in audits))}이다.
원본60파일의 SHA256을 저장했다. 실행 전후 V1 기존 파일 {run_status['v1_files_checked']:,}개의 크기/mtime 변화는 {len(run_status['v1_metadata_changes'])}개다. 기존 V1 전체를 전후 내용 해시로 비교한 검사는 아니다.
감사 후 원본 내용 SHA256 재검사 파일 수는 {final_integrity.get('raw_sha256_rechecked_files', '미실행')}개이며 결과는 {final_integrity.get('status', '미실행')}이다. 상세는 outputs/final_integrity.json에 있다.

{md_table(['장비','센서행','시간범위(raw)','중복추가행','중복센서/상태상충 시각','공백>12','전체열 Null 합'], rows)}

## 이벤트와 TTF
Fault 원본 {total('fault_raw_rows'):,}행 → 원문 기준 고유 {total('fault_unique_events'):,}건, 중복 {total('fault_duplicate_extra_rows')}행이다. 태그 이벤트는 {total('tagged_unique_events')}건이다.
태그 제외 유형별 고유 이벤트는 {dict((c,sum(a['event_counts'][c] for a in audits)) for c in FAULTS)}이다.
태그 제외 기준 TTF의 값 및 Null 위치와 다음 동종 이벤트까지 차이의 불일치 합계는 {sum(v['exclude_tag_mismatch'] for a in audits for v in a['ttf'].values())}행이다.
태그 포함 기준 불일치는 {sum(v['include_tag_mismatch'] for a in audits for v in a['ttf'].values()):,}행이다. 이 차이가 태그 제외의 물리적 정당성을 보장하지는 않는다.
동시각 센서값이 다른 경우가 {total('duplicate_sensor_or_state_conflict_times'):,}시각, Stage/Recipe 상충은 {total('duplicate_state_conflict_times')}시각이다. 평균 집계는 진단 가정이며 원본 센서 파형과 동시각 기록의 의미 확인이 필요하다.

## V1 축소 원인
V1 reports/03_dataset_summary.md는 82,189,440 원본행→82,071,907 시각→5,777,042 Anchor→이력충족2,331,977행→타깃 하나 이상 알려진1,466,770행을 보고했다.
이는 중복 집계, 비어 있는 bin이 없는 관측 기반 Anchor, 긴 공백을 끊는 Rolling, 과거이력/밀도 제한, 유한 TTF와 연속 미래 관측을 동시에 요구한 라벨 정책이 누적된 결과다.
V2는 관측/특징 준비 상태와 후보 라벨을 분리하고 원본행 가중 감사 및 빈 격자를 포함한 진단 Anchor를 보존한다. 클래스 균형이 좋아진다는 이유로 미검증 음성 정책을 채택하지 않았다.
""")
    labelrows = [[target, *[f"{v[k]:,}" for k in ['positive','negative','NA','excluded_lt60']]] for target,v in sums.items()]
    sevenrows = []
    for a in audits:
        if a["tool"] not in SEVEN:
            continue
        sevenrows.append([a['tool'], "/".join(str(a['event_counts'][c]) for c in FAULTS),
                          "/".join(f"{a['ttf'][c]['null']/a['raw_rows']:.1%}" for c in FAULTS),
                          a['v1_policy_recomputed']['h1800']['negative'], a['distributions_raw']['any_h600']['negative'],
                          a['sensitivity_any']['h600']['negative']])
    old = {key: sum(a['v1_policy_recomputed']['h600'][key] for a in audits) for key in ['positive','negative','NA','changed_raw_rows']}
    write("02_label_validation.md", f"""
# 라벨 검증 — 확정 라벨 배포 중단

## 결론과 경계
시간 단위·이력 완전성·정비 종료가 미확정이므로 최종 60초~10분 라벨은 승인하지 않는다. 아래는 **원시시간 단위 후보**다.
60≤d≤H를 양성, d>H와 유형별 유한 TTF/Fault 정합 근거가 있는 경우를 음성 후보로 계산했다. Null만으로 음성을 만들지 않았다.
d=0 및 0<d<60은 별도 제외다. 통합에서는 어떤 고장이든 60 미만이면 다른 유형의 양성보다 제외가 우선한다.
태그 포함/제외 차이 구간은 NA로 격리한다. 정비 후 구간은 아직 판정할 수 없으므로 이 표에 포함된 후보 전부 학습 승인이 없다.

## 20대 원본 행 기준 분포
각 행의 양성+음성+NA+제외 합계는 {total('raw_rows'):,}이다. 진단 Anchor 분포와 혼동하지 않는다.

{md_table(['유형_범위(raw)','양성후보','음성후보','NA','60미만제외'], labelrows)}

## 음성 없었던 7대
V1의 음성0은 선택된 학습 후보의 h1800 통합 기준이었다. 아래 V1정책 재계산은 **전체 원본행** 기준이므로 같은 분모의 재현 수치가 아니다.
유형별 미래 이벤트 또는 유한 TTF가 소진되면 세 유형 모두 음성이라는 기존 통합 조건이 성립하지 않는다. 따라서 음성0을 실제 정상 운전 부재로 해석할 수 없다.
센서 기록이 존재한다는 사실만으로 그 기간의 모든 고장 기록이 완전하다고 입증할 수 없어, 미검증 음성을 학습용으로 추가하지 않았다.
오른쪽 대안은 **센서 종료까지 Fault 이력이 완전하다는 가정**에서만 성립한다. 이 가정은 TTF Null의 미공개 검열 horizon으로부터 도출할 수 없다.

{md_table(['장비','이벤트 f1/f2/f3','TTF Null f1/f2/f3','V1정책 h1800 음성(raw)','보수적 h600 음성(raw)','완전이력 가정 h600 음성(raw)'], sevenrows)}

개별 원인은 다음과 같다. 02M02·03M01·07M02·09M02는 f3 이벤트가 없고 f3 TTF가 전부 Null이다.
08M02는 f1/f3 이벤트가 모두 없으며 두 유형 TTF가 전부 Null이다. 10M02는 f2/f3 이벤트가 없고 두 유형 TTF가 전부 Null이다.
04M02의 f2 이벤트 1건은 time=3,926,292로, 센서 시작 3,997,500보다 이르다. 따라서 센서 기록 구간의 미래 f2가 없고 f2 TTF가 전부 Null이다.
고장 미기록이 정상 운전의 검증인지 관측 범위 밖/검열인지 외부 이력 없이 확정할 수 없다는 점은 7대 모두 동일하다.

## V1 대비 정책 변화
동일 원본행에서 V1 정책을 다시 계산한 h600 통합은 양성 {old['positive']:,}, 음성 {old['negative']:,}, NA {old['NA']:,}이다.
V2 보수적 후보와 달라진 원본행은 {old['changed_raw_rows']:,}개다. 미래 센서 공백을 라벨의 필요조건에서 분리하고, 60미만 별도 제외와 태그 충돌 격리를 추가했기 때문이다.
이는 ml_v1 데이터를 로딩한 표본별 비교가 아니라 원본에서 정책을 재계산한 비교다. V1의 feature_ready/학습표본 필터까지 재현한 비교는 수행하지 않았다.
TTF/Fault가 서로 일치해도 고장 기록이 누락 없이 완전하다는 외부 검증은 아니다.

## 해제에 필요한 근거
1. 실제 시간 단위 또는 변환계수.
2. 장비별/유형별 Fault 로깅의 유효 시작·종료 범위와 미발생/검열/미관측의 구분.
3. 정비 시작~종료/재가동 이력. Stage=0만으로 정비 구간을 추정하지 않는다.
4. [NoWaferID] 이벤트와 동시각 다중 센서값의 의미 및 최종 집계 정책.

근거: [PHM Society 공식 설명]({SOURCE}). 원문은 TTF Null의 특정 horizon 수치를 제공하지 않는다.
""")
    pilotrows = [[p['tool'], f"{p['raw_rows']:,}", f"{p['normalized_times']:,}", f"{p['anchors']:,}",
                  p['events_with_raw_window'], p['events_with_positive_anchor'], p['total_unique_events']] for p in pilots]
    preservation_rows = []
    for p in pilots:
        ev = pl.read_parquet(DEST / f"{p['tool']}_event_preservation.parquet")
        n = ev.height
        raw_events = ev.filter(pl.col('raw_eligible_times') > 0).height
        observed_events = ev.filter(pl.col('observed_grid_anchors') > 0).height
        ready_events = ev.filter(pl.col('history900_positive_anchors') > 0).height
        preservation_rows.append([p['tool'], f"{raw_events}/{n} ({raw_events/n:.1%})",
                                  f"{observed_events}/{n} ({observed_events/n:.1%})",
                                  f"{ready_events}/{n} ({ready_events/n:.1%})"])
    foldrows = []
    for fold in range(5):
        for role in ['train','early_stopping','threshold_selection','evaluation']:
            groups = [s for check in checks for s in check['splits'] if s['scheme'] == f'fold_{fold}' and s['role'] == role]
            foldrows.append([fold, role, sum(s['anchors'] for s in groups), sum(s['positive'] for s in groups),
                             sum(s['negative'] for s in groups), sum(s['null_including_exclusions'] for s in groups), sum(s['positive_unique_events'] for s in groups)])
    write("03_dataset_validation.md", f"""
# 데이터셋 검증과 최종 판단

## 최종 판단: 모델 학습 시작 불가
V2는 V1과 별도 폴더에 구성했고 V1 기존 파일 변경 감지는 0개다. 모델 학습은 0회다.
시간 단위·이력 유효범위·정비 종료의 심각한 라벨 불확실성 때문에 전체20대 특징/확정 라벨 생성을 중단했다.
20대 원본 감사와 2대 진단 Parquet만 생성했다. **승인된 학습 데이터 규모는 0행**이다.
진단 Parquet는 data/processed/diagnostic_pilot/이며 candidate_*와 approved_for_training=false를 갖고 학습 target 컬럼은 없다.
미완료 산출물은 전체20대 확정 특징/라벨과 그에 대한 전체 Fold 검사다. 진단 코드 통과가 요청 전체 데이터셋 완성을 뜻하지 않는다.

## 대표2대 보존
{md_table(['장비','원본행','정규화시각','고정격자Anchor','60~600 창에 원본있는 이벤트','양성후보Anchor있는 이벤트','고유이벤트(태그포함)'], pilotrows)}

원본은 전부 유지하며 정규화 원본행 합계가 원본 수와 일치한다. 시간 격자 Anchor는 공백도 포함하여 관측 기반 V1 Anchor와 분모가 다르다.
고유 이벤트는 원문 유형·시각 기준으로 유지한다. 이벤트별 원본 창 관측/격자 수/센서 가용격자/양성후보/900이력 양성 수는 *_event_preservation.parquet에 보존한다.
양성 Anchor가 있는 비율은 위 표의 해당 수/고유이벤트 수다. 동일 이벤트의 900 이력 충족률도 파일에서 확인할 수 있다. 관측 없는 이벤트를 임의 삭제하지 않는다.
미래 고장 이력이 알려진 빈 격자에도 후보 라벨은 있을 수 있다. 사용 가능한 센서·과거 창 확보와 양성 라벨 보존은 별도 지표다.

{md_table(['장비','원본60~600창 관측 이벤트','센서가용 격자 이벤트','900이력 양성 격자 이벤트'], preservation_rows)}

## 특징·누수·재현 검증
A/B/C 특징은 8/32/44개다. 정확한 화이트리스트는 feature_schema.json에 있다. 센서/상태만 입력이고 Tool/Lot/절대시간/TTF/Fault/미래 메타데이터는 제외한다.
경계 라벨, 미래값 변경 불변성, 정확한 시간 창, 공백 초기화, Stage=0 보존, 동시각 상충, 배치/일괄 동등성, 입력 화이트리스트, 그룹 분할, purge, 배포 차단의 unittest 9개를 통과했다.
동일 원본·설정·코드로 대표2대 전체 후보를 재생성한 Parquet SHA256 동일 여부: {all(r['identical'] for r in reproducibility.values()) if reproducibility else '미실행'}. outputs/reproducibility.json에 전후 해시가 있다.
Notebook 실행 오류 합계: {sum(n['errors'] for n in notebook_checks) if notebook_checks else '미실행'}. 별도 배포 차단 검증 결과는 outputs/release_gate_test.json에서 확인한다.
실제2대에서 Anchor 고유성/고정격자, 출처 시각≤예측시점, 창 출처>(t-w), 공백 횡단0, 시간 지지 구간 중첩0, 동일 고장 이벤트 시간분할 중복0을 확인했다.
전체20대 장비 배치표는 서로 배타적인 역할을 갖도록 고정했으며 양성/음성/이벤트 수에 맞춰 재선택하지 않았다.
실제 전체20대 Anchor의 Fold 클래스·이벤트 검증은 아직 수행하지 않았다. 아래는 **2대 진단 표본에만** 해당하므로 0인 역할이 있다.

{md_table(['Fold','역할','진단Anchor','양성후보 h600','음성후보 h600','Null(제외포함)','유형별 양성고유이벤트 합'], foldrows)}

## 환경과 자원
실제 실행 Python: `{env['python']}`, Python {env['python_version'].split()[0]}.
CPU: {env['cpu']}, 물리{env['physical_cores']}/논리{env['logical_cores']}코어, RAM {env['ram_gib']:.2f}GiB.
GPU: {env['gpu']}. XGBoost {env['packages']['xgboost']} 설치를 메타데이터로 확인했으며 CUDA 학습/속도는 이번에 검증하지 않았다.
Polars {env['packages']['polars']} / PyArrow {env['packages']['pyarrow']} / DuckDB {env['packages']['duckdb']}, 실제 Polars {env['polars_threads']}스레드다. DuckDB는 설치 확인만 했고 실행 엔진은 Polars다.
최초 일괄 진단은 RSS 12.63GiB로 내부12GiB 한계를 넘어 다음 단계를 차단했다. 이후 블록 계산으로 수정했다.
최종 실행 샘플링 RSS 최대 {resource['peak_rss_gib']:.2f}GiB, 가용 RAM 최소 {resource['min_available_ram_gib']:.2f}GiB. 자원 경고: {resource['violation']}.
VRAM/사용률은 outputs/resources.json에 기록했다. GPU 전처리·학습은 하지 않았다. 2초 샘플링은 순간 피크를 놓칠 수 있다.

## 재현 명령
V2 루트에서 README의 절대 Conda Python으로 unittest → python -m src.run_pipeline → python -m src.report를 실행한다.
원본 감사는 코드/설정/원본 메타데이터가 같은 장비만 재사용한다. 후보 특징은 원본에서 다시 만든다. 모든 출력은 V2 안에 저장한다.
Notebook4개는 저장된 감사/후보 결과를 읽는 한국어 설명·그래프이며 원본 전체 재처리를 자동 실행하지 않는다.
""")
    build_notebooks()


def build_notebooks():
    import nbformat as nbf
    setup = '''from pathlib import Path
import os, json, sys
ROOT = Path.cwd()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
assert ROOT.name == 'PHM2018-Project-V2'
sys.path.insert(0, str(ROOT))
os.environ['MPLCONFIGDIR'] = str(ROOT / 'outputs/matplotlib')
import polars as pl
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
figdir = ROOT / 'reports/figures'
figdir.mkdir(parents=True, exist_ok=True)
load = lambda name: json.loads((ROOT / 'outputs' / name).read_text(encoding='utf-8'))
'''
    definitions = [
        ("01_raw_data_validation", "원본 감사", "학습20대의 센서·고장·TTF를 전수 감사한 결과입니다. 시각의 물리 단위는 미확정이며 원본을 변경하지 않았습니다.",
         '''audit = load('audit_summary.json')
fig, ax = plt.subplots(figsize=(12,4))
ax.bar([a['tool'] for a in audit], [a['raw_rows']/1e6 for a in audit])
ax.set(ylabel='원본 센서 행 수 (백만)', title='장비별 센서 원본 규모')
ax.tick_params(axis='x', rotation=60)
fig.tight_layout(); fig.savefig(figdir / '01_raw_rows.png', dpi=130); plt.show()
print('전체 행 수:', f"{sum(a['raw_rows'] for a in audit):,}")
print('태그 제외 TTF 불일치:', sum(v['exclude_tag_mismatch'] for a in audit for v in a['ttf'].values()))''',
         "막대 높이는 장비별 기록량 차이입니다. 기록량이나 Stage=0의 존재만으로 정상/정비 상태를 확정하지 않습니다. 중복·공백·결측의 상세 원인은 장비별 감사 JSON/Parquet에 보존했습니다."),
        ("02_fault_label_generation", "고장 라벨 후보와 불확실성", "라벨은 60~600 원시시간 단위의 진단 후보입니다. 실제 초 단위·정비 종료·Fault 이력 유효범위가 미확정이므로 확정 학습 라벨은 없습니다.",
         '''audit = load('audit_summary.json')
seven = {'02M02','03M01','04M02','07M02','08M02','09M02','10M02'}
selected = [a for a in audit if a['tool'] in seven]
x = list(range(len(selected)))
fig, ax = plt.subplots(figsize=(10,4))
ax.bar([i-.18 for i in x], [a['distributions_raw']['any_h600']['negative'] for a in selected], width=.36, label='보수적 음성 후보')
ax.bar([i+.18 for i in x], [a['sensitivity_any']['h600']['negative'] for a in selected], width=.36, label='이력 완전성 가정하의 음성 후보')
ax.set_xticks(x, [a['tool'] for a in selected]); ax.set_ylabel('원본 행 수'); ax.legend()
ax.set_title('음성이 없던 7대: 근거 조건에 따른 후보 수 차이')
fig.tight_layout(); fig.savefig(figdir / '02_seven_tools.png', dpi=130); plt.show()
load('label_distribution_raw.json')''',
         "대안 음성이 늘어나는 것은 정상 운전이 검증되었다는 뜻이 아닙니다. 세 유형의 유한 TTF를 모두 요구하는 조건과 이력 완전성 가정의 차이입니다. TTF Null을 자동으로 정상으로 바꾸지 않았으며 대안은 학습용으로 배포하지 않습니다."),
        ("03_feature_engineering", "과거 특징과 고정 Anchor", "전체 원본 관측에서 과거300/900 원시단위 창을 먼저 계산합니다. 60 간격 고정 격자 Anchor에서 현재값·통계·추세를 보존하며 공백을 넘는 보간은 하지 않습니다.",
         '''from src.features import feature_names
print({k:len(v) for k,v in feature_names().items()})
df = pl.read_parquet(ROOT / 'data/processed/diagnostic_pilot/01M01.parquet')
start = df.filter(pl.col('feature_status') == 'history_span_available')['time'][0]
part = df.filter((pl.col('time') >= start) & (pl.col('time') <= start+3600))
fig, ax = plt.subplots(figsize=(11,4))
ax.plot(part['time']-start, part['FLOWCOOLPRESSURE'], '.-', label='현재 관측값')
ax.plot(part['time']-start, part['FLOWCOOLPRESSURE_w300_mean'], label='과거300 평균')
ax.set(xlabel='구간 시작 후 원시시간', ylabel='익명화 센서값', title='과거 관측만 사용하는 센서 특징 예시')
ax.legend(); fig.tight_layout(); fig.savefig(figdir / '03_features.png', dpi=130); plt.show()
df.group_by('feature_status').len()''',
         "이동평균은 해당 시점까지의 원본 관측만 사용합니다. 결측/이력 부족 Anchor도 유지합니다. 그래프가 매끄럽다는 사실은 조기 예측 가능성이나 센서 물리 단위의 검증이 아닙니다. 특징 A/B/C는 8/32/44개이며 모델 학습은 수행하지 않았습니다."),
        ("04_dataset_validation", "데이터셋 검증과 배포 차단", "대표2대의 후보만 검증했습니다. 전체20대 확정 데이터 생성은 단위·정답 불확실성 때문에 중단되었습니다. V1에서 본 평가 구간을 독립 미사용 평가라고 주장하지 않습니다.",
         '''checks = load('validation.json')
assert all(not c['issues'] for c in checks)
assert all(c['approved_training_rows'] == 0 for c in checks)
pilot = load('pilot_summary.json')
fig, ax = plt.subplots(figsize=(8,4))
for i, p in enumerate(pilot):
    ax.bar(i-.18, p['events_with_raw_window'], width=.36, color='tab:blue', label='원본 창 있는 이벤트' if i==0 else None)
    ax.bar(i+.18, p['events_with_positive_anchor'], width=.36, color='tab:orange', label='양성 후보 Anchor 있는 이벤트' if i==0 else None)
ax.set_xticks(range(len(pilot)), [p['tool'] for p in pilot]); ax.set_ylabel('고유 이벤트 수')
ax.set_title('60~600 원시단위 창의 이벤트 보존'); ax.legend()
fig.tight_layout(); fig.savefig(figdir / '04_event_preservation.png', dpi=130); plt.show()
print('최종 판단: 모델 학습 시작 불가. 승인 데이터 0행.')
load('run_status.json')''',
         "미래 이벤트 이력과 입력 관측 가능성은 별개입니다. 따라서 센서 없는 격자에도 양성 후보가 있을 수 있으며, 사용 가능한 입력까지 확보한 비율은 이벤트별 observed_grid_anchors와 history900_positive_anchors를 함께 확인해야 합니다. 태그 충돌과 60 미만 제외도 보존율에 영향을 줍니다. 실제2대의 검증 통과는 전체20대 학습 데이터의 검증 완료를 의미하지 않습니다."),
    ]
    for name, title, intro, code, interpretation in definitions:
        nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}"),
            nbf.v4.new_code_cell(setup), nbf.v4.new_code_cell(code),
            nbf.v4.new_markdown_cell(f"## 그래프 해석과 한계\n\n{interpretation}\n\n상세 결과는 `reports/`의 한국어 보고서와 `outputs/`의 수치 파일을 참조하세요.")])
        nb.metadata['kernelspec'] = {"name":"phm2018-v2", "display_name":"PHM2018 V2 (local Conda)", "language":"python"}
        path = ROOT / "notebooks" / f"{name}.ipynb"
        if path.exists():
            previous = nbf.read(path, as_version=4)
            if [(c.cell_type, c.source) for c in previous.cells] == [(c.cell_type, c.source) for c in nb.cells]:
                continue  # 코드/설명이 같으면 실행 결과를 유지한다.
        nbf.write(nb, path)


if __name__ == "__main__":
    main()
