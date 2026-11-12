"""검증된 표·그림에 연결되는 한국어 보고서와 사용 설명서."""
from .common import *
import pandas as pd

def md(frame,limit=None):
    d=frame.head(limit) if limit else frame
    def cell(v):
        if pd.isna(v):return '—'
        if isinstance(v,(bool,np.bool_)):return '예' if v else '아니오'
        if isinstance(v,(int,np.integer)):return f'{v:,}'
        if isinstance(v,(float,np.floating)):return f'{v:,.4f}' if v!=int(v) else f'{int(v):,}'
        return str(v).replace('|',' / ').replace('\n',' ')
    lines=['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']
    lines += ['| '+' | '.join(cell(v) for v in r)+' |' for r in d.itertuples(index=False,name=None)]
    return '\n'.join(lines)

def img(path,caption):return f'![{caption}](../outputs/figures/{path}.png)\n\n{caption}\n'

def write(name,text):
    (ROOT/'reports'/name).write_text(text,encoding='utf-8')

def main():
    tool=pd.read_parquet(TABLES/'tool_summary.parquet');labels=pd.read_parquet(TABLES/'label_status_summary.parquet');feat=pd.read_parquet(TABLES/'feature_availability_summary.parquet');events=pd.read_parquet(TABLES/'events.parquet');ex=pd.read_parquet(TABLES/'event_exclusion_reasons.parquet');cases=json.loads((ROOT/'outputs/case_catalog.json').read_text(encoding='utf-8'));figs=json.loads((ROOT/'outputs/figure_manifest.json').read_text(encoding='utf-8'))
    raw_n=int(tool.raw_rows.sum());event_n=len(events);anchors=int(tool.anchors.sum())
    q=labels[labels.sample_level.eq('anchor')&labels.H.eq(600)&labels.fault_type.eq('any')].groupby('policy')[['positive','negative','NA','excluded','denominator']].sum();q['NA_ratio']=q.NA/q.denominator
    f=feat[feat.policy.eq('conservative')&feat.H.eq(600)&feat.fault_type.eq('any')].groupby(['cohort','feature_set'])[['available_anchors','positive_anchors','negative_anchors','positive_events','current_gap_anchors','history_gap_anchors']].sum().reset_index()
    summary=tool[['Tool','raw_rows','time_start','time_end','fault_events','duplicate_rows','conflict_times','long_gaps','ttf_null_ratio_low','ttf_null_ratio_high','ttf_null_ratio_leak','usable_events_C_h600_conservative']].rename(columns={'Tool':'장비','raw_rows':'원본 행','time_start':'시작 raw','time_end':'종료 raw','fault_events':'고유 고장','duplicate_rows':'중복 행','conflict_times':'충돌 시각','long_gaps':'긴 공백','ttf_null_ratio_low':'저압 Null 비율','ttf_null_ratio_high':'고압 Null 비율','ttf_null_ratio_leak':'누출 Null 비율','usable_events_C_h600_conservative':'보수적 H600 C 이벤트'})
    spec='''# V3 시각화 진단 명세

사용자가 별도 V3 기반 정의가 없음을 확인하여 이 문서와 config/preprocessing.json에 기준을 명시했다. V2의 A/B/C 개념을 참고했으며 기존 파생 데이터는 입력으로 사용하지 않았다. 모든 수치는 실제 원본 CSV에서 다시 생성했다. 모델 학습은 수행하지 않았다.

## 시간과 정답

- 시간은 raw이며 초/분 변환을 하지 않는다. 고장은 기록된 정비 정지 시각이다. 이상 시작·정비 종료·로그 완전성은 미확정이다.
- H=300/600/1800, L=60. 명시된 [tf-H,tf-L]을 적용해 d=60은 양성, 0≤d<60은 모든 유형과 통합에서 EXCLUDE를 최우선 적용한다. 요청의 “60 이내” 표현과 닫힌 구간의 경계 차이는 이 규약으로 해소했다.
- 고장 원문과 원본 위치를 보존하고 Tool/시각/원문 유형 중복만 합친다. event_id는 장비·시각·원문·원본 파일/행 목록의 SHA256에서 안정적으로 생성한다.
- [NoWaferID] 포함/제외 시 다음 고장이 달라지는 유형 라벨은 NA로 격리한다. 태그 이벤트도 원본 이벤트 분모와 임박 제외 검사에 포함한다.
- 보수적: 검증된 다음 동일 유형 이벤트가 60≤d≤H이면 POS, d>H이면 기록상 NEG. 후속 유형 이벤트가 없으면 NA. 입력 센서 공백을 이유로 알려진 정답을 삭제하지 않는다.
- 제한 연구용: 보수적 근거를 우선 사용한다. 그 외 첫~마지막 비태그 고장 사이이고 미래 H가 동일 연속 센서 구간에 포함된 시점에 한해 해당 유형 무기록을 NEG로 가정한다. 고장 로그 완전성이 입증된 것은 아니다.
- 완전 이력 가정: 센서 종료까지 고장 이력이 완전하다고 가정하여, t+H≤센서 종료이고 H 안에 해당 유형이 없으면 추가 NEG. 가정 음성은 검증 정상으로 표시하지 않는다.
- 통합: EXCLUDE 최우선, 하나라도 POS이면 POS, 모두 NEG이면 NEG, 나머지는 NA. 동시 발생 서로 다른 유형을 합쳐 하나의 이벤트로 만들지 않는다.

## 시간 정규화와 입력

- 원본 float64 센서값, 원본 행 번호, TTF와 공정 코드를 Parquet에 보존한다. 같은 시각 센서는 평균 집계하고 raw/normalized를 별도로 조회한다. 물리적으로 평균이 타당하다고 입증한 것은 아니다.
- recipe/stage/recipe_step/Lot/runnum의 동시각 불일치를 공정 충돌로 표시한다. Recipe/Stage만 검사한 과거 결과와 정의가 다르다. 단순 센서값 차이는 별도 sensor_disagreement다.
- dt>12 raw 또는 공정 충돌에서 세그먼트를 끊는다. 12는 진단 설정이며 검증된 물리 임계값이 아니다. 긴 공백은 보간하지 않고 그래프 선을 끊는다.
- 고정 구간 폭은 100,000 raw. 최빈 간격 4와 장비 범위 약 3천만~4천만 raw를 고려해 장비당 약 300~430개 구간을 표시한다. 연속 세그먼트도 별도 저장한다. OBSERVED/SPARSE/GAP/CONFLICT/FAULT_NEAR/UNKNOWN은 복수 플래그다.
- Anchor는 원점 0, 간격 60 raw의 고정 격자. 공백 중 Anchor도 남기며 가용성=false로 표시한다. 현재값은 t 이하 최근 관측, 최대 지연 12 raw만 허용한다.
- A: 센서 현재값6+recipe/stage=8. B: A+300 raw 평균/모표준편차/최소/최대=32. C: B+900 raw OLS기울기/처음-마지막 변화량=44.
- 모든 과거 창은 (t-w,t], Anchor 다운샘플링 전 전체 관측을 사용한다. 같은 연속 구간의 이력 길이와 각 센서 관측률≥90%를 요구한다. 과거60은 진단용으로 별도 계산한다. 미래 센서는 사용하지 않는다.
- A/B/C는 독립된 가용성 마스크와 최대 코호트로 집계한다. 공통 코호트는 세 가용성 마스크의 교집합이다. C가 B 특징을 포함하는 것은 특징 정의이며 A→B→C 데이터 처리 순서를 뜻하지 않는다.
- 전처리 통계/라벨에는 미래 정보가 존재하지만 특징 화이트리스트에는 TTF/Fault/절대시간/Tool/이벤트/미래 구간 끝이 없다. 후보는 연구 진단용이며 정비 종료를 확인한 정상 운전 데이터로 배포하지 않는다.

## 재현성과 조회

- 200,000행 청크, 명시적 dtype, 장비 순차 처리, Arrow 4스레드. 원본 전체를 메모리나 브라우저에 올리지 않는다.
- 원본 메타데이터·환경·설정·코드 해시·단계 진행률을 outputs에 기록한다. 원본 전체 내용 해시는 계산하지 않았고 크기/mtime를 비교했다.
- 상세 조회는 최대20,000 raw. 필요한 파일과 25,000행 row group만 읽고 행 필터를 적용한다. 이동 통계 계산에 필요한 과거900 raw는 의도적으로 추가 읽는다. 실제 읽기 범위를 대시보드에 표시한다.
- 센서 상세 그래프는 원본 해상도100%. 전체 그래프는 정확한 집계. 상대시간 비교는 60 raw bin의 이벤트별 중앙값을 먼저 구하고 동일 Tool/고장/Recipe/Stage/Step에서 이벤트 간 분위수를 계산한다. 미관측 bin은 보간하지 않는다.
- 공식 test는 정답이 없으므로 이 20대 학습 원본 진단의 분모에서 제외한다. 테스트 데이터로 설정을 선택하지 않았다.

공식 자료: [PHM Society](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/). 로컬 조회 구현은 [Arrow Parquet 문서](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html), 차트는 [Plotly API](https://plotly.com/javascript/plotlyjs-function-reference/)를 참고했다.
'''
    write('00_preprocessing_spec.md',spec)
    diagnostic=f'''# 09 시각화 중심 전처리 진단

원본 {raw_n:,}행, 장비 {len(tool)}대, 고유 고장 {event_n:,}개, 고정 격자 Anchor {anchors:,}개를 처리했다. 원본 1,236 고장 행의 중복25행을 합쳤으며 태그 이벤트1개는 원본 분모에 남겼다.

{md(summary)}

{img('tool_overview/raw_rows','원본 규모는 장비별로 다르다. 행 수가 많아도 예측 가능한 이벤트가 많다는 뜻은 아니다.')}
{img('tool_overview/ttf_null','TTF Null이 장비와 유형에 따라 집중된다. Null은 정상0으로 변환하지 않는다.')}

## 라벨 정책 비교

H=600, L=60, 고정 격자 Anchor 기준이다. 정책 간 차이는 데이터 복구가 아니라 정답 가정의 차이다.

{md(q.reset_index())}

{img('label_analysis/horizon_policies','예측 구간이 달라지면 라벨 분포도 달라진다. 연구용·민감도 음성은 확정 정상이 아니다.')}
{img('label_analysis/ttf_null_relation','임박 고장의 EXCLUDE 우선 규칙과 통합 논리 때문에 TTF Null과 라벨 NA를 같은 값으로 취급하면 안 된다.')}

## 특징 가용성

{md(f)}

{img('feature_availability/cohorts','현재값만 필요한 A와 과거 이력이 필요한 B/C의 가용 코호트가 다르다. 공통 코호트에서는 특징 효과를 비교할 수 있지만 이번 작업은 모델을 학습하지 않았다.')}

정확한 규칙은 [전처리 명세](00_preprocessing_spec.md), 세부 수치는 outputs/tables의 CSV/Parquet, 차트 출처와 분모는 outputs/figure_manifest.json에 있다.
'''
    write('09_visual_diagnostics.md',diagnostic)
    event_report='# 10 이벤트별 센서·정답 근거 분석\n\n원본에 존재하는 이벤트만 사용했다. 고장 이후는 참고 표시이며 해당 이벤트의 예측 입력에 포함하지 않는다. 아래 사례는 대표 01M01/03M02의 규칙 기반 선택이며 전체 고장 유형의 일반적인 전조를 증명하지 않는다.\n\n'
    segments=pd.read_parquet(TABLES/'time_segment_summary.parquet')
    for c in cases:
        meta=json.loads((ROOT/'outputs/cases'/c['case_id']/'metadata.json').read_text(encoding='utf-8'))
        anchors_df=pd.read_csv(ROOT/'outputs/cases'/c['case_id']/'anchor_evidence.csv') if meta['anchors'] else pd.DataFrame()
        reason=meta['exclusions'][0]['reason'] if meta['exclusions'] else '연결된 고장 이벤트 없음. Anchor별 입력/라벨 근거로 검토.'
        event_report+=f'## {c["case_id"]} — {c["category"]}\n\n장비 `{c["Tool"]}`, 범위 `{c["start"]:.0f}~{c["end"]:.0f} raw`, event_id `{c["event_id"] or "해당 없음"}`. {c["description"]}\n\n관측 사실: 원본 {meta["raw_rows"]:,}행, 고유 관측 {meta["observations"]:,}개, 범위 내 이벤트 {meta["unique_events_in_range"]}개, 긴 공백 {len(meta["gaps"])}개, 동시각 중복 {len(meta["duplicates"])}시각. 판단: {reason}\n\n'+img('fault_events/'+c['case_id'],'회색 공백을 선으로 연결하지 않았다. 센서 차이와 공정 전환의 동시 발생은 인과관계 입증이 아니다.')
        stats=pd.DataFrame(meta.get('sensor_summary',[]))
        if len(stats):
            pressure=stats.iloc[0]
            scope='선택 구간과 예측 구간의 교집합' if c['event_id'] else '선택 구간'
            event_report+=f'센서 변화 요약은 {scope} `{pressure.start:.0f}~{pressure.end:.0f} raw`의 실제 관측을 사용한다. '
            if pressure.observations:
                event_report+=f'Flowcool 압력은 첫 관측 {pressure.first_value:.6g}에서 마지막 {pressure.last_value:.6g}로 차이 {pressure.delta:+.6g}이며, 구간 내 최솟값 {pressure.minimum:.6g}, 최댓값 {pressure.maximum:.6g}다. 공정 조건 조합은 {int(pressure.process_conditions)}개다. '
            else:event_report+='이 범위에 센서가 없어 변화량을 계산할 수 없다. '
            event_report+='첫/끝 차이는 공백과 공정 전환의 영향을 포함할 수 있어 일정한 추세나 고장 전조로 판정하지 않는다.\n\n'
            event_report+=md(stats[['sensor','observations','first_time','last_time','first_value','last_value','delta','minimum','maximum','std']].rename(columns={'sensor':'센서','observations':'관측 수','first_time':'첫 raw','last_time':'끝 raw','first_value':'첫 값','last_value':'끝 값','delta':'차이','minimum':'최소','maximum':'최대','std':'표준편차'}))+'\n\n'
        if c['category'] in ['고장 직전 센서 존재','공정 전환과 센서 변화']:
            event_report+=img('fault_events/'+c['case_id']+'_changes','과거60 표준편차·raw당 변화율·평균60−평균300이다. 각 연속 구간 안에서만 계산하여 공백을 가로지르는 급변을 만들지 않았다. 공정 전환 표식과 동시에 변하는 센서는 공정 조건별로 분리해 검토한다.')
            event_report+=img('fault_events/'+c['case_id']+'_conditions','선택 구간에서 관측 수 상위6개 Recipe/Stage별 센서 분포다. 이상점 표시만 생략했고 관측값을 삭제하지 않았다. 조건 간 분포 차이를 고장 전조로 오인하지 않도록 모델 입력의 공정 조건 처리를 검토한다.')
        if len(anchors_df):event_report+=md(anchors_df[['time','current_lag','n_60','n_300','n_900','A_available','B_available','C_available','low','high','leak','any','candidate_reason']].head(8))+'\n\n표는 앞8개 Anchor만 표시. 전체 구간 상태·근거 표는 해당 [사례 폴더](../outputs/cases/'+c['case_id']+'/anchor_evidence.csv)에 보존했다.\n\n'
        selected_segments=segments[segments.Tool.eq(c['Tool'])&segments.kind.eq('fixed')&segments.start.le(c['end'])&segments.end.ge(c['start'])]
        selected_segments.to_csv(ROOT/'outputs/cases'/c['case_id']/'segment_status.csv',index=False,encoding='utf-8-sig')
        event_report+='이 상세 범위를 포함하는 100,000 raw 고정 구간의 상태다. 아래 구간 분모는 확대 센서 그림의 분모보다 크며, 복수 상태를 함께 유지한다.\n\n'+md(selected_segments[['start','end','sensor_observations','max_gap','process_changes','fault_count','ttf_null_ratio_low','ttf_null_ratio_high','ttf_null_ratio_leak','states']])+'\n\n'
        if meta['gaps']:event_report+=md(pd.DataFrame(meta['gaps']))+'\n\n'
        if meta['duplicates']:event_report+=md(pd.DataFrame(meta['duplicates']),8)+'\n\n'
    relatives=[x for x in figs if '/relative_' in x['file']]
    event_report+='## 공정 조건을 맞춘 상대시간 비교\n\n같은 장비·고장 유형·Recipe·Stage·Step에서 이벤트가 3개 이상인 집단 중 이벤트 수 상위6개만 표시한다. 각 상대시간 bin의 기여 이벤트 수는 relative_event_quantiles.csv에 남겼다.\n\n'
    for r in relatives:event_report+=f'![{r["title"]}](../{r["file"]})\n\n{r["context"]}\n\n'
    if not relatives:event_report+='조건을 충족하는 비교 집단이 없어 집계 전조 곡선을 만들지 않았다.\n'
    write('10_event_level_analysis.md',event_report)
    segment=f'''# 11 시간 구간 품질 검토

{img('timeline/all_tools','청록은 실제 연속 관측 세그먼트, 회색은 긴 관측 공백, 색 표식은 고장이다. 고장 기록이 센서 공백 안에도 존재하므로 이벤트와 센서 관측을 따로 추적한다.')}
{img('timeline/density','고정 구간과 연속 세그먼트는 서로 다른 분석 단위다. 밀도가 낮은 구간을 연속 센서 기록으로 대체하지 않는다.')}
{img('timeline/gap_distribution','장비별 공백 길이의 분포와 누적분포를 함께 보였다. 공백은 측정 중단 등의 가능성을 보여주지만 원인을 확인한 것은 아니다.')}
{img('timeline/duplicates_conflicts','같은 시각의 공정 충돌은 원본의 공정 코드 차이를 근거로 한다. 센서값 차이만 있는 중복과 구분한다.')}

## 실제 문제 구간

{md(pd.DataFrame(cases)[['case_id','Tool','category','start','end','event_id','description']])}

각 사례의 센서 그림, 구간 상태, 라벨, 제외 이유는 [이벤트 분석](10_event_level_analysis.md)에 함께 제시했다. 대시보드의 ‘검토 사례’에서 동일 구간을 다시 열 수 있다.

## 공정 조건과 결측

{img('feature_availability/process_conditions','같은 장비 안에서도 공정 조건에 따라 요구 이력의 확보 여부가 다르다. 전체 조건은 process_feature_summary.csv, 선택 구간의 TTF Null 비율은 대시보드 공정 조건 표에서 확인한다.')}

동일 시각 공정 충돌 총 {int(tool.conflict_times.sum()):,}시각. 본 V3는 recipe_step/Lot/runnum까지 확인하므로 Recipe/Stage만 검사한 기존 결과의 “충돌0”과 정의가 다르다. 원시 시간 단위, 정비 종료, 이력 완전성은 이 자료만으로 확정되지 않는다.
'''
    write('11_segment_quality_review.md',segment)
    final=f'''# 12 V3 시각화 진단 결과

장비20대의 원본 기반 집계와 대표2대의 상세 그림을 분리하여 구축했다. 모델 학습과 외부 업로드는 하지 않았다.

## 1. 전체 장비 현황 요약표

{md(summary)}

## 2. 장비별 데이터 관측 타임라인

{img('timeline/all_tools','실제 관측 세그먼트·긴 공백·고장 이벤트를 같은 원시시간축에 표시했다. 전체20대 집계이며 샘플링하지 않았다.')}

## 3. 시간 구간별 공백·충돌 분석

{img('timeline/density','구간 폭100,000 raw의 관측 밀도. 상세에서는 원본 해상도와 연속 세그먼트를 확인한다.')}
{img('timeline/duplicates_conflicts','공정 충돌은 원본의 실제 동일 시각 코드 불일치다. V3 검사 컬럼 확장으로 기존 검사보다 많은 충돌이 발견될 수 있다.')}

## 4. 고장 유형별 이벤트 분포

{img('tool_overview/fault_types','고장 행1,236개와 고유 이벤트1,211개를 구분했다. 태그 이벤트는 원본 분모에 남고 불확실한 라벨 범위는 격리한다.')}

## 5. 고장 이벤트 전 센서 변화

{img('fault_events/'+next(c['case_id'] for c in cases if c['category']=='고장 직전 센서 존재'),'대표 실제 이벤트. 고장 시각은 정지 기록이다. 센서 변화가 보이더라도 이상 시작·원인·예측 성능을 입증한 것은 아니다.')}

전체 대표 사례와 조건별 상대시간 비교는 [이벤트 보고서](10_event_level_analysis.md)에 있다.

## 6. 라벨 정책별 결과 비교

H600 / L60 / Anchor 기준:

{md(q.reset_index())}

{img('label_analysis/horizon_policies','동일 입력에 가정을 달리 적용한 민감도 비교다. 연구용·완전 이력 가정의 음성을 검증 정상으로 해석하지 않는다.')}

## 7. 전처리 단계별 데이터·이벤트 보존

{img('event_preservation/stage_rows','A/B/C를 독립 분기로 표시했다. 원본 행→고유 시각→Anchor의 분모 변경은 삭제와 구분한다.')}
{img('event_preservation/stage_events','경보 구간 센서 존재 → 해당 특징을 같은 Anchor에서 계산 가능 → 사용 가능한 양성 Anchor 존재 순으로 고유 이벤트를 센다. 한 이벤트의 서로 다른 시점에서 확보한 조건을 합쳐 가용하다고 판단하지 않는다.')}
{img('event_preservation/exclusion_reasons','행 수가 아닌 고유 이벤트를 한 번씩 계산한 배타적 사유다. 중첩된 원인은 event_exclusion_reasons.csv의 세부 가용성 열에서 함께 확인한다.')}

## 8. A/B/C 특징 가용성 비교

{md(f)}

{img('event_preservation/tool_policy_heatmap','특징 수가 증가할수록 더 좋은 모델이라고 결론 내리지 않는다. 긴 이력 요구로 사용 가능한 이벤트가 달라지므로 공통 코호트와 최대 코호트를 별도로 비교해야 한다.')}

## 9. 문제가 있는 대표 시간 구간

{md(pd.DataFrame(cases)[['case_id','Tool','category','start','end','event_id']])}

각 사례에 원본 센서·공정 상태·라벨 근거·제외 이유를 함께 저장했다. [상세 분석](10_event_level_analysis.md), [구간 품질 검토](11_segment_quality_review.md), 대시보드 ‘검토 사례’에서 확인한다.

## 10. 전체20대 확장 및 다음 단계

- 전체20대 원본 인덱스, 구간·고장·세 라벨 정책·A/B/C 가용성 집계가 완료되었다. 상세 조회도20대 모두 지원한다. 정적 상세 사례는 요청된01M01/03M02에 집중했다.
- 고장 기록1211개 중 연구 후보로 남는 이벤트 수는 정책/H/특징별로 다르다. 긴 이력 조건과 공정 충돌이 선택 편향을 만드는지 먼저 검토한다.
- 단위·정비 종료·고장 이력 완전성 확인 전에는 연구 후보를 운영용 정상/고장 정답으로 배포하지 않는다. 후속 모델 비교에서는 장비/시간 분할, 동일 이벤트 격리, 과거창/미래창 경계 분리, 검증 전용 임계값을 적용해야 한다.
- 평가 계획은 이벤트 검출률, 관측 가능한 운영시간당 경보 수, 선행시간을 포함한다. 이번 산출물에는 모델 성능 수치가 없다.

실행: [대시보드 안내](../dashboard/README.md). 검증 결과는 outputs/validation, 코드·환경·원본 메타데이터는 outputs에 있다. 전체 파일 내용 해시 검증과 실제 물리 단위 검증은 수행하지 않았다.
'''
    write('12_visualization_summary.md',final)
    print('보고서 00/09/10/11/12 생성 완료')

if __name__=='__main__':main()
