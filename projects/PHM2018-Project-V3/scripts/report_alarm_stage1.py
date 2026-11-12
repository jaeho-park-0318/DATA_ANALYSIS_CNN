"""실제 검증 분할의 경보 평가기 결과를 표·그림·보고서로 정리한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
from src.reporting import md
from src.visuals import plt,save,REGISTRY
from src.query import read_window,broken_line

NAMES={'temporal':'시간 순서 검증','group':'장비 그룹 검증'}
def main():
    s=pd.read_parquet(TABLES/'alarm_stage1_summary.parquet');states=pd.read_parquet(TABLES/'alarm_stage1_label_states.parquet');ev=pd.read_parquet(TABLES/'alarm_stage1_events.parquet');alarms=pd.read_parquet(TABLES/'alarm_stage1_episodes.parquet')
    cols=['recorded_events','scorable_events','available_events','detected_events','alarm_episodes','available_anchors','observed_support_raw']+[c for c in s if c.startswith('alarms_')]
    g=s.groupby(['scheme','feature_set','baseline'])[cols].sum().reset_index()
    g['recall_recorded']=g.detected_events/g.recorded_events.replace(0,np.nan);g['recall_available']=g.detected_events/g.available_events.replace(0,np.nan)
    g['alarm_episodes_per_10000_raw']=10000*g.alarm_episodes/g.observed_support_raw.replace(0,np.nan)
    save_table(g,'alarm_stage1_overview')
    source='실제 validation 분할 · H600/L60 · 최대 가용 코호트 · 모델 학습 없음 · holdout 점수화 없음'
    REGISTRY.clear()
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,(scheme,title) in zip(axes,NAMES.items()):
        q=g[g.scheme.eq(scheme)&g.baseline.eq('no_alarm')].set_index('feature_set').reindex(list('ABC'))
        ax.bar(list('ABC'),q.recorded_events,color='#dce4e8',label='분할 내 기록 이벤트');ax.bar(list('ABC'),q.available_events,color='#087e80',label='예측 구간 입력 존재');ax.set_title(title);ax.set_ylabel('고유 이벤트 수');ax.legend()
        for i,r in enumerate(q.itertuples()):ax.text(i,r.available_events+2,f'{r.available_events}/{r.recorded_events}',ha='center')
    save(fig,'alarm_stage1/event_coverage','학습 전에 확인한 이벤트 관측 커버리지','alarm_stage1_overview.csv',source+' · 입력 존재는 검출 성공과 다름')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,(scheme,title) in zip(axes,NAMES.items()):
        q=states[states.scheme.eq(scheme)].groupby(['feature_set','status']).anchors.sum().unstack(fill_value=0).reindex(index=list('ABC'),columns=['POS','NEG','NA','EXCLUDE'],fill_value=0)
        q.div(q.sum(axis=1),axis=0).plot.bar(stacked=True,ax=ax,color=[COLORS[k] for k in ['POS','NEG','NA','EXCLUDE']]);ax.set_title(title);ax.set_ylabel('전체 가용 Anchor 대비 비율');ax.set_xlabel('특징 구성');ax.legend(['양성','정책상 음성','미확인','제외'],fontsize=8)
    save(fig,'alarm_stage1/scoring_population','분류 후보와 전체 점수화 대상의 차이','alarm_stage1_label_states.csv',source+' · NA/EXCLUDE도 점수화하며 정상으로 변경하지 않음')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,(scheme,title) in zip(axes,NAMES.items()):
        q=g[g.scheme.eq(scheme)]
        for f,color in zip('ABC',['#087e80','#0072b2','#d55e00']):
            a=q[q.feature_set.eq(f)&q.baseline.eq('always_alarm')].iloc[0];ax.scatter(a.alarm_episodes_per_10000_raw,a.recall_available,color=color,s=80,label=f+' 항상 경보');ax.annotate(f,(a.alarm_episodes_per_10000_raw,a.recall_available),xytext=(5,5),textcoords='offset points')
        ax.scatter(0,0,color='#5e6c73',marker='x',s=80,label='무경보 (A/B/C)');ax.set_title(title);ax.set_xlabel('명목 관측 지지길이 10,000 raw당 에피소드');ax.set_ylabel('입력 있는 이벤트의 검출률');ax.set_ylim(-.03,1.05);ax.legend(fontsize=8)
    save(fig,'alarm_stage1/baseline_checks','고정 기준으로 확인한 경보 집계 동작','alarm_stage1_overview.csv',source+' · 고정 점수0/1의 시험 기준, 학습 성능·최적 임계값 곡선이 아님')
    cats=['MATCHED','KNOWN_NEGATIVE_START','UNKNOWN_START','LATE_OR_EXCLUDED_START','POSITIVE_UNMATCHED'];names=['이벤트 매칭','음성에서 시작','미확인에서 시작','임박/제외에서 시작','양성이지만 미매칭']
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,(scheme,title) in zip(axes,NAMES.items()):
        q=g[g.scheme.eq(scheme)&g.baseline.eq('always_alarm')].set_index('feature_set').reindex(list('ABC'));bottom=np.zeros(3)
        for c,n,color in zip(cats,names,['#087e80','#0072b2','#929aa4','#7b4ab5','#d55e00']):v=q['alarms_'+c];ax.bar(list('ABC'),v,bottom=bottom,label=n,color=color);bottom+=v
        ax.set_title(title);ax.set_ylabel('경보 에피소드 수');ax.legend(fontsize=7)
    save(fig,'alarm_stage1/alarm_categories','항상 경보 기준의 시작 시점별 분류','alarm_stage1_overview.csv',source+' · 시작 라벨 기준 배타 분류; 전체 에피소드의 미확인 포함 여부는 별도 열 보존')
    candidates=ev[ev.scheme.eq('temporal')&ev.Tool.eq('03M02')&ev.feature_set.eq('B')&ev.baseline.eq('always_alarm')&ev.available]
    pick=candidates[~candidates.detected];r=(pick if len(pick) else candidates).iloc[0];lo=r.fault_time-1800;hi=r.fault_time+120
    raw,_=read_window(r.Tool,lo,hi,'raw',columns=['time','FLOWCOOLPRESSURE'])
    a,_=read_window(r.Tool,lo,hi,'anchors',columns=['time','B_available','segment_id'])
    ep=alarms[alarms.scheme.eq('temporal')&alarms.Tool.eq(r.Tool)&alarms.feature_set.eq('B')&alarms.baseline.eq('always_alarm')&alarms.end.ge(lo)&alarms.start.le(hi)]
    fig,axes=plt.subplots(3,1,figsize=(13,8),sharex=True)
    x,y=broken_line(raw.time.to_numpy(),raw.FLOWCOOLPRESSURE.to_numpy());axes[0].plot(x,y,color='#087e80',lw=1);axes[0].set_ylabel('원본 압력\n익명화 값')
    axes[1].scatter(a.time,np.zeros(len(a)),c=np.where(a.B_available,'#087e80','#929aa4'),s=13);axes[1].set_yticks([0],['B 가용성']);axes[1].set_ylim(-1,1)
    for e in ep.itertuples():
        axes[2].plot([max(lo,e.start),min(hi,e.end)],[0,0],lw=5,color='#7b4ab5')
        if lo<=e.start<=hi:axes[2].scatter(e.start,0,marker='^',s=65,color='#d55e00')
    axes[2].set_yticks([0],['항상 경보 에피소드']);axes[2].set_ylim(-1,1);axes[2].set_xlabel('원시시간 (raw)')
    for ax in axes:
        ax.axvspan(r.fault_time-600,r.fault_time-60,color='#efc2aa',alpha=.4);ax.axvline(r.fault_time,color='#333');ax.axvline(r.fault_time-60,color='#d55e00',ls='--');ax.set_xlim(lo,hi);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    save(fig,'alarm_stage1/actual_event','03M02 실제 구간에서 경보 시작 시점 확인','원본 선택 구간 / alarm_stage1_events.csv / alarm_stage1_episodes.csv',f'{r.event_id} · 유형={r.fault_type} · {lo:.0f}~{hi:.0f} raw · 원본 {len(raw)}행 · 선택 이벤트1개 · 100% 해상도\n주황 음영=예측 구간 · 삼각형=새 경보 발행 · 보라=경보 유지 · 상태={r.status} · 학습 모델 아님')
    dump_json(ROOT/'outputs/alarm_stage1_figure_manifest.json',REGISTRY)
    selection=g[g.baseline.eq('always_alarm')][['scheme','feature_set','recorded_events','available_events','detected_events','alarm_episodes','observed_support_raw','alarm_episodes_per_10000_raw']]
    text='''# 15 첫 단계 결과 — 전체 구간 추론 로더와 경보 평가기

2026-11-05. 사용자와 정한 첫 단계만 실행했다. 모델 학습 및 holdout 점수화는 하지 않았다. 아래 결과는 실제 데이터에 고정 점수0(무경보)/1(항상 경보)을 넣어 평가기를 검증한 결과이며 예측 모델 성능이 아니다.

## 구현 내용

- `src.pretraining.iter_scoring_batches`: 가용 특징이 있는 모든 Anchor를 반환한다. POS/NEG뿐 아니라 NA/EXCLUDE를 유지한다. X에는 기존 특징 화이트리스트만 전달하고 audit_label은 metadata로 분리했다.
- `src.alarm_evaluation`: 같은 Tool·연속 세그먼트에서60raw 간격의 경보를 묶는다. 비경보 Anchor, 가용성 누락, 관측 세그먼트 변경에서 새 에피소드가 시작한다. 청크별로 경보를 따로 세지 않고 Tool의 시간축으로 결합한다.
- 고장 이전 `[tf-600,tf-60]` 안의 **에피소드 최초 발행**을 검출로 인정한다. 훨씬 이른 경보가 유지됐다는 이유로 뒤의 구간에서 새 검출로 소급하지 않는다. 최초 선행시간, 너무 늦은 경보, 입력 부재, 미탐을 분리한다. 다른 유형의 임박 고장으로 EXCLUDE인 시점에서 발행한 경보도 이후 이벤트의 유효 검출로 인정하지 않는다.
- 고장 유형별 event_id는 각각 검출을 평가하되, 한 에피소드가 여러 이벤트에 대응해도 운영 경보 횟수는1회로 센다. 태그 이벤트는 별도 미확정으로 보존한다.
- 미확인 시작·음성 시작·임박/제외 시작 경보를 분리한다. 음성 시작은 해당 보수적 정책 기준이며 현장 확인된 오탐이라는 뜻은 아니다. 에피소드 안에 NA/EXCLUDE가 포함되는지는 별도 플래그로 남긴다.

## 실제 검증 결과

분모는 각 방식의 validation 분할에 속한 모든 원본 이벤트다. 경계 격리 때문에 입력을 확보하지 못한 이벤트도 기록 분모에서 삭제하지 않았다. 두 실험의 원본 기간·장비가 다르므로 서로 더하기/직접 성능 순위 비교를 하지 않는다.

'''+md(selection)+'''

![이벤트 커버리지](../outputs/figures/alarm_stage1/event_coverage.png)

입력 존재는 학습 후보나 경보 성공과 다르다. NA 입력도 관측 가능성에 포함하므로 기존 POS 후보 이벤트 수와 분모가 다를 수 있다.

![점수화 분모](../outputs/figures/alarm_stage1/scoring_population.png)

확정 라벨만 읽으면 빠지는 구간을 보여준다. 모델 적용 구간에서는 이들을 누락하거나 정상0으로 바꾸지 않는다. 현재 시간 순서 검증/B에서 전체 가용745,401 Anchor 중 NA는727,937개(약97.66%)다. NEG16,771개, POS627개, EXCLUDE66개로, 분류 후보만의 점수로 전체 운영 경보 품질을 대표할 수 없다. 이 수치는 보수적 이력 검열과 선택 효과를 포함하며 실제 정상/고장 발생률이 아니다.

![고정 기준 검사](../outputs/figures/alarm_stage1/baseline_checks.png)

두 고정 기준은 평가기 동작 확인용이다. 특히 항상 켜진 오래된 경보는 유효 예측 구간 안의 새 발행이 아니므로 항상 경보가 최대 검출률을 뜻하지 않는다. 실제 검출률·경보 부담 곡선은 다음 단계의 학습 모델 점수가 나온 후 작성한다.

![경보 시작 분류](../outputs/figures/alarm_stage1/alarm_categories.png)

미확인에서 시작한 경보를 확정 오탐에 합치지 않았다. 에피소드별 시작 라벨과 전체 구간 내 미확인 여부를 CSV에서 확인할 수 있다.

![실제 이벤트](../outputs/figures/alarm_stage1/actual_event.png)

고장 시각·선행시간 경계·원본 센서·입력 가용성·경보 시작을 같은 시간축에 놓았다. 보라색 유지 구간이 예측 창에 들어오는 것과 새 경보가 예측 창에서 발행되는 것은 구분한다.

## 경보 부담의 시간 분모

각 실제 고유 관측 s를 명목 간격4raw의 `[s,s+4)`로 표현하고, 같은 세그먼트의 가용 Anchor t에서 점수를 최대60raw 유지하는 `[t,t+60)`와 겹치는 길이를 합산했다. 8/12raw 결측이나 긴 공백을 채우지 않는다. Anchor 수×60을 그대로 사용하지 않는다. 명목 관측 지지구간 길이이며 실제 가동시간이나 초/시간 단위가 아니다. 점수 유지60raw도 본 실험의 명시적 규약이다.

## 표와 Notebook

- [장비별 기준 집계](../outputs/tables/alarm_stage1_summary.csv), [전체 비교표](../outputs/tables/alarm_stage1_overview.csv)
- [전체 점수화 라벨 구성](../outputs/tables/alarm_stage1_label_states.csv)
- [이벤트별 검출·미탐·입력 부재](../outputs/tables/alarm_stage1_events.csv)
- [경보 에피소드](../outputs/tables/alarm_stage1_episodes.csv), [경보-이벤트 매칭](../outputs/tables/alarm_stage1_matches.csv)
- [실행된 Notebook](../notebooks/11_alarm_evaluation.ipynb)

## 검증 기록

전체 단위 테스트38개(경보 평가 관련16개 포함), 실제 입력의 분모·경계·스키마 검사72개, 첫 단계 산출물 대조331개가 통과했다. Notebook 코드 셀6개를 실제 실행했고 오류0개, 그림5개를 포함한다. 원본 크기/수정시각도 유지됐다. 검증 파일은 `outputs/validation/alarm_stage1_*`에 있다. 합성 경계 시험은 실제 데이터 표에 섞지 않았다.

## 로컬 자원과 CUDA

RTX3060 Laptop GPU6GB, RAM 약32GB, 물리8/논리16코어를 확인했다. CUDA 드라이버 장치 인식은 성공했고 XGBoost3.2.0은 CUDA 빌드다. 드라이버 API는12.3, XGBoost 빌드는12.9이므로 GPU 실제 학습 성공을 단정하지 않았다. 다음 단계에서 작은 실제 학습 후보로 GPU 실행과 VRAM을 먼저 확인하며, 오류 시 자동으로 CPU 학습한 결과를 GPU 결과처럼 표시하지 않는다.

설정은 `config/local_compute.json`: CUDA0 + hist, 학습 CPU12스레드, I/O4스레드, 학습 실험1개씩, 초기 max_bin128, 호스트 메모리 예산12GiB, GPU 여유1GiB. 이번 단계의 Parquet 읽기·경보 집계는 CPU 작업이며 GPU로 모델을 학습하지 않았다. 드라이버와 기존 패키지를 변경하지 않았다.

다음 단계는 이 평가기를 연결한 A/B×로지스틱/XGBoost의 첫 학습 비교다. 공식 test와 예약 holdout의 모델 점수는 아직 읽지 않았다.
'''
    (ROOT/'reports/15_alarm_evaluation_stage1.md').write_text(text,encoding='utf-8');print(selection.to_string(index=False))

if __name__=='__main__':main()
