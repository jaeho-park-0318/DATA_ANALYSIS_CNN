"""첫 CUDA 모델 비교의 실제 표·그림·한국어 보고서."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
from src.reporting import md
from src.visuals import plt,save,REGISTRY
from src.query import read_window,broken_line

OUT=ROOT/'outputs/stage2'
COL={'A_logistic':'#74a9cf','A_xgboost':'#0072b2','B_logistic':'#69b79d','B_xgboost':'#087e80'}

def main():
    s=pd.read_parquet(TABLES/'stage2_training_summary.parquet');curve=pd.read_parquet(TABLES/'stage2_alarm_curve.parquet');points=pd.read_parquet(TABLES/'stage2_operating_points.parquet');events=pd.read_parquet(TABLES/'stage2_events.parquet');tele=pd.read_parquet(TABLES/'stage2_resource_telemetry.parquet');unc=pd.read_parquet(TABLES/'stage2_uncertainty.parquet');unseen=pd.read_parquet(TABLES/'stage2_unseen_process.parquet')
    unique=points.drop_duplicates(['run_id','threshold']).copy();maximum=points[points.operating_point.eq('검출최대의100%')].copy();REGISTRY.clear()
    context='실제 temporal validation · H600/L60 · 최대 가용 코호트 · raw 단위 · holdout 미사용'
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,metric,title in zip(axes,['AP_known','ROC_AUC_known'],['알려진 라벨의 Average Precision','알려진 라벨의 ROC-AUC']):
        ax.bar(s.run_id,s[metric],color=[COL[r] for r in s.run_id]);ax.set_title(title);ax.set_ylim(0,1 if metric.startswith('ROC') else max(s[metric])*1.3);ax.tick_params(axis='x',rotation=15)
        for i,r in enumerate(s.itertuples()):ax.text(i,getattr(r,metric),f'{getattr(r,metric):.3f}',ha='center',va='bottom')
        if metric=='AP_known':ax.scatter(range(len(s)),s.validation_positive/s.validation_known_rows,marker='_',s=200,color='#d55e00',label='동일 후보 내 양성 비율');ax.legend(fontsize=8)
        else:ax.axhline(.5,ls='--',color='#999',label='무작위 순위 기준');ax.legend(fontsize=8)
    save(fig,'stage2/classification','분류 점수만으로 경보 모델을 선정할 수 없음','stage2_training_summary.csv',context+'\nA: 25,266개 중 POS776 · B: 17,398개 중 POS627 · NA/EXCLUDE는 분류 지표에서 제외')
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,metric,title in zip(axes,['recall_available','recall_recorded'],['입력 있는 고장 대비 검출률','전체 기록 고장 대비 검출률']):
        for rid,color in COL.items():
            q=curve[curve.run_id.eq(rid)];f=q[q.pareto].sort_values('alarm_episodes_per_10000_raw');ax.scatter(q.alarm_episodes_per_10000_raw,q[metric],color=color,s=18,alpha=.3);ax.plot(f.alarm_episodes_per_10000_raw,f[metric],'o-',color=color,label=rid,lw=1.5,ms=5)
        ax.set_title(title);ax.set_xlabel('명목 관측 지지길이 10,000 raw당 경보');ax.set_ylabel('고유 이벤트 검출 비율');ax.set_ylim(-.015,.65);ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'stage2/alarm_tradeoff','검출률과 경보 부담의 동시 비교','stage2_alarm_curve.csv',context+'\n진한 선: 탐색한 점의 비지배 후보 연결(보간 성능 아님) · A/B 분모·학습 코호트가 달라 순수 특징 효과는 아님')
    fig,axes=plt.subplots(1,2,figsize=(13,6));cats=['MATCHED','KNOWN_NEGATIVE_START','UNKNOWN_START','LATE_OR_EXCLUDED_START','POSITIVE_UNMATCHED'];names=['이벤트 매칭','정책상 음성 시작','미확인 시작','제외 시작','양성 미매칭']
    bottom=np.zeros(len(maximum))
    for cat,name,color in zip(cats,names,['#087e80','#0072b2','#a2aab1','#7b4ab5','#d55e00']):
        v=maximum['alarms_'+cat]/maximum.alarm_episodes;axes[0].bar(maximum.run_id,v,bottom=bottom,label=name,color=color);bottom+=v
    axes[0].tick_params(axis='x',rotation=15);axes[0].set_ylabel('전체 경보 에피소드 대비 비율');axes[0].set_title('모델별 탐색 최대 검출 운영점');axes[0].legend(fontsize=8,loc='lower left')
    z=unc[unc.operating_point.eq('검출최대의100%')&unc.denominator.eq('available')].set_index('run_id').reindex(s.run_id)
    axes[1].errorbar(np.arange(len(z)),z.recall,yerr=[z.recall-z.ci_low,z.ci_high-z.recall],fmt='o',capsize=5,color='#087e80');axes[1].set_xticks(range(len(z)),z.index,rotation=15);axes[1].set_ylim(0,.8);axes[1].set_ylabel('입력 있는 이벤트 검출률');axes[1].set_title('Tool 군집 재표집 95% 구간')
    save(fig,'stage2/alarm_uncertainty','경보 대부분은 정답 미확인 구간에서 시작','stage2_operating_points.csv / stage2_uncertainty.csv',context+'\n20 Tool 재표집 2,000회 · 선택 임계값 고정 조건부 구간 · 선택 낙관·장비쌍 의존성·이력 불완전성 미보정')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,f in zip(axes,'AB'):
        history=json.loads((OUT/f'{f}_xgboost_history.json').read_text(encoding='utf-8'))['validation_0']['aucpr'];r=s[s.run_id.eq(f+'_xgboost')].iloc[0]
        ax.plot(np.arange(1,len(history)+1),history,color=COL[f+'_xgboost']);ax.axvline(r.iterations,ls='--',color='#d55e00',label=f'채택 {r.iterations} trees / 평가 {len(history)}');ax.set_title(f+' XGBoost validation');ax.set_xlabel('부스팅 반복');ax.set_ylabel('XGBoost aucpr');ax.legend(fontsize=8)
    save(fig,'stage2/learning_history','검증 성능 기준 조기 종료 이력','A_xgboost_history.json / B_xgboost_history.json',context+'\naucpr은 XGBoost 내부 지표이며 sklearn AP와 다른 계산 · 학습 손실 곡선은 수집하지 않음')
    fig,axes=plt.subplots(1,2,figsize=(12,5));seconds=tele.unix_time-tele.unix_time.min()
    axes[0].plot(seconds,tele.rss_bytes/2**30,'o-',color='#087e80');axes[0].set_ylabel('학습 프로세스 RSS (GiB)');axes[0].set_xlabel('첫 측정 후 경과 (초)');axes[0].set_title('읽기·전처리·학습·저장 포함')
    axes[1].plot(seconds,tele.gpu_used_mib,'o-',color='#0072b2',label='GPU 전체 사용 메모리');axes[1].set_ylabel('MiB');axes[1].set_xlabel('첫 측정 후 경과 (초)');axes[1].set_title('다른 프로세스 사용량 포함');axes[1].legend(fontsize=8)
    save(fig,'stage2/local_resources','RTX3060 Laptop CUDA 실제 학습과 로컬 자원 사용','stage2_resource_telemetry.csv / cuda_training_smoke.json',f'CUDA smoke 및 A/B XGBoost device=cuda:0 확인 · CPU 최대12스레드 설정\n약2초 간격 {len(tele)}회 관측; 짧은 GPU 학습의 순간 최대 사용량/이용률을 보장하지 않음')
    r=maximum[maximum.run_id.eq('B_xgboost')].iloc[0];q=events[events.run_id.eq(r.run_id)&events.threshold.eq(r.threshold)&events.available]
    picks=[q[q.detected].sort_values('lead_raw').iloc[len(q[q.detected])//2],q[~q.detected].sort_values(['Tool','fault_time']).iloc[0]]
    pred=pd.read_parquet(OUT/'B_xgboost_validation.parquet');alarms=pd.read_parquet(TABLES/'stage2_selected_episodes.parquet');fig,axes=plt.subplots(2,2,figsize=(14,8));case_rows=[]
    for col,e in enumerate(picks):
        lo=e.fault_time-1500;hi=e.fault_time+120;raw,_=read_window(e.Tool,lo,hi,'raw',columns=['time','FLOWCOOLPRESSURE']);x,y=broken_line(raw.time.to_numpy(),raw.FLOWCOOLPRESSURE.to_numpy());axes[0,col].plot(x,y,lw=1,color='#087e80');axes[0,col].set_title(f'{e.Tool} / {e.event_id}\n{e.status}');axes[0,col].set_ylabel('원본 압력 (익명화 값)')
        p=pred[pred.Tool.eq(e.Tool)&pred.time.between(lo,hi)].sort_values('time');groups=((p.time.diff()!=60)|(p.segment_id.ne(p.segment_id.shift()))).cumsum()
        for _,g in p.groupby(groups):axes[1,col].plot(g.time,g.score,'o-',ms=3,lw=1,color='#0072b2')
        a=alarms[alarms.run_id.eq(r.run_id)&alarms.threshold.eq(r.threshold)&alarms.Tool.eq(e.Tool)&alarms.start.between(lo,hi)]
        axes[1,col].scatter(a.start,np.full(len(a),r.threshold),marker='^',color='#d55e00',s=45,label='새 경보 시작');axes[1,col].axhline(r.threshold,ls=':',color='#7b4ab5',label=f'임계값 {r.threshold:.7f}');axes[1,col].set_ylabel('위험 점수 (확률 보정 미검증)');axes[1,col].set_xlabel('원시시간 (raw)');axes[1,col].legend(fontsize=8)
        for ax in axes[:,col]:ax.axvspan(e.fault_time-600,e.fault_time-60,color='#efc2aa',alpha=.35);ax.axvline(e.fault_time,color='#333');ax.axvline(e.fault_time-60,color='#d55e00',ls='--');ax.set_xlim(lo,hi);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        case_rows.append(dict(event_id=e.event_id,Tool=e.Tool,status=e.status,fault_time=e.fault_time,window_start=lo,window_end=hi,raw_rows=len(raw),score_anchors=len(p),lead_raw=e.lead_raw,threshold=r.threshold))
    save(fig,'stage2/event_examples','B XGBoost의 실제 검출·미탐 구간','원본 선택 구간 / B_xgboost_validation.parquet / stage2_selected_episodes.csv',context+'\n좌: 검출 사례 선행시간 중앙 순위 · 우: Tool/시각 순 첫 미탐 · 회색 선=고장 · 음영=유효 예측 창 · 공백 연결 없음')
    save_table(pd.DataFrame(case_rows),'stage2_example_cases');dump_json(ROOT/'outputs/stage2_figure_manifest.json',REGISTRY)
    summary=s[['run_id','device','train_rows','train_positive','validation_known_rows','encoded_features','iterations','AP_known','ROC_AUC_known','train_seconds']].rename(columns={'run_id':'모델','device':'학습 장치','train_rows':'학습 행','train_positive':'학습 POS','validation_known_rows':'분류 검증 행','encoded_features':'인코딩 특징 수','iterations':'반복/LR 또는 채택tree','AP_known':'AP','ROC_AUC_known':'ROC-AUC','train_seconds':'fit초'})
    selected=unique[['run_id','threshold','detected_events','available_events','recall_recorded','recall_available','alarm_episodes','alarm_episodes_per_10000_raw','median_lead_raw']].rename(columns={'run_id':'모델','threshold':'임계값','detected_events':'검출','available_events':'입력 고장','recall_recorded':'전체 검출률','recall_available':'입력 검출률','alarm_episodes':'경보 횟수','alarm_episodes_per_10000_raw':'10000raw당 경보','median_lead_raw':'선행시간 중앙 raw'})
    selected['임계값']=selected['임계값'].map(lambda v:format(v,'.17g'))
    b=maximum[maximum.run_id.eq('B_xgboost')].iloc[0];a=maximum[maximum.run_id.eq('A_xgboost')].iloc[0]
    process=unseen.groupby('feature_set')[['rows','unknown_recipe','unknown_stage','unseen_combination']].sum().reset_index()
    text=f'''# 16 첫 모델 학습 결과 — CUDA A/B 기준 비교

2026-11-05. 경보 평가기를 독립 구현과 대조한 뒤 계획의 2단계인 A/B × 로지스틱 회귀/XGBoost 4개를 실제 학습했다. **추가 검토 후보는 B XGBoost이며, 최종 모델·운영 임계값은 아직 확정하지 않았다.**

## 핵심 결과

- 보완한 임계값 탐색에서 B XGBoost는 입력 있는 고장 {int(b.available_events)}개 중 **{int(b.detected_events)}개({b.recall_available:.1%})**를 검출했다. 전체 기록 232개 기준은 **{b.recall_recorded:.1%}**다.
- 해당 후보의 경보는 {int(b.alarm_episodes):,}회, 명목 관측 지지길이 10,000 raw당 **{b.alarm_episodes_per_10000_raw:.2f}회**다. A XGBoost의 탐색 최대 검출 후보는 {int(a.detected_events)}개, 같은 단위 경보 {a.alarm_episodes_per_10000_raw:.2f}회다. A/B 관측 기간·학습 후보가 달라 순수 특징 효과로 단정하지 않는다.
- B XGBoost 경보 중 {int(b.alarms_UNKNOWN_START):,}회({b.alarms_UNKNOWN_START/b.alarm_episodes:.1%})가 **미확인 라벨에서 시작**한다. 이를 확정 오경보로 바꾸지 않는다. 현장 운영 가능성을 입증한 결과가 아니다.
- 분류 AP는 A XGBoost가 가장 높지만 경보 지표의 순위는 다르다. 모든 모델의 알려진 라벨 ROC-AUC는 0.5 미만이다. 일부 상위 점수의 양성 집중과 전체 순위 성능은 다르며, 음성 선택/시점 변화/점수 분포를 후속 점검해야 한다.

## 경보 평가 검증

기존 단위 테스트 38개와 새 Pareto 선정 계약 테스트 2개, 총40개를 통과했다. 실제 validation 시각에 시험 점수 0/1/주기 점수를 얹은 27개 검사를 별도 순차 구현과 대조해 모두 일치했다. 이 시험 점수는 모델 성능 표에 섞지 않았다. 경보 첫 시작만 `[고장-600, 고장-60]`에 있어야 검출로 인정한다. EXCLUDE 시작은 다른 고장에도 검출로 인정하지 않는다. 긴 공백/세그먼트 변경/가용성 단절은 경보를 나눈다.

## 학습 구성과 분류 보조 지표

{md(summary)}

전처리 사전·표준화는 train에서만 추정했다. 숫자는 StandardScaler, recipe/stage는 One-Hot이며 미등록 범주는 all-zero로 처리한다. A8/B32는 인코딩 전 특징 수이고 위 표는 인코딩 후 수다. 시간·장비ID·TTF·라벨·미래 이력은 X에 넣지 않았다. train 후보 전체를 사용했고 SMOTE/음성 축소/클래스 가중치 증폭은 하지 않았다. H600/L60, conservative, temporal 60/20/20, 경계 격리 900/1800 raw, seed42다.

로지스틱은 C=1, L2, lbfgs, 최대1500회이며 둘 다 수렴했다. XGBoost는 depth3, eta0.05, 최대1000trees, subsample/colsample0.8, min_child_weight10, lambda5, max_bin128, CUDA hist, CPU12스레드다. 알려진 validation 라벨의 XGBoost aucpr로 50회 조기 종료했다. **채택 tree 수와 실제 평가한 tree 수는 다르다.** fit 시간은 전처리/읽기/예측/저장 제외이며 CPU/GPU 속도 비교 실험이 아니다.

![분류 지표](../outputs/figures/stage2/classification.png)

![조기 종료](../outputs/figures/stage2/learning_history.png)

B XGBoost의 채택 반복이 매우 이르다. 현재 설정에서 validation aucpr이 빠르게 나빠졌다는 관찰이며, 학습 손실은 수집하지 않았으므로 이를 학습 실패나 과적합으로 단정하지 않는다. 채택 모델의 전체 validation 점수는 16개의 고유 값으로만 나타나며 대부분 0.003 부근에 몰려 있다. 따라서 임계값 반올림이나 작은 변화가 경보 수를 크게 바꿀 수 있다. 양성 부족, 공정 분포 변화, 특징/정규화/가중치 영향을 후속 실험에서 분리한다. 초기 학습 후보의 POS 비율은 A 약0.255%, B 약0.298%; 알려진 validation의 POS 비율은 A 약3.07%, B 약3.60%로 선택 분포가 다르다. 위험 점수는 보정된 고장 확률이 아니다.

## 검출률과 경보 부담

{md(selected)}

표의 중복 임계값은 제거했다. 각 모델의 **탐색 격자 내 최대 검출수** 50/80/100% 이상을 달성하는 비지배 후보 중 경보 부담이 가장 적은 점을 남겼다. 같은 점이 여러 기준을 만족하면 고유 운영점은 3개보다 적다. 전역 최적 임계값이나 실제 요구 검출률을 뜻하지 않는다. 임계값 소수점 전체 값은 [운영점 CSV](../outputs/tables/stage2_operating_points.csv)에 있다.

고정 임계값12개와 전체 validation 점수 분위수16개를 합쳐 중복 제거했다. 첫 8분위수 탐색 후 낮은 점수의 밀집 구간을 확인하고 분위수를 보완했다. 항상 경보/무경보도 포함한다. 경보 첫 시작 규칙 때문에 임계값에 따른 검출률과 에피소드 수는 단조롭지 않다. 낮은 임계값의 성능은 가용성 단절로 생긴 재발행에 의존할 수 있다.

![경보 비교](../outputs/figures/stage2/alarm_tradeoff.png)

A 가용 Anchor 1,061,290개, 관측 지지길이 60,897,220 raw; B 745,401개, 42,090,658 raw다. 노출은 같은 segment의 점수 유지 `[t,t+60)`와 명목 관측 `[s,s+4)`의 교집합이며 실제 가동시간이 아니다. B 전체 입력의 약97.66%는 NA이며 계속 점수화했다. 무경보는 검출/부담0, 항상 경보는 A25/98(10.44회), B31/95(14.74회)다. 항상 경보가 최대 검출률은 아니다.

![경보 종류와 불확실성](../outputs/figures/stage2/alarm_uncertainty.png)

20 Tool 군집을 2,000회 seed42로 재표집해 95% 구간을 계산했다. 임계값을 고정한 조건부 변동 범위이며 검증 자료로 임계값을 고른 낙관, 장비쌍 의존성, 누락된 고장 이력은 보정하지 못한다. 선행시간 하위10%/중앙값과 장비별 결과는 [불확실성 표](../outputs/tables/stage2_uncertainty.csv), [장비별 이벤트 표](../outputs/tables/stage2_tool_event_results.csv)에 있다. 검출되지 않은 고장을 선행시간 통계에 넣지 않았다.

## 미등록 공정 진단

{md(process)}

분모는 해당 특징의 전체 validation 입력이다. recipe/stage 개별 범주와 두 범주의 조합 미등록을 따로 센다. train 공정 조합 목록은 학습 때와 동일한 원본·후보 계약으로 학습 후 재구성했고 validation을 fit하지 않았다. 숫자/공정별 원인 분리는 후속 공통 코호트 실험으로 남긴다.

## 실제 이벤트 사례

![실제 검출과 미탐](../outputs/figures/stage2/event_examples.png)

두 사례는 실제 원본을 전체 해상도로 조회했다. 검출 사례는 선행시간 중앙 순위, 미탐은 Tool/시각 순 첫 가용 이벤트를 선택했으며 전체 성능을 대표하도록 임의 일반화하지 않는다. 오른쪽 미탐 사례는 점수가 계속 임계값 이상이지만 유효 예측 창 안의 새 경보 발행이 없어 미탐이다. 경보는 전체 validation 시계열에서 계산한 뒤 확대했으므로 창 왼쪽에서 임의로 새 경보를 만들지 않았다. [사례 시각·행 수](../outputs/tables/stage2_example_cases.csv).

## 로컬 자원과 CUDA 확인

RTX3060 Laptop에서 실제 train 후보 50,000행/20trees smoke 학습 후 booster 설정의 `cuda:0`를 확인했다. A/B XGBoost 본학습도 같은 장치 설정을 확인했다. 로지스틱은 CPU12스레드 제한을 사용했다. GPU 학습은 순차 실행했으며 환경/드라이버/패키지를 변경하지 않았다.

관측된 프로세스 최대 RSS는 {tele.rss_bytes.max()/2**30:.2f}GiB, GPU 전체 사용 메모리는 {tele.gpu_used_mib.max():,.0f}MiB, 최소 여유는 {tele.gpu_free_mib.min():,.0f}MiB였다. 약2초 샘플링으로 짧은 학습의 순간 최고치는 놓칠 수 있다. GPU 사용량에는 다른 프로세스도 포함하므로 모델 전용 VRAM으로 해석하지 않는다. 학습이 짧아 높은 지속 이용률을 입증하지는 않는다.

![자원 기록](../outputs/figures/stage2/local_resources.png)

## 다음 방향

1. 계획대로 A/B/C **공통 코호트에서 각각 재학습**해 특징 효과와 입력 가용성 효과를 분리한다. 현재는 B XGBoost를 우선 검토하되 A와 단순 모델을 유지한다.
2. 공통 코호트 결과를 본 뒤 깊이/양성 가중치의 작은 탐색과 seed42/43/44 민감도를 비교한다. B의 이른 조기 종료와 미등록 공정/점수 분포 변화를 함께 진단한다.
3. 주 후보를 고정한 뒤 장비 그룹·H300/1800 일반화를 검토한다. 반복 경보/쿨다운 변경은 별도 평가 계약으로 비교한다.
4. 최종 모델·경보 규칙·운영점 후보를 validation에서 고정한 뒤 예약 평가를 한 번 수행한다. 이번에는 **holdout 점수를 계산하지 않았다**.

이번 validation은 조기 종료·임계값 탐색·모델 비교에 재사용했다. 보고된 수치는 탐색 결과이며 독립 일반화 성능이 아니다. 시간 단위, 정비 종료, 고장 이력 완전성 확인은 계속 필요하다.

## 재현과 산출물

- [실행 결과 포함 Notebook](../notebooks/12_initial_model_training.ipynb)
- [학습 요약 CSV](../outputs/tables/stage2_training_summary.csv), [전체 경보 곡선 CSV](../outputs/tables/stage2_alarm_curve.csv), [event_id별 결과 CSV](../outputs/tables/stage2_events.csv)
- [학습 실행 코드](../scripts/train_stage2.py), [경보 평가 코드](../scripts/evaluate_stage2.py), [사후 진단 코드](../scripts/diagnose_stage2.py)
- [모델·전처리·예측 저장 폴더](../outputs/stage2), [모델 파라미터](../outputs/stage2/model_parameters.json), [아티팩트 SHA256·실제 버전](../outputs/validation/stage2_artifacts.json)
- [독립 경보 검사](../outputs/validation/alarm_independent_audit.json), [실제 경보 집계 검사](../outputs/validation/stage2_evaluation.json)

Notebook은 저장된 실제 결과와 모델을 다시 읽고 검증하며, 열 때 전체 재학습을 자동 실행하지 않는다. 재학습은 `scripts/train_stage2.py`로 명시적으로 실행한다. 원본은 읽기 전용이고 내용 전체 해시는 미계산; 65개 CSV의 크기/mtime가 유지됨을 확인했다.
'''
    (ROOT/'reports/16_stage2_training_results.md').write_text(text,encoding='utf-8')
    print('REPORT COMPLETE: '+str(len(REGISTRY))+' figures')

if __name__=='__main__':main()
