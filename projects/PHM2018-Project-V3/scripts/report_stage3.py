"""같은 학습/평가 코호트에서 특징 효과를 비교한 표·그림·보고서."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
from src.reporting import md
from src.visuals import plt,save,REGISTRY

OUT=ROOT/'outputs/stage3'
COLORS3={'A':'#0072b2','B':'#087e80','C':'#d55e00'}

def main():
    def table(n):return pd.read_parquet(TABLES/('stage3_'+n+'.parquet'))
    s=table('training_summary');curve=table('alarm_curve');points=table('budget_points');deltas=table('paired_deltas');unc=table('uncertainty');counts=table('cohort_counts');tele=table('resource_telemetry');be=table('budget_events')
    old=pd.read_parquet(TABLES/'stage2_training_summary.parquet');oldmax=pd.read_parquet(TABLES/'stage2_operating_points.parquet');oldmax=oldmax[oldmax.operating_point.eq('검출최대의100%')]
    n_available=int(curve.available_events.iloc[0]);n_recorded=int(curve.recorded_events.iloc[0]);support=curve.observed_support_raw.iloc[0];order=[f+'_'+m for m in ['logistic','xgboost'] for f in 'ABC'];REGISTRY.clear()
    context=f'실제 temporal validation · 공통 코호트 · H600/L60 · 입력 고장 {n_available}/{n_recorded}개 · holdout 미사용'
    fig,axes=plt.subplots(1,2,figsize=(13,5));labels=['이전 A 최대','이전 B 최대','공통 A','공통 B','공통 C']
    nums=[int(old[old.run_id.eq('A_xgboost')].train_rows.iloc[0]),int(old[old.run_id.eq('B_xgboost')].train_rows.iloc[0]),*[int(s.train_rows.iloc[0])]*3]
    axes[0].bar(labels,nums,color=['#b8c6cd','#b8c6cd',*[COLORS3[f] for f in 'ABC']]);axes[0].set_ylabel('학습 후보 행');axes[0].tick_params(axis='x',rotation=12);axes[0].set_ylim(0,max(nums)*1.2)
    for i,v in enumerate(nums):axes[0].text(i,v,f'{v:,}',ha='center',va='bottom',fontsize=9)
    avail=[98,95,n_available,n_available,n_available];axes[1].bar(labels,[n_recorded]*5,color='#e4e8ec',label='전체 기록 이벤트');axes[1].bar(labels,avail,color=['#b8c6cd','#b8c6cd',*[COLORS3[f] for f in 'ABC']],label='예측 창 입력 존재');axes[1].set_ylabel('고유 고장 이벤트');axes[1].tick_params(axis='x',rotation=12);axes[1].legend(fontsize=8)
    for i,v in enumerate(avail):axes[1].text(i,v+3,str(v),ha='center')
    save(fig,'stage3/common_cohort','공통 코호트로 맞추면서 줄어든 학습량과 이벤트 커버리지','stage2_training_summary.csv / stage3_training_summary.csv / stage3_alarm_curve.csv',context+'\n이전 최대 코호트와 현재 공통 코호트의 점수 차이를 순수 특징 효과로 해석하지 않음')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,col,title in zip(axes,['AP_known','ROC_AUC_known'],['Average Precision (AP)','ROC-AUC']):
        for i,model in enumerate(['logistic','xgboost']):
            z=s[s.model.eq(model)].set_index('feature_set').reindex(list('ABC'));x=np.arange(3)+(i-.5)*.32;ax.bar(x,z[col],width=.32,label=model,color=['#82b5c5','#087e80'][i])
            for xx,yy in zip(x,z[col]):ax.text(xx,yy,f'{yy:.3f}',ha='center',va='bottom',fontsize=9)
        ax.set_xticks(range(3),list('ABC'));ax.set_title(title);ax.set_ylim(0,1 if col.startswith('ROC') else .35);ax.legend(fontsize=8)
        if col.startswith('ROC'):ax.axhline(.5,ls='--',color='#888')
    save(fig,'stage3/classification','같은 정답 후보에서 비교한 분류 보조 지표','stage3_training_summary.csv',context+f'\n분류 후보 {s.validation_known_rows.iloc[0]:,}개, POS {s.validation_positive.iloc[0]:,}개 · NA/EXCLUDE 제외 · 전처리 train 전용')
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),sharey=True)
    for ax,model in zip(axes,['logistic','xgboost']):
        for f,color in COLORS3.items():
            q=curve[curve.run_id.eq(f+'_'+model)];front=q[q.pareto].sort_values('alarm_episodes_per_10000_raw');ax.scatter(q.alarm_episodes_per_10000_raw,q.recall_available,color=color,s=15,alpha=.25);ax.plot(front.alarm_episodes_per_10000_raw,front.recall_available,'o-',label=f,color=color,ms=4)
        ax.set_title(model);ax.set_xlabel('명목 관측 지지길이 10,000 raw당 경보');ax.set_ylim(-.015,.55);ax.legend(title='특징');ax.grid(alpha=.2)
    axes[0].set_ylabel('입력 있는 고장 검출률')
    save(fig,'stage3/alarm_tradeoff','같은 입력·이벤트·관측 지지길이에서 경보 비교','stage3_alarm_curve.csv',context+f'\n공통 관측 지지길이 {support:,.0f} raw · 비지배 점 연결은 보간 성능을 뜻하지 않음 · 임계값별 비단조 허용')
    joint=table('joint_frontier');fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for ax,limit,title in zip(axes,[12,1.5],['전체 경보 부담 범위','매우 낮은 부담 구간 확대']):
        for f,color in COLORS3.items():
            z=joint[joint.run_id.eq(f+'_xgboost')&joint.alarm_episodes_per_10000_raw.le(limit)]
            ax.scatter(z.alarm_episodes_per_10000_raw,z.detected_events,s=65,color=color,label=f)
            for r in z.itertuples():ax.annotate(f'{int(r.detected_events)}/{n_available}',(r.alarm_episodes_per_10000_raw,r.detected_events),xytext=(4,5),textcoords='offset points',fontsize=8)
        ax.set_xlim(0,limit);ax.set_ylim(0,22 if limit>2 else 8);ax.set_title(title);ax.set_xlabel('명목 관측 지지길이 10,000 raw당 경보');ax.set_ylabel('검출 고장 수');ax.legend(title='XGBoost 특징');ax.grid(alpha=.2)
    save(fig,'stage3/joint_frontier','A/B/C를 함께 비교하면 남는 경보 운영 후보','stage3_joint_frontier.csv',context+'\n전체 탐색 점에서 검출 수↑·부담↓의 비지배 후보; 무경보 제외 · 5/10/20 상한 표와 별도의 전체 곡선 해석')
    fig,axes=plt.subplots(1,3,figsize=(14,6),sharey=True)
    for ax,budget in zip(axes,[5,10,20]):
        z=points[points.budget_per_10000_raw.eq(budget)].set_index('run_id').reindex(order);ax.barh(order,z.detected_events,color=[COLORS3[r[0]] for r in order]);ax.set_xlim(0,max(points.detected_events)*1.35);ax.set_title(f'경보 상한 {budget}회 / 10,000 raw');ax.set_xlabel('검출 고장 수')
        for y,r in enumerate(z.itertuples()):ax.text(r.detected_events+.15,y,f'{int(r.detected_events)}/{n_available}  ({r.alarm_episodes_per_10000_raw:.2f}회)',va='center',fontsize=8)
    axes[0].invert_yaxis()
    save(fig,'stage3/budget_comparison','같은 경보 부담 상한에서 비교한 검출 수','stage3_budget_points.csv',context+'\n5/10/20은 결과 확인 전에 정한 비교용 상한; 실제 운영 허용치 아님 · 괄호=선택점의 실제 경보 부담')
    fig,axes=plt.subplots(1,2,figsize=(13,7),sharex=True)
    for ax,model in zip(axes,['logistic','xgboost']):
        z=deltas[deltas.model.eq(model)].copy();labels=[f'{r.comparison}, 상한 {r.budget_per_10000_raw}' for r in z.itertuples()]
        ax.errorbar(z.recall_delta*100,np.arange(len(z)),xerr=[(z.recall_delta-z.ci_low)*100,(z.ci_high-z.recall_delta)*100],fmt='o',capsize=4,color='#087e80');ax.axvline(0,color='#888',ls='--');ax.set_yticks(range(len(z)),labels);ax.invert_yaxis();ax.set_title(model);ax.set_xlabel('입력 고장 검출률 차이 (%p)');ax.grid(axis='x',alpha=.2)
    save(fig,'stage3/paired_differences','특징 추가에 따른 대응 검출률 차이와 불확실성','stage3_paired_deltas.csv',context+'\n01~10 장비 번호 군집(M01/M02 함께) 재표집 2,000회 · 고정 임계값 조건부 95% 구간 · 다중 비교/선택 낙관 미보정')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for f,color in COLORS3.items():
        h=json.loads((OUT/f'{f}_xgboost_history.json').read_text(encoding='utf-8'))['validation_0']['aucpr'];it=int(s[s.run_id.eq(f+'_xgboost')].iterations.iloc[0]);axes[0].plot(np.arange(1,len(h)+1),h,color=color,label=f'{f}: 채택 {it} / 평가 {len(h)}');axes[0].scatter(it,h[it-1],color=color,s=45)
    axes[0].set_xlabel('부스팅 반복');axes[0].set_ylabel('XGBoost validation aucpr');axes[0].legend(fontsize=8);axes[0].set_title('같은 조기 종료 규칙, 서로 다른 채택 반복')
    seconds=tele.unix_time-tele.unix_time.min();axes[1].plot(seconds,tele.rss_bytes/2**30,'o-',ms=4,color='#087e80',label='프로세스 RSS');axes[1].set_xlabel('첫 측정 후 경과 (초)');axes[1].set_ylabel('RSS (GiB)');axes[1].set_title('로컬 읽기·전처리·학습·저장');axes[1].legend(fontsize=8)
    save(fig,'stage3/training_resources','CUDA 학습 이력과 로컬 메모리 사용','stage3_resource_telemetry.csv / 각 XGBoost history.json',context+f'\nA/B/C 실제 booster 장치 cuda:0 확인 · CPU12스레드 · 약2초 간격 {len(tele)}회 측정; 순간 최고치 미보장')
    # 같은 53개 이벤트의 탐색용 상한10 결과를 한 행씩 직접 비교한다.
    e=be[be.budget_per_10000_raw.eq(10)&be.available&~be.tagged];matrix=e.pivot(index='event_id',columns='run_id',values='detected').reindex(columns=order)
    meta=e[['event_id','Tool','fault_time']].drop_duplicates().sort_values(['Tool','fault_time']);matrix=matrix.reindex(meta.event_id)
    fig,ax=plt.subplots(figsize=(11,12));im=ax.imshow(matrix.to_numpy(int),aspect='auto',vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(6),order,rotation=15);ax.set_yticks(range(len(meta)),[f'{r.Tool} · {r.fault_time:.0f}' for r in meta.itertuples()],fontsize=7);fig.colorbar(im,ax=ax,ticks=[0,1],label='0 미탐 / 1 검출')
    save(fig,'stage3/event_matrix','같은 고장 이벤트별 검출·미탐 대조','stage3_budget_events.csv',context+'\n경보 상한10 비교용 운영점 · 행은 동일 event_id, 표시=Tool/고장 raw시각 · 입력 부재 이벤트는 이 그림에서 제외')
    dump_json(ROOT/'outputs/stage3_figure_manifest.json',REGISTRY)
    summary=s[['run_id','device','train_rows','train_positive','encoded_features','iterations','AP_known','ROC_AUC_known','train_seconds']].rename(columns={'run_id':'모델','device':'장치','train_rows':'학습 행','train_positive':'POS','encoded_features':'인코딩 특징','iterations':'LR반복/채택tree','AP_known':'AP','ROC_AUC_known':'ROC-AUC','train_seconds':'fit초'})
    shown=points[['run_id','budget_per_10000_raw','threshold','detected_events','recall_available','recall_recorded','alarm_episodes_per_10000_raw','median_lead_raw']].copy();shown['threshold']=shown.threshold.map(lambda v:format(v,'.17g'));shown=shown.rename(columns={'run_id':'모델','budget_per_10000_raw':'경보 상한','threshold':'임계값','detected_events':'검출','recall_available':'입력 검출률','recall_recorded':'전체 검출률','alarm_episodes_per_10000_raw':'실제 경보/10000raw','median_lead_raw':'중앙 선행 raw'})
    xpoints=points[points.run_id.str.endswith('xgboost')];best=points[points.budget_per_10000_raw.eq(10)].sort_values(['detected_events','alarm_episodes_per_10000_raw'],ascending=[False,True]).iloc[0]
    xd=deltas[deltas.model.eq('xgboost')&deltas.budget_per_10000_raw.eq(10)]
    v=counts[counts.partition.eq('validation_all')].iloc[0]
    text=f'''# 17 공통 코호트 재학습 — A/B/C 특징 효과 비교

2026-11-05. 계획의 3단계로 A/B/C × 로지스틱 회귀/XGBoost **6개 모델을 새로 학습**했다. 이전 최대 코호트 모델의 점수만 잘라 비교한 실험이 아니다. CUDA를 사용했고 holdout은 점수화하지 않았다.

## 주요 결과

- 모든 모델은 **train {int(s.train_rows.iloc[0]):,}행(POS {int(s.train_positive.iloc[0]):,})**, 전체 가용 validation **{int(s.validation_all_rows.iloc[0]):,}행**을 공유한다. 행·라벨·순서 해시와 전체 저장 예측의 metadata 일치를 확인했다.
- validation의 기록 고장 {n_recorded}개 중 공통 입력이 있는 고장은 **{n_available}개({n_available/n_recorded:.1%})**다. C의900 raw 이력을 요구하면서 이전 A98/B95개보다 커버리지가 줄었다.
- 탐색용 경보 상한10에서 검출 수가 가장 많고 동률 시 부담이 가장 적은 후보는 **{best.run_id}**, 검출 **{int(best.detected_events)}/{n_available}({best.recall_available:.1%})**, 실제 경보 **{best.alarm_episodes_per_10000_raw:.2f}/10,000 raw**다. 상한별 결과를 모두 아래에 제시한다.
- 분류 AP는 XGBoost A {s[s.run_id.eq('A_xgboost')].AP_known.iloc[0]:.3f}, B {s[s.run_id.eq('B_xgboost')].AP_known.iloc[0]:.3f}, C {s[s.run_id.eq('C_xgboost')].AP_known.iloc[0]:.3f}다. **복잡한 특징이 자동으로 더 좋은 결과를 내지는 않았다.** 최종 특징 선택은 경보 비교와 불확실성을 함께 본다.
- 비교용 경보 상한5에서는 B XGBoost 13/53, A 9/53으로 B가 앞섰다. 상한10에서는 A17/B13/C12개로 A가 앞섰다. 현재는 **상한5의 B와 검출 확대의 A를 주 후보로 유지**하는 것이 적절하다. C는 이 세 상한에서 앞서지 않았지만, 전체 곡선의 매우 낮은 경보 부담에서는 비지배 후보로 남는다.

## 동일 조건을 보장한 방법

`cohort=common`은 A/B/C 가용성의 교집합이며 현재 특징 정의에서 C 가용성과 같다. 각 특징마다 전처리와 모델을 독립적으로 train에서 새로 fit했다. 6개 모델의 학습 행·정답·순서, 평가 시각/세그먼트/라벨이 같다. 공유된 현재 센서6개의 StandardScaler 평균/스케일과 recipe/stage 범주 사전도 완전히 일치한다. 추가되는 특징만 A8→B32→C44로 달라진다.

seed42, 모델 파라미터, 전처리 방식, conservative 라벨, temporal 분할, H600/L60, 경계 격리900/1800 raw, 경보 첫 발행 규칙은 2단계와 동일하다. XGBoost는 depth3/eta0.05/max_bin128/hist/CUDA, 최대1000trees와 validation aucpr 50회 조기 종료다. 채택 tree 수는 각 특징의 validation 결과에 따라 다르므로 **같은 학습·선택 절차의 특징 구성 비교**이지 트리 구조·최종 tree 수까지 고정한 비교는 아니다. subsample/colsample0.8과 단일 seed의 변동도 남는다.

![코호트](../outputs/figures/stage3/common_cohort.png)

{md(counts)}

validation의 NA는 {int(v.NA):,}개({v.NA/v.rows:.2%})다. 분류 지표는 POS/NEG {int(s.validation_known_rows.iloc[0]):,}개에서만 계산하고 경보는 NA/EXCLUDE를 포함한 전체 가용 입력에서 평가했다. 입력이 없는 {n_recorded-n_available}개 고장은 전체 기록 분모에서 삭제하지 않았다.

## 학습과 분류 보조 지표

{md(summary)}

![분류 지표](../outputs/figures/stage3/classification.png)

XGBoost는 A/B/C 모두 실제 booster 설정의 `cuda:0`를 기록했다. 로지스틱은 CPU12스레드 제한으로 학습했고 수렴 경고가 없었다. 초기 가중치는 모두1이며 재표집·SMOTE·학습 음성 축소를 하지 않았다. fit초는 입출력/전처리/예측/저장을 제외하며 GPU/CPU 성능 벤치마크가 아니다.

## 같은 경보 부담 상한에서의 비교

{md(shown)}

5/10/20회는 결과 확인 전 설정한 **비교용 경보 상한**이며 현장 허용 오경보 수가 아니다. 각 상한 이하의 탐색 후보 중 검출 수 최대, 동률이면 경보 부담 최소를 선택했다. 후보는 stage2와 같은 고정 임계값12개+validation 점수 분위수16개(중복 제거)다. 점수 분포가 달라 실제 임계값은 모델마다 다르며 완전한 임계값 전수 탐색은 아니다. 정확한 임계값은 표/CSV에 저장했다.

![동일 경보 상한](../outputs/figures/stage3/budget_comparison.png)

모든 모델의 명목 관측 지지길이는 **{support:,.0f} raw**로 같다. 같은 segment에서 점수 유지 `[t,t+60)`와 관측 `[s,s+4)`의 교집합으로 계산하며 실제 가동시간이 아니다. 무경보 기준은 검출/부담0; 항상 경보 기준은 10/{n_available}, 15,732개 에피소드, 약8.82회/10,000 raw다. 항상 경보가 최대 검출률은 아니다.

![경보 곡선](../outputs/figures/stage3/alarm_tradeoff.png)

![모델 통합 비지배 후보](../outputs/figures/stage3/joint_frontier.png)

전체 곡선을 A/B/C 함께 대조하면 **C XGBoost의 약0.32회/10,000 raw, 검출5/53** 후보가 남는다. 경보를 매우 적게 허용하는 경우 검토할 가치가 있다. 이는 정해 둔5/10/20 상한 표에서 드러나지 않는 장단점이다. C의 추가 특징이 모든 부담 범위에서 불리하다고 해석하지 않는다. [전체 곡선의 통합 후보 표](../outputs/tables/stage3_joint_frontier.csv)를 보존했다. 이번 확대는 같은 탐색 점을 다시 비교한 것이며 새로운 임계값이나 모델을 추가 학습하지 않았다.

운영 경보는 에피소드 첫 발행으로 평가한다. 너무 이른 경보가 유지되는 동안 뒤의 경고 창에 진입해도 새 검출로 소급하지 않는다. 가용성 단절로 새 발행이 생길 수 있으며 미확인 경보를 확정 오경보라고 하지 않는다. [운영점별 경보 종류](../outputs/tables/stage3_budget_points.csv)의 `alarms_*` 열을 보존했다.

## 같은 이벤트의 대응 비교와 불확실성

경보 상한10에서 XGBoost의 대응 비교:

{md(xd[['comparison','detected_before','detected_after','gained_events','lost_events','recall_delta','ci_low','ci_high']])}

![대응 차이](../outputs/figures/stage3/paired_differences.png)

동일 event_id로 매칭하고 **01~10 장비 번호의10개 군집(M01/M02 함께)**을 2,000회 seed42로 재표집했다. 같은 재표집 인덱스로 앞/뒤 모델을 비교해 새로 검출한 이벤트와 놓치게 된 이벤트를 분리했다. 2단계의20 Tool 재표집보다 장비쌍 의존성을 보존하도록 개선했으며 이전 구간과 직접 비교하지 않는다. 임계값 고정 조건부95% 구간으로, validation 선택 낙관·이력 불완전성·다중 비교를 보정하지 않는다. 독립 군집이10개뿐이라 통계적 우위를 단정하지 않는다.

![이벤트 대응](../outputs/figures/stage3/event_matrix.png)

위 행렬은 상한10의 같은 {n_available}개 가용 고장을 모두 표시했다. [event_id별 결과](../outputs/tables/stage3_budget_events.csv), [대응 차이 표](../outputs/tables/stage3_paired_deltas.csv), [선행시간 하위10%·중앙값과 불확실성](../outputs/tables/stage3_uncertainty.csv)에서 확인할 수 있다. 선행시간은 검출된 이벤트만의 통계다.

## 로컬 자원과 재현성

![학습 이력과 메모리](../outputs/figures/stage3/training_resources.png)

실제 train의 최대50,000행/20trees CUDA smoke를 다시 통과했고 3개 XGBoost를 순차 학습했다. 관측 최대 RSS {tele.rss_bytes.max()/2**30:.2f}GiB, GPU 전체 사용 메모리 최대 {tele.gpu_used_mib.max():,.0f}MiB, 최소 여유 {tele.gpu_free_mib.min():,.0f}MiB였다. 약2초 샘플링은 짧은 학습의 순간 최고치를 놓칠 수 있고 GPU 값은 다른 프로세스도 포함한다. 환경/드라이버/패키지를 변경하지 않았다.

전처리·모델·native XGBoost 파일·booster 설정·이력·전체 검증 예측·공통 행 원장을 [stage3 저장 폴더](../outputs/stage3)에 저장했다. [행 일치 검사](../outputs/validation/stage3_common_contract.json), [아티팩트 SHA256과 패키지 버전](../outputs/validation/stage3_artifacts.json), [전체 산출물 검사](../outputs/validation/stage3_checks.json)를 함께 제공한다. 원본 CSV는 읽기 전용이며 크기/mtime 보존을 확인했다.

## 해석과 다음 방향

이번 결과는 긴 이력이 확보되는 제한된 코호트에서의 특징 효과다. 최대 가용 A/B 시스템의 전체 커버리지와는 다른 질문이다. **이전 B 최대 코호트의 우선 후보 지위를 이번 공통 코호트 결과 하나로 폐기하거나, C의 특징 수만 보고 승격하지 않는다.** 검출률·부담·입력 가용성·미확인 구간을 함께 유지한다.

다음 단계는 4단계의 제한적인 깊이/양성 가중치 조정과 seed42/43/44 민감도다. **A/B XGBoost를 주 조정 후보로 유지**하고 공통 코호트에서 설정 효과를 비교한다. C는 매우 낮은 경보 부담의 참조 후보로 보존해 seed 민감도를 확인한다. 최대 가용 코호트의 실제 커버리지 성능도 별도 비교하며, 공통 코호트 승자를 곧바로 전체 시스템 승자로 간주하지 않는다. 이후 장비 그룹 일반화/H 민감도를 검토하고, 모델·경보 규칙·운영점을 고정한 뒤 holdout을 한 번 평가한다. 이번 작업에서는 추가 튜닝이나 holdout 점수화까지 수행하지 않았다.

validation을 조기 종료·임계값·모델 비교에 재사용했으므로 독립 일반화 성능이 아니다. 위험 점수의 확률 보정, 시간 단위, 정비 종료, 고장 이력 완전성은 미검증이다.

## 실행 자료

- [실행 결과 포함 Notebook](../notebooks/13_common_cohort_training.ipynb)
- [학습 요약 CSV](../outputs/tables/stage3_training_summary.csv), [동일 경보 상한 CSV](../outputs/tables/stage3_budget_points.csv)
- [실험 설정](../config/stage3_experiment.json), [학습](../scripts/train_stage3.py), [경보 평가](../scripts/evaluate_stage3.py), [대응 진단](../scripts/diagnose_stage3.py)

Notebook은 저장된 실제 결과를 표시하고 6개 모델의 예측을 다시 계산해 일치를 확인한다. 전체 재학습을 자동 실행하지 않는다.
'''
    (ROOT/'reports/17_common_cohort_results.md').write_text(text,encoding='utf-8');print('STAGE3 REPORT: '+str(len(REGISTRY))+' figures')

if __name__=='__main__':main()
