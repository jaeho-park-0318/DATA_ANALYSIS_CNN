"""제한 탐색과 seed/고정 임계값 민감도 결과를 한국어로 정리한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
from src.reporting import md
from src.visuals import plt,save,REGISTRY

OUT=ROOT/'outputs/stage4'

def short(cid):
    f,d,w=cid.split('_',2)
    return f'{f} · {d} · 가중'+('1' if w=='unit' else '15.77')

def main():
    def table(n):return pd.read_parquet(TABLES/('stage4_'+n+'.parquet'))
    runs=table('training_summary');grid=table('grid_budget_points');retuned=table('retuned_points');fixed=table('fixed_points');ranges=table('seed_ranges');delta=table('paired_deltas');joint=table('joint_frontier');events=table('retuned_events');frozen=json.loads((OUT/'selection.json').read_text(encoding='utf-8'));configs=sorted(retuned.config_id.unique());palette=['#d55e00','#0072b2','#087e80','#7b4ab5'];colors=dict(zip(configs,palette));REGISTRY.clear()
    context='실제 temporal validation · 공통 코호트 · H600/L60 · 입력 있는 고장53/기록232 · holdout 미사용'
    order=runs[runs.seed.eq(42)&runs.feature_set.isin(['A','B'])].config_id.tolist();gridp=grid.pivot(index='config_id',columns='budget_per_10000_raw',values='detected_events').reindex(order)
    fig,axes=plt.subplots(1,2,figsize=(14,8));im=axes[0].imshow(gridp,aspect='auto',vmin=0,vmax=gridp.to_numpy().max(),cmap='YlGnBu');axes[0].set_xticks(range(len(gridp.columns)),[str(v) for v in gridp.columns]);axes[0].set_yticks(range(len(order)),[short(c) for c in order]);axes[0].set_xlabel('비교용 경보 상한 / 10,000 raw');axes[0].set_title('검출 고장 수 / 53')
    for i in range(len(order)):
        for j in range(len(gridp.columns)):axes[0].text(j,i,str(int(gridp.iloc[i,j])),ha='center',va='center',color='white' if gridp.iloc[i,j]>gridp.to_numpy().max()*.6 else '#222')
    ss=runs[runs.seed.eq(42)].set_index('config_id').reindex(order);axes[1].barh(np.arange(len(order)),ss.AP_known,color=['#0072b2' if c.startswith('A') else '#087e80' for c in order]);axes[1].set_yticks(range(len(order)),[short(c) for c in order]);axes[1].invert_yaxis();axes[1].set_xlabel('알려진 라벨의 AP');axes[1].set_title('분류 지표는 보조 지표');axes[1].set_xlim(0,ss.AP_known.max()*1.2)
    for i,v in enumerate(ss.AP_known):axes[1].text(v+.003,i,f'{v:.3f}',va='center',fontsize=8)
    save(fig,'stage4/search_grid','깊이와 양성 가중치의 제한 탐색: seed42','stage4_grid_budget_points.csv / stage4_training_summary.csv',context+'\n깊이2/3/4 × 가중1/√(train NEG/POS) × A/B =12설정 · 같은 관측 지지길이17,838,206 raw')
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True,sharey=True)
    for ax,budget in zip(axes.flat,[.5,5,10,20]):
        for cid in configs:
            z=retuned[retuned.config_id.eq(cid)&retuned.budget_per_10000_raw.eq(budget)].sort_values('seed');ax.plot(z.seed,z.detected_events,'o-',color=colors[cid],label=short(cid))
        ax.set_title(f'상한 {budget:g}회: seed마다 임계값 재선택');ax.set_xticks([42,43,44]);ax.set_ylim(0,max(retuned.detected_events)*1.25);ax.set_xlabel('seed');ax.set_ylabel('검출 고장 수 / 53');ax.grid(alpha=.2);ax.legend(fontsize=8)
    save(fig,'stage4/retuned_seed','seed를 바꾸면 검출 수가 얼마나 달라지는가','stage4_retuned_points.csv',context+'\n각 seed의 validation에서 임계값을 다시 고른 탐색 결과 · 3seed는 독립 고장 이벤트를 늘리지 않음')
    fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True)
    for ax,budget in zip(axes,[5,10]):
        for cid in configs:
            z=fixed[fixed.config_id.eq(cid)&fixed.budget_per_10000_raw.eq(budget)];ax.scatter(z.alarm_episodes_per_10000_raw,z.detected_events,color=colors[cid],s=65,label=short(cid))
            for r in z.itertuples():ax.annotate(str(r.seed),(r.alarm_episodes_per_10000_raw,r.detected_events),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.axvline(budget,color='#d55e00',ls='--',label=f'원래 비교 상한 {budget}');ax.set_title(f'seed42의 상한{budget} 임계값을 그대로 이전');ax.set_xlabel('실제 경보 / 명목 관측 지지길이10,000 raw');ax.set_ylabel('검출 고장 수 / 53');ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'stage4/fixed_threshold','같은 임계값을 다른 seed 모델에 옮겼을 때','stage4_fixed_points.csv',context+'\n숫자=seed · 임계값 재선택 없음 · 점수가 보정된 확률이 아니므로 재학습 후 경보 부담이 달라질 수 있음')
    compare=delta[delta.config_id.ne(delta.baseline_config)].copy();fig,axes=plt.subplots(1,2,figsize=(13,6))
    labels=[f'{short(r.config_id)}\nseed{r.seed} · 상한{r.budget_per_10000_raw:g}' for r in compare.itertuples()];y=np.arange(len(compare))
    axes[0].barh(y-.16,compare.detected_baseline,height=.32,color='#8ab1c8',label='같은 특징의 기존 깊이3/가중1');axes[0].barh(y+.16,compare.detected_tuned,height=.32,color='#d55e00',label='조정 설정');axes[0].set_yticks(y,labels);axes[0].set_xlabel('검출 고장 수 / 53');axes[0].legend(fontsize=8)
    axes[1].errorbar(compare.recall_delta*100,y,xerr=[(compare.recall_delta-compare.ci_low)*100,(compare.ci_high-compare.recall_delta)*100],fmt='o',capsize=5,color='#087e80');axes[1].axvline(0,ls='--',color='#888');axes[1].set_yticks(y,[str(s) for s in compare.seed]);axes[1].set_ylabel('seed');axes[1].set_xlabel('조정−기준 검출률 차이 (%p)');axes[1].set_title('장비쌍 군집 재표집 조건부95% 구간')
    save(fig,'stage4/paired_gain','선택된 조정의 개선이 다른 seed에서도 유지되는가','stage4_paired_deltas.csv',context+'\n각 seed의 임계값 재선택 결과를 대응 비교 · 10장비 번호 군집2,000회 · 선택 낙관/다중 비교 미보정')
    fig,axes=plt.subplots(1,2,figsize=(13,5.5));colfamily={'A':'#0072b2','B':'#087e80','C':'#7b4ab5'}
    for ax,limit,title in zip(axes,[max(12,joint.alarm_episodes_per_10000_raw.max()*1.1),1.5],['seed42 전체 탐색의 통합 후보','매우 낮은 부담 구간 확대']):
        for f,color in colfamily.items():
            z=joint[joint.feature_set.eq(f)&joint.alarm_episodes_per_10000_raw.le(limit)];ax.scatter(z.alarm_episodes_per_10000_raw,z.detected_events,s=55,color=color,label=f)
            for r in z.itertuples():ax.annotate(f'{int(r.detected_events)}',(r.alarm_episodes_per_10000_raw,r.detected_events),xytext=(3,4),textcoords='offset points',fontsize=8)
        ax.set_xlim(0,limit);ax.set_ylim(0,max(8,(joint[joint.alarm_episodes_per_10000_raw.le(limit)].detected_events.max()+3)));ax.set_xlabel('경보 / 명목 관측 지지길이10,000 raw');ax.set_ylabel('검출 수 / 53');ax.set_title(title);ax.legend();ax.grid(alpha=.2)
    save(fig,'stage4/joint_frontier','A/B 조정과 C 기준을 함께 본 비지배 후보','stage4_joint_frontier.csv',context+'\nseed42 탐색 점에 한정 · 무경보 제외 · C는 깊이3/가중1 기준만 포함 · 최종 운영점 아님')
    selected42=retuned[retuned.seed.eq(42)&retuned.budget_per_10000_raw.eq(10)].set_index('config_id').reindex(configs);fig,ax=plt.subplots(figsize=(12,5.5));bottom=np.zeros(len(configs))
    for cat,name,color in [('MATCHED','이벤트 매칭','#087e80'),('KNOWN_NEGATIVE_START','정책상 음성 시작','#0072b2'),('UNKNOWN_START','미확인 시작','#a2aab1'),('LATE_OR_EXCLUDED_START','제외 시작','#7b4ab5'),('POSITIVE_UNMATCHED','양성 미매칭','#d55e00')]:
        values=selected42['alarms_'+cat]/selected42.alarm_episodes.replace(0,np.nan);ax.bar([short(c) for c in configs],values,bottom=bottom,label=name,color=color);bottom+=values.fillna(0)
    ax.set_ylabel('경보 에피소드 대비 비율');ax.legend(fontsize=8,loc='lower left');ax.set_title('seed42, 비교용 경보 상한10')
    save(fig,'stage4/unknown_alarms','튜닝 후에도 미확인 구간의 경보를 분리한다','stage4_retuned_points.csv',context+'\nNA에서 시작한 경보를 확정 오경보로 계산하지 않음 · 전체53개 가용 고장의 검출과 다른 분모')
    telemetry=[]
    for phase in ['grid','sensitivity']:
        t=table(phase+'_resource_telemetry');t['phase']=phase;telemetry.append(t)
    tele=pd.concat(telemetry,ignore_index=True);fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for phase,color in [('grid','#0072b2'),('sensitivity','#087e80')]:
        t=tele[tele.phase.eq(phase)];axes[0].plot(t.unix_time-t.unix_time.min(),t.rss_bytes/2**30,'o-',ms=3,label=phase,color=color)
    axes[0].set_xlabel('각 학습 단계의 첫 측정 후 (초)');axes[0].set_ylabel('학습 프로세스 RSS (GiB)');axes[0].legend();axes[0].set_title('GPU 학습 순차 · CPU 최대12스레드')
    selected_tuned=next((c['config_id'] for c in frozen['selected_configs'] if c['config_id'] not in ['A_d3_unit','B_d3_unit']),frozen['selected_configs'][0]['config_id'])
    for seed in [42,43,44]:
        h=json.loads((OUT/(selected_tuned+f'_s{seed}_history.json')).read_text(encoding='utf-8'))['validation_0']['aucpr'];axes[1].plot(np.arange(1,len(h)+1),h,label=f'seed{seed}')
    axes[1].set_title(short(selected_tuned));axes[1].set_xlabel('부스팅 반복');axes[1].set_ylabel('validation aucpr');axes[1].legend()
    save(fig,'stage4/training_resources','CUDA 학습과 선택 설정의 조기 종료 변동','stage4_*_resource_telemetry.csv / 모델별 history.json',context+'\n메모리 약2초 표본이며 순간 최고치 미보장 · GPU 전체 사용량에는 다른 프로세스 포함')
    e=events[events.budget_per_10000_raw.eq(10)&events.available&~events.tagged];cols=[c+f'_s{s}' for c in configs for s in [42,43,44]];matrix=e.pivot(index='event_id',columns='run_id',values='detected');meta=e[['event_id','Tool','fault_time']].drop_duplicates().sort_values(['Tool','fault_time']);matrix=matrix.reindex(index=meta.event_id,columns=cols)
    fig,ax=plt.subplots(figsize=(15,12));im=ax.imshow(matrix.to_numpy(int),aspect='auto',vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(len(cols)),[short(c)+'\n'+str(s) for c in configs for s in [42,43,44]],rotation=55,ha='right',fontsize=8);ax.set_yticks(range(len(meta)),[f'{r.Tool} · {r.fault_time:.0f}' for r in meta.itertuples()],fontsize=7);fig.colorbar(im,ax=ax,ticks=[0,1],label='0 미탐 / 1 검출')
    save(fig,'stage4/event_matrix','같은 고장53개의 seed별 검출·미탐','stage4_retuned_events.csv',context+'\n상한10에서 seed별 임계값 재선택 · 같은 이벤트 반복 표시이며 고장 표본 수가 늘어난 것이 아님')
    dump_json(ROOT/'outputs/stage4_figure_manifest.json',REGISTRY)
    selected=pd.DataFrame(frozen['selected_by_budget']);selected['threshold']=selected.threshold.map(lambda v:format(v,'.17g'));shown=ranges[ranges.budget_per_10000_raw.isin([.5,5,10])].copy();summary=runs[['run_id','depth','positive_weight','seed','reused_stage3','iterations','AP_known','ROC_AUC_known','train_seconds']].copy()
    nnew=int((~runs.reused_stage3).sum());nreuse=int(runs.reused_stage3.sum());transfer=fixed[fixed.seed.ne(42)];exceed=int(transfer.exceeds_budget.sum());base10=retuned[retuned.budget_per_10000_raw.eq(10)].pivot(index='config_id',columns='seed',values='detected_events');base10.columns=[f'seed{s}' for s in base10.columns]
    text=f'''# 18 제한 튜닝과 seed 민감도 — 4단계

2026-11-05. A/B XGBoost의 깊이와 양성 가중치만 제한적으로 조정하고 seed42/43/44를 비교했다. **새 CUDA 학습 {nnew}개, 이전 단계 기준 재사용 {nreuse}개로 총 {len(runs)}개 실행 결과**를 정리했다. 모든 결과는 같은 공통 코호트 validation이며 holdout은 사용하지 않았다.

## 핵심 결과

seed42에서 설정을 선택해 동결한 결과:

{md(selected)}

상한5에서는 기존 B 깊이3/가중1이 유지됐다. 상한10에서는 A 깊이2/양성 가중치 약15.77이 21/53개를 검출해 기존 A17/53보다 많았다. 실제 부담은 각각 약6.60과6.14회/10,000 raw로 **같은 상한 아래의 비교이지 정확히 같은 경보 횟수 비교는 아니다**.

상한10에서 seed별로 임계값을 재선택한 실제 검출 수:

{md(base10.reset_index())}

seed42 임계값을 다른 seed 모델에 그대로 옮긴 {len(transfer)}개 조건 중 **{exceed}개에서 원래 비교 상한을 초과**했다. 다른 seed에서의 개선 여부와 임계값 이전 결과를 분리해 해석해야 한다. 최고 seed만 골라 결과를 대표하지 않았다.

**조정 A를 기존 A의 대체 모델로 승격하지 않는다.** 상한10의 기존 A는17/19/19개, 조정 A는21/17/15개로 seed43/44에서 오히려 줄었다. 기존 A/B를 후속 일반화 검증의 주 기준으로 유지하고 조정 A는 연구 비교 후보로 남긴다. C는 상한0.5에서5/7/5개를 검출해 저부담 참조 역할을 유지하지만, 이는 seed마다 임계값을 다시 선택한 결과다.

## 실험 계약

- 동일 train309,273행(POS1,239/NEG308,034), 전체 가용 validation309,203행, 알려진 라벨7,158행(POS392), 가용 고장53개/기록232개다. 원본 행·분할·특징·라벨을 변경하지 않았다.
- A/B × 깊이2/3/4 × 양성 가중치1/√(train NEG/POS) =12설정을 seed42에서 비교했다. 양성 가중치 **15.767530310943991**은 train에서만 계산해 XGBoost `scale_pos_weight`로 적용했다. 평가 분포에는 가중치/재표집을 적용하지 않았다.
- 나머지 설정은 eta0.05, max1000trees, subsample/colsample0.8, min_child_weight10, lambda5, max_bin128, CUDA hist, validation aucpr 조기 종료50회로 고정했다. 전처리는 동일 train에서 추정한 stage3 사전을 그대로 재사용하고 transform만 했다.
- 경보 비교 상한0.5/5/10/20을 사전에 설정했다. 설정 선택은5와10 두 상한에서 검출 최대→실제 경보 최소→낮은 깊이/단위 가중치/특징A 순으로 했다. 선택 설정 합집합을 `selection.json`에 동결한 뒤43/44를 학습했다.
- 선택 설정과 A/B/C의 깊이3/가중1 참조를 seed42/43/44로 비교했다. 중복 설정은 한 번만 평가했다. C는 새 깊이/가중치 탐색 없이 저부담 참조의 seed 민감도만 봤다.
- stage3의 A/B/C seed42 기준 파일과 검증 점수는 일치 확인 후 재사용했다. 표의 재사용 행 fit초0은 이번에 새 fit을 하지 않았다는 뜻이며 원래 학습 시간이0이라는 뜻이 아니다.

## 제한 탐색 결과

![제한 탐색](../outputs/figures/stage4/search_grid.png)

{md(summary)}

AP/ROC-AUC는 알려진 라벨의 보조 지표다. 가중치를 키우면 점수 분포도 달라지므로 같은 숫자 임계값으로 단순 비교하지 않는다. 임계값은 이전 단계의 고정12개+validation 점수 분위수16개를 합친 탐색이며, 모든 가능한 임계값을 전수 탐색한 것은 아니다.

![통합 후보](../outputs/figures/stage4/joint_frontier.png)

이는 seed42에서 발견한 비지배 후보이며 다른 seed에서도 동일한 우위를 보장하지 않는다. 전체 곡선과 C의 매우 낮은 부담 후보를 보존했고 사전에 정한 상한 표만으로 전 범위의 우위를 단정하지 않았다.

## seed별 임계값 재선택

![seed별 재선택](../outputs/figures/stage4/retuned_seed.png)

각 seed의 validation에서 같은 상한을 만족하는 임계값을 다시 선택했다. 모델이 낼 수 있는 탐색 성능의 민감도이며, 이미 고정한 운영 임계값의 안정성과 다르다. 3seed의 최소/최대/평균을 기록했지만 이를 모집단 신뢰구간이라고 부르지 않는다. 독립 고장 표본은 계속53개이며159개로 늘어나지 않는다.

## seed42 임계값을 고정해 이전

![임계값 이전](../outputs/figures/stage4/fixed_threshold.png)

각 설정의 seed42에서 고른 임계값을43/44에 그대로 적용했다. 상한을 초과해도 다시 조정하지 않고 초과 사실을 남겼다. seed42에서는 두 방식의 검출 수/부담이 일치함을 확인했다. 위험 점수는 보정된 확률이 아니므로 새 seed로 재학습한 뒤 같은 임계값이 같은 경보 수준을 뜻하지 않을 수 있다. 다음 일반화 평가에서도 모델과 임계값을 함께 고정해야 한다.

특히 B 기준의 상한5/10 임계값을 seed43 모델로 이전했을 때 **경보와 검출이 모두0**이었다. 상한을 넘지 않았다고 안정적인 것은 아니다. 원래의 검출 기능이 사라지는 경우도 함께 보아야 한다. C seed44의 상한10 임계값 이전에서는 실제 부담이 약10.23회로 상한을 넘었다.

주요 상한별 범위(원본 소수점은 CSV 보존):

{md(shown)}

## 기준 대비 대응 개선과 한계

![대응 개선](../outputs/figures/stage4/paired_gain.png)

{md(delta)}

같은 seed·상한·특징의 기존 깊이3/가중1과 대응 비교했다. B 선택은 기존 설정 그 자체라 차이가0이다. 01~10 장비 번호 군집(M01/M02 함께)을2,000회 재표집한 조건부95% 구간이며, validation 선택 낙관·다중 비교·고장 이력 불완전성을 보정하지 않는다. 단일 seed42에서의 개선만으로 튜닝 성공을 확정하지 않는다.

![이벤트별 변동](../outputs/figures/stage4/event_matrix.png)

![미확인 경보](../outputs/figures/stage4/unknown_alarms.png)

NA/EXCLUDE도 전체 입력 점수화에 포함했다. 대부분의 경보가 정답 미확인 구간에서 시작하므로 확정 오경보율/현장 정밀도를 산출할 수 없다. 명목 관측 지지길이17,838,206 raw는 실제 가동시간이 아니며 시간 단위도 미검증이다. 입력이 없는179개 고장을 전체 기록 분모에서 삭제하지 않았다.

## 로컬 자원·검증

![자원과 이력](../outputs/figures/stage4/training_resources.png)

GPU 학습은 순차 실행했고 모든 새 모델의 booster `cuda:0`를 확인했다. CPU12스레드 제한, 경보 평가는 CPU3프로세스를 사용했다. 관측 최대 학습 프로세스 RSS {tele.rss_bytes.max()/2**30:.2f}GiB, GPU 전체 사용 메모리 최대 {tele.gpu_used_mib.max():,.0f}MiB, 최소 여유 {tele.gpu_free_mib.min():,.0f}MiB다. 약2초 측정은 순간 최고치를 놓칠 수 있으며 GPU 값은 다른 프로세스를 포함한다. 패키지/드라이버를 변경하지 않았다.

행 원장은 stage3과 전체 일치 확인했다. 저장 모델 예측 재현, GPU 장치, 분류 지표 재계산, 경보 캐시의 입력/코드 해시, 이벤트 매칭 경계, 설정 선택 규칙, seed42 임계값 고정 이전, seed 범위, 모델 SHA256, 원본 크기/mtime 보존을 검사했다. 캐시는 모델별로 저장해 감도 평가 때 기존12설정의 경보 계산을 반복하지 않았다.

## 다음 단계

이번 단계는 제한 탐색과 seed 민감도까지다. 최고 seed로 교체하거나 holdout을 열어 재선택하지 않았다. 다음은 **후보 설정·경보 규칙·임계값 선택 절차를 고정한 장비 그룹 일반화 검증**이다. 기존 A/B 기준(깊이3/가중1)을 주 후보로 유지하고, 조정 A(깊이2/가중15.77)는 개선이 재현되는지 보는 보조 비교로 둔다. C는 낮은 경보 부담의 참조로 남긴다. 새 학습 모델에 기존 숫자 임계값을 그대로 이식하지 않으며, 새 장비 그룹의 validation에서 사전에 정한 절차로 임계값을 정하고 holdout에 고정 적용한다. 공통 코호트와 최대 가용 코호트의 커버리지 차이는 계속 분리한다.

validation을 조기 종료·설정 선택·임계값 탐색에 재사용했다. 본 결과는 독립 일반화 성능이나 운영 준비 완료를 의미하지 않는다. 시간 단위·정비 종료·고장 이력 완전성·확률 보정 확인은 남아 있다.

## 산출물과 재현

- [실행 Notebook](../notebooks/14_tuning_seed_sensitivity.ipynb)
- [실험 설정](../config/stage4_experiment.json), [동결 설정](../outputs/stage4/selection.json)
- [학습 표](../outputs/tables/stage4_training_summary.csv), [탐색 경보 표](../outputs/tables/stage4_grid_budget_points.csv), [seed 재선택](../outputs/tables/stage4_retuned_points.csv), [고정 임계값 이전](../outputs/tables/stage4_fixed_points.csv), [seed 범위](../outputs/tables/stage4_seed_ranges.csv)
- [모델·전처리·예측·이력](../outputs/stage4), [아티팩트 SHA256](../outputs/validation/stage4_artifacts.json), [검증 결과](../outputs/validation/stage4_checks.json)
- [학습 코드](../scripts/train_stage4.py), [경보 평가](../scripts/evaluate_stage4.py), [설정 선택](../scripts/select_stage4.py), [민감도 진단](../scripts/diagnose_stage4.py)

Notebook은 저장된 실제 결과를 읽고 모델 예측을 재계산한다. 전체 재학습을 자동 실행하지 않는다.
'''
    (ROOT/'reports/18_tuning_seed_results.md').write_text(text,encoding='utf-8');print('STAGE4 REPORT:',len(REGISTRY),'figures')

if __name__=='__main__':main()
