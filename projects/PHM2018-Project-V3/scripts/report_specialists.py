"""장비 전용 모델 파일럿의 실제 표/시각화/한국어 보고서."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.specialist_experiment import *
from src.reporting import md
from src.visuals import plt,save,REGISTRY
from src.query import read_window,broken_line

METHODS=['global_frozen','global_tool_threshold','specialist']
LABELS={'global_frozen':'공통·기존 임계값','global_tool_threshold':'공통·장비별 임계값','specialist':'장비 전용'}
PALETTE=['#a8b8c5','#0072b2','#d55e00']


def table(name):return pd.read_parquet(TABLES/('stage7_'+name+'.parquet'))


def main():
    verify_previous();REGISTRY.clear();selection=readjson(OUT/'selection.json');tools=selection['selected_tools']
    # 현재 파일럿은 사전 기준을 통과한03M02 한 대. 추가 실험은 새 버전으로 기록한다.
    assert tools==['03M02'];tool=tools[0]
    eligibility=table('eligibility');train=table('training_summary');v=table('validation_points');h=table('later_summary');classification=table('later_classification');checks=readjson(OUT/'checks.json')
    summary=h[['Tool','method','budget','threshold','detected_events','recorded_events','available_events','alarm_episodes','alarm_episodes_per_10000_raw','alarms_UNKNOWN_START']].copy()
    summary['method']=summary.method.map(LABELS);summary['threshold']=summary.threshold.map(lambda x:format(x,'.17g'));summary.columns=['장비','방식','검증 비교상한','동결 임계값','검출','기록 고장','입력 가용 고장','전체 경보','경보/10000 raw','미확인 시작 경보']
    save_table(summary,'stage7_comparison_display')
    context='03M02 · B32 · H600/L60 · 기존 후반20% 재사용 연구 · 독립 최종 시험 아님 · 시간 단위 raw'

    keys=list(PROTOCOL['eligibility']);values=eligibility[keys].to_numpy();limits=np.array([PROTOCOL['eligibility'][k] for k in keys]);passed=values>=limits
    fig,ax=plt.subplots(figsize=(13,10));ax.imshow(passed.astype(float),aspect='auto',vmin=0,vmax=1,cmap='RdYlBu')
    ax.set_xticks(range(6),['학습 고장군집 ≥8','검증 고장군집 ≥3','학습 양성 ≥100','검증 양성 ≥20','학습 음성 ≥1,000','검증 음성 ≥1,000']);ax.set_yticks(range(20),[r.Tool+(' ★선정' if r.selected else '') for r in eligibility.itertuples()])
    for i in range(20):
        for j in range(6):ax.text(j,i,f'{int(values[i,j]):,}',ha='center',va='center',fontsize=8,color='white')
    ax.set_title('파랑: 조건 충족 · 빨강: 부족 · 값은 실제 건수 · 후반 결과로 선정하지 않음')
    save(fig,'stage7/eligibility','20대 중 최소 학습·검증 조건을 충족한 장비는03M02', 'stage7_eligibility.csv / stage7_specialists.json','군집: 양성 입력이 있는 고장의 인접 간격600 raw 초과 시 분리 · 독립 표본 보장은 아님\n08M01 검증 음성984개는 사전 기준1,000개 미달로 제외 · 선정 수를 늘리기 위해 기준을 변경하지 않음')

    r=eligibility[eligibility.Tool.eq(tool)].iloc[0];fig,axes=plt.subplots(1,2,figsize=(13,5));part=['학습 구간','검증 구간','후반 연구 구간'];recorded=[r.train_recorded_events,r.validation_recorded_events,h.recorded_events.iloc[0]];available=[r.train_available_events,r.validation_available_events,h.available_events.iloc[0]]
    axes[0].bar(part,available,label='예측창 입력 있음',color='#087e80');axes[0].bar(part,np.array(recorded)-available,bottom=available,label='예측창 입력 없음',color='#d8dfe3')
    for i,(a,n) in enumerate(zip(available,recorded)):axes[0].text(i,n+4,f'{int(a)} / {int(n)}',ha='center')
    axes[0].set_ylim(0,280);axes[0].set_ylabel('고장 이벤트 수');axes[0].set_title('입력 가용 / 기록 고장');axes[0].legend(fontsize=9)
    positive=[r.train_positive_rows,r.validation_positive_rows,classification.positive.iloc[0]];negative=[r.train_negative_rows,r.validation_negative_rows,classification.known_rows.iloc[0]-classification.positive.iloc[0]]
    x=np.arange(3);b1=axes[1].bar(x-.18,positive,.36,label='양성',color='#d55e00');b2=axes[1].bar(x+.18,negative,.36,label='정책상 음성',color='#0072b2');axes[1].set_yscale('symlog',linthresh=1);axes[1].set_ylim(0,500000);axes[1].set_xticks(x,part);axes[1].bar_label(b1,labels=[f'{int(n):,}' for n in positive],padding=3);axes[1].bar_label(b2,labels=[f'{int(n):,}' for n in negative],padding=3);axes[1].set_title('라벨이 알려진 입력 행');axes[1].set_ylabel('행 수 (0 포함 로그)');axes[1].legend(fontsize=9)
    save(fig,'stage7/coverage','학습 데이터가 충분해도 후반 평가 입력은 부족할 수 있다','stage7_eligibility.csv / stage7_later_summary.csv / stage7_later_classification.csv',context+'\n후반27개 고장 중 입력 가용3개 · 알려진 정답21행이 모두 양성 → AP/ROC-AUC 계산하지 않음')

    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for row,(d,period) in enumerate([(v,'validation: 임계값 선택에 사용'),(h,'후반: 동결 후 연구 비교')]):
        for j,(metric,title) in enumerate([('detected_events','검출 고장 수'),('alarm_episodes_per_10000_raw','경보 / 관측 지지길이10,000 raw')]):
            ax=axes[row,j]
            for k,method in enumerate(METHODS):
                q=d[d.method.eq(method)].sort_values('budget');bar=ax.bar(np.arange(2)+(k-1)*.24,q[metric],.24,label=LABELS[method],color=PALETTE[k]);ax.bar_label(bar,labels=[str(int(z)) if j==0 else f'{z:.2f}' for z in q[metric]],padding=3,fontsize=8)
            ax.set_xticks([0,1],['검증 상한5 기준','검증 상한10 기준']);ax.set_title(period);ax.set_ylabel(title);ax.set_ylim(0, max(d[metric].max()*1.35,1));ax.legend(fontsize=8)
            if j==1:
                for x,budget in enumerate([5,10]):ax.plot([x-.4,x+.4],[budget,budget],color='#5c6872',ls=':',lw=1)
    save(fig,'stage7/comparison','전용 모델 학습 효과와 장비별 임계값 조정 효과의 비교','stage7_validation_points.csv / stage7_later_summary.csv',context+'\n기존 공통 임계값은 전체 장비 기준으로 선정되어03M02의 상한10을 보장하지 않음 · 점선=비교 상한')

    fig,axes=plt.subplots(1,2,figsize=(13,5));history=readjson(OUT/(tool+'_history.json'))['validation_0']['aucpr'];best=int(train.iterations.iloc[0]);axes[0].plot(np.arange(1,len(history)+1),history,color='#0072b2');axes[0].axvline(best,color='#d55e00',ls='--',label=f'선택 트리 수{best}');axes[0].set_xlabel('학습 반복');axes[0].set_ylabel('XGBoost validation aucpr');axes[0].set_title('50회 개선 없음으로 조기 종료');axes[0].legend()
    vp=pd.read_parquet(OUT/(tool+'_validation_predictions.parquet'));hp=pd.read_parquet(OUT/(tool+'_later_predictions.parquet'))
    for frame,label,color in [(vp,'validation','#0072b2'),(hp,'후반 연구','#d55e00')]:
        counts=frame.score.value_counts(normalize=True).sort_index();axes[1].plot(counts.index,np.cumsum(counts),'o-',ms=4,label=label,color=color)
    axes[1].set_xlabel('전용 모델 점수');axes[1].set_ylabel('전체 입력 누적 비율');axes[1].set_title('고정 모델 점수 분포 변화');axes[1].legend();axes[1].grid(alpha=.2)
    save(fig,'stage7/learning','전용 모델의 조기 종료와 시점별 점수 분포','03M02_history.json / 저장 validation·후반 예측',context+'\n선택 트리1개 · 더 깊은 모델로 결과를 사후 교체하지 않음 · 학습 aucpr와 sklearn AP는 계산 방식이 다름')

    fig,axes=plt.subplots(1,2,figsize=(13,5));cats=['MATCHED','KNOWN_NEGATIVE_START','POSITIVE_UNMATCHED','UNKNOWN_START','LATE_OR_EXCLUDED_START'];colors=['#087e80','#0072b2','#e4a668','#b5c1ca','#ae70a5']
    for ax,budget in zip(axes,[5,10]):
        q=h[h.budget.eq(budget)].set_index('method').reindex(METHODS);bottom=np.zeros(3)
        for category,color,label in zip(cats,colors,['고장 매칭','정책상 음성 시작','양성 미매칭','미확인 시작','임박 제외 시작']):
            vals=q['alarms_'+category].to_numpy();ax.barh(range(3),vals,left=bottom,color=color,label=label);bottom+=vals
        for i,n in enumerate(bottom):ax.text(n+70,i,f'{int(n):,}',va='center',fontsize=9)
        ax.set_yticks(range(3),[LABELS[m] for m in METHODS]);ax.invert_yaxis();ax.set_title(f'검증 상한{budget} 기준');ax.set_xlim(0,6200);ax.set_xlabel('후반 전체 경보 에피소드 수')
    axes[1].legend(loc='lower right',fontsize=8)
    save(fig,'stage7/alarm_burden','공통 모델의 두 고장 검출에는 높은 미확인 경보 부담이 동반됐다','stage7_later_summary.csv',context+'\n전용 모델 경보228/289개 모두 미확인 시작 · 공통 기존 상한10 기준은5,411개 중5,409개 미확인 시작 · 확정 오경보가 아님')

    ev=table('later_events');ep=table('later_episodes');ge=ev[ev.method.eq('global_frozen')&ev.budget.eq(10)]
    chosen=[ge[ge.detected].sort_values(['fault_time','event_id']).iloc[0],ge[ge.available&~ge.detected].sort_values(['fault_time','event_id']).iloc[0]]
    globalp=pd.read_parquet(ROOT/'outputs/stage6/holdout_predictions.parquet');globalp=globalp[globalp.Tool.eq(tool)];fig,axes=plt.subplots(3,2,figsize=(14,10));cases=[]
    for col,(event,title) in enumerate(zip(chosen,['공통 모델만 검출한 첫 사례','두 모델 모두 놓친 입력 가용 사례'])):
        ft=event.fault_time;start=ft-1200;end=ft+120;data,audit=read_window(tool,start,end,columns=['time','segment_id',SENSORS[0]])
        for _,part in data.groupby('segment_id',sort=False):
            tx,vy=broken_line(part.time.to_numpy()-ft,part[SENSORS[0]].to_numpy());axes[0,col].plot(tx,vy,color='#0072b2',lw=1)
        axes[0,col].set_ylabel('FLOWCOOLPRESSURE',fontsize=8);axes[0,col].set_title(title+'\n'+event.fault_type+' · '+str(int(ft))+' raw',fontsize=10)
        for row,(frame,method) in enumerate([(globalp,'global_frozen'),(hp,'specialist')],start=1):
            q=frame[frame.time.between(start,end)].sort_values('time')
            for _,part in q.groupby('segment_id',sort=False):
                tx,vy=broken_line(part.time.to_numpy()-ft,part.score.to_numpy(),threshold=60);axes[row,col].plot(tx,vy,'o-',ms=2,lw=1,color='#253847')
            for budget,color in [(5,'#0072b2'),(10,'#d55e00')]:
                threshold=h[h.method.eq(method)&h.budget.eq(budget)].threshold.iloc[0];axes[row,col].axhline(threshold,color=color,ls='--',label=f'검증 상한{budget} 기준')
            a=ep[ep.method.eq(method)&ep.budget.eq(10)&ep.start.between(start,end)]
            if len(a):
                z=a[['start']].merge(q[['time','score']],left_on='start',right_on='time',how='left');axes[row,col].scatter(z.start-ft,z.score,marker='^',s=45,color='#d55e00',label='경보 시작',zorder=4)
            axes[row,col].set_ylabel(LABELS[method]+' 점수');axes[row,col].legend(fontsize=7)
        for ax in axes[:,col]:ax.axvspan(-600,-60,color='#d3e9de',alpha=.5);ax.axvline(0,color='#ac3045',ls=':');ax.set_xlim(-1200,120);ax.grid(alpha=.15)
        axes[2,col].set_xlabel('고장 시각 기준 상대 raw');cases.append(dict(event_id=event.event_id,fault_time=float(ft),selection=title,query=audit))
    save(fig,'stage7/event_cases','동일 센서 구간에서 공통·전용 모델의 점수 비교','정규화 센서 / 저장 점수 / stage7_later_episodes.csv',context+'\n초록=유효 사전 예측창[-600,-60] · 공백과 세그먼트 경계 선 연결 금지 · 사례는 사후 설명용으로만 선정')
    dump_json(OUT/'case_examples.json',cases);dump_json(OUT/'figure_manifest.json',REGISTRY)

    eligibility_display=eligibility[['Tool','train_positive_clusters','validation_positive_clusters','train_positive_rows','validation_positive_rows','train_negative_rows','validation_negative_rows','selected']].copy();eligibility_display.columns=['장비','학습 고장군집','검증 고장군집','학습 양성','검증 양성','학습 음성','검증 음성','선정']
    validation=v[['method','budget','threshold','detected_events','available_events','recorded_events','alarm_episodes_per_10000_raw']].copy();validation['method']=validation.method.map(LABELS);validation.columns=['방식','비교 상한','임계값','검출','입력 가용','기록','실제 경보/10000 raw']
    def picture(n):return f'![{n}](../outputs/figures/stage7/{n}.png)\n'
    text=f'''# 20 장비 전용 모델 후속 프로젝트 — 03M02 파일럿

사용자 요청에 따라 약20분 내 완료할 범위로 장비 수를 최대3대로 제한하고, 학습·검증 자료가 충분한 장비만 전용 모델을 만들었습니다. 사전 최소 조건을 모두 통과한 장비는 **03M02 한 대**입니다. CUDA 학습과 후반48,164행 예측을 완료했지만 **이번 전용 모델은 개선을 보이지 않았습니다.** 기존 공통 모델과 최종 결과는 바이트 단위로 보존했습니다.

## 1. 장비 선정과 제외 이유

학습은 기존 시간 앞60%, validation은 다음20%를 사용합니다. B 최대가용 입력, 보수적 라벨, H600/L60, 과거900/미래1800 raw guard를 그대로 유지했습니다. train에는 해당 장비의 알려진 양성/음성만 사용하고, 미확인/임박 제외 구간은 학습에서 제외하되 검증·후반 점수화에는 유지했습니다.

최소 조건은 학습/검증 고장군집8/3개, 양성 입력100/20행, 음성 입력1,000/1,000행입니다. 군집은 양성 입력이 있는 비태그 고장을 시각순 정렬해 인접 간격600 raw 초과일 때 나눕니다. 동일 고장의 반복 양성 행만으로 데이터가 충분하다고 판단하지 않기 위한 조건입니다. 독립 고장 사건 수를 보장하거나 통계적 충분성을 입증하는 기준은 아닙니다. 자격 충족 장비가3대를 넘으면 학습 고장군집 수→학습 양성 수→장비명 순으로 고릅니다. 후반 결과는 선정에 사용하지 않았습니다.

{md(eligibility_display)}

{picture('eligibility')}

여러 장비는 현재 보수적 통합 라벨에서 음성이0개입니다. 이는 정상 운전이 없다는 의미가 아니라, 모든 고장 유형의 후속 기록을 요구하는 정책과 분할 경계에 의해 정상 근거를 확보하지 못했다는 뜻입니다. 08M01은 validation 음성984개로 기준1,000개에16개 못 미쳤습니다. 이 차이가 물리적인 충분/불충분 경계라는 뜻은 아니지만, 이번 사전 기준을 결과에 맞춰 바꾸지 않았습니다. 다른 라벨 정책 또는 최소 건수의 민감도 실험은 별도 프로토콜로 진행해야 합니다.

03M02는 학습 양성500행/음성123,296행(합계123,796행), validation 양성133행/음성5,982행입니다. 양성 입력이 있는 고장군집은 학습83개/검증23개였습니다. 학습 구간의 전체 입력134,198행 중 라벨이 알려진 행만 fit했습니다.

{picture('coverage')}

## 2. 세 비교 방식과 공정한 입력

- **공통·기존 임계값**: 기존 동결 B 모델과 전체 장비 validation에서 고른 숫자 임계값을 그대로 사용합니다.
- **공통·장비별 임계값**: 공통 모델 점수는 그대로 두고03M02 validation에서 임계값만 선택합니다.
- **장비 전용**:03M02 학습 데이터만으로 전처리와 모델을 fit하고 동일 장비 validation에서 임계값을 선택합니다.

세 방식의 후반 입력48,164행, 라벨, 예측창, 고장 이벤트, 관측 지지길이2,671,668 raw는 동일합니다. 장비 ID/TTF/시각/미래 라벨은 모델 입력에 넣지 않았습니다. 기준 모델과 같은 B32 특징, 깊이3, 양성 가중1, eta0.05, 최대1,000회, 조기 종료50회, seed42를 사용합니다. 별도 튜닝과 seed 탐색은 하지 않았습니다. 학습은 실제 cuda:0, CPU12스레드이며 환경/드라이버를 변경하지 않았습니다.

전용 모델의 validation aucpr 최고점은 첫 트리에서 발생해 선택 트리 수는1개였습니다. 학습·저장 booster의 전체 트리 수와 다를 수 있으며 실제 예측은 선택 반복까지만 사용합니다. fit 소요는{train.train_seconds.iloc[0]:.3f}초이고 입력 조회·경보 평가·보고서 시간은 포함하지 않습니다. 최대1,000개를 강제로 사용하거나 결과를 보고 더 깊은 모델로 교체하지 않았습니다.

{picture('learning')}

## 3. validation에서 선택한 숫자 임계값

고정12개+점수 분위수16개 후보(중복 제거)에서 비교 상한5/10 내 검출 최대, 실제 경보 부담 최소, 높은 임계값 순으로 선정했습니다. 장비별 두 방식은 같은 선택 규칙을 사용하지만 점수에 따른 후보 숫자는 다릅니다. 전체 임계값 공간의 최적해라는 뜻은 아닙니다. 기존 공통 모델의 전체 장비 기준 임계값은03M02의 개별 경보 상한을 보장하지 않습니다.

{md(validation)}

전용 모델/전처리/숫자 임계값을 후반 예측 전에 별도 [동결 manifest](../outputs/stage7/frozen_manifest.json)에 기록했습니다. train+validation 재학습이나 후반 결과 기반 재선택은 하지 않았습니다. 공통 모델의 validation/후반 점수는 기존 저장값을 재사용했습니다.

## 4. 후반 구간 연구 비교 결과

{md(summary)}

{picture('comparison')}

전용 모델은 두 설정 모두0/27개(입력 가용3개 중0개)를 검출했고 경보228/289개를 냈습니다. 공통 모델의 장비별 임계값도0개를 검출했으나 경보141개로 부담이 더 낮았습니다. 따라서 이번 파일럿에서 전용 학습의 개선 근거는 없습니다.

기존 공통 모델의 상한10 기준 설정은2/27개(입력 가용3개 중2개)를 검출했지만 경보5,411개, 부담20.253회/10,000 raw로 장비별 비교 상한10을 넘었습니다. 이를 같은 경보 부담에서 공통 모델이 우월하다는 증거로 제시하지 않습니다. 더 많은 경보를 허용한 결과입니다. 기록27개 중24개는 예측창 입력이 없으므로 전용 모델만 바꿔 해결할 수 있는 범위를 넘어섭니다.

{picture('alarm_burden')}

전용 모델의 모든 경보는 미확인 라벨에서 시작했고, 기존 공통 상한10 기준도5,409개가 미확인 시작입니다. 이를 확정 오경보로 분류하지 않았습니다. 알려진 정답은21행 모두 양성이고 음성이 없으므로 이 구간의 AP/ROC-AUC는 **미산출**로 저장했습니다. 0이나1로 성능을 꾸미지 않았습니다. 장비1대/입력 가용 고장3개만으로 신뢰할 만한 장비 일반화 결론이나 유의성 검정을 제시하지 않습니다.

## 5. 실제 구간 예측

{picture('event_cases')}

공통 기존 상한10 설정에서 검출한 첫 고장과, 공통·전용 모두 놓친 입력 가용 고장을 시각순으로 골랐습니다. 센서와 저장된 점수를 표시하며 추가 후반 추론을 하지 않았습니다. 사후 설명용 사례이며 선정·튜닝에 되먹임하지 않습니다. [사례 ID와 조회 기록](../outputs/stage7/case_examples.json)을 남겼습니다.

## 6. 결과의 사용 범위와 다음 결정

**이번 결과는 독립 Holdout 성능이 아닙니다.** 이 마지막20% 구간은 이전 최종 평가에서 이미 확인한 자료이며, 사용자가 요청한 후속 연구 비교에 재사용했습니다. 장비와 임계값은 이번 후반 평가 전에 고정했지만 이미 알려진 데이터라는 사실은 없어지지 않습니다. 원래 stage6의 동결 모델/점수/결과/접근 기록은 별도 유지하며 이번 전용 모델로 대체하지 않았습니다.

현 단계에서는03M02 전용 모델을 연구 산출물로 보존하고 공통 모델을 대체하지 않습니다. 다음 우선순위는03M02 후반24개 고장의 입력 공백 원인과, 다른 장비의 음성 라벨 부족 원인을 확인하는 것입니다. 시간 단위, 정비 종료, 고장 로그 완전성은 여전히 미검증입니다. 이후 라벨 정책/데이터 범위를 개선할 때 새 평가 자료를 확보해야 하며 이번 구간에서 계속 조정한 결과를 독립 검증으로 부르지 않습니다.

## 7. 산출물과 검증

실제 검사{len(checks['checks'])}개를 통과했습니다. 장비 선정식, 데이터 분할 원장, 과거/미래 guard, 장비 전용 학습, CUDA 설정, 저장 모델 validation128행 재현, 세 방식의 동일 평가 행, 고정 임계값, 경보 시작·연속성·원시각 고장 매칭, 이전 산출물 해시를 확인했습니다. 원본65개 CSV의 크기/수정시각도 유지했습니다. 전체 원본 내용 해시를 계산한 것은 아닙니다.

- [실행 Notebook](../notebooks/16_equipment_specialists.ipynb)
- [전용 모델](../outputs/stage7/03M02_model.joblib), [전처리](../outputs/stage7/03M02_preprocessor.joblib), [UBJ 모델](../outputs/stage7/03M02_model.ubj)
- [후반48,164행 저장 예측](../outputs/stage7/03M02_later_predictions.parquet)
- [장비 선정·제외 전체 CSV](../outputs/tables/stage7_eligibility.csv), [비교 결과 CSV](../outputs/tables/stage7_comparison_display.csv)
- [고장별 비교 CSV](../outputs/tables/stage7_later_events.csv), [경보별 CSV](../outputs/tables/stage7_later_episodes.csv)
- [선정 기록](../outputs/stage7/selection.json), [후속 연구 접근 기록](../outputs/stage7/evaluation_access.json), [검사 기록](../outputs/stage7/checks.json)

모델은 입력 B32 특징을 기존 정의대로 인과적으로 계산한 뒤 저장 전처리의 transform만 적용하고, 선택 반복 수까지만 예측해야 합니다. 32열 이름/순서는 기존 features.schema()['B']와 일치해야 합니다. 원시 센서 CSV를 model.predict에 바로 전달하는 모델은 아닙니다. Notebook은 저장 validation 입력에서 이 절차를 실제 재현합니다.
'''
    (ROOT/'reports/20_equipment_specialists.md').write_text(text,encoding='utf-8');print('전용 모델 보고서 및 그림',len(REGISTRY),'개 완료')


if __name__=='__main__':main()
