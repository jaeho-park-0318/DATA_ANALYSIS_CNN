"""Measured results, concise Korean report and figures; original study remains read-only."""
import os
from common import *
os.environ['MPLCONFIGDIR']=str(CACHE/'.matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def md(df):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.4f}' if np.isfinite(v) else '해당 없음'
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def prepare():
    full=pd.read_csv(OUT/'tables/full_summary.csv')
    assert len(full)==15 and set(full.seed)=={42,43,44}
    baseline=json.loads((OUT/'baseline.json').read_text())
    phm=json.loads((OUT/'sources/phm_definition_audit.json').read_text())
    control=json.loads((OUT/'tables/summary_partial_control_r0.250_s42.json').read_text())
    columns=['evaluable_fault_runs','unavailable_fault_runs','partial_fault_runs','detected_fault_runs',
        'conditional_detection_rate','overall_detection_rate','detection_roc_auc','detection_f1',
        'cause_A_no_input','cause_B_input_but_no_detection','baseline_detections_lost','median_delay_minutes',
        'removed_raw_observation_fraction','analytical_MCAR_detection_rate','normal_alarm_episodes_per_hour']
    means=full.groupby('rate')[columns].mean().reset_index()
    for field in ['conditional_detection_rate','overall_detection_rate','detection_roc_auc','detection_f1']:
        means[field+'_sd']=full.groupby('rate')[field].std(ddof=1).to_numpy()
    means['total_fault_runs']=10000
    means['unavailable_fraction']=means.unavailable_fault_runs/10000
    means.to_csv(OUT/'tables/summary_mean_sd.csv',index=False)
    comparisons=[dict(dataset='PHM2018 V3',condition='기존 고정 운영점10',total=phm['total_events'],available=phm['available_events'],
        unavailable=phm['unavailable_events'],partial=np.nan,detected=phm['detected_events'],
        conditional_rate=phm['conditional_detection_rate'],overall_rate=phm['overall_detection_rate'],
        unavailable_fraction=phm['unavailable_fraction'],task='고장 전 [tf-600,tf-60] raw 조기예측',evidence='V3 실제 Holdout 표·코드 검증')]
    for rate in [0.,.531]:
        q=means[means.rate==rate].iloc[0]
        comparisons.append(dict(dataset='TEP',condition=f'평가 불가능 실행 {rate:.1%}',total=10000,available=q.evaluable_fault_runs,
            unavailable=q.unavailable_fault_runs,partial=q.partial_fault_runs,detected=q.detected_fault_runs,
            conditional_rate=q.conditional_detection_rate,overall_rate=q.overall_detection_rate,
            unavailable_fraction=q.unavailable_fraction,task='고장 주입 후40h 이상탐지',evidence='3개 seed의 실제 측정 평균'))
    comparison=pd.DataFrame(comparisons);comparison.to_csv(OUT/'tables/phm_tep_comparison.csv',index=False)
    pe=pd.read_csv(OUT/'tables/events_partial_control_r0.250_s42.csv')
    paired=[]
    for name,mask in [('공백 선택 실행',pe.selected),('공백 미선택 실행',~pe.selected),('전체 실행',np.ones(len(pe),bool))]:
        q=pe[mask&pe.detected&pe.baseline_detected]
        paired.append(dict(group=name,paired_detected_runs=len(q),baseline_delay_median=q.baseline_delay_minutes.median(),
            masked_delay_median=q.delay_minutes.median(),paired_delay_change_median=(q.delay_minutes-q.baseline_delay_minutes).median()))
    paired=pd.DataFrame(paired);paired.to_csv(OUT/'tables/partial_paired_delays.csv',index=False)
    return full,means,baseline,phm,control,comparison,paired

def figures(full,means,baseline,control,comparison):
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,'figure.dpi':140})
    x=100*means.rate
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for field,label,color in [('overall_detection_rate','전체 실행 기준','#c27d35'),('conditional_detection_rate','입력 가용 실행 기준','#257d91')]:
        axes[0].errorbar(x,100*means[field],yerr=100*means[field+'_sd'],fmt='o-',color=color,label=label,capsize=4)
    axes[0].plot(x,100*means.analytical_MCAR_detection_rate,'--',color='#999999',label='가정 계산: Baseline×(1-r)')
    axes[0].set(xlabel='평가 불가능하도록 선택한 고장 실행 비율(%)',ylabel='실행 검출률(%)',ylim=(0,105),xticks=x,title='전체 검출 범위와 조건부 검출률')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    for field,label,color in [('detected_fault_runs','검출','#257d91'),('cause_B_input_but_no_detection','B: 입력 있음·미검출','#d59a53'),('cause_A_no_input','A: 입력 없음','#a4afb5')]:
        bottom=np.zeros(len(means)) if field=='detected_fault_runs' else means.detected_fault_runs/100 if field=='cause_B_input_but_no_detection' else (means.detected_fault_runs+means.cause_B_input_but_no_detection)/100
        axes[1].bar(np.arange(len(means)),means[field]/100,bottom=bottom,label=label,color=color)
    axes[1].set(xticks=np.arange(len(means)),xticklabels=[f'{v:.1f}' for v in x],xlabel='평가 불가능 실행 비율(%)',ylabel='전체 10,000개 실행 내 비율(%)',ylim=(0,105),title='전체 실행 결과의 배타적 분해')
    axes[1].legend(fontsize=8)
    fig.suptitle('TEP 공식 Test / 고장20종×500실행 / seed42·43·44 평균±SD\n출처: full_summary.csv · 모델·임계값 고정, 재학습 없음',fontsize=10)
    fig.savefig(OUT/'figures/coverage_and_detection.png');plt.close(fig)

    faults=pd.read_csv(OUT/'tables/fault_results.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,7),layout='constrained')
    for ax,col,title in [(axes[0],'overall_detection_rate','전체 실행 기준'),(axes[1],'conditional_detection_rate','입력 가용 실행 기준')]:
        a=faults.groupby(['fault','rate'])[col].mean().unstack().reindex(index=range(1,21),columns=means.rate)
        im=ax.imshow(a,aspect='auto',vmin=0,vmax=1,cmap='YlGnBu')
        ax.set(xticks=range(5),xticklabels=[f'{v:.1%}' for v in means.rate],yticks=range(20),yticklabels=range(1,21),xlabel='평가 불가능 실행 목표 비율',ylabel='Fault ID',title=title)
        for i in range(20):
            for j in range(5):ax.text(j,i,f'{a.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if a.iloc[i,j]>.6 else 'black')
    fig.colorbar(im,ax=axes,label='실행 검출률(무차원)')
    fig.suptitle('고장별 검출률 변화 / 공식 Test / 3개 seed 평균\n출처: fault_results.csv',fontsize=10)
    fig.savefig(OUT/'figures/fault_detection_heatmap.png');plt.close(fig)

    masks=[];labels=[]
    for name,first,last in [('원본',0,0),('주 실험: 종료까지 공백',141,960),('부분 가용 대조: 짧은 공백',141,180)]:
        raw=np.ones(960,bool) if first==0 else raw_mask(first,last)
        valid=np.r_[np.zeros(19,bool),valid_windows(raw)]
        masks.extend([raw,valid]);labels.extend([name+' / 센서',name+' / 20시점 창'])
    fig,axes=plt.subplots(1,2,figsize=(14,4.5),layout='constrained')
    for ax,zoom in [(axes[0],(0,48)),(axes[1],(6.5,11))]:
        ax.imshow(np.asarray(masks),aspect='auto',extent=[.05,48,5.5,-.5],cmap='Greys',vmin=0,vmax=1,interpolation='nearest')
        ax.axvline(8,color='#c27d35',ls='--',label='명목 고장8h');ax.set(xlim=zoom,yticks=range(6),yticklabels=labels,xlabel='명목 시각(h)')
        ax.legend(loc='lower right',fontsize=8)
    axes[0].set_title('전체48h');axes[1].set_title('고장 주변 확대: 검정=입력 가용, 흰색=불가')
    fig.suptitle('설계한 결측 마스크와 실제 윈도우 유효성 계산\n출처: config.json + valid_windows(); 최초19관측은 공통 warm-up',fontsize=10)
    fig.savefig(OUT/'figures/availability_masks.png');plt.close(fig)

    base=pd.read_csv(OUT/'tables/baseline_events.csv');partial=pd.read_csv(OUT/'tables/events_partial_control_r0.250_s42.csv')
    selected=partial.selected&partial.detected&partial.baseline_detected
    full531=pd.read_csv(OUT/'tables/events_full_censor_r0.531_s42.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for data,label in [(base.delay_minutes.dropna(),'Baseline 검출 실행'),(full531.delay_minutes.dropna(),'53.1% 공백 후 검출 실행')]:
        a=np.sort(data);axes[0].step(a,np.arange(1,len(a)+1)/len(a),where='post',label=label)
    for col,label in [('baseline_delay_minutes','같은 실행의 원본'),('delay_minutes','짧은 공백 후')]:
        a=np.sort(partial.loc[selected,col]);axes[1].step(a,np.arange(1,len(a)+1)/len(a),where='post',label=label)
    for ax in axes:ax.set(xlabel='명목 고장 후 탐지 지연(분)',ylabel='검출 실행의 누적 비율',xlim=(0,600),ylim=(0,1));ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].set_title('주 실험: 검출 실행 구성 변화, seed42');axes[1].set_title(f'짧은 공백 선택·공동 검출 {selected.sum():,}개 실행')
    fig.suptitle('검출된 실행만 비교 / 미검출·입력 없음은 지연에서 제외\n출처: baseline_events.csv 및 events_*.csv · 그림은0~600분 확대, 표에는 전체 지연 포함',fontsize=10)
    fig.savefig(OUT/'figures/delay_comparison.png');plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
    labels=['PHM V3 / 고장 전 예측','TEP 원본 / 고장 후 탐지','TEP 53.1% 입력불가 / 고장 후 탐지']
    left=np.zeros(3)
    for vals,label,color in [(comparison.detected/comparison.total*100,'검출','#257d91'),((comparison.available-comparison.detected)/comparison.total*100,'입력 있음·미검출','#d59a53'),(comparison.unavailable/comparison.total*100,'입력 없음','#a4afb5')]:
        bars=ax.barh(labels,vals,left=left,color=color,label=label)
        for bar,value in zip(bars,vals):
            if value>4:ax.text(bar.get_x()+bar.get_width()/2,bar.get_y()+bar.get_height()/2,f'{value:.2f}%',ha='center',va='center',fontsize=9)
        left+=vals.to_numpy()
    ax.set(xlim=(0,100),xlabel='각 데이터셋의 전체 기록 이벤트/실행 내 비율(%)',title='입력 가용성의 분모 비교 — 서로 다른 과제의 모델 순위 비교가 아님')
    ax.invert_yaxis();ax.legend(loc='lower right',fontsize=8)
    fig.suptitle('출처: PHM V3 stage6 실측 + TEP 3개 seed 실측 평균\nPHM raw 시간은 미검증, TEP는 규칙적3분 간격; 선행예측과 사후탐지는 직접 비교 불가',fontsize=10)
    fig.savefig(OUT/'figures/phm_tep_availability.png');plt.close(fig)

def run():
    full,means,b,phm,control,comparison,paired=prepare()
    assert json.loads((OUT/'sources/pilot_gate.json').read_text())['passed']
    assert json.loads((OUT/'sources/preservation_check.json').read_text())['all_unchanged']
    figures(full,means,b,control,comparison)
    main=means[means.rate==.531].iloc[0]
    compact=means[['rate','total_fault_runs','evaluable_fault_runs','unavailable_fault_runs','partial_fault_runs','detected_fault_runs','conditional_detection_rate','overall_detection_rate','detection_roc_auc','detection_f1']].copy()
    compact.columns=['평가불가 목표율','전체 실행','평가 가능','평가 불가','부분 가용','검출 평균','조건부 검출률','전체 검출률','가용시점 ROC-AUC','가용시점 F1']
    for column in ['평가불가 목표율','조건부 검출률','전체 검출률']:
        compact[column]=compact[column].map(lambda x:f'{x:.2%}')
    for column in ['전체 실행','평가 가능','평가 불가','부분 가용']:
        compact[column]=compact[column].astype(int)
    compact['검출 평균']=compact['검출 평균'].map(lambda x:f'{x:,.2f}')
    compact.to_csv(OUT/'tables/compact_results.csv',index=False)
    report=f'''# TEP 센서 입력 가용성 후속 실험

정리일: 2026-11-12. 기존 학습·분할·라벨·체크포인트·임계값과00~10번 노트북을 보존했다. 이 실험은 입력 가용성 변화만 다루며 재학습, 무작위 센서 노이즈, 라벨 변경, 학습자료 축소를 하지 않았다.

## 핵심 결과

고장 실행의53.1%에 유효 입력이 전혀 없게 하자 전체 검출률은 **98.51% → {main.overall_detection_rate:.2%}**로 낮아졌다. 반면 가용 실행의 조건부 검출률은 **{main.conditional_detection_rate:.2%}**였다. 감소의 핵심은 모델 분류 능력 변화가 아니라 **예측 기회를 잃은 실행**이다.

{md(compact)}

위 표는 seed42·43·44의 실제 측정 평균이다. 정수 분모는 세 번 모두 같고 검출 수 평균은 소수가 될 수 있다. 검출률은 백분율, ROC-AUC/F1은 0~1 단위다. SD와 seed별 정수 결과는 `tables/summary_mean_sd.csv`, `tables/full_summary.csv`에 있다. ROC-AUC/F1은 실제 유효 입력이 남은 시점에서만 계산했다.

![가용성과 검출](figures/coverage_and_detection.png)

## 1. Baseline 재현과 고정 조건

기존`CNN1D_Temporal_balanced_s42`, Temporal20×52, threshold={b['threshold']}를 사용했다. 원본 Test 전체9,880,500개 유효 시점을 새로 추론했고 저장된21클래스 확률과의 최대 차이는 **0.0**이었다. 이 새 추론 실측 시간은 **{b['fresh_full_test_inference_seconds']:.2f}초**이며 원본 체크섬 검사와 후처리 시간은 제외한다. 이 측정은 중단 전 실행 결과를 재사용했다.

ROC-AUC={b['detection_metrics']['detection_roc_auc']:.10f}, F1={b['detection_metrics']['detection_f1']:.10f}, 검출9,851/10,000, 지연 중앙값48분·평균{b['mean_delay_minutes']:.2f}분·P90={b['p90_delay_minutes']:.0f}분을 재현했다. 이벤트별 검출 여부·최초 검출 sample·지연도 기존10,000개 기록과 같았다. 숫자 비교 허용오차1e-10과 이벤트 정확 일치가 모두 통과한 뒤에만 실험했다.

정상 전용500실행의 경보는 **705 episodes / 23,525 관측시간 = {b['normal_alarm_episodes_per_hour']:.6f} episodes/h**, 샘플 FAR={b['normal_sample_false_alarm_rate']:.4%}다. 정상 실행은 가리지 않았으므로 모든 조건에서 동일했다. 고장 실행의 주입 전 정상 구간까지 합친 기존 전체 정상 FAR1.11%와 이 정상 전용 FAR1.24%는 분모가 다르다.

## 2. PHM V3 판정 코드를 직접 확인한 내용

PHM2018-Project-V3의`src/alarm_evaluation.py`는 이벤트별`[tf-600,tf-60]` 닫힌 사전 예측구간 안에 사용 가능한score anchor가 하나라도 있으면available=True로 판정한다. `eligible_anchors=0`이고 유효 기회가 없으면NO_INPUT이다. `src/features.py`의 B입력은 현재센서6종+recipe/stage와 과거300raw 평균·표준편차·최소·최대이며, 현재 lag≤12raw·같은 세그먼트·90%이상 coverage 등으로 유효성을 판정한다.

최종`validation_budget_10`의194개 이벤트를 원표에서 직접 확인했다: 입력 가용91개, NO_INPUT103개, 검출33개. available과eligible_anchors>0이 모두 일치했다. PHM의raw 시간은 초/분으로 환산하지 않았다. 참조7개 파일 경로·SHA256과 실제194개 이벤트는`sources/phm_definition_audit.json`, `tables/phm_verified_events.csv`에 보존했다.

이 실험이 대응시키는 것은 **평가구간 내 유효 입력 기회가0개라는 의미**다. PHM의 고장 전 시간창·B특징 규칙과 TEP의 고장 후40h·20×52 규칙은 같지 않다. PHM 원자료의 공백 길이 분포나 실제 발생 기전을 복원한 실험도 아니다. [PHM 공식 과제 설명](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/) 역시 고장 이전 예측 과제로 설명한다.

## 3. 누락 구간·실행 선정·부분 가용의 정의

각 Fault의500개 실행에서 균형 있게 비복원 선택했다. 조건은0/10/25/40/53.1%, seed42/43/44다. 같은 seed에서는 높은 비율의 선택 집합이 낮은 비율 집합을 포함한다.53.1%는전체10,000개 중5,310개로 정확히 맞췄고 Fault별265 또는266개다. PHM의103/194=53.0928%와 반올림 목표53.1%의 미세한 차이를 구분한다.

주 실험은 선택 실행의 **sample141~960의52개 채널 전체를 논리적으로 누락**시킨다. 명목 고장 sample160 전후에 걸친 연속 공백이며 이후 종료시점까지 지속된다. 원본 파일에NaN을 쓰거나 삭제하지 않았다. sample161~960의800개 post-fault시점 중20개 연속 입력이 모두 남은 시점이0개인 경우에만 평가 불가다. 미래값 보간·결측 대체·불가 시점의 정상/고장 예측값 생성은 하지 않았다. 경보3연속 계산에서는 결측 시점이 경보 연속성을 끊으며, 남은 예측을 시간축에서 압축하여 이어 붙이지 않았다.

단순히 고장 주변 짧은 구간만 지우면 이후 유효창이 복구돼 평가 가능하므로, 주 실험에서0기회를 보장하려면 TEP의긴40h 평가구간을 덮는 설계가 필요하다. 이는 의도적인 평가범위 검열 가정이다. **53.1%는 센서 샘플 삭제율이 아니다.** 이 조건에서전체 Test 원 관측의 제거 비율은{main.removed_raw_observation_fraction:.2%}다. 설계한 마스크의 입력값 자체는 가정이고 아래 결과는 그 마스크를 실제 예측·윈도우에 적용한 실측이다.

![누락과 유효창](figures/availability_masks.png)

## 4. 마스크 방식의 검증과 파일럿 순서

원 시계열 복사본에 실제NaN을 넣고20×52 창의 유한성으로 얻은 유효성 마스크가 누적합 방식 마스크와 완전히 같음을 확인했다. 남은 창의52×20개 입력값은 원본과 정확히 같았고, 같은 배치에서 원본창/NaN복사본의 유효창 예측도 완전히 같았다. 상태가 없는CNN의 고정된 한 창 입력을 재사용하므로 저장 예측의 유효성 필터링이 가능하다. GPU 배치 재배치에서 작은 부동소수 반올림 차이는 별도 기록했으며, 본 성능지표는 동일한 원 예측값을 필터링한 수치다.

0/25/53.1%×3seed 파일럿 및 짧은 공백 대조가 통과한 뒤10/40%까지 확장했다.25%·seed42의ROC-AUC/F1을sklearn로 독립 재계산하여 tie-group 집계와 차이가0임을 확인했다. 유효창이딱1개/2개인 사례는 평가 가능으로 남기고, 불가 시점을 가로질러 경보를 연결하지 않는 검사도 통과했다. 이런 실행은 입력이 조금 있어도3연속 경보에는 부족할 수 있으며, 가용 미검출이 모두 모델 분류 능력 문제라는 뜻은 아니다.

## 5. 원인 A/B 분해와 지연

53.1%조건의 전체 실행10,000개는 평균 **검출{main.detected_fault_runs:.2f}개 + A:입력 없음5,310개 + B:입력은 있으나 미검출{main.cause_B_input_but_no_detection:.2f}개**로 분해된다. A에는원래도 미검출이었던 실행이 포함될 수 있으므로 A수5,310개를 신규 검출손실 수로 오인하면 안 된다. Baseline에서 검출하던 실행의 손실은 평균{main.baseline_detections_lost:.2f}개이며 주 실험에서는 모두 입력 기회 상실로 설명된다.

주 실험의 입력이 남은 동일 검출 실행은 예측·임계값·시간축이 같으므로 **짝지은 지연 변화0분**이다. seed별 집계 지연 중앙값도 모두48분이었다. 이는 입력 누락이 지연에 영향 없다는 일반적 결론이 아니라, 이 설계가 선택 실행의 평가 기회 전체를 없앴기 때문이다.

고장별 전체·조건부 검출률은 다음과 같다. 기존 Fault15의 모델 미검출은 가용 조건에서도 남는다.

![고장별 검출률](figures/fault_detection_heatmap.png)

## 6. 짧은 공백 대조: 부분 가용 실행

25%·seed42의같은2,500개 선택 실행에서sample141~180만 가렸다. sample200부터20개 연속 관측이 복구돼 post-fault유효창은 **761/800=95.125%**다. 결과는평가 불가0개, 부분 가용2,500개, 완전 가용7,500개다. 부분 가용은 평가 가능 분모에 포함했다.

전체 검출은9,849/10,000(98.49%), 검출 지연 전체 중앙값은48→72분이었다. 전체 median은 비교대상 구성도 달라질 수 있으므로, 선택됐고 전후 모두 검출된 동일2,471개 실행을 추가로 대조했다: baseline 지연 중앙값48분, 공백 후126분, **실행별 지연 차이의 중앙값+81분**이다. 두 중앙값의 차이78분과 짝지은 차이 중앙값81분은 다른 통계다.

{md(paired)}

![지연 변화](figures/delay_comparison.png)

짧은 공백 후 가용 시점만의F1은{control['detection_f1']:.4f}로 오히려 증가했지만, 검출 실행2개를 잃었고 일부 지연이 늘었다. 평가 시점이 바뀐 F1 상승을 모델 개선으로 해석하면 안 된다.

## 7. PHM과 같은 형식의 분모 비교

{md(comparison[['dataset','condition','total','available','unavailable','partial','detected','conditional_rate','overall_rate','task']])}

PHM은실제 이벤트 정수 결과, TEP는3seed 실측 평균이다. PHM은고장 전 조기예측, TEP는고장 주입 후 이상탐지이므로 검출률 숫자로 모델 우열을 정하지 않는다. 공통으로 관찰하는 관계는 **전체 검출률=평가 가능 비율×조건부 검출률**이다. 부분 가용도 평가 가능에 포함한다. PHM의 총 경보9.8986/10,000raw와 TEP의 정상 오경보episodes/h는 정의·분모·단위가 달라 직접 비교하지 않는다.

![PHM과 TEP 가용성](figures/phm_tep_availability.png)

그림의Baseline×(1-r) 점선은 입력 상실이 검출 가능성과 독립이라는 가정의 계산값이다. 이론적 예상치와seed별 실제 균형 선택 결과를 구분했다. 실제 평균의 미세한 차이는 선택된 실행 구성에서 나온다.

## 8. 한계와 산출물

재학습하지 않았으므로 센서 누락의 학습 영향은 검증하지 않았다. 주 실험은 평가 범위 감소를 측정한 것으로 모델의 분류 함수가 나빠졌음을 직접 입증하지 않는다. 공백 발생은 고장별 균형 랜덤 가정이며 실제 PHM처럼 공정 조건·장비·고장 징후와 연관된 누락을 재현하지 않았다.3개 seed SD는누락 선택의 변동이며 TEP 독립성에 대한 신뢰구간이 아니다. PHM의raw 단위·이력 완전성과 실제 원인 발생 시점도 미확인이다.

기존 TEP 보호대상147개 파일과 PHM 참조7개 파일의SHA256 보존 검사를 통과했다. 실험 코드·새 노트북은`experiments/sensor_availability/`, 새 보고·표·그림은`reports/sensor_availability/`에만 작성했다. 원본 모델·라벨·분할·00~10번 노트북은 수정하지 않았다.

- `experiments/sensor_availability/11_availability_baseline.ipynb`
- `experiments/sensor_availability/12_availability_degradation.ipynb`
- `experiments/sensor_availability/13_phm_comparison.ipynb`
- 핵심수치: `tables/compact_results.csv`; seed별 전체 결과: `tables/full_summary.csv`
- 실행별 입력 가용 비율·미검출 원인·지연: `tables/events_*.csv`; 고장별: `tables/fault_results.csv`
- Baseline·마스크·파일럿·보존 검증: `sources/` 및 `tables/baseline_reproduction.csv`
'''
    (OUT/'final_report.md').write_text(report,encoding='utf-8')
    print('Report,5 figures and summary tables written',flush=True)

if __name__=='__main__':run()
