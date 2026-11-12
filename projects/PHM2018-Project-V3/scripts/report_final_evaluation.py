"""보호된 그룹 검증과 단일 Holdout 평가의 표·그림·한국어 보고서."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import *
from src.reporting import md
from src.visuals import plt, save, REGISTRY
from src.query import read_window, broken_line

OUT = ROOT / 'outputs/stage6'
NAMES = {'A_base_maximum':'A 최대가용', 'B_base_maximum':'B 최대가용',
         'A_base_common':'A 공통', 'B_base_common':'B 공통',
         'A_tuned_common':'조정 A 공통', 'C_base_common':'C 공통'}
COLORS_FINAL = ['#0072b2', '#d55e00']


def table(name):
    return pd.read_parquet(TABLES / (name + '.parquet'))


def main():
    REGISTRY.clear()
    frozen = json.loads((OUT / 'frozen_manifest.json').read_text(encoding='utf-8'))
    access = json.loads((OUT / 'holdout_access.json').read_text(encoding='utf-8'))
    assert access['status'] == 'complete'
    runs = table('stage5_training_summary'); group = table('stage5_budget_points')
    rank = table('stage6_selection_ranking'); evidence = table('stage6_selection_evidence')
    summary = table('stage6_holdout_summary').sort_values('validation_budget_per_10000_raw')
    events = table('stage6_holdout_events'); episodes = table('stage6_holdout_episodes')
    pertool = table('stage6_holdout_by_tool'); ci = table('stage6_holdout_uncertainty')
    classification = json.loads((OUT / 'holdout_classification.json').read_text(encoding='utf-8'))
    context = '최종 시간 Holdout · B 최대가용 · seed42 · 고정 H600/L60 · 기록194개 / 입력 가용91개 · 시간 단위 raw'
    points = summary.operating_point.tolist(); labels = ['검증 상한5 기준', '검증 상한10 기준']

    # 개념도: 각 장비의 실제 절대 시각은 다르므로 비율 축만 사용한다.
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, prefix in enumerate(['01–06 (12대)', '07–08 (4대)', '09–10 (4대)']):
        color = ['#087e80', '#0072b2', '#b9c2c9'][i]
        ax.broken_barh([(0, 60), (60, 20)], (i-.3, .6), facecolors=color, edgecolors='white')
        ax.broken_barh([(80, 20)], (i-.3, .6), facecolors='#eed4c6', edgecolors='white')
        ax.text(40, i, ['그룹 학습', '그룹 검증', '그룹 실험에서 미사용'][i], ha='center', va='center', color='white' if i<2 else '#172f3d')
        ax.text(90, i, '보호된\nHoldout', ha='center', va='center', fontsize=9)
    ax.axvline(60, color='white', lw=2); ax.axvline(80, color='#d55e00', ls='--')
    ax.set_yticks(range(3), ['01–06 (12대)', '07–08 (4대)', '09–10 (4대)']); ax.invert_yaxis()
    ax.set_xlim(0,100); ax.set_xticks([0,60,80,100]); ax.set_xlabel('장비별 관측 시간 범위의 비율 (%) · 행 수 비율 아님')
    ax.set_title('그룹 실험은 앞 80% 내에서만 수행 · 각 시간 분할의 900/1800 raw 경계 보호 유지')
    save(fig, 'final_evaluation/protected_split', '미래 Holdout을 보호한 장비 그룹 검증',
         'stage5_final_protocol.json / pretraining_split_boundaries.csv',
         '개념도(guard 폭은 생략) · 최종 모델은 이전 stage2의 전체20대 시간 train 모델 · 그룹 모델은 별도의 일반화 진단')

    order = list(NAMES); fig, axes = plt.subplots(1,3,figsize=(15,6))
    for ax, budget in zip(axes[:2], [5,10]):
        z = group[group.budget_per_10000_raw.eq(budget)].set_index('run_id').reindex(order)
        bars = ax.barh(np.arange(6), z.detected_events, color=['#0072b2','#087e80']*3)
        ax.set_yticks(range(6), [NAMES[x] for x in order]); ax.invert_yaxis(); ax.set_xlim(0,9)
        ax.bar_label(bars, labels=[f'{int(r.detected_events)} / {int(r.available_events)}' for r in z.itertuples()], padding=3)
        ax.set_title(f'비교 상한 {budget}: 검출 / 입력 가용'); ax.set_xlabel('검출 이벤트 수 (기록 분모=121)')
    z = group[group.budget_per_10000_raw.eq(10)].set_index('run_id').reindex(order)
    axes[2].barh(np.arange(6),z.available_events,color='#087e80',label='입력 가용')
    axes[2].barh(np.arange(6),121-z.available_events,left=z.available_events,color='#d8dfe3',label='예측창 입력 없음')
    axes[2].set_yticks(range(6),[NAMES[x] for x in order]); axes[2].invert_yaxis(); axes[2].set_xlabel('기록 이벤트 수'); axes[2].legend(fontsize=8); axes[2].set_title('최대가용56개 / 공통37개')
    save(fig,'final_evaluation/group_results','다른 장비에서의 검출력은 여전히 제한적',
         'stage5_budget_points.csv','장비07/08의4대 · 그룹 학습 장비와 분리 · validation 조기 종료/임계값 선택 포함 · 독립 최종 시험 아님')

    fig, axes = plt.subplots(1,2,figsize=(13,5))
    for i, family in enumerate(['A','B']):
        z=evidence[evidence.family.eq(family)]
        for j, validation in enumerate(['temporal_validation','protected_group_validation']):
            q=z[z.validation.eq(validation)].sort_values('budget_per_10000_raw')
            axes[0].plot(q.budget_per_10000_raw,q.recall_recorded*100,marker='o',ls=['-','--'][j],color=COLORS_FINAL[i],label=family+' '+['시간 검증','그룹 검증'][j])
    axes[0].set_xticks([5,10]); axes[0].set_xlabel('검증 비교 상한'); axes[0].set_ylabel('기록 고장 검출률 (%)'); axes[0].legend(fontsize=9); axes[0].grid(alpha=.2)
    bars=axes[1].bar(rank.family,rank.mean_worst_recall*100,color=['#087e80','#a9b8c4'])
    axes[1].bar_label(bars,fmt='%.3f%%',padding=3); axes[1].set_ylim(0,4.2); axes[1].set_ylabel('두 상한에서 최저 검출률의 평균 (%)'); axes[1].set_title('사전 규칙으로 B 선택 · 유의한 우월성 주장은 아님')
    save(fig,'final_evaluation/selection','Holdout을 열기 전에 선택한 모델과 기준',
         'stage6_selection_evidence.csv / stage6_selection_ranking.csv','시간·그룹 검증 중 낮은 기록 검출률을 상한5/10에서 평균 · 최종 B stage2 저장 모델 그대로 사용 · 재학습 없음')

    fig, axes=plt.subplots(1,3,figsize=(15,5))
    for j, den in enumerate(['recorded','available']):
        v=summary['recall_'+den].to_numpy()*100
        u=ci[ci.denominator.eq(den)].set_index('operating_point').reindex(points)
        axes[j].bar(labels,v,color=COLORS_FINAL)
        axes[j].errorbar(np.arange(2),v,yerr=np.vstack([v-u.ci_low.to_numpy()*100,u.ci_high.to_numpy()*100-v]),fmt='none',ecolor='#253847',capsize=5)
        for i,(y,n) in enumerate(zip(v,summary.detected_events)):axes[j].text(i,y+1.3,f'{int(n)}개 · {y:.1f}%',ha='center',fontsize=9)
        axes[j].set_ylim(0,60); axes[j].set_ylabel('검출률 (%)'); axes[j].set_title(['전체 기록194개 기준','입력 가용91개 기준'][j])
    actual=summary.alarm_episodes_per_10000_raw.to_numpy(); validation=[p['validation_actual_burden'] for p in frozen['operating_points']]
    x=np.arange(2);axes[2].bar(x-.18,validation,.36,label='시간 validation',color='#a9b8c4');axes[2].bar(x+.18,actual,.36,label='최종 Holdout',color=COLORS_FINAL)
    for i,v in enumerate(actual):axes[2].text(i+.18,v+.3,f'{v:.3f}',ha='center',fontsize=9)
    axes[2].set_xticks(x,labels);axes[2].set_ylim(0,12);axes[2].set_ylabel('전체 경보 / 관측 지지길이10,000 raw');axes[2].set_title('동일 숫자 임계값을 그대로 이전');axes[2].legend(fontsize=9)
    save(fig,'final_evaluation/holdout_results','동결한 두 운영점의 최종 시간 Holdout 결과',
         'stage6_holdout_summary.csv / stage6_holdout_uncertainty.csv',context+'\n오차막대: 장비쌍10개 군집 부트스트랩2,000회 조건부95% 구간 · 가용성/선택 편향을 보정한 외부 신뢰구간 아님')

    fig,axes=plt.subplots(1,2,figsize=(14,6));cats=['MATCHED','KNOWN_NEGATIVE_START','POSITIVE_UNMATCHED','UNKNOWN_START','LATE_OR_EXCLUDED_START']
    catnames=['고장 매칭','정책상 음성 시작','양성·매칭 없음','미확인 시작','임박·제외 시작']
    for i,r in enumerate(summary.itertuples()):
        vals=[getattr(r,'alarms_'+c) for c in cats]; bars=axes[0].barh(np.arange(5)+(i-.5)*.36,vals,.36,label=labels[i],color=COLORS_FINAL[i]);axes[0].bar_label(bars,labels=[f'{int(v):,}' for v in vals],padding=3,fontsize=8)
    axes[0].set_xscale('symlog',linthresh=1);axes[0].set_xlim(0,160000);axes[0].set_yticks(range(5),catnames);axes[0].invert_yaxis();axes[0].set_xlabel('경보 에피소드 수 (0 부근 선형, 이후 로그)');axes[0].legend(fontsize=8);axes[0].set_title('미확인 경보를 확정 오경보로 간주하지 않음')
    counts=[classification['label_counts'].get(str(k),0) for k in [0,1,2,3]]
    bars=axes[1].bar(['정책상 음성','양성','미확인 NA','임박 제외'],counts,color=['#0072b2','#d55e00','#a9b8c4','#b77fac']);axes[1].set_yscale('log');axes[1].set_ylim(10,2000000);axes[1].bar_label(bars,labels=[f'{v:,}' for v in counts],padding=4,fontsize=9);axes[1].set_ylabel('입력 Anchor 수 (로그)');axes[1].set_title(f'알려진 라벨은 {classification["known_anchors"]:,} / {classification["all_anchors"]:,}개')
    save(fig,'final_evaluation/unknown_burden','경보 부담과 정답 불확실성을 함께 해석해야 한다',
         'stage6_holdout_summary.csv / holdout_classification.json',context+'\n관측 지지길이는 명목4 raw 센서 지지와 점수 유지 구간의 교집합 · 실제 가동시간/시간당 오경보율 아님')

    fig,axes=plt.subplots(2,1,figsize=(14,9),sharex=True)
    for ax,point,label in zip(axes,points,labels):
        q=pertool[pertool.operating_point.eq(point)].sort_values('Tool');x=np.arange(len(q));det=q.detected_events.to_numpy();miss=(q.available_events-q.detected_events).to_numpy();no=(q.recorded_events-q.available_events).to_numpy()
        ax.bar(x,det,color='#087e80',label='검출');ax.bar(x,miss,bottom=det,color='#e3a16a',label='입력 있으나 미검출');ax.bar(x,no,bottom=det+miss,color='#d8dfe3',label='예측창 입력 없음')
        ax.set_title(label);ax.set_ylabel('기록 고장 수');ax.legend(ncol=3,fontsize=9);ax.set_xticks(x,q.Tool,rotation=45)
    save(fig,'final_evaluation/by_tool','장비별 검출·미검출·입력 공백',
         'stage6_holdout_by_tool.csv',context+'\n공통 B 모델을 모든 장비에 적용 · 장비별 임계값 조정 없음 · 무고장 장비의 경보 부담은 별도 표에 보존')

    telemetry=table('stage5_resource_telemetry');fig,axes=plt.subplots(1,2,figsize=(13,5))
    for i,point in enumerate(points):
        leads=np.sort(events.loc[events.operating_point.eq(point)&events.detected,'lead_raw'].to_numpy());axes[0].step(leads,np.arange(1,len(leads)+1)/len(leads),where='post',label=labels[i]+f' ({len(leads)}개)',color=COLORS_FINAL[i])
    axes[0].set_xlim(60,600);axes[0].set_ylim(0,1.05);axes[0].set_xlabel('최초 유효 경보의 선행시간 (raw)');axes[0].set_ylabel('검출 이벤트의 누적 비율');axes[0].legend(fontsize=9);axes[0].grid(alpha=.2);axes[0].set_title('검출된 고장에 한정된 선행시간')
    elapsed=(telemetry.unix_time-telemetry.unix_time.min())/60
    axes[1].plot(elapsed,telemetry.rss_bytes/2**30,label='프로세스 RAM GiB',color='#0072b2');axes[1].plot(elapsed,telemetry.gpu_used_mib/1024,label='전체 GPU 사용 GiB',color='#087e80');axes[1].set_xlabel('그룹 학습 경과 (분)');axes[1].set_ylabel('메모리 (GiB)');axes[1].legend(fontsize=9);axes[1].set_title('6개 모델 모두 cuda:0 학습 · CPU12스레드')
    save(fig,'final_evaluation/lead_resources','선행시간과 로컬 CUDA 사용 기록',
         'stage6_holdout_events.csv / stage5_resource_telemetry.csv','좌: 최종 Holdout 검출된 고장만 포함 · 우: 그룹 학습의 약2초 간격 표본, 순간 최대치 보장 없음\nGPU 전체 사용량은 다른 프로세스를 포함할 수 있음 · GPU 학습 순차, 경보 평가 CPU3프로세스')

    # 사후 설명용 두 사례. 모델이나 운영점 선택에 되먹임하지 않는다.
    p=pd.read_parquet(OUT/'holdout_predictions.parquet');chosen=events[events.operating_point.eq(points[1])]
    detected=chosen[chosen.detected].copy();detected['median_distance']=(detected.lead_raw-detected.lead_raw.median()).abs()
    cases=[detected.sort_values(['median_distance','Tool','fault_time']).iloc[0],chosen[chosen.available&~chosen.detected].sort_values(['Tool','fault_time']).iloc[0]]
    fig,axes=plt.subplots(3,2,figsize=(15,10),sharex='col');case_meta=[]
    for col,(event,title) in enumerate(zip(cases,['검출 사례: 선행시간 중앙값에 가장 가까움','미검출 사례: Tool·시각 정렬의 첫 가용 고장'])):
        ft=event.fault_time;start=ft-1200;end=ft+120
        data,audit=read_window(event.Tool,start,end,columns=['time','segment_id']+SENSORS[:2])
        q=p[p.Tool.eq(event.Tool)&p.time.between(start,end)].sort_values('time')
        for row,sensor in enumerate(SENSORS[:2]):
            for _,part in data.groupby('segment_id',sort=False):
                tx,vy=broken_line(part.time.to_numpy()-ft,part[sensor].to_numpy());axes[row,col].plot(tx,vy,lw=.9,color='#0072b2')
            axes[row,col].set_ylabel(sensor,fontsize=8)
        for _,part in q.groupby('segment_id',sort=False):
            tx,vy=broken_line(part.time.to_numpy()-ft,part.score.to_numpy(),threshold=60);axes[2,col].plot(tx,vy,'o-',ms=2,lw=.9,color='#253847')
        for i,point in enumerate(frozen['operating_points']):axes[2,col].axhline(point['threshold'],ls='--',color=COLORS_FINAL[i],label=labels[i])
        alarm=episodes[episodes.operating_point.eq(points[1])&episodes.Tool.eq(event.Tool)&episodes.start.between(start,end)]
        if len(alarm):
            scores=alarm[['start']].merge(q[['time','score']],left_on='start',right_on='time',how='left');axes[2,col].scatter(scores.start-ft,scores.score,marker='^',s=50,color='#d55e00',label='상한10 기준 경보 시작',zorder=6)
        for ax in axes[:,col]:ax.axvspan(-600,-60,color='#d3e9de',alpha=.5);ax.axvline(0,color='#a22c40',ls=':');ax.grid(alpha=.15);ax.set_xlim(-1200,120)
        axes[0,col].set_title(title+'\n'+event.Tool+' · '+event.fault_type+' · '+str(int(ft))+' raw',fontsize=10)
        axes[2,col].set_xlabel('고장 시각 기준 상대 raw');axes[2,col].set_ylabel('저장된 모델 점수');axes[2,col].legend(fontsize=7)
        case_meta.append(dict(event_id=event.event_id,Tool=event.Tool,fault_time=float(ft),detected=bool(event.detected),lead_raw=float(event.lead_raw) if pd.notna(event.lead_raw) else None,selection=title,query=audit))
    save(fig,'final_evaluation/event_cases','실제 센서와 저장 예측으로 확인한 검출·미검출 사례',
         'normalized 센서 / holdout_predictions.parquet / stage6_holdout_episodes.csv',context+'\n초록: 유효 사전 예측창 [-600,-60] · 공백/세그먼트 경계에서 선 끊음 · 두 센서만 표시 · 사례 선정은 사후 설명용')
    dump_json(OUT/'case_examples.json',case_meta)
    dump_json(ROOT/'outputs/final_evaluation_figure_manifest.json',REGISTRY)

    result=summary[['operating_point','threshold','detected_events','recorded_events','available_events','recall_recorded','recall_available','alarm_episodes','alarm_episodes_per_10000_raw','median_lead_raw']].copy()
    result['threshold']=result.threshold.map(lambda x:format(x,'.17g'))
    result=result.rename(columns={'operating_point':'운영점','threshold':'동결 임계값','detected_events':'검출','recorded_events':'기록','available_events':'입력 가용','recall_recorded':'기록 검출률','recall_available':'가용 검출률','alarm_episodes':'경보 수','alarm_episodes_per_10000_raw':'경보/10000 raw','median_lead_raw':'중앙 선행 raw'})
    gp=group[group.budget_per_10000_raw.isin([5,10])][['run_id','budget_per_10000_raw','detected_events','available_events','recorded_events','alarm_episodes_per_10000_raw']].copy()
    gp['run_id']=gp.run_id.map(NAMES);gp.columns=['그룹 모델','비교 상한','검출','입력 가용','기록','실제 경보/10000 raw']
    rt=runs[['run_id','device','train_rows','validation_all_rows','validation_known_rows','iterations','train_seconds','AP_known']].copy();rt.columns=['그룹 모델','장치','학습 행','검증 입력','검증 정답 확인','선택 트리 수','학습 초','확인 라벨 AP']
    uncertainty=ci.copy();uncertainty.columns=['운영점','분모','95% 하한','95% 상한','유효 재표집 수','선행 p10 raw','선행 중앙 raw']
    def picture(name):return f'![{name}](../outputs/figures/final_evaluation/{name}.png)\n'
    text=f'''# 19 보호된 장비 그룹 검증과 최종 시간 Holdout

사전 규칙에 따라 **B 최대가용 XGBoost 기준 모델**을 선정하고, 실제 모델·전처리·두 숫자 임계값을 동결한 뒤 각 장비의 마지막20% 시간 구간을 한 번 추론했습니다. 검출은 각각 **7/194개, 33/194개**였으며, 입력 가용91개 기준으로는 **7.7%, 36.3%**입니다. 현재 결과는 자동 운영 전환을 뒷받침하기에 부족합니다. 미확인 정답과 입력 공백, 약한 장비 일반화가 남아 있습니다.

## 1. 무엇을 검증했는가

장비마다 별도 모델을 만들지 않고 여러 장비의 데이터를 함께 학습한 공통 모델을 사용합니다. 장비 ID·시각·TTF·미래 고장 이력은 입력 특징이 아닙니다. B는 현재 센서6종+recipe/stage에 과거300 raw 평균/표준편차/최소/최대를 추가한32개 원래 특징입니다. 범주형 인코딩 후 열 수는 학습 집합에 따라 달라집니다.

그룹 실험의 학습 장비는01–06의12대, 검증 장비는07–08의4대입니다. 모두 기존 시간 train/validation 영역만 쓰고 각 시간 분할의 과거900/미래1800 raw 경계 보호와 분할 내 라벨을 유지했습니다. 그룹 실험은09–10 장비나 최종 시간 Holdout을 사용하지 않았습니다. 최대가용 A/B와 동일 입력 공통 A/B/조정A/C의6개 모델을 새로 CUDA 학습했습니다.

{picture('protected_split')}

그룹 validation은 조기 종료와 임계값 선택에 사용했습니다. 앞선 연구에서는 이 장비들을 이미 봤으므로 완전히 새로운 외부 장비 시험이 아닙니다. 장비쌍2개뿐이므로 일반화 성공의 통계적 증거로 보지 않습니다. 최종 Holdout은 **20대 기존 장비의 미래 시간 성능**이며 미학습 장비 최종 시험과는 다른 질문입니다.

## 2. 보호된 그룹 검증 결과

{md(gp)}

{picture('group_results')}

B 최대가용은 비교 상한5/10 모두 같은 후보 임계값에서5/56개(입력 가용8.9%),5/121개(기록4.1%)를 검출했습니다. 공통 코호트의 B/C는7/37개지만 입력이 적고 분모가 다르므로 최대가용 모델보다 우월하다고 단정할 수 없습니다. C/조정A는 사전 프로토콜상 최종 선택 대상이 아니며 Holdout 결과를 보고 후보로 추가하지 않았습니다.

{md(rt)}

모든6개 모델의 저장 booster 설정은 cuda:0입니다. RTX3060 Laptop에서 GPU 학습을 순차 수행하고 CPU12스레드, 경보 평가3프로세스를 사용했습니다. 학습 초는 fit 구간만 포함하며 데이터 읽기·변환·경보 계산을 포함하지 않습니다.

## 3. 선택 절차와 실제 동결

상한5/10 각각에서 시간·그룹 validation의 **기록 고장 검출률 중 작은 값**을 취하고 두 상한의 평균이 큰 A/B를 선택했습니다. B의 점수0.033592, A의 점수0.028017로 B를 택했습니다. 이는 정해 둔 선택 규칙의 결과이며 유의한 성능 차이를 입증하지 않습니다. 그룹 모델의 점수 척도를 최종 모델로 옮기지 않았습니다.

{picture('selection')}

최종 artifact는 `outputs/stage2/B_xgboost.joblib`의 기존 시간 train 모델(20대, seed42, 깊이3, 양성 가중치1, 선택 트리2개)과 train 전용 전처리입니다. train+validation 재학습을 하지 않았습니다. 임계값은 이 동일 artifact의 기존 시간 validation 후보들에서 검출 최대→실제 부담 최소→높은 임계값 순으로 선택했습니다. 상한5는 실제 경보0.991이었으므로 상한을 꽉 채운 최적 임계값이라는 의미가 아닙니다. 후보 격자 밖 전역 최적화도 아닙니다.

동결 시각(UTC): `{frozen['frozen_at_utc']}`. 최초 Holdout 접근 시각: `{access['opened_at_utc']}`. 두 운영점 모두 사전에 고정했고 Holdout에서 둘 중 하나를 새로 채택하지 않았습니다. 상한5/10은 연구 비교 기준이며 실제 현장 허용량으로 승인된 값이 아닙니다.

- [동결 manifest와 SHA256](../outputs/stage6/frozen_manifest.json)
- [선택 근거 전체 표](../outputs/tables/stage6_selection_evidence.csv)
- [최종 모델](../outputs/stage6/frozen_model.joblib), [전처리](../outputs/stage6/frozen_preprocessor.joblib)
- [단일 추론 접근 기록](../outputs/stage6/holdout_access.json)

## 4. 최종 시간 Holdout의 고정 운영점

{md(result)}

{picture('holdout_results')}

기록194개 중103개는 유효 사전 예측창에 입력이 없어 검출 기회가 없습니다. 91개만 분모로 제시하면 전체 시스템의 공백을 가리므로 두 분모를 병기했습니다. 경보 부담 분모는 실제 센서의 명목4 raw 지지구간과 같은 세그먼트의60 raw 점수 유지구간의 교집합으로 합계41,583,806 raw입니다. 시간 단위를 검증하지 않았으므로 시간당 경보나 실제 가동시간으로 바꾸지 않습니다.

{md(uncertainty)}

장비쌍 접두사10개를 군집 단위로2,000회 seed42 재표집했습니다. 위95% 구간은 고정 모델·임계값과 관측된 장비쌍에 조건부이며, 후보 선택·학습 변동·입력 누락·외부 환경 변화를 보정하지 않습니다. 선행시간은 검출한 고장만을 대상으로 하며 전체 고장의 기대 선행시간이 아닙니다.

## 5. 경보와 정답 불확실성

{picture('unknown_burden')}

상한5 기준 경보3,389개 중 미확인 시작3,302개(97.4%), 상한10 기준41,162개 중38,878개(94.5%)입니다. 미확인 경보를 정상 구간의 확정 오경보로 간주하지 않았습니다. 정책상 음성 시작 경보는 각각77개/2,246개이며 이 역시 보수적 라벨 정의에 따른 수치입니다. 전체 경보 부담을 함께 제시해 NA 경보가 운영 부담에서 사라지지 않도록 했습니다. 고장 매칭 에피소드는8개/34개, 검출 고장은7개/33개로 일치하지 않을 수 있습니다. 한 고장에 여러 에피소드가 매칭될 수 있고 동일 에피소드가 여러 이벤트에 매칭될 수도 있습니다.

입력736,646행 중 알려진 라벨32,645행(양성635), NA703,931행, 임박 제외70행입니다. 알려진 라벨만의 AP={classification['AP_known']:.6f}, ROC-AUC={classification['ROC_AUC_known']:.6f}입니다. 양성 비율은{classification['positive']/classification['known_anchors']:.2%}이며 AP만으로 전 구간 성능이나 경보 품질을 주장하지 않습니다. 점수는 보정된 고장 확률로 해석하지 않습니다.

{picture('by_tool')}
{picture('lead_resources')}

## 6. 실제 구간 사례

{picture('event_cases')}

상한10 기준 검출 사례는 검출 선행시간 중앙값에 가장 가까운 고장, 미검출 사례는 Tool/시각 순 첫 입력 가용 미검출 고장을 선택했습니다. 두 사례는 사후 설명용이며 전체 성능의 대표성을 주장하지 않습니다. 원본 정규화 센서 두 종류와 이미 저장된 점수만 읽고 추가 추론하지 않았습니다. 공백과 세그먼트 경계를 연결하지 않았습니다. [사례 ID와 조회 감사 기록](../outputs/stage6/case_examples.json)을 보존했습니다.

## 7. 검증과 기술적 복구

보호된 그룹 산출물1,170개 검사, 단위 테스트46개, 최종 Holdout123개 검사를 통과했습니다. 최종 검사는 원장 시각·라벨·분할, 각 경보의 실제 시작과 직전 발화,60–600 raw 고장 매칭, 경보 분류 합계, 지표 재계산, SHA256 동결 보존을 확인했습니다. 원본65개 CSV의 크기/수정시각이 그대로이며 원본 전체 내용 해시를 계산한 것은 아닙니다.

첫 Holdout 추론은736,646행을 저장한 뒤 이벤트 표를 합치는 단계에서 bool 열이 object로 변해 `~tagged`가 실패했습니다. 동결 평가 스크립트를 수정하지 않고 별도 복구 스크립트에서 **저장 예측의 tagged/available/detected 자료형만 명시적으로 bool로 변환**하여 같은 고정 임계값·경보 규약·부트스트랩으로 집계했습니다. 전체 추론1회, 추가 추론0회, 추가 학습0회, Holdout 임계값 탐색0회입니다. 예측 파일 해시와 모든 동결 파일 해시는 복구 전후 일치합니다. 원래 실패 로그를 보존했습니다.

- [그룹 검사](../outputs/validation/stage5_checks.json), [최종 검사](../outputs/validation/stage6_checks.json)
- [원래 집계 오류 로그](../outputs/stage6_holdout_evaluation.log), [복구 감사](../outputs/stage6/postprocessing_recovery.json)
- [결과 요약 CSV](../outputs/tables/stage6_holdout_summary.csv), [장비별 CSV](../outputs/tables/stage6_holdout_by_tool.csv)
- [고장별 CSV](../outputs/tables/stage6_holdout_events.csv), [경보별 CSV](../outputs/tables/stage6_holdout_episodes.csv), [매칭 CSV](../outputs/tables/stage6_holdout_matches.csv)
- [실행 Notebook](../notebooks/15_final_holdout.ipynb)

## 8. 연구 결론과 이후 방향

현재 모델과 두 운영점은 연구 기준선으로 보존하고 자동 배포하지 않습니다. 다음 연구의 우선순위는 입력 가용성이 낮은103개 고장의 원인, 미확인 정답 구간의 이력 완전성·정비 종료·시간 단위 확인, 공정 조건 변화에 따른 장비 일반화 실패를 규명하는 것입니다. 그 후 새 데이터 또는 별도로 확보한 외부/미래 구간을 검증 대상으로 정하고 개선 모델을 비교해야 합니다. 이번 Holdout은 이미 결과를 확인했으므로 반복 튜닝 후 같은 구간을 독립 검증으로 다시 주장하지 않습니다.

이번 독립성은 예측점수·모델·임계값 선택에서 보류한 미래 구간에 한정됩니다. 초기 전처리·원본 품질·상수 경보 진단에서는 전체 기간을 이미 확인했으므로 완전 맹검 외부 시험은 아닙니다. 동결된 preprocessing 설정의 초기 DIAGNOSTIC_ONLY 상태 문구와 이전 보고서의 Holdout 미사용 문구는 당시 기록입니다. 현재 실행 상태의 근거는 이 보고서와 stage6 접근·검사 기록입니다.
'''
    (ROOT/'reports/19_group_and_holdout.md').write_text(text,encoding='utf-8')
    print('그림',len(REGISTRY),'개 및 최종 보고서 생성 완료')


if __name__ == '__main__':
    main()
