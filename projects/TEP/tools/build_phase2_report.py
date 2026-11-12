"""Korean reports, figures, model card, research questions, and Phase 2 notebooks."""
from pathlib import Path
import sys,json,platform,importlib.metadata,datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.phase2 import CACHE,CFG,trajectories
from src.evaluation.run_phase2 import M,T,P,Predictor
from src.models.temporal import TemporalNet
from sklearn.metrics import precision_recall_fscore_support
import torch
R=ROOT/'reports/markdown';F=ROOT/'reports/figures/phase2';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,'figure.dpi':140})
SOURCES='''
## 출처 및 재현

- 데이터: Harvard Dataverse V1.0, https://doi.org/10.7910/DVN/6C3JR1 . RData 4개의 공식 MD5 일치를 Phase 1에서 확인했다.
- 공정·공개 고장 참조: https://github.com/Blound04/tennessee-eastman-dataset . Dataset A는 학습에 사용하지 않았다. A의 물리 정의를 B 구현과 완전히 동일하다고 가정하지 않았다.
- API: [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [XGBoost](https://xgboost.readthedocs.io/en/stable/gpu/index.html), [PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.rnn.LSTM.html).
- 실제 설정: configs/phase2.json, 모델 선택 기록: models/phase2/selection_lock.json. 표·그림은 reports/tables/phase2, reports/figures/phase2에 있다.
'''
def md(df):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.4f}' if np.isfinite(v) else '미검출/해당 없음'
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
def write(name,body): (R/name).write_text(body+'\n'+SOURCES,encoding='utf-8')
def pct(x):return f'{100*x:.2f}%'

def summarize():
    lock=json.loads((M/'selection_lock.json').read_text());dense=json.loads((P/'dense_metrics.json').read_text())
    test=pd.read_csv(T/'test_comparison.csv')
    val=pd.concat([pd.read_csv(T/'validation_baselines.csv'),pd.read_csv(T/'validation_temporal.csv')],ignore_index=True)
    records=[]
    for (family,feature,weight),group in val.groupby(['family','feature','weight']):
        if group.seed.nunique()!=3:continue
        matching=test[test.model_id.isin(group.model_id)]
        record=dict(model=family,features=feature,weight=weight,seeds=len(group),
            validation_selection_mean=group.selection_score.mean(),detection_f1_mean=matching.detection_f1.mean(),detection_f1_sd=matching.detection_f1.std(ddof=1),
            diagnosis_macro_f1_mean=matching.diagnosis_macro_f1.mean(),diagnosis_macro_f1_sd=matching.diagnosis_macro_f1.std(ddof=1),
            sample_FAR_mean=matching.sample_false_alarm_rate.mean(),end_to_end_macro_f1_mean=matching.end_to_end_macro_f1.mean(),
            training_seconds_mean=group.training_seconds.mean(),inference_seconds_mean=matching.inference_seconds.mean())
        records.append(record)
    summary=pd.DataFrame(records).sort_values('validation_selection_mean',ascending=False)
    summary.to_csv(T/'model_performance_summary.csv',index=False)
    return lock,dense,test,val,summary

def figures(lock,dense,summary):
    s=summary.iloc[::-1]
    fig,ax=plt.subplots(figsize=(11,7),layout='constrained');iy=np.arange(len(s))
    labels=s.model+' / '+s.features
    ax.barh(iy-.17,s.detection_f1_mean,.32,label='이상 탐지 F1',xerr=s.detection_f1_sd,color='#257d91')
    ax.barh(iy+.17,s.diagnosis_macro_f1_mean,.32,label='고장 분류 Macro F1',xerr=s.diagnosis_macro_f1_sd,color='#d89140')
    ax.set_yticks(iy,labels);ax.set(xlim=(0,1),xlabel='F1 (무차원)',title='공식 Test / 동일 997,500개 관측 시점 / 3개 seed 평균 ± 표준편차\n출처: Harvard V1.0, 설정·모델 선택은 Validation에서 확정');ax.legend(loc='lower right')
    fig.savefig(F/'model_comparison.png');plt.close(fig)
    cm=np.load(T/'primary_dense_confusion.npy');norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
    fig,ax=plt.subplots(figsize=(11,9),layout='constrained');im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues');fig.colorbar(im,ax=ax,label='실제 클래스 내 비율')
    ax.set_xticks(range(21));ax.set_yticks(range(21));ax.set(xlabel='경보 임계값을 적용한 예측 (0=정상)',ylabel='명목 실제 상태 (0=정상)',title=f'{lock["primary_model"]}\n공식 Test / 모든 유효 9,880,500시점 / 출처: Harvard V1.0')
    for i in range(21):
        for j in range(21):
            if norm[i,j]>=.1:ax.text(j,i,f'{norm[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if norm[i,j]>.5 else 'black')
    fig.savefig(F/'primary_confusion.png');plt.close(fig)
    pf=pd.read_csv(T/'primary_dense_per_fault.csv');delay=pd.read_csv(T/'dense_detection_delays.csv');d=delay[delay.system=='joint_primary']
    fig,axes=plt.subplots(2,1,figsize=(12,9),layout='constrained')
    axes[0].plot(pf.fault,pf.detection_recall,'o-',label='샘플 탐지 Recall');axes[0].plot(pf.fault,pf.conditional_diagnosis_recall,'s-',label='고장 조건부 분류 Recall');axes[0].plot(d.fault,d.run_detection_rate,'^-',label='3연속 경보 실행 검출률')
    axes[0].set(ylim=(0,1.05),ylabel='비율',title='고장별 성능 / 공식 Test / Fault별 500개 실행');axes[0].legend()
    axes[1].bar(d.fault,d.median_delay_minutes,color='#257d91');axes[1].set(xlabel='Fault ID',ylabel='검출된 실행의 지연 중앙값 (분)',title='명목 주입 8h 기준 / 3개 연속 경보 / 미검출 실행은 지연에서 제외')
    for ax in axes:ax.set_xticks(range(1,21));ax.grid(axis='y',alpha=.2)
    fig.suptitle('출처: Harvard V1.0, 처음 19개 관측 warm-up 제외. 미검출률은 별도 표 참조',fontsize=10)
    fig.savefig(F/'fault_performance_delay.png');plt.close(fig)
    rankings=pd.read_csv(T/'shap_sensor_rankings.csv');p=rankings[rankings.model_id==lock['primary_model']]
    importance=p.groupby('variable').mean_absolute_attribution.mean().nlargest(15).iloc[::-1]
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained');ax.barh(importance.index,importance,color='#257d91')
    ax.set(xlabel='고장 class logit의 평균 절대 SHAP 기여 (시차 합)',title='주 모델 GradientSHAP / Test 실행1~5, 시점240/480/960\nTrain 배경64개, nsamples64 / 출처: Harvard V1.0 / 물리 원인 순위 아님')
    fig.savefig(F/'primary_shap.png');plt.close(fig)
    # Standardized actual B responses; no unverified raw units are attached.
    ref=np.load(CACHE/'healthy_train_reference.npz');raw=trajectories('test');resp=pd.read_csv(T/'b_fault_sensor_response.csv')
    sensors=json.loads((CACHE/'sensor_names.json').read_text())
    for fault in range(1,21):
        ranked=p[p.fault==fault].sort_values('rank').head(5)
        if len(ranked)==0:ranked=rankings[rankings.fault==fault].sort_values('rank').head(5)
        fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
        for variable in ranked.variable:
            j=sensors.index(variable);z=(raw[fault,0,:,j]-ref['mean'][j])/ref['std'][j]
            ax.plot(np.arange(1,961)*3/60,z,label=variable,linewidth=.8)
        ax.axvline(8,color='#444444',ls='--',label='공식 명목 주입 8h');ax.legend(ncol=3)
        ax.set(xlabel='명목 sample×3분 / 시간 (h)',ylabel='정상 Train 기준 표준화 편차 (무차원)',title=f'Fault {fault} / Test simulationRun 1 / 주 모델 SHAP 상위5 변수\n출처: Harvard V1.0, 모델 기여 변수와 실제 원인은 별개')
        fig.savefig(F/f'fault_{fault:02d}_shap_response.png');plt.close(fig)

def build():
    lock,dense,test,val,summary=summarize();figures(lock,dense,summary)
    a=dense['joint_primary'];b=dense['dedicated_binary'];pf=pd.read_csv(T/'primary_dense_per_fault.csv');delays=pd.read_csv(T/'dense_detection_delays.csv');alarm=pd.read_csv(T/'dense_alarm_episodes.csv')
    primary_delay=delays[delays.system=='joint_primary'];primary_alarm=alarm[(alarm.system=='joint_primary')&(alarm.fault>0)]
    ci=pd.read_csv(T/'primary_cluster_bootstrap_ci.csv')
    rankings=pd.read_csv(T/'shap_sensor_rankings.csv');top=rankings[(rankings.model_id==lock['primary_model'])&(rankings['rank']<=3)]
    response=pd.read_csv(T/'b_fault_sensor_response.csv')
    catalog=pd.read_csv(ROOT/'reports/tables/fault_catalog.csv')
    physical=[]
    for fault in range(1,21):
        c=catalog[catalog.fault==fault].iloc[0]
        rr=response[response.fault==fault].copy();rr['abs_change']=rr.post_minus_pre_over_healthy_train_sd.abs()
        observed=rr.nlargest(3,'abs_change')
        explained=top[top.fault==fault].sort_values('rank')
        physical.append(dict(fault=fault,public_scenario_reference=c.cause,scenario_source='A 공개 카탈로그; B 구현 동일성 별도 미확인',
            observed_top3=', '.join(observed.variable),shap_top3=', '.join(explained.variable),
            interpretation='변화·예측기여·원인은 서로 다름; 정량 원인 적중률 미계산'))
    physical=pd.DataFrame(physical);physical.to_csv(T/'fault_cause_comparison.csv',index=False)
    binaryval=pd.read_csv(T/'validation_binary.csv');binarytest=test[test.model_id.str.startswith('BinaryXGBoost')]
    binary_compare=binaryval[['model_id','weight','seed','train_rows','detection_f1']].merge(binarytest[['model_id','detection_f1','sample_false_alarm_rate']],on='model_id',suffixes=('_validation','_test'))
    binary_compare.to_csv(T/'imbalance_comparison.csv',index=False)
    unknown=test[test.model_id.str.startswith('Unknown')]
    ck=torch.load(M/(lock['primary_model']+'.pt'),map_location='cpu',weights_only=False);net=TemporalNet(ck['kind']);net.load_state_dict(ck['state_dict'])
    parameters=sum(p.numel() for p in net.parameters())
    dense_runtime=json.loads((P/'dense_complete.json').read_text())
    faultfree_alarm=alarm[(alarm.system=='joint_primary')&(alarm.fault==0)]
    alarm_rate=faultfree_alarm.pre_alarm_episodes.sum()/faultfree_alarm.normal_exposure_hours.sum()
    perclass=[];cm=np.load(T/'primary_dense_confusion.npy')
    for i in range(21):
        tp=cm[i,i];precision=tp/max(1,cm[:,i].sum());recall=tp/max(1,cm[i].sum());f1=2*precision*recall/max(1e-15,precision+recall)
        perclass.append(dict(fault=i,precision=precision,recall=recall,f1=f1,support=int(cm[i].sum())))
    pd.DataFrame(perclass).to_csv(T/'primary_dense_classification_report.csv',index=False)
    pd.DataFrame(cm,index=[f'true_{i}' for i in range(21)],columns=[f'pred_{i}' for i in range(21)]).to_csv(T/'primary_dense_confusion.csv')
    # Counterpart: diagnostic-only confusion on all true post-fault samples.
    probs=np.load(CACHE/'primary_dense_probabilities.npy',mmap_mode='r')
    diagcm=np.zeros((20,20),dtype=np.int64)
    for fault in range(1,21):
        block=np.asarray(probs[fault*500*941:(fault+1)*500*941]).reshape(500,941,21)[:,141:,:]
        pred=block[:,:,1:].argmax(2).ravel();diagcm[fault-1]=np.bincount(pred,minlength=20)
    np.save(T/'primary_conditional_diagnosis_confusion.npy',diagcm)
    pd.DataFrame(diagcm,index=[f'true_{i}' for i in range(1,21)],columns=[f'pred_{i}' for i in range(1,21)]).to_csv(T/'primary_conditional_diagnosis_confusion.csv')
    write('07_phase2_experiment_design.md',f'''# Phase 2 실험 설계와 누수 방지

사용자 승인: 2026-11-10. Dataset B만 모델 학습에 사용했다. 52개 공정 변수 외 faultNumber/simulationRun/sample은 입력에 넣지 않았다.

## 분할과 라벨

공식 Training 500개 simulationRun을 seed2026으로 학습400·검증100개로 나눴다. 같은 run 번호의 모든 Fault와 정상 실행을 함께 묶었다. 공식 Test 500개 RNG family는 최종 평가에 보존했다. 정확한 목록은 phase2_split_manifest.json에 있다. 공유 초기20샘플 해시가 Train/Validation/Test 사이에 겹치지 않는지 자동 검사했다.

학습용 전체 Training 실행은 8,400개 궤적(21상태×400), Validation은 2,100개 궤적, Test는 10,500개 궤적이다. 궤적 수와 독립 난수 family 수(각400/100/500)를 구분한다. 원 seed 자체는 제공되지 않아 해시/공식 seed 설명을 근거로 삼았다.

명목 라벨은 Training sample≤20 및 Test sample≤160을 정상0으로, 이후는 해당 Fault ID1~20으로 처리했다. 실제 센서 반응이 늦게 시작해도 라벨은 주입 기준을 유지했다. 조기 반응이 없는 구간도 평가에서 제외하지 않았다. 관측시간은 sample×3분, 주입시각은 Training1h/Test8h로 정의했다. 첫 기록의 실제 0/3분 관례 미확인에 따른 최대 한 샘플 시간 해석 불확실성이 있다.

## 공정 특징과 표본

- Current: 현재 52개 값.
- Statistical: Current + 과거20개 관측의 평균·표준편차·최소·최대·변화율로 총312개 값. 변화율 분모는 첫/끝 관측 간57분이다.
- Temporal: 과거20×52 배열, 인과적 창. 시간 순서는 유지하고 실행 경계를 넘지 않는다.

세 집합의 비교 관측 시점은 동일하다. 처음19개 샘플은 warm-up으로 제외하며 Test에서199,500개 샘플이 제외된다. Training은 sample20,60,…,500을 추출하고 고장 궤적의 정상 prefix는 정상 실행과 중복되지 않도록 제거했다. 결과 학습 창은101,200개, Validation 비교 창102,900개다. 모든8400개 학습 궤적을 포함하지만 모든 행을 학습한 것은 아니다.

**학습 stride40(120분)은 이번 실험의 제한이다.** Training 주입 직후 sample21~59의 창을 직접 학습하지 않아 첫 약2시간의 초기 과도 사례가 충분하지 않다. Test에서는 이 구간을 포함하여 평가했다. 향후 초기 구간을 촘촘하게 학습하려면 새로운 미사용 평가 실행으로 재검증해야 한다.

전 모델의 공통 Test 비교는 sample20부터10개 간격으로997,500개 시점을 사용했다. 선택한 주 모델과 별도 이진 탐지기는 매 샘플의 유효9,880,500개 시점에서도 평가했다. 비교 표와 dense 결과는 같은 지표 이름이라도 표본 집합이 다르므로 구분해서 읽는다.

## 모델·선택·실행 환경

Logistic Regression, Random Forest, XGBoost는 Current/Statistical을 비교했다. natural/balanced를 seed42로 먼저 비교한 후 검증 우수 설정을 seed7/2026으로 반복했다. Logistic Regression은 lbfgs 결정적 해법이어서 seed 반복값이 같을 수 있다. LSTM·GRU·1D-CNN은 동일 창·동일 정규화로 각각3개 seed를 사용했다. 최대8 epoch, patience2, hidden64, Adam lr0.001이다. Transformer는 이번 규모에서 추가하지 않았다.

정규화는 Train 선택 관측으로만 fit했다. 시계열 모델은 채널별 Train 평균·표준편차를 공유한다. Random Forest/XGBoost는 원 특징을 사용했다. 클래스 가중치 및 over/under sampling은 Train에서만 계산했다. SMOTE는 사용하지 않았다.

RTX3060 Laptop6GB에서 PyTorch2.5.1+cu121을 실행했다. 설치된 XGBoost3.4.1의 실제 GPU probe는 CPU로 fallback했으므로 XGBoost는 CPU8 threads로 명시했다. 기존 Conda 환경을 변경하지 않고 프로젝트 .deps에 설치했다. GPU 연산은 seed를 고정했지만 비트 단위 재현까지 보장하도록 deterministic 옵션을 강제하지는 않았다.

주 모델은 검증 탐지F1과 고장분류Macro F1 평균을3개 seed에 걸쳐 평균해 선택했다. 주 모델: `{lock['primary_model']}`. 대표 seed42는 사전 지정했으며3개 모델의 ensemble이 아니다. 별도 이진 탐지기: `{lock['binary_model']}`. 모델·임계값 고정시각(UTC): {lock['locked_utc']}. 이후 Test 수치로 설정을 바꾸지 않았다.

## 경보와 통계

검증에서 정상 샘플 FAR≤1%를 만족하는 후보 중 탐지F1 최대 임계값을 정했다. Test FAR1%를 보장하는 규칙은 아니다. 연속3개 임계값 초과를 경보 확인으로 정의하고, 연속 초과 묶음을1 episode로 센다. 정상 판정1개가 나오면 episode가 끊긴다. 이벤트 지연은 주입 경계에서 카운터를 초기화한 후 post-fault3개 연속 경보의 확인시각과 명목 주입시각 차이다. 주입 전 경보가 이후로 이어지는 경우도3개 post 샘플이 있어야 검출로 센다. 최소 지연은9분이며 warm-up은 별도다.

지연 통계는 검출된 실행만 포함하며 미검출률과 함께 보고한다. 500개 Test run family 전체(모든 Fault 포함)를 단위로500회 cluster bootstrap하여95% 구간을 계산했다. 동일 실행의 행을 독립 표본으로 bootstrap하지 않았다. 일반 Test 지표는 서로 같은 정상 prefix도 각 시나리오의 관측으로 포함하며, 정상 전용 실행 FAR와 주입 전 FAR도 따로 제공한다.
''')
    write('08_model_performance.md',f'''# 모델 성능 비교

## 공통 Test 시점, 3개 seed 평균

{md(summary[['model','features','detection_f1_mean','detection_f1_sd','diagnosis_macro_f1_mean','diagnosis_macro_f1_sd','sample_FAR_mean','end_to_end_macro_f1_mean']])}

![모델 비교](../figures/phase2/model_comparison.png)

모델 선정은 Validation 점수로 수행했다. Test 순위가 다르더라도 주 모델을 바꾸지 않았다. conditional diagnosis는 실제 고장 구간을 알고 있다고 가정한20종 분류이며, gated end-to-end는 탐지 실패까지 반영한21상태 지표다.

## 선택 주 모델의 모든 유효 Test 시점

{md(pd.DataFrame([a]))}

{md(ci)}

정상 전용 실행 FAR: {pct(a['normal_faultfree_FAR'])}. 고장 실행의 주입 전 FAR: {pct(a['pre_injection_FAR'])}. 위 수치는 단일 샘플 임계값 초과율이다. 정상 전용 경보 episode는 시간당 {alarm_rate:.5f}개다. 연속3개 규칙을 적용한 실행 검출률과는 다르다.

![Confusion matrix](../figures/phase2/primary_confusion.png)

정밀도/재현율/F1/support 전체는 primary_dense_classification_report.csv, 조건부 고장 Confusion Matrix는 primary_conditional_diagnosis_confusion.npy, 경보 적용21상태 matrix는 primary_dense_confusion.npy다. PR-AUC는 average precision 정의를 사용하며 사다리꼴 적분값과 구분한다. Macro ROC-AUC는 OVR 평균이다.

## 별도 이진 탐지기와 불균형 처리

{md(pd.DataFrame([b]))}

{md(binary_compare)}

이진 탐지기는 정상/고장만 예측하며 고장 ID를 주 모델 대신 결정하지 않는다. 두 모델을 결합한 새로운 cascade의 성능을 이 보고서에서 주장하지 않았다. oversampling/undersampling 비교는 seed42 탐색이며 natural/balanced는3개 seed로 확인했다. Joint21클래스는 정상과 개별 고장이 비슷한 크기라 balanced 효과가 작을 수 있지만, 이진 문제에서는 고장 총수가 훨씬 크다.
''')
    write('09_fault_diagnosis_and_delay.md',f'''# 고장별 진단과 탐지 지연

{md(pf)}

{md(delays)}

![고장 성능과 지연](../figures/phase2/fault_performance_delay.png)

주 모델 전체 고장 실행의3연속 경보 검출률은 {pct(primary_alarm.detected.mean())}, 검출된 실행의 지연 중앙값은 {primary_alarm.detection_delay_minutes.median():.1f}분이다. 미검출 실행의 지연을0으로 채우지 않았고 평균/중앙값 계산에서 제외했다. 고장마다 분포가 다르므로 전체 중앙값만으로 조기 검출을 판단하면 안 된다.

Fault별 Recall은 모든800개 post-injection 샘플을 포함한다. 주입 직후 실제 반응 전 구간을 임의로 제거하지 않았다. 진단 오류와 미탐지를 구분한다. threshold는 Test에서 변경하지 않았다. 알려진 까다로운 고장을 제거한 점수도 주 결과로 사용하지 않았다.
''')
    # Separate unknown-fault detection from supervised 20-class diagnosis.
    write('10_explainability_and_unknown_faults.md',f'''# 설명과 미지 고장 탐지

## 주 모델과 TreeSHAP

주 모델은 GradientSHAP으로, 별도 XGBoost 비교 모델은 TreeSHAP으로 설명했다. 각 고장별 Test 실행1~5, sample240/480/960의15개 관측 창을 사용했다. 시계열 설명 배경64개는 Train에서만 뽑았다. GradientSHAP은 nsamples64이므로 근사 설명이며 전체 Test 대표성을 보장하지 않는다. 알려진 실제 Fault의 class logit을 설명한 것으로, 잘못 분류된 창에서도 그 클래스에 대한 기여가 계산된다.

원 변수별 순위는 파생 통계나 시차의 절대 기여를 합산해 계산했다. 절대값 합은 시차/특징 상쇄를 제거하므로 원 logit의 가산 분해 합과 같지 않다. 모델 기여도, 관측 센서 반응, 실제 원인 변수는 서로 다른 개념이다.

{md(top[['fault','rank','variable','mean_absolute_attribution']])}

![주 모델 SHAP](../figures/phase2/primary_shap.png)

## 실제 B 센서 변화와 물리 해석 범위

정상 Train 실행400개 전체에서 센서별 기준 표준편차를 계산했다. 각 Fault의500개 Test 실행에서 주입 전 sample121~160 평균과 주입 후 sample161~960 평균 차이를 이 기준으로 표준화했다. B 반응 표는 b_fault_sensor_response.csv에 있다. 모델이 참조한 센서가 실제 변화했는지 이 표와20개 시계열 그림을 대조할 수 있다.

A 공개 카탈로그의 Fault1~15는 원료·냉각·동역학·밸브 시나리오 해석의 참고다. 다만 B의 시뮬레이터 구현·운전모드·원인 변수의 측정 열 대응이 충분히 공개되지 않아 공통 ID만으로 원인 적중률을 계산하지 않았다. 예를 들어 냉각수 출구 온도는 냉각수 입구 온도 원인과 다르며, 제어입력은 고장 원인일 수도 보상 반응일 수도 있다. Fault16~20의 공개 원인은 UNKNOWN으로 유지했다. Source가 충돌하는 A Fault21은 B에 없고 학습 대상도 아니다.

{md(physical[['fault','public_scenario_reference','observed_top3','shap_top3']])}

'''+ '\n'.join(f'- [Fault {i} SHAP 상위 센서의 실제 시계열](../figures/phase2/fault_{i:02d}_shap_response.png)' for i in range(1,21))+f'''

## 미지 고장 실험 (독립 과제)

Fault3/9/15의 모든 양성 학습 예시를 제외하고 별도 이진 XGBoost를 학습했다. 검증 임계값 결정에서도 해당 고장 양성 구간을 제외했다. 정상 예시와 나머지 고장만 학습했다. unknown detector metadata에 실제 학습 라벨 목록을 기록했고 자동 테스트로 제외를 검증했다. Test에서3/9/15를 처음 본 이상으로 평가했다.

{md(unknown[['model_id','unseen_faults_detection_recall','unseen_faults_detection_f1','unseen_faults_detection_roc_auc','seen_faults_detection_recall','sample_false_alarm_rate']])}

여기서 미지 고장 탐지는3종 고장이 정상과 구분되는지를 묻는다.20종 지도 고장 분류 정확도와 동일하지 않다. 이 특정3종 외 모든 새로운 고장으로 일반화한다고 결론낼 수 없다. recall은 고장별 신호 크기와 센서 반응 자체에도 제한된다.
''')
    bestfault=pf.sort_values('end_to_end_recall',ascending=False).head(5);weakfault=pf.sort_values('end_to_end_recall').head(5)
    doc=f'''# TEP 공정 이상 탐지 및 고장 진단 최종 연구 보고서

## 1. 연구 배경 및 목적

TEP 시뮬레이션에서 이상 탐지,20종 고장 분류 및 모델 설명을 재현 가능한 Python 프로젝트로 구현했다. Phase1 감사 이후 사용자 승인으로 Phase2를 실행했다. 모델 예측 중요도를 실제 고장 원인으로 단정하지 않는 것을 원칙으로 했다.

## 2. TEP 공정 구조

Reactor·Condenser·Separator·Stripper·Compressor, 냉각수·원료 공급 및 제어밸브가 결합된 폐루프 공정이다. 주요 측정값은 유량·온도·압력·수위·조성이며 조작값은 밸브/제어입력이다. 상세 공정/변수 사전은02 보고서에 있다.

## 3. 데이터셋 구성

A는214개 Excel/168개 고유 시계열로 공정 이해에만 사용했다. 모델은 Harvard B의1,533만 행,52개 공정 변수와3개 메타 열에서 학습했다. 공식 Training400개 RNG family 학습·100개 검증, Test500개 family 평가다. 고장 간 공유 정상 prefix가 있어 같은 run을 함께 분할했다.21,000개 전체 궤적을 독립 난수 실험21,000개로 간주하지 않았다.

## 4. 고장 유형 및 물리적 원인

Fault1~20을 예측했다. Fault16~20의 공개 원인은 UNKNOWN이다. 원료·냉각·반응·밸브 시나리오와 SHAP/관측 반응은 정성적으로 대조했으나 B 구현과 직접 측정 원인 대응이 미확인인 부분이 있어 정량 원인 적중률은 계산하지 않았다.

## 5. 데이터 전처리

Training20/Test160까지 정상, 이후 명목 고장으로 재라벨링했다. 반응 지연을 이유로 초기 고장 구간을 제거하지 않았다. 과거20개 관측(끝점 간57분)을 이용했고19개 warm-up을 명시했다. Train에만 정규화·가중치를 fit했다. 학습은 모든8400개 궤적에서 고정 간격101,200개 창을 뽑았다. 원본을 모두 읽었지만 모든 행으로 학습한 모델은 아니다.

## 6. 모델 학습 방법

Current52·Statistical312·Temporal20×52를 비교했다. Logistic Regression·Random Forest·XGBoost와 LSTM·GRU·1D-CNN을 학습했다. 대표 설정은 각3개 seed를 사용했다. 별도 이진 XGBoost에서 불균형 처리4종도 비교했다. CNN/LSTM/GRU는 GPU, 기타 모델은 CPU를 사용했다.

## 7. 실험 설계

모델 선정 기준은 Validation 탐지F1과 고장분류Macro F1 평균의3개 seed 평균이다. threshold는 검증 정상 FAR≤1% 조건에서 정했다. Test 결과를 본 뒤 모델/임계값을 바꾸지 않았다. 모든 모델은 동일997,500개 Test 시점으로 비교했고 선택한 주 모델과 별도 이진 탐지기는 매 유효 시점9,880,500개로 재평가했다.500개 RNG family 단위 cluster bootstrap으로 불확실성을 확인했다.

## 8. 모델 성능 비교

**검증 기준으로 선택한 주 모델은 {lock['primary_model']}다.** 대표 seed42 단일 모델이며3개 seed ensemble이 아니다. 입력은20×52, 구조는 Conv1d(52→64,k5)·ReLU·Conv1d(64→64,k3)·ReLU·시간 평균·현재52값 연결·Dense64·Dropout0.1·21출력이다. 학습 파라미터 {parameters:,}개, checkpoint {(M/(lock['primary_model']+'.pt')).stat().st_size/1024:.1f}KiB다.

전체 유효 Test에서 탐지 Precision {pct(a['detection_precision'])}, Recall {pct(a['detection_recall'])}, F1 {a['detection_f1']:.4f}, ROC-AUC {a['detection_roc_auc']:.4f}, PR-AUC(AP) {a['detection_pr_auc']:.4f}다. 고장 조건부 분류 Macro F1은 {a['diagnosis_macro_f1']:.4f}, 탐지 실패까지 포함한21상태 end-to-end Macro F1은 {a['end_to_end_macro_f1']:.4f}다. 정상 샘플 오경보율은 {pct(a['sample_false_alarm_rate'])}다.

{md(ci)}

별도 이진 XGBoost는 F1 {b['detection_f1']:.4f}, Recall {pct(b['detection_recall'])}, 정상 FAR {pct(b['sample_false_alarm_rate'])}다. 고장 ID 분류 기능은 없다. 두 모델을 임의로 결합한 cascade 성능은 평가하지 않았다.

{md(summary[['model','features','detection_f1_mean','diagnosis_macro_f1_mean','sample_FAR_mean']])}

## 9. 고장별 원인 분석 및 조기 탐지

고장 실행의3연속 경보 검출률은 {pct(primary_alarm.detected.mean())}다. 이는 주입 후40h 안에 한 번이라도3개 연속 경보가 나왔다는 뜻이며 올바른 Fault ID를 맞췄다는 뜻은 아니다. 검출된 실행의 지연 중앙값은 {primary_alarm.detection_delay_minutes.median():.1f}분이며 미검출 실행은 이 지연 통계에서 제외했다. 고장별 검출률·미검출률·지연을 함께 봐야 한다.

상대적으로 높은 end-to-end Recall 고장:

{md(bestfault)}

낮은 end-to-end Recall 고장:

{md(weakfault)}

SHAP은 모델이 사용한 신호의 설명이다. 관측 센서 변화는10 보고서에서 별도로 제시했다. 직접 원인 변수와 측정 열이 대응하지 않는 고장에서 물리 원인 적중을 주장하지 않았다. 미지 Fault3/9/15 제외 학습 실험도 별도의 이진 과제로 평가했다.

## 10. 연구 의의와 모델 특징

공유 초기 궤적이 있는 자료에서 실행 family 분할을 적용했고, 단순 현재값·통계·시계열 입력을 동일 시점으로 비교했다. 선택 CNN은 순환 연산 없이 과거 창의 국소 변화와 현재값을 함께 처리하는 작은 모델이다. 시차 패턴을 활용하면서도 고장 ID·실행번호를 입력에서 제외했다. 경보·진단·설명을 서로 다른 결과로 보고하여 높은 조건부 분류 성능이 전체 공정 경보 성능을 보장하지 않는 점을 확인했다.

## 11. 한계 및 향후 개선 방향

- 학습 endpoint를120분 간격으로 추출해 최초2시간 고장 과도 창이 충분히 학습되지 않았다. 현재 결과의 조기 탐지 한계에 포함된다. 다음 실험은 이 구간의 Train 창을 촘촘히 구성하되 미사용 평가 family로 재검증해야 한다.
- Training 고장주입1h·Test8h 및25/48h 운전시간 차이가 있다. Validation FAR 조건이 Test에서 그대로 보장되지 않는다.
- Test의 같은 run 정상 prefix는 여러 시나리오에서 공유된다. 표본 수가 독립 실험 수가 아니며 bootstrap은500개 family 단위다.
- SHAP은15개 창/고장의 제한된 설명 cohort와 근사 GradientSHAP에 기반한다. 원인 식별 성능으로 해석할 수 없다.
- 운전모드 변경, 센서 고장/결측, 실제 산업 설비, 반도체 공정 일반화는 검증하지 않았다. 모든 입력은 유한값을 요구한다.
- 3개 seed와 제한된 구조/고정 하이퍼파라미터를 비교했으며 Transformer나 대규모 최적화를 수행하지 않았다. GPU seed 고정은 bitwise 재현 보장이 아니다.
- Phase1에서 Test 데이터 품질/일부 기술통계를 이미 조사했다. Phase2 설정은 Train/Validation로 정했지만 완전히 비노출한 신규 외부 Test는 아니다.

학습 모델, 고정 threshold, 정규화,52열 schema와 추론 코드는 models/phase2/TEP_final_model.zip에 제공한다. 실행 방법은 README.md와 run_phase2.ps1을 따른다. 모델을 새 데이터에 적용하기 전에 해당 운전조건에서 별도 검증이 필요하다.
'''
    write('11_final_research_report.md',doc)
    card=f'''# 선택 모델 성능 및 특징

모델: **{lock['primary_model']}**,21상태 joint CNN. 별도 이진 탐지기: **{lock['binary_model']}**.

| 항목 | 주 모델 전체 Test 결과 |
| --- | --- |
| 입력 |52개 공정 변수의 과거20개 관측,3분 간격 |
| 평가 |500개 공식 Test RNG family,10,500개 궤적,유효9,880,500개 시점 |
| 탐지 Precision / Recall | {pct(a['detection_precision'])} / {pct(a['detection_recall'])} |
| 탐지 F1 | {a['detection_f1']:.4f} |
| 탐지 ROC-AUC / PR-AUC(AP) | {a['detection_roc_auc']:.4f} / {a['detection_pr_auc']:.4f} |
| 정상 샘플 FAR | {pct(a['sample_false_alarm_rate'])} |
|20종 고장 조건부 Macro F1 | {a['diagnosis_macro_f1']:.4f} |
| 탐지+분류21상태 Macro F1 | {a['end_to_end_macro_f1']:.4f} |
|3연속 경보 실행 검출률 | {pct(primary_alarm.detected.mean())} |
| 검출된 실행 지연 중앙값 | {primary_alarm.detection_delay_minutes.median():.1f}분,명목 주입 기준 |
| 파라미터 / checkpoint | {parameters:,}개 / {(M/(lock['primary_model']+'.pt')).stat().st_size/1024:.1f}KiB |
| warm-up | 최초19개 관측,57분 |

CNN은 국소 시계열 패턴과 현재값을 함께 사용한다. 작은 checkpoint는 추론용 패키지가 작다는 의미이며 Python/PyTorch 런타임 크기를 포함하지 않는다. 단일 창 CPU/GPU 지연은 별도 latency_benchmark.csv를 참조한다. 전체 dense 추론 시간은 특징 계산과 두 모델 실행을 함께 포함하므로 단일 모델 지연으로 환산하면 안 된다.

중요한 약점은 Fault9/15다. Fault9의 샘플 탐지 Recall은16.49%, 탐지와 정확한 고장 ID를 함께 맞춘 Recall은0.04%다. Fault15는 각각1.40%와0.00%다. Fault15에서 검출된 실행의 지연 중앙값도858분이었다. 따라서98.51% 실행 검출률을 조기·정확 진단률로 해석하면 안 된다. 학습에서 제외한 Fault3/9/15의 별도 미지 고장 탐지 Recall은 평균약2.60%로 낮았다.

CPU 추론과 GPU dense 평가를181개 창에서 대조한 최대 확률 차이는 약0.000552였고 해당 창의 경보 판정은100% 일치했다. 이는 전체 Test의 CPU/GPU 판정 동일성을 보장하지 않는다. 수치 정밀도·커널 차이로 임계값 근처에서는 판정이 달라질 수 있다.

약한 고장과 검출 지연의 차이가 커서 전체F1 하나로 적합성을 판단하면 안 된다.3개 seed 평균 비교와 고장별 결과는08~09 보고서, 물리 해석 범위와 미지 고장은10 보고서, 전체 연구 서술은11 보고서에 있다. 모델은 시뮬레이션 조건에 한해 검증했다. 실제 원인 식별기 또는 실제 설비 검증 모델로 표현하지 않는다.
'''
    write('12_model_card.md',card)
    research=pd.DataFrame([
        ['현재값만으로 탐지할 수 있는가?','가능하나 현재값 모델의 고장별 성능 차이가 크다.08 비교표의 Current 결과를 확인한다.'],
        ['과거 정보를 추가하면 개선되는가?','동일 Test 시점의 Statistical/Temporal과 Current 차이로 확인했다. 구조와 특징이 함께 달라진 비교에서는 순수 시차 효과로 단정하지 않는다.'],
        ['XGBoost와 LSTM 중 무엇이 적합한가?','이번 설정에서는 시계열 LSTM의 공통 Test 탐지/진단 점수가 통계 XGBoost보다 높았다. 주 모델은 Validation으로 선택한 CNN이며 계산비용도 별도 표로 제공한다.'],
        ['불균형 처리가 개선하는가?','이진 문제의 natural/balanced/over/under 결과는 imbalance_comparison.csv에 있다. seed42 한 번인 resampling을 통계적 우위로 단정하지 않는다.'],
        ['고장마다 난도가 다른가?','고장별 Recall·실행 검출률·지연이 크게 달랐다.09 보고서에 전체20개 고장을 유지했다.'],
        ['모델 설명이 실제 원인과 일치하는가?','예측 기여와 센서 변화는 대조했지만 직접 원인 대응과 B 구현이 불명확하여 정량 원인 적중은 미평가다.'],
        ['탐지와 원인 진단 성능은 다른가?','샘플 탐지F1, 고장 조건부 분류Macro F1, 탐지 실패를 포함한 end-to-end Macro F1이 서로 달랐다. 원인 적중률은 이 분류 점수가 아니다.'],
        ['제외한 고장을 새로운 이상으로 검출하는가?',f'Fault3/9/15 제외 학습에서 평균 Recall은 {unknown.unseen_faults_detection_recall.mean():.4f}였다. 이 실험은 미지 고장 탐지에 한계를 보였다.']
    ],columns=['연구 질문','이번 실험의 답'])
    write('13_research_questions.md','# 연구 질문별 결론\n\n'+md(research))
    make_notebooks()
    import importlib.metadata as im
    packages={n:im.version(n) for n in ['numpy','pandas','scikit-learn','xgboost','torch','shap','pyreadr','matplotlib','joblib']}
    (P/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':packages,'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()},indent=2),encoding='utf-8')
    distributions={d.metadata['Name']:d.version for d in im.distributions(path=[str(ROOT/'.deps')])}
    for name in ['numpy','pandas','openpyxl']:distributions[name]=im.version(name)
    (ROOT/'requirements-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(distributions.items(),key=lambda x:x[0].lower()))+'\n',encoding='utf-8')
    print('Phase 2 reports and figures complete',flush=True)

def make_notebooks():
    specs=[('06_modeling_dataset','07_phase2_experiment_design.md','../sources/phase2_split_manifest.json'),('07_baseline_training','08_model_performance.md','phase2/validation_baselines.csv'),('08_advanced_training','08_model_performance.md','phase2/validation_temporal.csv'),('09_fault_diagnosis','09_fault_diagnosis_and_delay.md','phase2/primary_dense_per_fault.csv'),('10_explainability','10_explainability_and_unknown_faults.md','phase2/shap_sensor_rankings.csv')]
    for title,report,data in specs:
        code="from pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display, Markdown, Image\nROOT=Path.cwd()\nif not (ROOT/'reports').exists(): ROOT=ROOT.parent\n"
        code+=f"display(Markdown((ROOT/'reports/markdown/{report}').read_text(encoding='utf-8')))\n"
        if data.endswith('.csv'):code+=f"display(pd.read_csv(ROOT/'reports/tables/{data}').head(70))\n"
        else:code+="display(json.loads((ROOT/'reports/sources/phase2_split_manifest.json').read_text()))\n"
        cells=[{'cell_type':'markdown','id':'intro','metadata':{},'source':[f'# {title}\n','실제 학습/평가 결과를 읽는 실행 가능한 분석 노트북. 원자료 학습 재실행은 run_phase2.ps1 참조.\n']},{'cell_type':'code','id':'results','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]
        if title.startswith(('07','08','09','10')):
            name='model_comparison.png' if title.startswith(('07','08')) else 'fault_performance_delay.png' if title.startswith('09') else 'primary_shap.png'
            cells.append({'cell_type':'code','id':'figure','metadata':{},'execution_count':None,'outputs':[],'source':[f"display(Image(filename=str(ROOT/'reports/figures/phase2/{name}')))\n"]})
        notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'nbformat':4,'nbformat_minor':5}
        (ROOT/'notebooks'/f'{title}.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':build()
