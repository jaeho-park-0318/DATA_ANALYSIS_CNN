"""실행된 4.5단계 결과만 표·그림·한국어 보고서로 정리한다."""
import json,hashlib,winreg
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.improvement_common import *

def table(frame):
    def value(x):
        if pd.isna(x):return 'NA'
        if isinstance(x,(float,np.floating)):return f'{x:.5f}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(value(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])

def collect():
    objects=[read_json(p) for p in sorted((OUT/'cv').glob('fold_*/*/result.json'))]
    rows=[];types=[];grids=[]
    for r in objects:
        for policy,m in r['evaluation'].items():
            rows.append({'fold':r['fold'],'model':r['name'],'policy':policy,'threshold':r['thresholds'][policy],**{k:v for k,v in m.items() if not isinstance(v,(dict,list))},'fit_seconds':r['resources']['seconds'],'peak_rss_gib':r['resources']['peak_rss_bytes']/2**30,'train_ap':r['training']['pr_auc_ap'],'best_iteration':r['best_iteration']})
            for c,v in m['per_type'].items():types.append({'fold':r['fold'],'model':r['name'],'policy':policy,'fault_code':c,**v})
        for g in r['calibration_grid']:grids.append({'fold':r['fold'],'model':r['name'],**{k:v for k,v in g.items() if not isinstance(v,(dict,list))}})
    return objects,pd.DataFrame(rows),pd.DataFrame(types),pd.DataFrame(grids)

def plots(metrics,na,grids,hardware):
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'figure.dpi':130})
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    def save(name,fig):fig.tight_layout();fig.savefig(directory/f'{name}.png',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));d=hardware['median_seconds'];ax.bar(list(d),list(d.values()),label='3회 중앙값')
    ax.set(title='CPU/CUDA 실제 학습 시간: 15만 행·83특징·100회',ylabel='학습 시간(초)',xlabel='학습 장치');ax.legend();save('hardware',fig)
    reasons=na[(na.fault_code=='any')&(na.reason!='known')].groupby('reason').unknown.sum()
    names={'at_fault_timestamp':'고장 기록 시각','gap_or_unverified':'공백 또는 검증 불가','one_or_more_type_history_ended':'유형별 후속 고장 기록 종료'}
    fig,ax=plt.subplots(figsize=(8,4));ax.barh([names.get(k,k) for k in reasons.index],reasons.values/1e6,label='전체 정규화 시각')
    ax.set(title='통합 600 라벨 NA 원인: 우선순위로 배타적 분류',xlabel='관측 수(백만 행)',ylabel='NA 사유');ax.legend();save('na_reasons',fig)
    tools=na[na.fault_code=='any'].groupby('Tool')[['rows','unknown']].sum()
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(tools.index,tools.unknown/tools.rows,label='통합 600 NA 비율');ax.tick_params(axis='x',rotation=60)
    ax.set(title='장비별 정답 미확인 비율: 전체 정규화 시각',ylabel='NA 비율',xlabel='장비');ax.legend();save('na_tools',fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key,title in zip(axes,['pr_auc_ap','f1','event_recall'],['PR-AUC(AP)','F1','Event Recall']):
        for policy,offset in [('f1',-.12),('event_far',.12)]:
            g=metrics[metrics.policy==policy].groupby('model')[key].agg(['mean','std'])
            ax.errorbar(np.arange(len(g))+offset,g['mean'],yerr=g['std'],fmt='o',capsize=3,label=policy)
            ax.set_xticks(range(len(g)),g.index,rotation=25)
        ax.set(title=title,xlabel='특징 구성',ylabel='Fold 평균 ± 표준편차');ax.legend()
    fig.suptitle('개발 장비 12대·외부 4-Fold: 기존 평가 장비 제외');save('cv_comparison',fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for model,g in grids.groupby('model'):
        g=g[g.fold==0].sort_values('threshold')
        axes[0].plot(g.threshold,g.event_recall,marker='.',label=model)
        axes[1].plot(g.threshold,g.false_alarms_per_active_hour,marker='.',label=model)
    axes[0].set(ylabel='Event Recall',title='이벤트 탐지');axes[1].set(ylabel='확인된 가동 음성 시간당 경보',title='조건부 오경보');axes[1].set_yscale('symlog',linthresh=.1)
    from matplotlib.ticker import FuncFormatter
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
    for ax in axes:ax.set_xlabel('경보 임계값');ax.legend()
    fig.suptitle('Fold 0 보정 장비에서만 계산한 임계값 민감도');save('thresholds',fig)
    diag=read_json(OUT/'diagnostics.json');f=pd.DataFrame(diag['old_tool_xgb_fault_scores'])
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(f.fault_code,f.score_max,label='고장 이전 최대 점수');ax.axhline(diag['old_threshold'],color='red',label='기존 검증 임계값')
    ax.set(title='기존 장비 평가 XGB_A: 유형별 미탐지 원인 진단',xlabel='고장 유형',ylabel='모델 점수(확률 보정 미실시)');ax.legend();save('old_fault_scores',fig)
    distribution=pd.read_parquet(OUT/'tool_state_distribution.parquet');sensors=[c for c in distribution if c.endswith('_median')]
    d=distribution[distribution.state=='Stage>0'].set_index('Tool')[sensors]
    scale=d.std().replace(0,np.nan);z=(d-d.mean())/scale
    fig,ax=plt.subplots(figsize=(10,6));im=ax.imshow(z,cmap='coolwarm',aspect='auto',vmin=-3,vmax=3)
    ax.set_xticks(range(len(sensors)),[s.replace('__current_median','') for s in sensors],rotation=35,ha='right');ax.set_yticks(range(len(z)),z.index)
    ax.set(title='Stage>0 센서 중앙값의 장비 간 차이: 표시용 표준화',xlabel='센서(물리 단위 익명화)',ylabel='장비');fig.colorbar(im,ax=ax,label='장비 중앙값의 표준화 차이');save('tool_shift',fig)

def build():
    objects,m,t,g=collect();assert len(objects)==16,'완료되지 않은 실험이 있음'
    m.to_parquet(OUT/'cv_metrics.parquet',index=False);t.to_parquet(OUT/'cv_fault_metrics.parquet',index=False);g.to_parquet(OUT/'threshold_metrics.parquet',index=False)
    keys=['precision','recall','f1','pr_auc_ap','event_recall','false_alarms_per_active_hour','lead_time_median_raw']
    summary=m.groupby(['model','policy'])[keys].agg(['mean','std']).reset_index();summary.columns=['_'.join(filter(None,c)) for c in summary.columns]
    summary.to_parquet(OUT/'cv_summary.parquet',index=False)
    na=pd.read_parquet(OUT/'na_audit.parquet');hardware=read_json(OUT/'hardware.json');diag=read_json(OUT/'diagnostics.json')
    plots(m,na,g,hardware)
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    trace=read_json(OUT/'resource_trace.json');gpu_memory=[float(x['gpu'].split(',')[0]) for x in trace if x['gpu']]
    save_json(OUT/'hardware_supplement.json',{'cpu_name':cpu,'peak_system_gpu_mib':max(gpu_memory),'min_available_ram_gib':min(x['available_ram'] for x in trace)/2**30,'note':'GPU 사용량은 다른 화면·프로세스를 포함한 장치 전체 값이다. CPU 모델명 조회 CIM은 권한 제한으로 실패했고 레지스트리 읽기로 확인했다.'})
    hw=f'''# 4.5단계 로컬 하드웨어 벤치마크

분석 실행 파일은 `{hardware['python']}`이며 Python 3.11, Conda `phm2018`이다. 로컬 Windows 프로세스·파일과 NVIDIA 장치를 직접 사용했다. Codex 화면의 Local 선택 상태 자체는 API로 확인하지 않았으므로 UI 표시까지 검증했다고 주장하지 않는다. 실제 계산은 로컬이다.

- CPU: {cpu}, 물리 {hardware['physical_cores']}코어 / 논리 {hardware['logical_cores']}스레드.
- RAM: {hardware['memory']['total']/2**30:.2f}GiB, 최초 가용 {hardware['memory']['available']/2**30:.2f}GiB.
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, VRAM 6144MiB, Driver 546.92, nvidia-smi CUDA 12.3.
- XGBoost {hardware['xgboost']}, CUDA 지원 빌드 12.9. `device="cuda", tree_method="hist"` 실제 성공. Booster `cuda:0`, CUDA 경고 없음.
- 같은 150,000행·83개 float32 특징, 100 boosting round, depth=4, max_bin=128, 4 CPU 스레드, seed=42. 5회 워밍업 트리 이후 각 장치 3회, Early Stopping 없음. 데이터 로딩은 시간에서 제외하고 fit의 호스트→GPU 전송은 포함한다. CPU 먼저 실행한 순서 및 열 상태 영향은 남는다.

{table(pd.DataFrame([{'device':r['device'],'repeat':r['repeat'],'seconds':r['seconds'],'peak_process_rss_gib':r['peak_rss_bytes']/2**30} for r in hardware['runs']]))}

중앙값 CPU {hardware['median_seconds']['cpu']:.4f}초, GPU {hardware['median_seconds']['cuda']:.4f}초. CPU/GPU 시간 비율 {hardware['speedup_cpu_over_gpu']:.3f}배다. 따라서 이번 학습은 GPU를 선택했다. 작은 단일 벤치마크이므로 모든 작업의 동일 배속을 보장하지 않는다. CPU/GPU 예측 성능의 동등성을 시험한 결과가 아니다.

교차검증 16개 모두 실제 학습 장치: {sorted(set(r['actual_device'] for r in objects))}. CUDA 실패 후 CPU 전환 {sum(r['fallback'] is not None for r in objects)}건. 최대 프로세스 RSS {m.peak_rss_gib.max():.2f}GiB. 실행 중 전체 GPU 메모리 최대 {max(gpu_memory):.0f}MiB이며 다른 데스크톱 앱 사용량도 포함된다. 가용 RAM 최저 {min(x['available_ram'] for x in trace)/2**30:.2f}GiB. 자원 감시는 0.5~1초 주기로 샘플링하므로 순간 피크를 놓칠 수 있다.

NumPy 호스트 행렬의 추론은 CPU로 명시하여 반복 전송·디바이스 불일치 경고를 피했다. 학습 장치는 저장된 Booster 설정으로 별도 검증했다. 모델별 fit 시간/RSS, 전체 작업 경과시간, GPU 사용률·온도는 `outputs/stage4_5/` JSON에 있다. 모델은 순차 실행하고 DuckDB/XGBoost 스레드는 4개로 제한했다. 데이터 캐시 생성과 지표 계산 시간이 순수 모델 학습보다 크다.

Jupyter는 작업 폴더 전용 커널 사양에서 같은 절대 Python 경로를 사용한다. 기본 등록된 python3 커널의 argv는 `python`이므로 기본 이름만으로 환경을 단정하지 않았다. 드라이버·Conda 패키지는 업데이트하지 않았다.

버전 표기가 달라도 CUDA 12 계열의 minor compatibility가 가능한 조건이 있다. 실제 실행이 성공했으므로 버전 차이만으로 비호환이라고 진단하지 않는다. [NVIDIA 호환성 설명](https://docs.nvidia.com/deploy/cuda-compatibility/610/minor-version-compatibility.html), [CUDA 12.9 릴리스 노트](https://docs.nvidia.com/cuda/archive/12.9.1/pdf/CUDA_Toolkit_Release_Notes.pdf). 처음 CPU 이름을 읽는 하위 프로세스에서 문자 디코딩 오류가 있었으나 CUDA 학습과는 무관하며 레지스트리 조회로 보완했다.
'''
    (ROOT/'reports/04_5_hardware_benchmark.md').write_text(hw,encoding='utf-8')
    counts=pd.read_parquet(OUT/'cache_counts.parquet');reason=na[na.fault_code=='any'].groupby('reason')[['rows','unknown']].sum().reset_index()
    stage=na[na.fault_code=='any'].groupby('state')[['rows','positive','negative','unknown']].sum().reset_index();stage['na_rate']=stage.unknown/stage.rows
    toolna=na[na.fault_code=='any'].groupby('Tool')[['rows','positive','negative','unknown']].sum().reset_index();toolna['na_rate']=toolna.unknown/toolna.rows
    types=na.groupby('fault_code')[['rows','positive','negative','unknown']].sum().reset_index();types['na_rate']=types.unknown/types.rows
    foldcounts=pd.concat([pd.DataFrame(read_json(OUT/f'fold_{i}_counts.json')).assign(fold=i) for i in range(4)],ignore_index=True)
    splitcounts=foldcounts.groupby(['fold','role'])[['rows','positive','negative','unknown','f1','f2','f3']].sum().reset_index()
    old=[]
    for scheme in ['split_tool','split_time']:
        for name in ['A','B','C']:
            r=read_json(ROOT/f'outputs/stage4/experiments/{scheme}_h600/XGB_{name}/result.json')
            old.append({'scheme':scheme,'features':name,'validation_ap':r['metrics']['validation']['pr_auc_ap'],'evaluation_ap':r['metrics']['evaluation']['pr_auc_ap']})
    main=summary[['model','policy']+[f'{k}_{s}' for k in ['precision','recall','f1','pr_auc_ap','event_recall','false_alarms_per_active_hour'] for s in ['mean','std']]]
    paired=[]
    for key in ['pr_auc_ap','f1','event_recall','false_alarms_per_active_hour']:
        wide=m[m.policy=='f1'].pivot(index='fold',columns='model',values=key)
        for left,right in [('B','A'),('C','B'),('C_regularized','C')]:
            delta=wide[left]-wide[right]
            paired.append({'metric':key,'comparison':left+' - '+right,'mean_difference':delta.mean(),'std_difference':delta.std(),'positive_difference_folds':int((delta>0).sum())})
    contrast=pd.read_parquet(OUT/'fault_sensor_contrasts.parquet')
    contrast=contrast[contrast.Tool.str[:2].astype(int)<=6].groupby(['fault_code','sensor']).median_difference_over_negative_iqr.agg(['median','count']).reset_index()
    importance=[]
    for r in objects:
        if r['name']=='C':
            for c in configs()['C'][-3:]:importance.append({'fold':r['fold'],'feature':c,'gain_fraction':r['importance'][c]})
    report=f'''# 4.5단계 데이터 신뢰성·장비 일반화·경보 개선 결과

## 실행 범위와 결론

로컬 phm2018에서 CUDA 파일럿 1개와 본 실험 XGBoost 16개(4-Fold × A/B/C/C_regularized)를 실행했다. 각 본 모델에서 F1 최대화와 Event Recall/FAR 제약의 두 경보 정책을 비교했다. 기존 4단계 모델·데이터·Notebook은 보존했고 AGENTS.md만 요청된 실행 정책을 추가했다. LSTM, SHAP, 패키지·드라이버 업데이트, 외부 업로드는 하지 않았다.

**미확인 라벨을 정상으로 바꾸는 개선은 수행하지 않았다. 일반화 성능의 확실한 개선을 입증하지 못했으며 현장 배포와 LSTM 확대를 권고하지 않는다.** 이번 결과는 기존 개발 장비의 후향적 그룹 교차검증이다. 이미 수행한 시간 분할 평가와 장비 일부가 겹치므로 완전히 새로운 미관측 데이터 성능이 아니다. 기존 장비 평가 09/10 계열은 새 모델 학습·튜닝·선택에서 제외했다.

## 1. 하드웨어 및 실행 효율

{cpu}, {hardware['physical_cores']}코어/{hardware['logical_cores']}스레드, RAM {hardware['memory']['total']/2**30:.2f}GiB, RTX 3060 Laptop 6GB. 실제 CUDA 학습 성공 및 Booster cuda:0 확인. 동일 조건 CPU {hardware['median_seconds']['cpu']:.4f}초 / GPU {hardware['median_seconds']['cuda']:.4f}초, 약 {hardware['speedup_cpu_over_gpu']:.2f}배. 속도 비교와 예측 성능 비교는 별개다. 상세는 `04_5_hardware_benchmark.md`에 있다.

## 2. NA 원인 및 라벨 정책

전체 정규화 시각 82,071,907행을 DuckDB의 필요한 컬럼 집계로 다시 확인했다. 아래 NA 사유는 우선순위로 배타적 분류했으므로 후속 이벤트 부재와 공백이 동시에 있는 경우 전자를 우선한다. 유형별 상세 `na_audit.parquet`에는 공백 전 고장(gap_before_event), 전체 horizon 미확보(incomplete_horizon), 후속 유형 이벤트 부재 등이 분리되어 있다.

{table(reason)}

{table(types)}

{table(stage)}

{table(toolna)}

TTF Null은 제공된 기록 안에서 이후 같은 유형의 이벤트가 없다는 사실과 부합한다. 장비별 처음/마지막 최대 4096 시각을 별도로 대조한 불일치는 0건이며 전체 원본 대조는 기존 3단계 검증을 재사용했다. 이것이 미래에 고장이 없다는 증명은 아니다. `[NoWaferID]` 제외, 중복 이벤트 제거, 고장 시각 NA, 공백을 넘는 라벨 보류를 유지했다. 기존 ml_v1 라벨 변경 0행, ml_v2 미생성. 라벨을 완화할 외부 완전성 근거가 없어 새 버전 생성이 불필요했다.

개선한 것은 **평가 범위의 투명성**이다. 12개 타깃 중 하나가 알려졌다는 기존 training 파일 포함 조건을 추론에서는 제거하고, 모든 feature_ready 2,331,977개 anchor를 float32 Parquet 캐시로 보존했다. NA도 경보량에는 포함하고 학습과 샘플 성능에는 제외했다. 알려진 타깃은 기존과 동일하며 표본의 시간 순서를 유지했다. 고장 공백을 해소하거나 새로운 정상 정답을 확보한 것은 아니다.

{table(counts)}

확인 음성이 없는 장비 {int((counts.negative==0).sum())}대: {', '.join(counts.loc[counts.negative==0,'Tool'])}. 분할별 클래스 비율과 장비 정체성이 얽혀 있다. TTF가 없는 후반부·공백 주변은 평가에서 과소대표되어 낮은 FAR를 전체 운영 FAR로 해석할 수 없다.

## 3. 그룹 교차검증 설계와 분할

01~06 계열 12대만 사용, 장비명으로 사전 고정한 4-Fold이며 각 외부 평가에 3대씩 한 번 등장한다. 나머지 9대 중 5대 학습, 2대 Early Stopping, 2대 임계값 보정으로 구분한다. A/B/C와 정규화 강화 C를 사전 지정하여 모두 보고하며 외부 Fold 성능으로 모델을 다시 튜닝하지 않았다. Fold 상세 장비 목록은 `fold_plan.json`이다. 장비 그룹이 완전히 분리되어 동일 이벤트, 과거 특징 구간, 중첩 윈도가 역할을 넘지 않는다. 대치·스케일링·오버샘플링 없음. 원본 특징의 source_max_time≤time를 검사하고 83개 허용 컬럼만 입력한다.

{table(splitcounts)}

고장 수 f1/f2/f3은 해당 유형의 검증된 양성 anchor를 가진 고유 이벤트 수다. 고장 표에 기록된 전체 수와 다르다. Fold별 희소 이벤트와 음성 부족은 재분할로 숨기지 않았다. 이 단일 고정 4-Fold는 반복 그룹 CV나 완전한 nested 하이퍼파라미터 탐색이 아니다.

## 4. 특징 및 모델 설정

A 현재 센서·공정 8개, B 과거 센서 통계 포함 80개, C 공정 이력 포함 83개. C_regularized는 같은 83개에 depth 3, min_child_weight 30, reg_lambda 20을 적용한 사전 지정 후보다. 기본 후보는 기존 depth 4, min_child_weight 10, lambda 5를 유지했다. 500회 상한, Early Stopping 30, learning_rate .05, 학습 음성/양성 비율 scale_pos_weight, seed 42, hist/max_bin 128. 학습과 CPU 호스트 행렬 추론 장치를 구분했다. 모든 실험별 입력 컬럼·설정·학습시간·경고·Booster 설정을 저장했다.

입력 구성 전체:

```json
{json.dumps(configs(),ensure_ascii=False,indent=2)}
```

TTF/라벨/next_event/Tool/time/Lot/사용량은 입력에서 제외한다. 추론 캐시에 포함된 정답 메타데이터는 평가에만 쓴다. 센서 물리 단위는 익명화 상태이고 time의 초 해석은 4단계 근거를 유지한다.

## 5. Fold별 성능 및 변동성

F1 정책은 보정 장비의 F1 최대화다. event_far 정책은 보정 장비에서 확인된 가동 음성 구간 FAR≤0.1/h를 만족하는 후보 중 Event Recall 최대화, 동률 시 미확인 경보가 적고 Precision/임계값이 큰 순서다. 0.1/h는 사전 지정한 실험 제약이며 사용자가 확정한 운영 허용치가 아니다. 이 제약은 외부 장비의 FAR를 보장하지 않는다.

{table(main)}

표는 4개 Fold의 단순 평균과 표본 표준편차다. 정의 불가능한 Precision 등은 NA를 유지하며 평균에서 제외한다. 원자료 `cv_metrics.parquet`에서 Fold별 유효 지표 수를 확인할 수 있다. PR-AUC는 average precision이다.

{table(m.groupby(['model','policy'])[keys].count().reset_index())}

{table(m[['fold','model','policy','n','positive','negative','precision','recall','f1','pr_auc_ap','eligible_events','detected_events','event_recall','false_alarms_per_active_hour','lead_time_median_raw','unknown_episodes']])}

유형별 이벤트 탐지:

{table(t.groupby(['model','policy','fault_code'])[['eligible','detected']].sum().reset_index())}

연속 경보는 같은 장비·관측 세그먼트·가동 상태에서 합치며, 72초 초과 공백·음성 예측·상태 변화 때 끊고 600초마다 재발행한다. 한 경보는 가장 이른 미래 이벤트 시각에만 매칭한다. 같은 시각의 다중 유형은 함께 기록하고 순차 이벤트 중복 공로를 막는다. 정답을 예측 특징에 사용하지 않고 점수 생성 후 매칭한다. Lead Time은 탐지 이벤트에만 정의한다. NA가 포함된 미매칭 경보는 false로 세지 않고 unknown으로 분리한다. FAR 분모는 같은 세그먼트의 인접 확인 음성 anchor(간격≤72)의 Stage>0 시간이다. **전체 운영 시간당 실제 FAR가 아니다.**

## 6. A/B/C 차이와 과적합

기존 4단계의 검증·평가 AP:

{table(pd.DataFrame(old))}

시간 분할에서 A의 평가 AP가 더 높았지만 당시 선택 기준은 검증 AP였으므로 C 선택 자체가 오류는 아니다. 검증/평가의 장비·시간·공정 분포 및 클래스 비율이 달라 순위가 뒤바뀐 관찰이다. 원인을 하나로 확정할 수 없다. 이번 장비 분리 CV에서 B−A는 과거 센서 특징의 효과, C−B는 공정 이력 3개의 추가 효과를 같은 Fold에서 비교한다. 모델별 학습 AP와 외부 AP 차이, Fold 변동성이 크면 과적합/분포 이동 가능성을 함께 본다. 단일 seed와 서로 다른 구성의 early stopping 결과이므로 인과적 특징 효과로 단정하지 않는다.

실제 평균 AP는 A 0.01226, B 0.00933, C 0.00949로 과거 특징 추가가 AP를 높이지 못했다. 반면 F1 정책의 평균 F1은 A 0.00488에서 B 0.02193, C 0.02220으로 달라져 순위 지표와 경보 지표의 목적이 다름을 확인했다. 학습 AP는 A 0.16776/B 0.17396/C 0.20082로 외부 AP보다 훨씬 높다. 정규화 강화 C도 AP 0.00586으로 일반화를 개선하지 못했다.

{table(pd.DataFrame(paired))}

C의 공정 이력 특징 gain 비율(모델 내 상대값, 인과 중요도 아님):

{table(pd.DataFrame(importance))}

{table(m[m.policy=='f1'][['fold','model','train_ap','pr_auc_ap','best_iteration','fit_seconds','peak_rss_gib']])}

Stage=0에서도 고장이 존재하므로 제거하지 않았다. `tool_state_distribution.parquet`은 장비·Stage별 클래스와 센서 중앙값을, `fault_sensor_contrasts.parquet`은 유형별 양성/음성 센서 중앙값 및 IQR 대비 차이를 제공한다. 이는 물리 단위 복원이나 공정 교란 제거가 아닌 기술 통계다. 장비·공정 중앙값 차이 그림은 분포 이동을 보여주지만 고장의 원인이라고 단정하지 않는다. 모델별 gain 중요도는 result.json에 저장했고 SHAP 상세 분석은 수행하지 않았다.

개발 12대의 유형별 `(양성 중앙값 - 음성 중앙값)/음성 IQR`을 장비 간 중앙값으로 집계하면 f1은 ETCHBEAMVOLTAGE 약 +0.492, ETCHBEAMCURRENT +0.451, f2는 FLOWCOOLFLOWRATE -0.503, f3는 IONGAUGEPRESSURE -0.839가 상대적으로 컸다. 장비에 두 클래스가 없거나 음성 IQR이 0이면 제외하므로 유효 장비 수를 함께 제시한다. 특히 f1의 FLOWCOOLPRESSURE 차이는 -0.011로 작았지만 이것만으로 전조 부재를 결론짓지 않는다. Recipe/Stage 구성과 센서 상호작용을 통제하지 않았기 때문이다.

{table(contrast)}

## 7. 압력 저하 30건 미탐지 및 임계값 진단

{table(pd.DataFrame(diag['old_tool_xgb_fault_scores']))}

기존 XGB_A의 f1 양성 242개 시점/30개 이벤트에서 최대 점수는 0.62040이고 검증 임계값은 0.65273이므로 경보가 0개였다. 누출 f3 역시 임계값 아래다. 이는 직접 확인한 임계값 미통과 원인이며, 센서에 전조가 없다는 증명은 아니다. 기존 평가 점수에 맞춰 임계값을 낮추거나 재평가하여 개선 성능으로 제시하지 않았다. 기존 검증 장비 07/08 계열의 임계값별 Precision/Recall/F1/AP/Event Recall/FAR/Lead Time/unknown 경보는 `old_validation_threshold_grid.json`에만 기록했다. 새 Fold별 보정 결과는 `threshold_metrics.parquet`에 있다.

## 8. 기존 대비 개선 여부 및 다음 단계

기존 장비 XGB AP 0.08518/F1 0.13008/이벤트 4/45, 시간 XGB AP 0.27880/F1 0.26476/17/56은 그대로 보존했다. 새 그룹 CV는 장비·학습량·분할이 다르므로 수치의 직접 증감을 독립 평가 개선으로 주장하지 않는다. 완료한 개선은 CUDA 속도 검증, NA 범위 공개, 별도 Early Stopping/보정 장비, 전체 준비 anchor의 미확인 경보 추적, 재개 가능한 캐시와 자원 기록이다. 확실한 예측 성능 향상은 아직 미입증이다.

우선 고장 이력 완전성·관측 공백·TTF 종료의 외부 근거를 확보하고 장비별 검증 음성 부족을 해결해야 한다. 그다음 공정 상태별 보정 및 유형별 목표, 운영 FAR/Lead Time 요구사항을 사전 합의하고 새로운 장비/기간에서 고정 모델을 평가하는 순서가 적절하다. **현재는 LSTM 학습을 권고하지 않는다.** 라벨 범위 문제와 장비 일반화 불안정성을 먼저 해결해야 하며 복잡한 시계열 모델이 불완전한 정답을 보완하지 못한다.

## 9. 재현·실행 상태

`python -m src.hardware_benchmark`, `python -m src.dataset_audit45`, `python -m src.ttf_audit45`, `python -m src.group_validation45 --pilot`, `python scripts/run_improvement45.py`, `python -m src.improvement_diagnostics`, `python -m src.improvement_report`, `python scripts/build_improvement_notebooks.py` 순서로 같은 phm2018 절대 실행 파일을 사용한다. 캐시/완료 실험은 재사용한다. 새 정책·코드로 변경할 경우 기존 경로 대신 새 버전 디렉터리를 사용해야 한다.

생성 Notebook은 09_dataset_audit, 10_group_cross_validation, 11_threshold_analysis이며 학습 재실행 대신 저장된 실제 결과를 읽어 표·그림·한국어 해석을 재현한다. Notebook 실행 상태와 보존·테스트 검증은 `delivery_checks.json`에 기록한다. 반복 seed·완전 nested 탐색·새 독립 평가·LSTM은 미실행이다.
'''
    (ROOT/'reports/04_5_model_improvement_summary.md').write_text(report,encoding='utf-8')
    save_json(OUT/'summary.json',{'experiments':len(objects),'folds':4,'labels_changed':0,'summary':summary.to_dict('records'),'gpu_all':all(r['actual_device'].startswith('cuda') for r in objects)})
    progress('4.5단계 보고서와 7개 그래프 생성 완료')

if __name__=='__main__':build()
