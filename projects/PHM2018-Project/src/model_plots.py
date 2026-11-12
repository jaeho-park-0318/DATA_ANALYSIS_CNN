"""실제 평가 점수에서 한국어 비교 그래프를 생성한다. 예시는 시간 순서로 선정한다."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from src.model_common import OUT,read_json,save_json
from src.evaluate_models import episodes
from src.dataset_common import event_table


def setup():
    plt.rcParams.update({"font.family":"Malgun Gothic","axes.unicode_minus":False,"font.size":10,"figure.dpi":110})


def save(fig,name):
    folder=OUT/"figures";folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/(name+".png"),dpi=130,bbox_inches="tight");plt.show()


def results(scheme,h=600):
    names=["Dummy","Pressure","RF_A","RF_B","RF_C","XGB_A","XGB_B","XGB_C"]
    return [read_json(OUT/"experiments"/f"{scheme}_h{h}"/n/"result.json") for n in names]


def comparison(scheme,h=600):
    rows=results(scheme,h);names=[r['name'] for r in rows]
    fig,ax=plt.subplots(figsize=(14,6),layout="constrained");x=np.arange(len(rows))
    for j,(key,label) in enumerate((('precision','정밀도 Precision'),('recall','재현율 Recall'),('f1','F1-score'))):
        values=[r['metrics']['evaluation'][key] if r['metrics']['evaluation'][key] is not None else np.nan for r in rows]
        ax.bar(x+(j-1)*.25,values,width=.25,label=label)
    ax.set_xticks(x,names,rotation=30);ax.set_ylim(0,1)
    n=rows[0]['metrics']['evaluation']['n']
    ax.set(title=f"{scheme} / h{h}raw / 정답 확인 평가 {n:,}행\n검증 데이터에서 고정한 임계값; 정의 불가 값은 빈 막대",xlabel="모델 / 입력 구성",ylabel="점수 (0~1)")
    ax.legend();ax.grid(axis='y',alpha=.2);save(fig,f"{scheme}_metrics")


def pr_curves(scheme,h=600):
    rows=results(scheme,h);fig,ax=plt.subplots(figsize=(11,7),layout="constrained")
    for r in rows:
        f=pd.read_parquet(OUT/"experiments"/f"{scheme}_h{h}"/r['name']/"evaluation_predictions.parquet")
        y=f[f'label_any_h{h}raw'].to_numpy(dtype=int)
        p,recall,_=precision_recall_curve(y,f.score)
        ax.plot(recall,p,label=f"{r['name']} AP={r['metrics']['evaluation']['pr_auc_ap']:.3f}")
    ax.axhline(y.mean(),ls='--',color='gray',label=f"양성 비율 {y.mean():.4f}")
    ax.set(title=f"{scheme} / h{h}raw / 정답 확인 평가 세트 PR 곡선",xlabel="재현율 Recall",ylabel="정밀도 Precision",xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=9);ax.grid(alpha=.2);save(fig,f"{scheme}_pr_curves")


def confusion(scheme,h=600):
    rows=results(scheme,h);fig,axes=plt.subplots(2,4,figsize=(16,8),layout="constrained")
    largest=max(max(r['metrics']['evaluation'][k] for k in ('tn','fp','fn','tp')) for r in rows)
    for ax,r in zip(axes.flat,rows):
        m=r['metrics']['evaluation'];c=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
        im=ax.imshow(np.log1p(c),cmap='Blues',vmin=0,vmax=np.log1p(largest))
        for i in range(2):
            for j in range(2):ax.text(j,i,f"{c[i,j]:,}",ha='center',va='center',color='white' if np.log1p(c[i,j])>np.log1p(c.max())*.65 else 'black')
        ax.set(title=f"{r['name']} / 임계값 {r['threshold']:.3g}",xlabel="예측 라벨",ylabel="실제 라벨",xticks=[0,1],yticks=[0,1])
    fig.suptitle(f"{scheme} / h{h}raw / 정답 확인 평가 혼동행렬 — 숫자는 행 수, 색은 log(1+행 수)")
    fig.colorbar(im,ax=list(axes.flat),label="색 범례: log(1+관측 수)",shrink=.8)
    save(fig,f"{scheme}_confusion")


def score_histograms(scheme,h=600):
    rows=[r for r in results(scheme,h) if r['name'].startswith(('RF_','XGB_'))]
    fig,axes=plt.subplots(3,2,figsize=(14,13),layout="constrained")
    for ax,r in zip(axes.flat,rows):
        f=pd.read_parquet(OUT/"experiments"/f"{scheme}_h{h}"/r['name']/"evaluation_predictions.parquet")
        for y,label in ((0,'음성'),(1,'양성')):
            s=f.loc[f[f'label_any_h{h}raw']==y,'score']
            if len(s):ax.hist(s,bins=np.linspace(0,1,31),density=True,alpha=.5,label=f"{label} n={len(s):,}")
        ax.axvline(r['threshold'],ls='--',color='black',label='검증 임계값')
        ax.set(title=r['name'],xlabel="미보정 예측 확률/점수",ylabel="클래스별 확률 밀도",yscale='log');ax.legend(fontsize=8)
    fig.suptitle(f"{scheme} / h{h}raw / 정답 확인 평가 — 불균형 가중치 사용, 확률 보정 미실행")
    save(fig,f"{scheme}_score_distribution")


def importance(scheme,h=600):
    rows=[r for r in results(scheme,h) if r['name'].startswith(('RF_','XGB_'))]
    fig,axes=plt.subplots(3,2,figsize=(16,16),layout="constrained")
    for ax,r in zip(axes.flat,rows):
        path=OUT/"experiments"/f"{scheme}_h{h}"/r['name']/"importance.json"
        s=pd.Series(read_json(path)).nlargest(12).sort_values()
        ax.barh([c.replace('__','\n') for c in s.index],s.values,label='학습된 내장 중요도')
        ax.set(title=r['name'],xlabel="상대 중요도 (모델별 척도)",ylabel="입력 특징");ax.tick_params(axis='y',labelsize=8);ax.legend(fontsize=8)
    fig.suptitle(f"{scheme} / h{h}raw 학습 모델 중요도 — RF 불순도 감소 / XGBoost gain\n척도가 다르고 상관·범주 수 편향이 있으므로 인과 설명으로 해석하지 않음")
    save(fig,f"{scheme}_importance")


def case_plots(scheme):
    result=read_json(OUT/"prospective"/scheme/"result.json")
    frame=pd.read_parquet(OUT/"prospective"/scheme/"predictions.parquet")
    frame=episodes(frame,result['threshold'],600)
    events=event_table();events=events[~events.tagged].sort_values(['time','Tool'])
    selected=None;description={'scheme':scheme,'model':result['selected_name'],'threshold':result['threshold']}
    for e in events.itertuples():
        part=frame[(frame.Tool==e.Tool)&(frame.time>=e.time-1800)&(frame.time<e.time)]
        if len(part)>=3 and (part[f'label_{e.fault_code}_h600raw']==1).any():selected=(e,part);break
    fig,ax=plt.subplots(figsize=(13,5),layout="constrained")
    if selected:
        e,part=selected;y=part.score.copy();y.loc[part.time.diff()>72]=np.nan
        ax.plot(part.time-e.time,y,marker='.',label='실제 과거 시점의 예측 점수')
        ax.axvline(0,color='crimson',ls='--',label='기록된 정지 시각')
        ax.axhline(result['threshold'],ls='--',color='black',label='검증 임계값')
        ax.set(title=f"{scheme} 평가 / {result['selected_name']} / {e.Tool} {e.fault_code}\n관측 조건을 만족하는 시간상 첫 이벤트; 점수로 사례를 고르지 않음",xlabel="고장 시각 대비 원시시간 (초 해석 지지)",ylabel="미보정 예측 점수",ylim=(-.02,1.02))
        last=part[part.time>=e.time-600]
        description.update({'event_Tool':e.Tool,'event_code':e.fault_code,'event_time':float(e.time),'points':len(part),
                            'score_min':float(part.score.min()),'score_max':float(part.score.max()),
                            'score_first':float(part.score.iloc[0]),'score_last':float(part.score.iloc[-1]),
                            'last_horizon_max':float(last.score.max()),'threshold_crossed_last_horizon':bool((last.score>=result['threshold']).any())})
    else:ax.text(.5,.5,'관측 조건을 만족하는 이벤트 사례 없음',ha='center')
    ax.legend();ax.grid(alpha=.2);save(fig,f"{scheme}_pre_fault_scores")
    false=None
    for ep,g in frame[frame.alarm].groupby('episode_id',sort=False):
        if g.label_any_h600raw.notna().all() and (g.label_any_h600raw==0).all():false=g;break
    fig,ax=plt.subplots(figsize=(13,5),layout="constrained")
    if false is not None:
        t=float(false.time.min());tool=false.Tool.iloc[0]
        part=frame[(frame.Tool==tool)&(frame.time>=t-600)&(frame.time<=false.time.max()+600)]
        y=part.score.copy();y.loc[part.time.diff()>72]=np.nan
        ax.plot(part.time-t,y,marker='.',label='실제 예측 점수')
        ax.axhline(result['threshold'],ls='--',color='black',label='검증 임계값')
        ax.axvspan(0,float(false.time.max()-t),color='orange',alpha=.25,label='확인된 음성 구간의 오경보')
        ax.set(title=f"{scheme} 평가 / {result['selected_name']} / {tool}\n시간상 첫 확인 가능한 오경보 사례; NA 구간은 정상으로 판정하지 않음",xlabel="경보 시작 대비 원시시간 (초 해석 지지)",ylabel="미보정 예측 점수",ylim=(-.02,1.02))
        description.update({'false_alarm_Tool':tool,'false_alarm_start':t,'false_alarm_score_max':float(false.score.max())})
    else:ax.text(.5,.5,'확인 가능한 음성 구간의 오경보 사례 없음\n정답 미확인 구간의 무고장을 뜻하지 않음',ha='center')
    ax.set_yticks(np.linspace(0,1,6));ax.set_ylabel('미보정 예측 점수')
    ax.legend();ax.grid(alpha=.2);save(fig,f"{scheme}_false_alarm_case")
    save_json(OUT/'figures'/f'{scheme}_cases.json',description)


def all_plots():
    setup()
    for scheme in ('split_tool','split_time'):
        comparison(scheme);pr_curves(scheme);confusion(scheme);score_histograms(scheme);importance(scheme);case_plots(scheme)
