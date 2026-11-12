"""Notebook에서 사용하는 한국어 EDA 그래프. 수치 단위를 임의 복원하지 않는다."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.eda_common import OUT, SENSORS, FAULTS

SHORT_FAULTS = {FAULTS[0]:"FlowCool 저압",FAULTS[1]:"FlowCool 고압",FAULTS[2]:"FlowCool 누출"}


def setup():
    plt.rcParams.update({"font.family":"Malgun Gothic","axes.unicode_minus":False,
                         "figure.dpi":110,"font.size":10,"axes.titlepad":12})


def save(fig,name):
    path=OUT/"figures"; path.mkdir(parents=True,exist_ok=True)
    fig.savefig(path/(name+".png"),dpi=130,bbox_inches="tight")
    plt.show()


def histograms(detail):
    fig,axes=plt.subplots(3,2,figsize=(14,12),layout="constrained")
    for ax,s in zip(axes.flat,SENSORS):
        h=detail["hist"][s]; edges=np.array(h["edges"])
        ax.stairs(h["counts"],edges,fill=True,alpha=.7,label="대표 장비 전체 행")
        ax.set(title=s,xlabel="센서값 (익명화 수치, 물리 단위 미공개)",ylabel="관측 수 (행)",yscale="log")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle(f"{detail['Tool']} 센서 분포 — 전체 {detail['rows']:,}행, 로그 빈도",fontsize=15)
    save(fig,"sensor_histograms")


def timeline(detail):
    frame=pd.DataFrame(detail["timeline"]).sort_values("time")
    # 빈 시간 bin을 NA로 넣어 장기 공백을 선으로 연결하지 않는다.
    first=frame.time.min(); width=detail["timeline_bin_raw"]
    frame=frame.set_index("time").reindex(np.arange(first,frame.time.max()+width/2,width))
    fig,axes=plt.subplots(3,2,figsize=(14,12),layout="constrained")
    for ax,s in zip(axes.flat,SENSORS):
        ax.plot((frame.index-first)/3600,frame[s],lw=.8,label=f"{width:g} 원시시간 단위 bin 평균")
        ax.set(title=s,xlabel="첫 bin 이후 경과 (시간, time=초 가정)",ylabel="센서값 (익명화 수치)")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle(f"{detail['Tool']} 장기 추이 — bin 평균, 빈 bin은 연결하지 않음",fontsize=15)
    save(fig,"sensor_timeline")


def correlation(detail):
    x=np.array(detail["correlation"])
    fig,ax=plt.subplots(figsize=(11,8),layout="constrained")
    im=ax.imshow(x,vmin=-1,vmax=1,cmap="RdBu_r")
    ax.set_xticks(range(6),SENSORS,rotation=40,ha="right"); ax.set_yticks(range(6),SENSORS)
    for i in range(6):
        for j in range(6):ax.text(j,i,f"{x[i,j]:.3f}",ha="center",va="center",color="white" if abs(x[i,j])>.65 else "black")
    ax.set(title=f"{detail['Tool']} 전체 완전 관측 행의 Pearson 상관 (n={detail['correlation_complete_rows']:,})",
           xlabel="센서",ylabel="센서")
    fig.colorbar(im,ax=ax,label="Pearson r (무차원)")
    save(fig,"sensor_correlation")


def process_means(overview,detail):
    groups=pd.DataFrame(overview["groups"])
    groups=groups[(groups.split=="train")&(groups.Tool==detail["Tool"])].copy()
    groups["condition"]=groups.recipe+" / "+groups.stage
    groups=groups.nlargest(10,"rows").sort_values("rows",ascending=False)
    fig,axes=plt.subplots(3,2,figsize=(15,12),layout="constrained")
    for ax,s in zip(axes.flat,SENSORS):
        ax.bar(groups.condition,groups[s],label="조건별 평균")
        ax.tick_params(axis="x",rotation=55)
        ax.set(title=s,xlabel="Recipe / Stage (관측 수 상위 10조건)",ylabel="평균 센서값 (익명화 수치)")
        ax.legend(fontsize=8); ax.grid(axis="y",alpha=.2)
    fig.suptitle(f"{detail['Tool']} 공정 조건별 센서 평균 — 전체 대표 장비 집계",fontsize=15)
    save(fig,"process_sensor_means")


def event_plot(frame,event,control=None,name="pre_event",note=""):
    ft=event["time"]
    fig,axes=plt.subplots(3,2,figsize=(15,12),layout="constrained")
    for ax,s in zip(axes.flat,SENSORS):
        # 관측 공백은 선으로 연결하지 않는다. 데이터 행은 제거하지 않는다.
        x=frame.copy(); gaps=x.time.diff()>12
        y=x[s].copy(); y.loc[gaps]=np.nan
        ax.plot((x.time-ft)/60,y,lw=1,label="고장 전 원본 순서 (공백에서 단절)")
        if control is not None and len(control):
            ax.axhline(control[s].mean(),color="tab:orange",ls="--",label="비교 구간 평균")
        ax.axvline(0,color="crimson",ls="--",label="기록된 정지 시각")
        for h in (-5,-10,-30):ax.axvline(h,color="gray",lw=.5,alpha=.4)
        ax.set(title=s,xlabel="고장 시각 대비 (분, time=초 가정)",ylabel="센서값 (익명화 수치)",xlim=(-60,1))
        ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle(f"{SHORT_FAULTS.get(event['fault_name'],event['fault_name'])} / time={ft:g}\n{note}",fontsize=14)
    save(fig,name)


def fault_distribution(faults):
    table=pd.crosstab(faults.Tool,faults.fault_name).reindex(columns=FAULTS,fill_value=0)
    table=table.rename(columns=SHORT_FAULTS)
    fig,axes=plt.subplots(1,2,figsize=(16,7),layout="constrained")
    counts=table.sum(); axes[0].bar(counts.index,counts.values,label="태그 분리 후 이벤트 수")
    for i,n in enumerate(counts):axes[0].text(i,n,str(n),ha="center",va="bottom")
    axes[0].set(title="FlowCool 고장 3종 분포",xlabel="고장 유형",ylabel="이벤트 수 (건)");axes[0].legend()
    table.plot.bar(stacked=True,ax=axes[1],rot=60)
    axes[1].set(title="장비별 고장 유형",xlabel="Tool ID",ylabel="이벤트 수 (건)")
    axes[1].legend(title="고장 유형",fontsize=8)
    save(fig,"fault_distribution")


def process_timeline(pre,control,event):
    fig,axes=plt.subplots(2,1,figsize=(13,7),layout="constrained")
    for ax,col in zip(axes,["recipe","stage"]):
        ax.step((pre.time-event["time"])/60,pre[col],where="post",label="고장 전")
        if len(control):ax.step((control.time-control.time.max())/60,control[col],where="post",alpha=.6,label="비교 구간 (종료를 0에 정렬)")
        ax.set(title=col,xlabel="각 구간 종료 대비 (분, time=초 가정)",ylabel="공정 범주 ID (단위 없음)")
        ax.legend();ax.grid(alpha=.2)
    save(fig,"event_process_conditions")
