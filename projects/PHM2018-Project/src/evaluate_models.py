"""샘플·고장 이벤트·경보 에피소드 지표. 미래 정답은 평가에만 사용한다."""
from __future__ import annotations
from pathlib import Path
import sys
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_recall_curve,average_precision_score,roc_auc_score,
                             confusion_matrix,accuracy_score)
from src.model_common import OUT,save_json,read_json
from src.dataset_common import event_table,REPORT


def choose_threshold(y,score):
    """검증 세트 F1 최대화. 동률이면 더 높은 임계값을 고른다."""
    y=np.asarray(y,dtype=int);score=np.asarray(score,dtype=float)
    if not np.any(y==1):return float(np.inf)
    p,r,t=precision_recall_curve(y,score)
    f=np.divide(2*p[:-1]*r[:-1],p[:-1]+r[:-1],out=np.zeros(len(t)),where=p[:-1]+r[:-1]>0)
    best=np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12))[-1]
    return float(t[best])


def sample_metrics(y,score,threshold):
    y=np.asarray(y,dtype=int);score=np.asarray(score);pred=score>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    precision=float(tp/(tp+fp)) if tp+fp else None
    recall=float(tp/(tp+fn)) if tp+fn else None
    f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else None
    both=len(np.unique(y))==2
    return {"n":len(y),"positive":int(y.sum()),"negative":int((y==0).sum()),
            "precision":precision,"recall":recall,"f1":f1,
            "pr_auc_ap":float(average_precision_score(y,score)) if np.any(y==1) else None,
            "roc_auc":float(roc_auc_score(y,score)) if both else None,"accuracy":float(accuracy_score(y,pred)),
            "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}


def episodes(frame,threshold,horizon=600):
    """동일 Tool/관측 세그먼트의 연속 양성 anchor를 하나의 경보로 묶는다.

    anchor 간격 72 초과, 세그먼트/가동 상태 변화, 또는 중간 음성 anchor가 있으면 끊는다.
    경보 지속 중의 모든 실제 예측 시각을 매칭에 사용한다. H 이상 계속되면
    H마다 새 경보로 재발행해 영구 경보가 한 건으로 축소되는 문제를 막는다.
    """
    data=frame.sort_values(["Tool","time"],kind="stable").copy()
    data["alarm"]=data.score>=threshold
    breaks=(data.Tool!=data.Tool.shift())|(data.segment_id!=data.segment_id.shift())|(data.time.diff()>72)|(~data.alarm.shift(fill_value=False))|(data.stage.gt(0)!=data.stage.shift().gt(0))
    base=breaks.cumsum()
    first=data.groupby(base,sort=False).time.transform("min")
    rearm=np.floor((data.time-first)/horizon).astype(int)
    data["episode_id"]=pd.factorize(pd.MultiIndex.from_arrays([base,rearm]))[0]
    return data


def operational_metrics(frame,threshold,horizon,scheme,split):
    """평가 가능한 이벤트와 전체 기록 이벤트를 분리하고 NA 구간 경보는 판정 보류한다."""
    data=episodes(frame,threshold,horizon)
    target=f"label_any_h{horizon}raw"
    events=event_table();events=events[~events.tagged].copy()
    recorded=read_json(REPORT/"dataset_summary.json")["fault_events"]
    recorded=pd.DataFrame(recorded);recorded=recorded[(~recorded.tagged)&recorded.inside_sensor_range]
    ownership="split_tool" if scheme=="split_tool" else "event_time_split"
    recorded=recorded[recorded[ownership]==split]
    eligible={};candidates={};all_candidates={}
    for event in recorded.itertuples():
        own=data[(data.Tool==event.Tool)&(data.time>=float(event.event_key.split(':')[1])-horizon)&(data.time<float(event.event_key.split(':')[1]))]
        fcol=f"label_{event.fault_code}_h{horizon}raw"
        # 가까운 연속 고장의 이전 이벤트 라벨을 다음 이벤트의 근거로 재사용하지 않는다.
        evaluable=own[(own[fcol]==1)&(own['next_event_'+event.fault_code]==event.event_key)]
        eligible[event.event_key]=len(evaluable)>0
        alarms=evaluable[evaluable.alarm]
        if len(alarms):
            ft=float(event.event_key.split(':')[1])
            for ep,g in alarms.groupby('episode_id'):
                candidates.setdefault(int(ep),[]).append((event.event_key,ft,float(g.time.min())))
    # 모든 알려진 미래 고장에 대한 경보 시간 매칭. 이 작업은 예측 후 평가용이다.
    for event in recorded.itertuples():
        ft=float(event.event_key.split(':')[1])
        own=data[(data.Tool==event.Tool)&data.alarm&(data.time>=ft-horizon)&(data.time<ft)]
        for ep,g in own.groupby('episode_id'):
            all_candidates.setdefault(int(ep),[]).append((event.event_key,ft,float(g.time.min())))
    def match(candidate_map):
        detected={};assigned={}
        for ep,values in candidate_map.items():
            earliest=min(v[1] for v in values)
            # 한 경보는 가장 이른 미래 이벤트 시각에만 대응한다. 동시각 다중 유형은 함께 기록한다.
            for key,ft,alarm_time in values:
                if ft==earliest:
                    detected[key]=max(detected.get(key,0),ft-alarm_time)
                    assigned.setdefault(ep,set()).add(key)
        return detected,assigned
    detected,episode_matches=match(candidates)
    all_detected,_=match(all_candidates)
    all_hits=set(all_detected)
    alarm_rows=data[data.alarm]
    ep_count=alarm_rows.episode_id.nunique();false_episodes=[];unknown_episodes=[]
    for ep,g in alarm_rows.groupby("episode_id"):
        if int(ep) in episode_matches:continue
        if g[target].notna().all() and (g[target]==0).all():false_episodes.append(int(ep))
        else:unknown_episodes.append(int(ep))
    # 인접한 알려진 음성 anchor 사이의 실제 관측 시간만 노출량으로 센다.
    prev=data.shift();dt=data.time-prev.time
    exposure=(data.Tool==prev.Tool)&(data.segment_id==prev.segment_id)&(dt>0)&(dt<=72)&(data[target]==0)&(prev[target]==0)
    active=exposure&(data.stage>0)&(prev.stage>0)
    negative_raw=float(dt[exposure].sum());active_raw=float(dt[active].sum())
    false_alarm_raw=float(dt[exposure & (data.alarm|prev.alarm.fillna(False))].sum())
    active_false=0
    for ep in false_episodes:
        group=alarm_rows[alarm_rows.episode_id==ep]
        if (group.stage>0).all():active_false+=1
    leads=list(detected.values());eligible_keys=[k for k,v in eligible.items() if v]
    per_type={}
    for code in ("f1","f2","f3"):
        keys=[k for k in eligible_keys if k.endswith(':'+code)]
        hits=[k for k in keys if k in detected]
        per_type[code]={"events":len(keys),"detected":len(hits),"recall":len(hits)/len(keys) if keys else None}
    return {"evaluable_events":len(eligible_keys),"detected_events":len(detected),
            "event_recall":len(detected)/len(eligible_keys) if eligible_keys else None,
            "recorded_events":len(recorded),"recorded_events_hit":len(all_hits),
            "recorded_recall_with_observation_limit":len(all_hits)/len(recorded) if len(recorded) else None,
            "event_coverage":len(eligible_keys)/len(recorded) if len(recorded) else None,
            "alarm_episodes":int(ep_count),"false_alarm_episodes":len(false_episodes),"unknown_alarm_episodes":len(unknown_episodes),
            "negative_exposure_raw":negative_raw,"false_alarms_per_3600raw":len(false_episodes)/(negative_raw/3600) if negative_raw>0 else None,
            "false_alarm_time_fraction":false_alarm_raw/negative_raw if negative_raw>0 else None,
            "active_negative_exposure_raw":active_raw,"active_false_alarm_episodes":active_false,
            "false_alarms_per_active_hour_seconds_interpretation":active_false/(active_raw/3600) if active_raw>0 else None,
            "lead_time_median_raw":float(np.median(leads)) if leads else None,
            "lead_time_min_raw":float(np.min(leads)) if leads else None,"lead_time_max_raw":float(np.max(leads)) if leads else None,
            "lead_times":detected,"per_type":per_type,"false_episode_ids":false_episodes[:30]}


def evaluate_and_save(name,scheme,horizon,config,frames,scores,threshold,details):
    folder=OUT/"experiments"/f"{scheme}_h{horizon}"/name
    folder.mkdir(parents=True,exist_ok=True)
    metrics={}
    for split,frame in frames.items():
        target=f"label_any_h{horizon}raw";y=frame[target].to_numpy(dtype=int)
        metrics[split]=sample_metrics(y,scores[split],threshold)
        if split!="train":
            keep=["Tool","time","row_id","segment_id","stage",target]+[f"label_{c}_h{horizon}raw" for c in ("f1","f2","f3")]+["next_event_"+c for c in ("f1","f2","f3")]
            predictions=frame[keep].copy();predictions["score"]=scores[split]
            predictions.to_parquet(folder/f"{split}_predictions.parquet",index=False,compression="zstd")
            metrics[split].update(operational_metrics(predictions,threshold,horizon,scheme,split))
    result={"name":name,"scheme":scheme,"horizon":horizon,"config":config,"threshold":threshold,
            "details":details,"metrics":metrics,"completed":True}
    save_json(folder/"result.json",result)
    print(f"완료 {scheme}/h{horizon}/{name}: 평가 F1={metrics['evaluation']['f1']}, AP={metrics['evaluation']['pr_auc_ap']}",flush=True)
    return result
