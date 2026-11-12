"""대표 장비의 전체 분포와 시간 순서를 보존한 고장 전 구간 분석."""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.eda_common import (DATA, OUT, SENSORS, FAULTS, csv_chunks, load_faults,
                            read_json, write_json)

REP_FILE = "01_M01_DC_train.csv"


def coverage(frame, start, end, step):
    """[start,end) 구간의 엄격한 관측 충분성. 보간/임의 재표본화 없음."""
    t = np.sort(frame.time.unique())
    expected = (end-start)/step
    result = {"rows": len(frame), "unique_times": len(t), "expected_at_mode": expected,
              "coverage_ratio": min(1, len(t)/expected),
              "start_gap": float(t[0]-start) if len(t) else None,
              "end_gap": float(end-t[-1]) if len(t) else None,
              "max_gap": float(np.diff(t).max()) if len(t)>1 else None}
    result["sufficient"] = bool(len(t)>1 and result["coverage_ratio"]>=.9
                               and result["start_gap"]<=2*step and result["end_gap"]<=2*step
                               and result["max_gap"]<=3*step)
    return result


def hist_quantile(counts, edges, q):
    target = counts.sum()*q
    j = min(len(counts)-1, int(np.searchsorted(np.cumsum(counts),target)))
    prior = counts[:j].sum()
    fraction = (target-prior)/counts[j] if counts[j] else .5
    return float(edges[j] + fraction*(edges[j+1]-edges[j]))


def representative(force=False):
    """대표 장비만 2회 순회하고 과거 비교 구간을 추가 탐색한다. 무작위 표본 없음."""
    from hashlib import sha256
    overview = read_json("overview.json")
    sig = sha256((overview["signature"]+PathText()).encode()).hexdigest()
    if not force and (OUT/"detail.json").exists() and read_json("detail.json").get("signature")==sig:
        print("대표 장비의 검증된 로컬 집계 캐시 사용")
        return read_json("detail.json")
    info = next(x for x in overview["tools"] if x["file"]==REP_FILE and x["split"]=="train")
    if info["time_reversals"]:
        raise ValueError("대표 장비의 역전 시각을 먼저 해결해야 합니다.")
    tool=info["Tool"]; step=info["mode_dt"]
    faults=load_faults(); events=faults[faults.Tool==tool].sort_values("time")
    stats={x["sensor"]:x for x in overview["sensor_by_tool"] if x["Tool"]==tool}
    edges={s:np.linspace(stats[s]["min"],stats[s]["max"] if stats[s]["max"]>stats[s]["min"] else stats[s]["min"]+1,121) for s in SENSORS}
    hist={s:np.zeros(120,dtype=np.int64) for s in SENSORS}
    count=0; sums=np.zeros(6); sums2=np.zeros(6); cross=np.zeros((6,6))
    retained=[]; bins={}; width=max(step,int(np.ceil(info["duration_raw"]/2000/step))*step)
    for chunk in csv_chunks(DATA/"train"/REP_FILE):
        x=chunk[SENSORS].to_numpy()
        for j,s in enumerate(SENSORS): hist[s]+=np.histogram(x[np.isfinite(x[:,j]),j],bins=edges[s])[0]
        complete=x[np.isfinite(x).all(1)]
        count+=len(complete); sums+=complete.sum(0); sums2+=(complete*complete).sum(0); cross+=complete.T@complete
        for bucket, group in chunk.groupby(np.floor((chunk.time-info["min_time"])/width).astype(int),sort=False):
            acc=bins.setdefault(int(bucket), {"n":0,"sum":np.zeros(6),"valid":np.zeros(6,dtype=int)})
            gx=group[SENSORS].to_numpy(); valid=np.isfinite(gx)
            acc["n"]+=len(gx); acc["sum"]+=np.where(valid,gx,0).sum(0); acc["valid"]+=valid.sum(0)
        keep=np.zeros(len(chunk),dtype=bool); t=chunk.time.to_numpy()
        for event in events.itertuples(): keep|=(t>=event.time-18000)&(t<event.time)
        if keep.any(): retained.append(chunk.loc[keep].copy())
    cov=cross-np.outer(sums,sums)/count
    variance=np.maximum(0,sums2-sums*sums/count)
    corr=np.divide(cov,np.sqrt(np.outer(variance,variance)),out=np.full((6,6),np.nan),where=np.outer(variance,variance)>0)
    timeline=[]
    for bucket,acc in sorted(bins.items()):
        row={"time":float(info["min_time"]+bucket*width+width/2),"rows":acc["n"]}
        row.update({s:float(acc["sum"][j]/acc["valid"][j]) if acc["valid"][j] else None for j,s in enumerate(SENSORS)})
        timeline.append(row)
    # 청크마다 이벤트 구간의 합집합 마스크를 쓰므로 원본 행은 한 번만 들어온다.
    # 동시각의 서로 다른 원본 행은 제거하지 않는다. 충분성은 고유 시각 기준이다.
    context=pd.concat(retained,ignore_index=True).sort_values("time",kind="stable")
    candidates=[]
    for event in events.itertuples():
        windows=[]
        for minutes in (5,10,30,60):
            sub=context[(context.time>=event.time-minutes*60)&(context.time<event.time)]
            windows.append(dict(coverage(sub,event.time-minutes*60,event.time,step),minutes_assuming_seconds=minutes))
        hour=context[(context.time>=event.time-3600)&(context.time<event.time)]
        candidates.append({"event_id":int(event.event_id),"time":float(event.time),"fault_name":event.fault_name,
                           "windows":windows,"sufficient_windows":sum(w["sufficient"] for w in windows),
                           "stage_nonzero_ratio":float((hour.stage>0).mean()) if len(hour) else 0})
    # 센서 변화 크기와 무관하게 관측 충분성, 시간 순서로 선정한다.
    selected=max(candidates,key=lambda a:(a["sufficient_windows"],min(w["coverage_ratio"] for w in a["windows"]),-a["time"]))
    ft=selected["time"]
    pre=context[(context.time>=ft-3600)&(context.time<ft)].copy()
    controls=[]
    # 같은 장비, 고장 시각보다 과거인 1시간 블록 중 모든 알려진 fault에서 ±3600 이상 떨어진 구간.
    for back in (7200,10800,14400,18000):
        start=ft-back; end=start+3600
        if ((events.time>=start-3600)&(events.time<=end+3600)).any(): continue
        ctrl=context[(context.time>=start)&(context.time<end)].copy()
        check=coverage(ctrl,start,end,step)
        overlap=set(map(tuple,pre[["recipe","stage"]].drop_duplicates().to_numpy())) & set(map(tuple,ctrl[["recipe","stage"]].drop_duplicates().to_numpy()))
        controls.append((check["sufficient"],len(overlap),check["coverage_ratio"],start,end,ctrl,check))
    best=max(controls,key=lambda c:c[:3]) if controls else None
    control=best[5] if best else context.iloc[:0].copy()
    control_info={"start":best[3],"end":best[4],**best[6]} if best else {"sufficient":False,"reason":"고장 미인접 비교 구간 없음"}
    pre.to_json(OUT/"pre_event.json",orient="records",force_ascii=False)
    control.to_json(OUT/"control.json",orient="records",force_ascii=False)
    # 정상 보장 대신, stage>0이고 알려진 고장에 인접하지 않은 운전 비교 후보를 별도로 제시한다.
    eligible=[c for c in candidates if c["sufficient_windows"]>=2 and c["stage_nonzero_ratio"]>=.5]
    secondary=max(eligible,key=lambda c:(c["sufficient_windows"],-c["time"])) if eligible else None
    active_control=context.iloc[:0].copy(); active_control_info={"sufficient":False}
    limit=min(ft,secondary["time"] if secondary else ft)
    # 비교 구간도 대상 이벤트보다 과거로 제한한다. 센서값으로 선별하지 않고 첫 적격 블록을 쓴다.
    for source in csv_chunks(DATA/"train"/REP_FILE):
        for bucket, block in source.groupby(np.floor(source.time/3600).astype(int),sort=True):
            start=float(bucket*3600); end=start+3600
            if end>=limit or ((events.time>=start-3600)&(events.time<=end+3600)).any(): continue
            quality=coverage(block,start,end,step)
            if quality["sufficient"] and (block.stage>0).mean()>=.9:
                active_control=block; active_control_info={"start":start,"end":end,**quality,
                                                          "stage_nonzero_ratio":float((block.stage>0).mean())}
                break
        if len(active_control):break
    active_control.to_json(OUT/"active_control.json",orient="records",force_ascii=False)
    secondary_pre=context[(context.time>=secondary["time"]-3600)&(context.time<secondary["time"])] if secondary else context.iloc[:0]
    secondary_pre.to_json(OUT/"secondary_pre.json",orient="records",force_ascii=False)
    # 히스토그램 보간 분위수로 IQR 경계를 정의한 뒤, 그 경계 밖의 전체 대표 장비 행을 정확히 집계.
    fences={}
    for s in SENSORS:
        q1=hist_quantile(hist[s],edges[s],.25); q3=hist_quantile(hist[s],edges[s],.75)
        fences[s]={"q1_hist_approx":q1,"q3_hist_approx":q3,"lower":q1-1.5*(q3-q1),"upper":q3+1.5*(q3-q1),"flagged":0}
    for chunk in csv_chunks(DATA/"train"/REP_FILE):
        for s in SENSORS:
            a=fences[s]; x=chunk[s].to_numpy(); a["flagged"]+=int((np.isfinite(x)&((x<a["lower"])|(x>a["upper"]))).sum())
    comparisons=[]
    for minutes in (5,10,30,60):
        win=pre[pre.time>=ft-minutes*60]
        quality=next(w for w in selected["windows"] if w["minutes_assuming_seconds"]==minutes)
        for s in SENSORS:
            comparisons.append({"minutes_assuming_seconds":minutes,"sensor":s,"pre_mean":float(win[s].mean()),
                                "pre_std":float(win[s].std()),"control_mean":float(control[s].mean()),
                                "difference":float(win[s].mean()-control[s].mean()),
                                "window_sufficient":quality["sufficient"],"control_sufficient":control_info["sufficient"]})
    matched=[]
    for (recipe,stage), group in pre.groupby(["recipe","stage"]):
        ctrl=control[(control.recipe==recipe)&(control.stage==stage)]
        for s in SENSORS:
            matched.append({"recipe":float(recipe),"stage":float(stage),"sensor":s,"pre_n":int(group[s].count()),
                            "control_n":int(ctrl[s].count()),"pre_mean":float(group[s].mean()),"control_mean":float(ctrl[s].mean()),
                            "difference":float(group[s].mean()-ctrl[s].mean()),"comparable":bool(group[s].count()>=10 and ctrl[s].count()>=10)})
    summary={"signature":sig,"Tool":tool,"file":REP_FILE,"rows":info["rows"],"mode_dt":step,
             "hist":{s:{"edges":edges[s].tolist(),"counts":hist[s].tolist()} for s in SENSORS},
             "correlation_complete_rows":count,"correlation":corr.tolist(),"timeline_bin_raw":width,
             "timeline":timeline,"outlier_fences":fences,"event_candidates":candidates,"selected_event":selected,
             "control":control_info,"active_control":active_control_info,"secondary_event":secondary,
             "comparisons":comparisons,"matched_comparisons":matched}
    write_json("detail.json",summary)
    print(f"대표 {tool}: {info['rows']:,}행 분포/상관 집계; 선택 event={selected['event_id']}, 충분한 윈도 {selected['sufficient_windows']}/4")
    return summary


def PathText():
    from pathlib import Path
    return Path(__file__).read_text(encoding="utf-8")


def audit_time_keys(force=False):
    """센서/TTF 전체 시각열의 순서·중복을 포함한 digest를 청크로 비교한다."""
    from hashlib import sha256
    sig=read_json("overview.json")["signature"]
    if not force and (OUT/"time_keys.json").exists() and read_json("time_keys.json").get("signature")==sig:
        return read_json("time_keys.json")
    rows=[]
    for path in sorted((DATA/"train").glob("*.csv")):
        digests=[]; counts=[]
        for source in (path,DATA/"train/train_ttf"/path.name):
            digest=sha256(); n=0
            for chunk in pd.read_csv(source,usecols=["time"],dtype={"time":"float64"},chunksize=200_000):
                digest.update(chunk.time.to_numpy(dtype="<f8").tobytes()); n+=len(chunk)
            digests.append(digest.hexdigest()); counts.append(n)
        rows.append({"file":path.name,"sensor_rows":counts[0],"ttf_rows":counts[1],
                     "ordered_time_keys_equal":digests[0]==digests[1],"sensor_time_sha256":digests[0],"ttf_time_sha256":digests[1]})
        print(f"시간 키 검증 {path.name}: {digests[0]==digests[1]}",flush=True)
    result={"signature":sig,"files":rows}
    write_json("time_keys.json",result)
    return result


def audit_tagged_ttf():
    """태그 이벤트를 포함/제외한 두 기준을 실제 TTF와 대조한다."""
    faults=load_faults(); all_results=[]
    flagged=faults[faults.fault_raw.str.contains("[NoWaferID]",regex=False)]
    for event in flagged.itertuples():
        path=DATA/"train/train_ttf"/event.source.replace("_train_fault_data.csv","_DC_train.csv")
        exact=np.sort(faults.loc[(faults.Tool==event.Tool)&(faults.fault_raw==event.fault_name),"time"].unique())
        normalized=np.sort(faults.loc[(faults.Tool==event.Tool)&(faults.fault_name==event.fault_name),"time"].unique())
        result={"Tool":event.Tool,"tagged_time":float(event.time),"rows":0,"mismatch_including_tag":0,
                "mismatch_excluding_tag":0,"examples":[]}
        for chunk in pd.read_csv(path,usecols=["time","TTF_"+event.fault_name],dtype="float64",chunksize=200_000):
            t=chunk.time.to_numpy(); x=chunk["TTF_"+event.fault_name].to_numpy(); result["rows"]+=len(t)
            for f,key in [(normalized,"mismatch_including_tag"),(exact,"mismatch_excluding_tag")]:
                i=np.searchsorted(f,t,side="left"); available=i<len(f)
                e=np.full(len(t),np.nan); e[available]=f[i[available]]-t[available]
                bad=np.isfinite(x)&(~np.isclose(x,e,atol=1e-6,rtol=0))
                result[key]+=int(bad.sum())
                if key=="mismatch_including_tag" and bad.any() and len(result["examples"])<5:
                    for j in np.flatnonzero(bad)[:5-len(result["examples"])]:
                        result["examples"].append({"time":float(t[j]),"ttf_actual":float(x[j]),
                                                  "expected_including_tag":float(e[j]),"ttf_implied_event_time":float(t[j]+x[j])})
        all_results.append(result)
    write_json("tagged_ttf.json",all_results)
    return all_results


def horizon_labels(times, fault_times, observed_end, horizon, valid_ttf=None, followup_complete=None):
    """검증용 타깃만 생성: 1=미래 이벤트 확인, 0=전체 구간 관측, NA=검열/불명.

    valid_ttf는 이미 fault와 정합성을 확인한 TTF만 전달한다. 결측 TTF를
    직접 0으로 바꾸지 않는다. 관측 종료, 내부 결측 구간, fault 기록 완전성을
    검증한 행만 followup_complete=True로 전달한다. 기본값은 미확인이다.
    """
    t=np.asarray(times,dtype=float); f=np.sort(np.unique(fault_times))
    idx=np.searchsorted(f,t,side="right"); next_delta=np.full(len(t),np.nan)
    available=idx<len(f); next_delta[available]=f[idx[available]]-t[available]
    y=pd.array([pd.NA]*len(t),dtype="Int8")
    positive=(next_delta>0)&(next_delta<=horizon)
    y[positive]=1
    followup=np.zeros(len(t),dtype=bool) if followup_complete is None else np.asarray(followup_complete,dtype=bool)
    negative=(~positive)&(t+horizon<=observed_end)&followup
    y[negative]=0
    # 고장 시각 자체는 조기 예측 타깃에서 제외.
    y[np.isin(t,f)]=pd.NA
    if valid_ttf is not None:
        x=np.asarray(valid_ttf,dtype=float)
        # TTF가 알려진 미래 이벤트와 모순되면 임의 정상화하지 않는다.
        conflict=np.isfinite(x)&(~np.isclose(x,next_delta,atol=1e-6,rtol=0))
        y[conflict]=pd.NA
    return y
