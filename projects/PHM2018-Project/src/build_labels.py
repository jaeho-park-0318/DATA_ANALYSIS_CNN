"""원본을 청크로 정합 검증하고 정규화 관측 및 보수적 라벨을 Parquet로 만든다."""
from __future__ import annotations

import argparse
from itertools import zip_longest
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.dataset_common import *


def normalize_tool(path, tool):
    """동시각 배치를 청크 경계에서도 완결한 뒤 평균 집계한다."""
    directory=DEST/"observations"/tool
    if directory.exists() and any(directory.glob("*.parquet")):
        raise FileExistsError(f"기존 파생 파일은 덮어쓰지 않습니다: {directory}")
    sensor_iter=pd.read_csv(path,usecols=["time","Tool","recipe","stage"]+SENSORS,
                            dtype={**{c:"float64" for c in ["time","recipe","stage"]+SENSORS},"Tool":"str"},chunksize=CHUNK)
    ttf_iter=pd.read_csv(DATA/"train/train_ttf"/path.name,dtype="float64",chunksize=CHUNK)
    events=event_table();own=events[events.Tool==tool]
    timing={int(r.event_id):{"event_id":int(r.event_id),"event_key":r.event_key,"Tool":tool,
                            "fault_code":r.fault_code,"fault_time":float(r.time),"tagged":bool(r.tagged),
                            "sensor_time":-np.inf,"stage":None,"recipe":None} for r in own.itertuples()}
    carry=None;last_time=None;last_ambiguous=False;sid=-1;row_id=0;part=0
    segments={};raw_rows=0;ttf_mismatches=0;ambiguous=0;duplicates=0
    exact_times={}
    for f,code in FAULT_CODES.items():
        exact_times[code]=np.sort(own.loc[(own.fault_name==f)&(~own.tagged),"time"].unique())

    def consume(raw):
        nonlocal last_time,last_ambiguous,sid,row_id,part,ambiguous,duplicates
        if raw.empty:return
        # 이전/현재 시각만 사용한 동시각 집계이며 미래 시각 관측은 혼합하지 않는다.
        grouped=raw.groupby("time",sort=False,dropna=False)
        normalized=grouped[SENSORS+["ttf_f1","ttf_f2","ttf_f3"]].mean()
        normalized["recipe"]=grouped.recipe.first()
        normalized["stage"]=grouped.stage.first()
        normalized["duplicate_count"]=grouped.size()
        ambiguous_mask=(grouped.recipe.nunique(dropna=False)>1)|(grouped.stage.nunique(dropna=False)>1)
        normalized["state_ambiguous"]=ambiguous_mask.to_numpy()|normalized[["recipe","stage"]].isna().any(axis=1).to_numpy()
        normalized=normalized.reset_index()
        t=normalized.time.to_numpy();am=normalized.state_ambiguous.to_numpy()
        dt=np.diff(np.r_[last_time,t] if last_time is not None else np.r_[t[0]-GAP-1,t])
        if np.any(dt<=0):raise ValueError(f"시간 역전 또는 미완결 동시각 배치: {tool}")
        starts=(dt>GAP)|am|np.r_[last_ambiguous,am[:-1]]
        ids=sid+np.cumsum(starts.astype(np.int64));sid=int(ids[-1])
        if ids[0]<0:raise AssertionError("첫 세그먼트 ID 오류")
        normalized["segment_id"]=ids
        normalized["row_id"]=np.arange(row_id,row_id+len(t),dtype=np.int64)
        normalized["Tool"]=tool
        row_id+=len(t);last_time=float(t[-1]);last_ambiguous=bool(am[-1])
        ambiguous+=int(am.sum());duplicates+=int((normalized.duplicate_count-1).sum())
        for seg,group in normalized.groupby("segment_id",sort=False):
            seg=int(seg)
            if seg not in segments:segments[seg]={"segment_id":seg,"start":float(group.time.min()),"end":float(group.time.max()),"rows":0}
            segments[seg]["end"]=float(group.time.max());segments[seg]["rows"]+=len(group)
        for event in timing.values():
            idx=np.searchsorted(t,event["fault_time"],side="right")-1
            if idx>=0 and t[idx]>event["sensor_time"]:
                row=normalized.iloc[int(idx)]
                event.update(sensor_time=float(row.time),stage=None if row.state_ambiguous else float(row.stage),
                             recipe=None if row.state_ambiguous else float(row.recipe))
        write_part(normalized,directory,part);part+=1

    for sensor,ttf in zip_longest(sensor_iter,ttf_iter):
        if sensor is None or ttf is None or not np.array_equal(sensor.time.to_numpy(),ttf.time.to_numpy()):
            raise ValueError(f"센서/TTF 행·시각 불일치: {tool}")
        if not (sensor.Tool==tool).all():raise ValueError(f"Tool 불일치: {path}")
        if not np.isfinite(sensor.time).all():raise ValueError(f"시각 결측: {tool}")
        raw_rows+=len(sensor)
        for fault,code in FAULT_CODES.items():
            x=ttf["TTF_"+fault].to_numpy();ft=exact_times[code]
            idx=np.searchsorted(ft,sensor.time.to_numpy(),side="left");valid=idx<len(ft)
            expected=np.full(len(x),np.nan);expected[valid]=ft[idx[valid]]-sensor.time.to_numpy()[valid]
            matching=np.isclose(x,expected,rtol=0,atol=1e-6,equal_nan=True)
            ttf_mismatches+=int((~matching).sum())
            if not matching.all():raise ValueError(f"기본 정책의 TTF 불일치: {tool}/{code}")
            sensor["ttf_"+code]=x
        combined=pd.concat([carry,sensor],ignore_index=True) if carry is not None else sensor
        # 다음 청크에도 같은 시각이 있을 수 있으므로 마지막 시각 배치를 보류한다.
        tail=combined.time.iloc[-1];carry=combined[combined.time==tail].copy()
        consume(combined[combined.time!=tail])
    if carry is not None:consume(carry)
    segment_frame=pd.DataFrame(list(segments.values()))
    dest=DEST/"segments";dest.mkdir(parents=True,exist_ok=True)
    segment_frame.to_parquet(dest/f"{tool}.parquet",index=False,compression="zstd")
    for event in timing.values():
        event["lag_raw"]=event["fault_time"]-event["sensor_time"]
        event["state_known"]=bool(event["lag_raw"]<=GAP and event["stage"] is not None)
        if not event["state_known"]:event["stage"]=None;event["recipe"]=None
    return {"Tool":tool,"raw_rows":raw_rows,"normalized_rows":row_id,"collapsed_duplicate_rows":duplicates,
            "ambiguous_state_times":ambiguous,"segments":len(segments),"ttf_mismatches":ttf_mismatches,
            "events":list(timing.values())}


def classify(t, delta, finite_evidence, segment_end, ambiguous, at_fault, horizon):
    """모순/누락은 NA, 양성은 이벤트까지, 음성은 horizon 전체의 연속 관측을 요구한다."""
    labels=np.full(len(t),-1,dtype=np.int8)
    positive=(delta>0)&(delta<=horizon)&finite_evidence&(t+delta<=segment_end+GAP)
    negative=(delta>horizon)&finite_evidence&(t+horizon<=segment_end)
    labels[positive]=1;labels[negative]=0
    labels[ambiguous|at_fault]=-1
    return pd.array(np.where(labels<0,None,labels),dtype="Int8")


def label_tool(tool):
    directory=DEST/"labels"/tool
    if directory.exists() and any(directory.glob("*.parquet")):raise FileExistsError(str(directory))
    segments=pd.read_parquet(DEST/"segments"/f"{tool}.parquet").set_index("segment_id")
    seg_end=segments["end"].to_numpy();low=float(segments.start.min());high=float(segments.end.max())
    events=event_table();own=events[events.Tool==tool];base=own[~own.tagged]
    all_fault_times=np.sort(own.time.unique())
    counters=[];changed={};n=0
    for part,frame in enumerate(read_batches(DEST/"observations"/tool)):
        t=frame.time.to_numpy();end=seg_end[frame.segment_id.to_numpy()]
        at_fault=np.isin(t,all_fault_times);ambiguous=frame.state_ambiguous.to_numpy()
        out=frame[["row_id","Tool","time","recipe","stage","segment_id","state_ambiguous"]].copy()
        out["segment_end_raw"]=end
        out["split_tool"]=tool_split(tool);out["split_time"]=temporal_split(t,low,high)
        for fault,code in FAULT_CODES.items():
            ev=base[base.fault_name==fault].sort_values("time")
            ft=ev.time.to_numpy();keys=ev.event_key.to_numpy()
            idx=np.searchsorted(ft,t,side="right");available=idx<len(ft)
            delta=np.full(len(t),np.nan);delta[available]=ft[idx[available]]-t[available]
            event_keys=np.full(len(t),None,dtype=object);event_keys[available]=keys[idx[available]]
            x=frame["ttf_"+code].to_numpy()
            verified=np.isfinite(x)&np.isclose(x,delta,rtol=0,atol=1e-6)
            out["next_event_"+code]=event_keys
            out["ttf_verified_"+code]=verified
            for horizon in HORIZONS:
                col=f"label_{code}_h{horizon}raw"
                out[col]=classify(t,delta,verified,end,ambiguous,at_fault,horizon)
        # 태그 포함 대안은 fault 이력 우선의 후보다. 제공 TTF와 충돌하는 행은 별도 표시한다.
        ev=own[own.fault_code=="f2"].sort_values("time");ft=ev.time.to_numpy()
        idx=np.searchsorted(ft,t,side="right");available=idx<len(ft)
        delta=np.full(len(t),np.nan);delta[available]=ft[idx[available]]-t[available]
        mismatch=np.isfinite(frame.ttf_f2)&(~np.isclose(frame.ttf_f2,delta,rtol=0,atol=1e-6))&(~at_fault)
        out["alt_include_tag_ttf_conflict"]=mismatch
        for horizon in HORIZONS:
            alt=classify(t,delta,np.isfinite(frame.ttf_f2).to_numpy(),end,ambiguous,at_fault,horizon)
            out[f"alt_label_f2_h{horizon}raw"]=alt
            cols=[f"label_{f}_h{horizon}raw" for f in ("f1","f2","f3")]
            matrix=np.column_stack([out[c].fillna(-1).to_numpy(dtype=np.int8) for c in cols])
            any_values=np.where((matrix==1).any(axis=1),1,np.where((matrix==0).all(axis=1),0,-1))
            out[f"label_any_h{horizon}raw"]=pd.array(np.where(any_values<0,None,any_values),dtype="Int8")
        write_part(out,directory,part);n+=len(out)
    return {"Tool":tool,"label_rows":n,"time_min":low,"time_max":high}


def build(tools=None):
    ensure_compatible()
    REPORT.mkdir(parents=True,exist_ok=True)
    policy_path=DEST/"policy.json"
    if policy_path.exists():
        import json
        if json.loads(policy_path.read_text(encoding="utf-8"))!=POLICY:raise ValueError("정책이 다른 기존 출력 경로입니다.")
    save_json(policy_path,POLICY)
    before=snapshot();summaries=[]
    paths=sorted((DATA/"train").glob("*.csv"))
    for path in paths:
        tool=path.stem[:6].replace("_","")
        if tools and tool not in tools:continue
        start=time.monotonic();done=REPORT/f"labels_{tool}.json"
        if done.exists():
            import json
            summaries.append(json.loads(done.read_text(encoding="utf-8")));print(f"완료된 라벨 재사용: {tool}",flush=True);continue
        summary=normalize_tool(path,tool);summary.update(label_tool(tool));summary["elapsed_seconds"]=time.monotonic()-start
        save_json(done,summary);summaries.append(summary)
        print(f"{tool}: 정규화 {summary['normalized_rows']:,}행 / 라벨 완료 ({summary['elapsed_seconds']:.1f}초)",flush=True)
    assert before==snapshot(),"원본 메타데이터 변경"
    save_json(REPORT/"source_snapshot.json",before)
    save_json(REPORT/"label_build.json",summaries)
    return summaries


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--tools",nargs="*");args=parser.parse_args()
    build(args.tools)
