"""센서·공정의 과거 관측만 읽어 일정 원시시간 간격의 특징을 생성한다."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer

from src.dataset_common import *

INPUT_COLUMNS=["row_id","Tool","time","recipe","stage","segment_id","state_ambiguous"]+SENSORS


class RawTimeIndexer(BaseIndexer):
    """물리 단위 변환 없이 원시 시각으로 계산한 rolling 경계."""
    def __init__(self, starts, ends):
        super().__init__();self.starts=starts;self.ends=ends

    def get_window_bounds(self,num_values=0,min_periods=None,center=None,closed=None,step=None):
        return self.starts,self.ends


def make_features(frame, fresh_start=0, stride=STRIDE):
    """현재 시각 t를 포함한 (t-w,t]와 이전 (t-2w,t-w]를 비교한다."""
    if not set(frame.columns).issubset(set(INPUT_COLUMNS)):
        raise ValueError("특징 입력에 허용하지 않은 컬럼이 있습니다.")
    t=frame.time.to_numpy(dtype=float)
    if len(t)==0:return pd.DataFrame()
    if np.any(np.diff(t)<=0):raise ValueError("특징 입력 시각은 엄격히 증가해야 합니다.")
    buckets=np.floor(t/stride).astype(np.int64)
    choose=np.r_[True,buckets[1:]!=buckets[:-1]]
    choose[:fresh_start]=False
    anchors=np.flatnonzero(choose)
    if not len(anchors):return pd.DataFrame()
    out=frame.iloc[anchors][["row_id","Tool","time","recipe","stage","segment_id"]].reset_index(drop=True)
    x=frame[SENSORS].to_numpy(dtype=float)
    valid=np.isfinite(x);z=np.where(valid,x,0)
    sums=np.vstack([np.zeros(6),np.cumsum(z,axis=0)])
    squares=np.vstack([np.zeros(6),np.cumsum(z*z,axis=0)])
    counts=np.vstack([np.zeros(6,dtype=np.int64),np.cumsum(valid,axis=0)])
    for j,s in enumerate(SENSORS):out[s+"__current"]=x[anchors,j]
    ready=~frame.state_ambiguous.to_numpy()[anchors]
    end_all=np.arange(1,len(t)+1,dtype=np.int64);end=end_all[anchors]
    for w in WINDOWS:
        start_all=np.searchsorted(t,t-w,side="right").astype(np.int64)
        start=start_all[anchors]
        previous_start=np.searchsorted(t,t[anchors]-2*w,side="right")
        n=counts[end]-counts[start];prev_n=counts[start]-counts[previous_start]
        total=sums[end]-sums[start];previous=sums[start]-sums[previous_start]
        mean=np.divide(total,n,out=np.full(total.shape,np.nan),where=n>0)
        prev_mean=np.divide(previous,prev_n,out=np.full(previous.shape,np.nan),where=prev_n>0)
        variance=np.divide(squares[end]-squares[start],n,out=np.full(total.shape,np.nan),where=n>0)-mean*mean
        std=np.sqrt(np.maximum(variance,0))
        rolling=pd.DataFrame(x,columns=SENSORS).rolling(RawTimeIndexer(start_all,end_all),min_periods=1)
        minimum=rolling.min().to_numpy()[anchors];maximum=rolling.max().to_numpy()[anchors]
        for j,s in enumerate(SENSORS):
            values={"mean":mean[:,j],"std":std[:,j],"min":minimum[:,j],"max":maximum[:,j],
                    "delta_mean":mean[:,j]-prev_mean[:,j],"rate_mean":(mean[:,j]-prev_mean[:,j])/w}
            for kind,value in values.items():out[f"{s}__{kind}_{w}raw"]=value
        ready&=(n>=.9*w/4).all(axis=1)&(prev_n>=.9*w/4).all(axis=1)
    # 이력 충분성은 특징과 동일하게 현재·과거 시각/상태만으로 판단한다.
    old=np.searchsorted(t,t[anchors]-LOOKBACK,side="right")-1
    safe=np.maximum(old,0)
    sid=frame.segment_id.to_numpy()
    ready&=(old>=0)&(sid[safe]==sid[anchors])&((t[anchors]-LOOKBACK-t[safe])<=8)
    out["history_min_time"]=t[safe]
    out["source_max_time"]=t[anchors]
    stage=frame.stage.to_numpy();recipe=frame.recipe.to_numpy()
    begin=np.searchsorted(t,t[anchors]-300,side="right");finish=anchors+1
    for name,value in [("stage_zero_fraction_300raw",(stage==0).astype(float)),
                       ("stage_change_count_300raw",np.r_[0,(stage[1:]!=stage[:-1]).astype(int)]),
                       ("recipe_change_count_300raw",np.r_[0,(recipe[1:]!=recipe[:-1]).astype(int)])]:
        prefix=np.r_[0,np.cumsum(value)]
        total=prefix[finish]-prefix[begin]
        out[name]=total/(finish-begin) if "fraction" in name else total
    ready&=np.isfinite(out[feature_columns()].to_numpy()).all(axis=1)
    out["feature_ready"]=ready
    return out


def build_tool(tool):
    directory=DEST/"features"/tool
    if directory.exists() and any(directory.glob("*.parquet")):raise FileExistsError(str(directory))
    halo=None;rows=0;usable=0;part=0
    for fresh in read_batches(DEST/"observations"/tool,INPUT_COLUMNS):
        data=pd.concat([halo,fresh],ignore_index=True) if halo is not None else fresh
        features=make_features(data,len(halo) if halo is not None else 0)
        if len(features):
            write_part(features,directory,part);part+=1
            rows+=len(features);usable+=int(features.feature_ready.sum())
        halo=data[data.time>=data.time.iloc[-1]-LOOKBACK-GAP].copy()
    return {"Tool":tool,"anchor_rows":rows,"feature_ready_rows":usable,"features":len(feature_columns())}


def build(tools=None):
    ensure_compatible()
    summaries=[]
    for folder in sorted((DEST/"observations").iterdir()):
        if not folder.is_dir() or (tools and folder.name not in tools):continue
        done=REPORT/f"features_{folder.name}.json"
        if done.exists():
            import json
            summaries.append(json.loads(done.read_text(encoding="utf-8")));print(f"완료된 특징 재사용: {folder.name}",flush=True);continue
        start=time.monotonic();summary=build_tool(folder.name);summary["elapsed_seconds"]=time.monotonic()-start
        save_json(done,summary);summaries.append(summary)
        print(f"{folder.name}: anchor {summary['anchor_rows']:,} / 특징 준비 {summary['feature_ready_rows']:,} ({summary['elapsed_seconds']:.1f}초)",flush=True)
    save_json(DEST/"feature_columns.json",feature_columns())
    save_json(REPORT/"feature_build.json",summaries)
    return summaries


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--tools",nargs="*");args=parser.parse_args()
    build(args.tools)
