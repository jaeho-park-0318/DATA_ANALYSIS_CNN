"""원본 CSV의 1회 청크 읽기, 원본값 보존 Parquet와 시간 인덱스 생성."""
from collections import Counter
from itertools import zip_longest
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from .common import *

INGEST_VERSION=1

def ingest_tool(tool, events):
    dest=CACHE/'source'/tool;dest.mkdir(parents=True,exist_ok=True)
    marker=dest/'manifest.json'
    if marker.exists():
        m=json.loads(marker.read_text(encoding='utf-8'))
        if m['source_signature']!=source_signature(tool) or m['version']!=INGEST_VERSION:
            raise ValueError('입력 변경: 새로운 캐시 버전 필요 '+tool)
        print(f'{tool}: 원본 인덱스 캐시 재사용',flush=True);return m
    t0=time.monotonic();stem=f'{tool[:2]}_{tool[2:]}_DC_train.csv'
    cols=['time','Tool']+CONTEXT+SENSORS
    dtype={c:'float64' for c in cols if c not in ['Tool','Lot','time']}
    dtype.update(time='int64',Tool='string',Lot='string')
    reader=pd.read_csv(RAW/'train'/stem,usecols=cols,dtype=dtype,chunksize=CFG['chunk_rows'])
    treader=pd.read_csv(RAW/'train/train_ttf'/stem,dtype='float64',chunksize=CFG['chunk_rows'])
    base=event_arrays(events,tool);tagged=event_arrays(events,tool,True)
    raw_index=[];norm_index=[];gaps=[];duplicates=[];segments={};bins={};dt_hist=Counter()
    raw_rows=0;norm_rows=0;part=0;carry=None;last_t=None;last_state=None;last_conflict=False;sid=-1;segment_start=None
    ttf_null={k:0 for k in FAULTS};ttf_mismatch={k:0 for k in FAULTS};sensor_missing={s:0 for s in SENSORS}
    min_time=None;max_time=None

    def consume(d):
        nonlocal last_t,last_state,last_conflict,sid,segment_start,norm_rows,part
        if d.empty:return
        g=d.groupby('time',sort=False,dropna=False)
        n=g[SENSORS+['ttf_'+k for k in FAULTS]].mean()
        for c in CONTEXT:n[c]=g[c].first()
        n['duplicate_n']=g.size();n['raw_row_first']=g.raw_row.min()
        n['state_conflict']=False
        for c in CONTEXT:n['state_conflict']|=g[c].nunique(dropna=False)>1
        n['sensor_disagreement']=(g[SENSORS].max()-g[SENSORS].min()).abs().max(axis=1)>0
        n=n.reset_index();t=n.time.to_numpy();conf=n.state_conflict.to_numpy()
        dt=np.diff(np.r_[t[0] if last_t is None else last_t,t])
        if last_t is None:dt[0]=0
        if np.any(dt[1:]<=0) or (last_t is not None and dt[0]<=0):raise ValueError('시간 정렬 오류')
        values,counts=np.unique(dt[dt>0],return_counts=True);dt_hist.update(dict(zip(values.tolist(),counts.tolist())))
        for j in np.flatnonzero(dt>CFG['nominal_interval']):
            gaps.append({'Tool':tool,'start':int(t[j]-dt[j]),'end':int(t[j]),'gap_raw':int(dt[j]),'long_gap':bool(dt[j]>CFG['gap_threshold'])})
        starts=(dt>CFG['gap_threshold'])|conf|np.r_[last_conflict,conf[:-1]]
        if last_t is None:starts[0]=True
        ids=sid+np.cumsum(starts);n['segment_id']=ids
        st=np.where(starts,t,segment_start if segment_start is not None else t[0]);n['segment_start']=np.maximum.accumulate(st)
        changed=np.zeros(len(n),dtype=bool)
        state={}
        for c in CONTEXT:
            v=n[c].astype('string').fillna('<NA>').to_numpy()
            previous=v[0] if last_state is None else last_state[c]
            changed|=v!=np.concatenate(([previous],v[:-1]));state[c]=v[-1]
        n['process_change']=changed;n['dt_raw']=dt;n['within_segment_dt']=np.where(starts,0,dt);n['Tool']=tool
        n['obs_id']=np.arange(norm_rows,norm_rows+len(n));norm_rows+=len(n)
        for k in FAULTS:
            expected,_=next_events(t,base[k]);incl,_=next_events(t,tagged[k])
            n['ttf_verified_'+k]=np.isclose(n['ttf_'+k],expected,rtol=0,atol=1e-6,equal_nan=True)
            n['tag_conflict_'+k]=~np.isclose(expected,incl,rtol=0,atol=1e-6,equal_nan=True)
            n['null_'+k]=n['ttf_'+k].isna().astype('int64')
        for r in n[n.duplicate_n>1].itertuples():
            duplicates.append({'Tool':tool,'time':r.time,'duplicate_n':r.duplicate_n,'state_conflict':r.state_conflict,'sensor_disagreement':r.sensor_disagreement,'raw_row_first':r.raw_row_first})
        n['bin_start']=(n.time//CFG['bin_width'])*CFG['bin_width']
        for group_col,store in [('segment_id',segments),('bin_start',bins)]:
            gapcol='within_segment_dt' if group_col=='segment_id' else 'dt_raw'
            agg={'time':['min','max','size'],'duplicate_n':'sum',gapcol:'max','process_change':'sum','state_conflict':'sum',**{'null_'+k:'sum' for k in FAULTS}}
            a=n.groupby(group_col,sort=False).agg(agg)
            for key,row in a.iterrows():
                q={'start':int(row[('time','min')]),'end':int(row[('time','max')]),'sensor_observations':int(row[('time','size')]),'raw_rows':int(row[('duplicate_n','sum')]),'max_gap':float(row[(gapcol,'max')]),'process_changes':int(row[('process_change','sum')]),'conflicts':int(row[('state_conflict','sum')]),**{'null_'+k:int(row[('null_'+k,'sum')]) for k in FAULTS}}
                key=int(key)
                if key not in store:store[key]=q
                else:
                    old=store[key];old['end']=q['end'];old['max_gap']=max(old['max_gap'],q['max_gap'])
                    for c in ['sensor_observations','raw_rows','process_changes','conflicts']+['null_'+k for k in FAULTS]:old[c]+=q[c]
        n=n.drop(columns=['bin_start']+['null_'+k for k in FAULTS])
        name=f'normalized-{part:04d}.parquet';write_parquet(n,dest/name)
        norm_index.append({'file':name,'start':int(t[0]),'end':int(t[-1]),'rows':len(n)})
        sid=int(ids[-1]);segment_start=int(n.segment_start.iloc[-1]);last_t=int(t[-1]);last_state=state;last_conflict=bool(conf[-1]);part+=1

    for batch,(d,ttf) in enumerate(zip_longest(reader,treader)):
        if d is None or ttf is None or not np.array_equal(d.time.to_numpy(),ttf.time.to_numpy()):raise ValueError('센서/TTF 시간 불일치 '+tool)
        if not d.Tool.eq(tool).all() or (np.diff(d.time)<0).any():raise ValueError('장비/시간 무결성 오류')
        d['raw_row']=np.arange(raw_rows,raw_rows+len(d));raw_rows+=len(d)
        if min_time is None:min_time=int(d.time.iloc[0])
        max_time=int(d.time.iloc[-1])
        for k,f in FAULTS.items():
            d['ttf_'+k]=ttf['TTF_'+f].to_numpy();ttf_null[k]+=int(d['ttf_'+k].isna().sum())
            expected,_=next_events(d.time.to_numpy(),base[k])
            ttf_mismatch[k]+=int((~np.isclose(expected,d['ttf_'+k],equal_nan=True,rtol=0,atol=1e-6)).sum())
        for s in SENSORS:sensor_missing[s]+=int((~np.isfinite(d[s])).sum())
        name=f'raw-{batch:04d}.parquet';write_parquet(d,dest/name)
        raw_index.append({'file':name,'start':int(d.time.iloc[0]),'end':int(d.time.iloc[-1]),'rows':len(d)})
        combined=pd.concat([carry,d],ignore_index=True) if carry is not None else d
        tail=combined.time.iloc[-1];carry=combined[combined.time==tail].copy();consume(combined[combined.time!=tail])
        if batch%5==0:print(f'{tool}: 원본 {raw_rows:,}행 / {time.monotonic()-t0:.1f}s',flush=True)
    consume(carry)
    gapdf=pd.DataFrame(gaps,columns=['Tool','start','end','gap_raw','long_gap'])
    dupdf=pd.DataFrame(duplicates,columns=['Tool','time','duplicate_n','state_conflict','sensor_disagreement','raw_row_first'])
    write_parquet(gapdf,dest/'gaps.parquet');write_parquet(dupdf,dest/'duplicates.parquet')
    own=events[events.Tool.eq(tool)]
    rows=[]
    for key in range(min_time//CFG['bin_width']*CFG['bin_width'],max_time+1,CFG['bin_width']):
        b=bins.get(key,{'sensor_observations':0,'raw_rows':0,'max_gap':np.nan,'process_changes':0,'conflicts':0,**{'null_'+k:0 for k in FAULTS}}).copy()
        b.update(Tool=tool,kind='fixed',segment_id=-1,start=max(key,min_time),end=min(key+CFG['bin_width']-1,max_time))
        rows.append(b)
    for key,b in segments.items():rows.append(dict(b,Tool=tool,kind='continuous',segment_id=key))
    sdf=pd.DataFrame(rows)
    for k in FAULTS:sdf['ttf_null_ratio_'+k]=sdf['null_'+k]/sdf.sensor_observations.replace(0,np.nan)
    sdf['density']=sdf.sensor_observations/((sdf.end-sdf.start)/CFG['nominal_interval']+1)
    times=own.fault_time.to_numpy()
    sdf['fault_count']=[int(((times>=a)&(times<=b)).sum()) for a,b in zip(sdf.start,sdf.end)]
    sdf['fault_near']=[bool(((times>=a-CFG['horizons'][-1])&(times<=b+CFG['horizons'][-1])).any()) for a,b in zip(sdf.start,sdf.end)]
    # GAP 상태는 긴 공백의 열린 내부와 겹칠 때만 부여한다.
    lg=gapdf[gapdf.long_gap] if len(gapdf) else gapdf
    starts=lg.start.to_numpy();ends=lg.end.to_numpy()
    sdf['has_gap']=[bool(((starts<b)&(ends>a)).any()) for a,b in zip(sdf.start,sdf.end)]
    def flags(r):
        v=[]
        if r.sensor_observations:v.append('OBSERVED')
        if r.sensor_observations and r.density<0.8:v.append('SPARSE')
        if r.has_gap and r.kind=='fixed':v.append('GAP')
        if r.conflicts:v.append('CONFLICT')
        if r.fault_near:v.append('FAULT_NEAR')
        if not r.sensor_observations:v.append('UNKNOWN')
        return '|'.join(v)
    sdf['states']=[flags(r) for r in sdf.itertuples()];write_parquet(sdf,dest/'segments.parquet')
    meta={'version':INGEST_VERSION,'Tool':tool,'source_signature':source_signature(tool),'raw_rows':raw_rows,'normalized_rows':norm_rows,'duplicate_rows':raw_rows-norm_rows,'duplicate_times':len(dupdf),'conflict_times':int(dupdf.state_conflict.sum()) if len(dupdf) else 0,'min_time':min_time,'max_time':max_time,'mode_dt':dt_hist.most_common(1)[0][0],'long_gaps':int(gapdf.long_gap.sum()) if len(gapdf) else 0,'max_gap':float(gapdf.gap_raw.max()) if len(gapdf) else 0,'ttf_null':ttf_null,'ttf_mismatch':ttf_mismatch,'sensor_missing':sensor_missing,'raw_index':raw_index,'normalized_index':norm_index,'dt_hist':dict(dt_hist),'elapsed_seconds':time.monotonic()-t0}
    dump_json(marker,meta);print(f'{tool}: 원본 인덱스 완료 {raw_rows:,}행 / {meta["elapsed_seconds"]:.1f}s',flush=True)
    return meta

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--tools',nargs='*');args=parser.parse_args()
    events=events_from_raw();save_table(events,'events')
    dump_json(ROOT/'outputs/source_snapshot_before.json',snapshot());dump_json(ROOT/'outputs/environment.json',environment())
    tools=args.tools or sorted(events.Tool.unique())
    for tool in tools:ingest_tool(tool,events)
    dump_json(ROOT/'outputs/source_snapshot_after_ingest.json',snapshot())

if __name__=='__main__':main()
