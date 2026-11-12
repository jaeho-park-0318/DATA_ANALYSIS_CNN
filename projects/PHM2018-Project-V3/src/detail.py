"""선택 구간의 원본·집계값·정답 근거를 함께 반환한다."""
import numpy as np
import pandas as pd
from .common import *
from .query import read_window,manifest,broken_line
from .labels import make_labels,STATUS_NAMES

def records(frame):
    return json.loads(frame.to_json(orient='records',force_ascii=False))

def summarize_sensors(observations,start,end):
    """진단 구간의 실제 관측만 요약한다. 첫/끝 차이는 추세나 전조 판정이 아니다."""
    q=observations[observations.time.between(start,end)]
    result=[]
    for sensor in SENSORS:
        z=q.loc[np.isfinite(q[sensor].to_numpy(dtype=float)),['time',sensor]].astype({'time':'float64',sensor:'float64'})
        result.append({'sensor':sensor,'start':start,'end':end,'observations':len(z),
                       'first_time':z.time.iloc[0] if len(z) else np.nan,'last_time':z.time.iloc[-1] if len(z) else np.nan,
                       'first_value':z[sensor].iloc[0] if len(z) else np.nan,'last_value':z[sensor].iloc[-1] if len(z) else np.nan,
                       'delta':z[sensor].iloc[-1]-z[sensor].iloc[0] if len(z) else np.nan,
                       'minimum':z[sensor].min(),'maximum':z[sensor].max(),'std':z[sensor].std(ddof=0),
                       'max_observed_gap':z.time.diff().max(),'process_conditions':len(q[CONTEXT].drop_duplicates()),
                       'conflict_observations':int(q.state_conflict.sum())})
    return records(pd.DataFrame(result))

def get_detail(tool,start,end,event_id=None,policy='conservative',h=600,family='C'):
    if policy not in CFG['policies'] or h not in CFG['horizons'] or family not in 'ABC':raise ValueError('선택 정책/특징/H 오류')
    if end-start>CFG['max_detail_width']:raise ValueError('상세 구간은 20,000 raw 이하로 확대하세요.')
    raw,ra=read_window(tool,start,end,'raw')
    norm,na=read_window(tool,start-900,end,'normalized',enforce_limit=False)
    anchors,aa=read_window(tool,start,end,'anchors')
    events=pd.read_parquet(TABLES/'events.parquet');own=events[events.Tool.eq(tool)]
    selected=own[own.event_id.eq(event_id)] if event_id else own.iloc[:0]
    event=None if selected.empty else records(selected)[0]
    n=norm[norm.time.between(start,end)].copy();series={}
    for s in SENSORS:
        x,y=broken_line(n.time.to_numpy(),n[s].to_numpy(float))
        rx,ry=broken_line(raw.time.to_numpy(),raw[s].to_numpy(float)) if len(raw) else ([],[])
        curves=[]
        for _,g in norm.groupby('segment_id',sort=False):
            if g.empty:continue
            index=pd.to_timedelta(g.time.to_numpy(),unit='ns');v=pd.Series(g[s].to_numpy(),index=index)
            mean=v.rolling('60ns',closed='right',min_periods=2).mean();std=v.rolling('60ns',closed='right',min_periods=2).std(ddof=0)
            long_mean=v.rolling('300ns',closed='right',min_periods=2).mean()
            rate=g[s].diff()/g.time.diff()
            q=pd.DataFrame({'time':g.time.to_numpy(),'mean':mean.to_numpy(),'std':std.to_numpy(),'rate':rate.to_numpy(),'short_long':(mean-long_mean).to_numpy()})
            curves.append(q[q.time.between(start,end)])
        cur=pd.concat(curves,ignore_index=True) if curves else pd.DataFrame(columns=['time','mean','std','rate','short_long'])
        series[s]={'x':x,'normalized':y,'raw_x':rx,'raw':ry,'mean60':dict(zip(['x','y'],broken_line(cur.time.to_numpy(),cur['mean'].to_numpy(float)))),'std60':dict(zip(['x','y'],broken_line(cur.time.to_numpy(),cur['std'].to_numpy(float)))),'rate':dict(zip(['x','y'],broken_line(cur.time.to_numpy(),cur['rate'].to_numpy(float)))),'short_long':dict(zip(['x','y'],broken_line(cur.time.to_numpy(),cur['short_long'].to_numpy(float))))}
    if len(anchors):
        segment=pd.read_parquet(CACHE/'source'/tool/'segments.parquet');se=segment[segment.kind.eq('continuous')].set_index('segment_id').end
        labels,_,_,_,evidence=make_labels(anchors.time.to_numpy(),events,tool,manifest(tool)['max_time'],anchors.segment_id.map(se).fillna(-np.inf).to_numpy(),include_evidence=True)
        cols=['time','last_observation','current_lag','n_60','n_300','n_900','coverage_60','coverage_300','coverage_900','A_available','B_available','C_available','recipe','stage','recipe_step']
        table=anchors[cols].copy()
        for k in list(FAULTS)+['any']:
            key=f'{policy}__{k}__h{h}';table[k]=STATUS_NAMES[labels[key]];table[k+'_evidence']=evidence[key]
        table['candidate_reason']=np.select([~table.A_available,~table[family+'_available'],table['any'].eq('EXCLUDE'),table['any'].eq('NA')],['현재값 없음/12 raw 초과/공정충돌','요구 이력·관측률·유한 특징 부족','최소 선행시간 미달','라벨 근거 부족'],default='해당 정책의 연구 후보')
        strips={p:{k:STATUS_NAMES[anchors[f'label__{p}__{k}__h{h}'].to_numpy(int)].tolist() for k in list(FAULTS)+['any']} for p in CFG['policies']}
    else:table=pd.DataFrame();strips={}
    gap=pd.read_parquet(CACHE/'source'/tool/'gaps.parquet');gap=gap[gap.long_gap&(gap.end>start)&(gap.start<end)]
    dup=pd.read_parquet(CACHE/'source'/tool/'duplicates.parquet');dup=dup[dup.time.between(start,end)]
    ex=pd.read_parquet(TABLES/'event_exclusion_reasons.parquet')
    ex=ex[ex.Tool.eq(tool)&ex.policy.eq(policy)&ex.H.eq(h)&ex.feature_set.eq(family)]
    if event:ex=ex[ex.event_id.eq(event_id)]
    else:ex=ex[ex.fault_time.between(start,end)]
    condition=[]
    if len(n):
        for (r,st),g in n.groupby(['recipe','stage'],dropna=False):
            condition.append({'recipe':r,'stage':st,'observations':len(g),**{s:float(g[s].median()) for s in SENSORS},**{'ttf_null_'+k:float(g['ttf_'+k].isna().mean()) for k in FAULTS}})
    analysis_start=max(start,event['fault_time']-h) if event else start
    analysis_end=min(end,event['fault_time']-CFG['minimum_lead']) if event else end
    return {'Tool':tool,'start':start,'end':end,'event':event,'raw_rows':len(raw),'observations':len(n),'anchors':len(anchors),'unique_events_in_range':int(own.fault_time.between(start,end).sum()),'sampling':'선택 구간 원본 해상도 100%; 선은 공백>12 raw에서 끊음','source':f'원본 {tool[:2]}_{tool[2:]}_DC_train.csv → V3 시간 인덱스 Parquet','policy':policy,'H':h,'L':CFG['minimum_lead'],'feature_set':family,'series':series,'anchor_times':anchors.time.tolist() if len(anchors) else [],'label_strips':strips,'anchor_table':records(table),'gaps':records(gap),'duplicates':records(dup),'events':records(own[own.fault_time.between(start,end)]),'process_changes':n.loc[n.process_change,'time'].tolist(),'ttf':{k:{'x':n.time.tolist(),'valid':n['ttf_'+k].notna().astype(int).tolist()} for k in FAULTS},'exclusions':records(ex),'conditions':condition,'sensor_summary':summarize_sensors(n,analysis_start,analysis_end),'load_audit':[ra,na,aa],'empty':len(raw)==0,'post_fault_note':'고장 이후 값은 참고 표시이며 선택 이벤트의 예측 입력에 포함되지 않음'}
