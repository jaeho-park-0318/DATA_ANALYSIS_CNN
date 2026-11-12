"""고정 Anchor의 위험 점수에서 이벤트별 경보를 평가한다. 학습·보간 없음."""
import numpy as np
import pandas as pd
from .common import CFG
from .labels import NEG,POS,NA,EXCLUDE

EPISODE_COLUMNS=['Tool','episode_id','start','end','anchors','start_label','contains_unknown','contains_exclude','segment_id']

def alarm_episodes(frame,threshold,stride=60):
    required=['Tool','time','segment_id','score','audit_label']
    if not set(required)<=set(frame):raise ValueError('경보 입력 열 부족')
    if not np.isfinite(threshold):raise ValueError('유한 임계값 필요')
    if frame.duplicated(['Tool','time']).any():raise ValueError('중복 Anchor')
    if not frame.audit_label.isin([NEG,POS,NA,EXCLUDE]).all():raise ValueError('라벨 코드 오류')
    rows=[]
    for tool,g in frame.groupby('Tool',sort=True):
        g=g.sort_values('time');t=g.time.to_numpy();seg=g.segment_id.to_numpy();scores=g.score.to_numpy(float)
        if not np.isfinite(scores).all():raise ValueError('누락된 예측을 무경보로 변환할 수 없음')
        firing=scores>=threshold
        starts=firing & np.r_[True,(~firing[:-1])|(np.diff(t)!=stride)|(seg[1:]!=seg[:-1])]
        ids=np.cumsum(starts)
        z=g.loc[firing].copy();z['episode']=ids[firing]
        if z.empty:continue
        z['unknown']=z.audit_label.eq(NA);z['exclude']=z.audit_label.eq(EXCLUDE)
        a=z.groupby('episode',sort=True).agg(start=('time','first'),end=('time','last'),anchors=('time','size'),start_label=('audit_label','first'),contains_unknown=('unknown','any'),contains_exclude=('exclude','any'),segment_id=('segment_id','first')).reset_index()
        a['Tool']=tool;a['episode_id']=[f'{tool}-alarm-{int(eid):06d}' for eid in a.episode]
        rows.extend(a[EPISODE_COLUMNS].to_dict('records'))
    return pd.DataFrame(rows,columns=EPISODE_COLUMNS)

def evaluate(frame,events,threshold,h=600,lead=60):
    """events는 호출 측에서 선택 분할로 제한한다. 태그 이벤트는 별도 집계한다."""
    if not 0<lead<=h:raise ValueError('0<L<=H 필요')
    if events.event_id.duplicated().any():raise ValueError('이벤트 ID 중복')
    alarms=alarm_episodes(frame,threshold,CFG['anchor_stride']);result=[];matches=[]
    for e in events.itertuples():
        q=frame[frame.Tool.eq(e.Tool)]
        valid=q.time.between(e.fault_time-h,e.fault_time-lead)
        a=alarms[alarms.Tool.eq(e.Tool)&alarms.start.between(e.fault_time-h,e.fault_time-lead)&alarms.start_label.ne(EXCLUDE)]
        tagged=bool(getattr(e,'tagged',False));scorable=not tagged
        detected=scorable and len(a)>0
        if detected:
            matches.extend(dict(event_id=e.event_id,episode_id=r.episode_id,lead_raw=e.fault_time-r.start) for r in a.itertuples())
        status='TAG_UNVERIFIED' if tagged else 'NO_INPUT' if not valid.any() else 'DETECTED' if detected else 'MISSED'
        late=alarms[alarms.Tool.eq(e.Tool)&alarms.start.gt(e.fault_time-lead)&alarms.start.le(e.fault_time)]
        result.append(dict(event_id=e.event_id,Tool=e.Tool,fault_type=e.fault_type,fault_time=e.fault_time,tagged=tagged,eligible_anchors=int(valid.sum()),available=bool(valid.any()),detected=detected,status=status,first_alarm=float(a.start.min()) if detected else np.nan,lead_raw=float(e.fault_time-a.start.min()) if detected else np.nan,late_alarm_episodes=len(late)))
    per_event=pd.DataFrame(result,columns=['event_id','Tool','fault_type','fault_time','tagged','eligible_anchors','available','detected','status','first_alarm','lead_raw','late_alarm_episodes'])
    links=pd.DataFrame(matches,columns=['event_id','episode_id','lead_raw'])
    matched=set(links.episode_id);alarms['matched_event_count']=alarms.episode_id.map(links.episode_id.value_counts()).fillna(0).astype(int)
    names={NEG:'KNOWN_NEGATIVE_START',POS:'POSITIVE_UNMATCHED',NA:'UNKNOWN_START',EXCLUDE:'LATE_OR_EXCLUDED_START'}
    alarms['category']=['MATCHED' if r.episode_id in matched else names[r.start_label] for r in alarms.itertuples()]
    denominator=per_event[~per_event.tagged] if len(per_event) else per_event
    available=denominator[denominator.available] if len(denominator) else denominator
    summary={'recorded_events':len(per_event),'tagged_events':int(per_event.tagged.sum()) if len(per_event) else 0,'scorable_events':len(denominator),'available_events':len(available),'detected_events':int(per_event.detected.sum()) if len(per_event) else 0,'alarm_episodes':len(alarms)}
    summary['recall_recorded']=summary['detected_events']/len(per_event) if len(per_event) else np.nan
    summary['recall_scorable']=summary['detected_events']/len(denominator) if len(denominator) else np.nan
    summary['recall_available']=summary['detected_events']/len(available) if len(available) else np.nan
    for k in ['MATCHED',*names.values()]:summary['alarms_'+k]=int(alarms.category.eq(k).sum())
    summary['median_lead_raw']=per_event.lead_raw.median() if len(per_event) else np.nan
    return summary,per_event,alarms,links

def observed_support(times,segments,anchors,stride=60,nominal=4,next_time=None):
    """점수를 최대60raw 유지하고 관측[t,t+4)와 겹치는 길이만 센다.
    명목 관측 지지구간의 길이이며 실제 가동시간은 아니다. 8/12raw 결측도 보간하지 않는다.
    """
    if anchors.empty or len(times)==0:return 0.0
    if not 0<nominal<=stride:raise ValueError('0<명목 간격<=Anchor 간격 필요')
    at=anchors.time.to_numpy();aseg=anchors.segment_id.to_numpy()
    t=np.asarray(times);seg=np.asarray(segments);idx=np.searchsorted(at,t,side='right')-1
    end=np.minimum(t+nominal,np.r_[t[1:],next_time if next_time is not None else t[-1]+nominal])
    # 첫 Anchor가 관측 지지구간 중간에 들어오는 경우도 구간을 분할한다.
    total=0.0
    for shift in [0,1]:
        ix=idx+shift;safe=np.clip(ix,0,len(at)-1)
        lo=np.maximum(t,at[safe]);hi=np.minimum(end,at[safe]+stride)
        ok=(ix>=0)&(ix<len(at))&(aseg[safe]==seg)&(hi>lo)
        total+=np.maximum(hi-lo,0)[ok].sum()
    return float(total)
