"""동일한 경보 계약의 구간 지지길이 및 고정 임계값 평가."""
from .common import *
from .alarm_evaluation import evaluate,observed_support
from .query import manifest

def exposure_by_tool(frame):
    supports={}
    for tool,a in frame.groupby('Tool',sort=True):
        a=a.sort_values('time');index=manifest(tool)['normalized_index'];total=0.
        for i,item in enumerate(index):
            if item['end']<a.time.min()-4 or item['start']>a.time.max()+60:continue
            n=pd.read_parquet(CACHE/'source'/tool/item['file'],columns=['time','segment_id']);n=n[n.time.ge(a.time.min()-4)&n.time.le(a.time.max()+60)]
            following=index[i+1]['start'] if i+1<len(index) else None
            total+=observed_support(n.time.to_numpy(),n.segment_id.to_numpy(),a,next_time=following)
        supports[tool]=total
    return supports

def at_threshold(frame,events,threshold,supports):
    ss=[];ee=[];aa=[];ll=[]
    # 입력이 없는 장비도 기록 이벤트 분모에 남긴다.
    for tool in sorted(set(events.Tool)|set(frame.Tool)):
        q=frame[frame.Tool.eq(tool)];ev=events[events.Tool.eq(tool)];s,e,a,l=evaluate(q,ev,float(threshold));s.update(Tool=tool,observed_support_raw=supports.get(tool,0.));ss.append(s);ee.append(e);aa.append(a);ll.append(l)
        assert l.lead_raw.between(60,600).all() and not a[a.episode_id.isin(l.episode_id)].start_label.eq(3).any()
    e=pd.concat(ee,ignore_index=True);a=pd.concat(aa,ignore_index=True);l=pd.concat(ll,ignore_index=True);s=pd.DataFrame(ss);cols=['recorded_events','tagged_events','scorable_events','available_events','detected_events','alarm_episodes','observed_support_raw']+[k for k in s if k.startswith('alarms_')];row=s[cols].sum().to_dict()
    for den in ['recorded','scorable','available']:row['recall_'+den]=row['detected_events']/row[den+'_events'] if row[den+'_events'] else np.nan
    row['alarm_episodes_per_10000_raw']=10000*row['alarm_episodes']/row['observed_support_raw'] if row['observed_support_raw'] else np.nan;row['median_lead_raw']=e.lead_raw.median();row['available_anchors']=len(frame);row['threshold']=float(threshold)
    return row,e,a,l,s
