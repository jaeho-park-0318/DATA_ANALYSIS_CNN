"""고장 정답 근거와 센서 입력 품질을 분리하는 세 진단 정책."""
import numpy as np
from .common import *

NEG,POS,NA,EXCLUDE=0,1,2,3
STATUS_NAMES=np.array(['NEG','POS','NA','EXCLUDE'])

def make_labels(times,events,tool,source_end,segment_end=None,include_evidence=False):
    t=np.asarray(times,dtype=float);own=events[events.Tool.eq(tool)]
    base=event_arrays(events,tool);incl=event_arrays(events,tool,True)
    delta={};ids={};tag_conflict={}
    for k in FAULTS:
        delta[k],ids[k]=next_events(t,base[k]);di,_=next_events(t,incl[k])
        tag_conflict[k]=~np.isclose(delta[k],di,rtol=0,atol=1e-6,equal_nan=True)
    all_table=own.sort_values('fault_time');all_delta,_=next_events(t,all_table)
    imminent=np.isfinite(all_delta)&(all_delta>=0)&(all_delta<CFG['minimum_lead'])
    first=own.loc[~own.tagged,'fault_time'].min();last=own.loc[~own.tagged,'fault_time'].max()
    segend=np.full(len(t),-np.inf) if segment_end is None else np.asarray(segment_end)
    result={};evidence={}
    for policy in CFG['policies']:
        for h in CFG['horizons']:
            group=[]
            for k in FAULTS:
                d=delta[k];s=np.full(len(t),NA,dtype=np.int8);reason=np.full(len(t),'미래 유형 이력 없음',dtype=object) if include_evidence else None
                pos=(d>=CFG['minimum_lead'])&(d<=h);neg=d>h
                s[pos]=POS;s[neg]=NEG
                if include_evidence:reason[pos]='확인된 다음 고장';reason[neg]='검증 TTF/다음 고장이 H 밖'
                if policy=='research':
                    mask=(s==NA)&(t>=first)&(t+h<=last)&(t+h<=segend)
                    s[mask]=NEG
                    if include_evidence:reason[mask]='가정: 제한 구간의 유형 무기록'
                if policy=='sensitivity':
                    mask=(s==NA)&(t+h<=source_end)
                    s[mask]=NEG
                    if include_evidence:reason[mask]='가정: 센서 종료까지 완전 고장 이력'
                s[tag_conflict[k]]=NA;s[imminent]=EXCLUDE
                if include_evidence:reason[tag_conflict[k]]='NoWaferID 포함/제외 충돌';reason[imminent]='어떤 고장이든 0<=d<L (최우선 제외)'
                key=f'{policy}__{k}__h{h}';result[key]=s
                if include_evidence:evidence[key]=reason
                group.append(s)
            a=np.stack(group);combined=np.full(len(t),NA,dtype=np.int8)
            combined[(a==POS).any(axis=0)]=POS;combined[(a==NEG).all(axis=0)]=NEG
            combined[imminent]=EXCLUDE
            key=f'{policy}__any__h{h}';result[key]=combined
            if include_evidence:evidence[key]=np.select([imminent,(a==POS).any(axis=0),(a==NEG).all(axis=0)],['어떤 고장이든 0<=d<L (최우선 제외)','하나 이상의 유형이 양성','모든 유형이 해당 정책상 음성'],default='확정되지 않은 유형 존재')
    return result,delta,ids,tag_conflict,evidence
