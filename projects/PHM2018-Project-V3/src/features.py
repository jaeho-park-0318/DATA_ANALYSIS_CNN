"""전체 관측에서 고정 격자 Anchor의 과거 특징을 계산한다."""
import numpy as np
import pandas as pd
from .common import *

def schema():
    a=['recipe','stage']+[s+'__current' for s in SENSORS]
    b=a+[f'{s}__{f}_300' for s in SENSORS for f in ['mean','std','min','max']]
    c=b+[f'{s}__{f}_900' for s in SENSORS for f in ['slope','delta']]
    return {'A':a,'B':b,'C':c}

def calculate(frame,anchors):
    t=frame.time.to_numpy(float);a=np.asarray(anchors,dtype=float)
    out=pd.DataFrame({'time':a.astype('int64')})
    if not len(a):return out
    ix=np.searchsorted(t,a,side='right')-1;safe=np.maximum(ix,0)
    lag=a-t[safe];current=(ix>=0)&(lag<=CFG['gap_threshold'])&(~frame.state_conflict.to_numpy()[safe])
    out['last_observation']=np.where(ix>=0,t[safe],np.nan);out['current_lag']=np.where(ix>=0,lag,np.nan)
    out['segment_id']=np.where(ix>=0,frame.segment_id.to_numpy()[safe],-1)
    out['segment_start']=np.where(ix>=0,frame.segment_start.to_numpy()[safe],np.nan)
    for c in ['recipe','stage','recipe_step']:out[c]=np.where(ix>=0,frame[c].to_numpy()[safe],np.nan)
    x=frame[SENSORS].to_numpy(float);finite=np.isfinite(x);z=np.where(finite,x,0)
    for j,s in enumerate(SENSORS):out[s+'__current']=np.where(current,x[safe,j],np.nan)
    out['A_available']=current&np.isfinite(out[schema()['A']]).all(axis=1)
    right=np.searchsorted(t,a,side='right');origin=t[0];tx=t-origin
    def prefix(v):return np.vstack([np.zeros((1,v.shape[1])),np.cumsum(v,axis=0)])
    pn=prefix(finite.astype(float));ps=prefix(z);ps2=prefix(z*z)
    pt=prefix(finite*tx[:,None]);pt2=prefix(finite*tx[:,None]**2);ptx=prefix(z*tx[:,None])
    for w in CFG['history_windows']:
        left=np.searchsorted(t,a-w,side='right')
        # 공백 이전/충돌 이전 관측이 이력으로 섞이지 않도록 세그먼트 시작으로 제한한다.
        left=np.maximum(left,np.searchsorted(t,out.segment_start.fillna(np.inf),side='left'))
        left=np.minimum(left,right)
        n=pn[right]-pn[left];count=right-left
        out[f'n_{w}']=count;out[f'coverage_{w}']=n.min(axis=1)/(w/CFG['nominal_interval'])
        out[f'history_{w}']=out.A_available&(a-out.segment_start>=w)&(out[f'coverage_{w}']>=CFG['minimum_coverage'])
        out[f'source_min_{w}']=np.where(count>0,t[np.minimum(left,len(t)-1)],np.nan)
        out[f'source_max_{w}']=np.where(count>0,t[np.maximum(right-1,0)],np.nan)
        sums=ps[right]-ps[left];sums2=ps2[right]-ps2[left]
        with np.errstate(divide='ignore',invalid='ignore'):
            mean=sums/n;std=np.sqrt(np.maximum(0,sums2/n-mean**2))
        if w==300:
            # Anchor를 관측 시간 인덱스에 삽입해서 정확한 (t-w,t] 최솟값/최댓값을 구한다.
            index=pd.to_timedelta(t,unit='ns');ai=pd.to_timedelta(a,unit='ns');union=index.union(ai).sort_values()
            rolling=pd.DataFrame(x,index=index,columns=SENSORS).reindex(union).rolling(pd.Timedelta(w,unit='ns'),closed='right',min_periods=1)
            lo=rolling.min().reindex(ai).to_numpy();hi=rolling.max().reindex(ai).to_numpy()
            for j,s in enumerate(SENSORS):
                for f,v in [('mean',mean),('std',std),('min',lo),('max',hi)]:out[f'{s}__{f}_300']=np.where(out.history_300,v[:,j],np.nan)
        if w==900:
            st=pt[right]-pt[left];st2=pt2[right]-pt2[left];stx=ptx[right]-ptx[left]
            with np.errstate(divide='ignore',invalid='ignore'):
                den=st2-st**2/n;slope=(stx-st*sums/n)/den
            delta=x[np.maximum(right-1,0)]-x[np.minimum(left,len(t)-1)]
            for j,s in enumerate(SENSORS):
                out[f'{s}__slope_900']=np.where(out.history_900,slope[:,j],np.nan)
                out[f'{s}__delta_900']=np.where(out.history_900,delta[:,j],np.nan)
    out['B_available']=out.A_available&out.history_300&np.isfinite(out[schema()['B']]).all(axis=1)
    out['C_available']=out.A_available&out.history_300&out.history_900&np.isfinite(out[schema()['C']]).all(axis=1)
    out['common_available']=out[['A_available','B_available','C_available']].all(axis=1)
    return out
