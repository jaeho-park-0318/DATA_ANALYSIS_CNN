"""최종 시간 Holdout을 배제한 장비 그룹 입력 계약."""
from .common import *
from .pretraining import iter_batches,iter_scoring_batches

PROTOCOL=json.loads((ROOT/'config/stage5_final_protocol.json').read_text(encoding='utf-8'))

def protected_mask(meta,bounds):
    h=bounds[bounds.scheme.eq('temporal')&bounds.split.eq('holdout')].set_index('Tool').start
    return meta.time.to_numpy()+1800 < meta.Tool.map(h).to_numpy()

def load(role,family,cohort,known=False):
    if role not in ['train','validation']:raise ValueError('허용된 그룹 역할 아님')
    allowed=PROTOCOL['group_'+role+'_tool_prefixes'];bounds=pd.read_parquet(TABLES/'pretraining_split_boundaries.parquet');tools=sorted(bounds[bounds.Tool.str[:2].isin(allowed)].Tool.unique());xs=[];ms=[]
    for tool in tools:
        for part in PROTOCOL['allowed_temporal_partitions']:
            if known:
                iterator=((x,m.assign(audit_label=y.to_numpy())) for x,y,m in iter_batches('temporal',part,family,600,cohort,tool=tool))
            else:iterator=iter_scoring_batches('temporal',part,family,600,cohort,tool=tool)
            for x,m in iterator:
                b=bounds[bounds.scheme.eq('temporal')&bounds.split.eq(part)&bounds.Tool.eq(tool)].iloc[0]
                assert m.time.ge(b.start+900).all() and m.time.lt(b.end_exclusive-1800).all()
                assert protected_mask(m,bounds).all()
                m=m.copy();m['temporal_partition']=part;xs.append(x);ms.append(m)
    x=pd.concat(xs,ignore_index=True);m=pd.concat(ms,ignore_index=True);order=m.sort_values(['Tool','time']).index
    x=x.loc[order].reset_index(drop=True);m=m.loc[order].reset_index(drop=True)
    assert not m.duplicated(['Tool','time']).any() and m.Tool.str[:2].isin(allowed).all()
    return x,m

def validation_events():
    events=pd.read_parquet(TABLES/'events.parquet');bounds=pd.read_parquet(TABLES/'pretraining_split_boundaries.parquet');h=bounds[bounds.scheme.eq('temporal')&bounds.split.eq('holdout')].set_index('Tool').start
    return events[events.Tool.str[:2].isin(PROTOCOL['group_validation_tool_prefixes'])&events.fault_time.lt(events.Tool.map(h))].copy()
