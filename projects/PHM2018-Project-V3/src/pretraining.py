"""모델 학습 전용 분할 계약. X와 정답/출처를 분리하며 어떠한 fit도 수행하지 않는다."""
from .common import *
from .features import schema
from .labels import NEG,POS,NA,EXCLUDE

PRE=json.loads((ROOT/'config/pretraining.json').read_text(encoding='utf-8'))
PRE_DIR=ROOT/'data/pretraining'
SPLITS=['train','validation','holdout']

def partitions(tool,start,end,scheme):
    end=end+1  # 모든 구간 [start,end)
    if scheme=='temporal':
        a=start+(end-start)*PRE['temporal_fractions'][0]
        b=start+(end-start)*sum(PRE['temporal_fractions'][:2])
        return [(0,start,a),(1,a,b),(2,b,end)]
    if scheme!='group':raise ValueError('분할 방식 오류')
    code=next(i for i,s in enumerate(SPLITS) if tool[:2] in PRE['group_'+s])
    return [(code,start,end)]

def assign_and_label(frame,parts,h):
    """동일한 900/1800 guard를 모든 A/B/C와 H에 적용한다."""
    t=frame.time.to_numpy();split=np.full(len(t),-1,dtype=np.int8)
    bounded=np.full(len(t),NA,dtype=np.int8)
    for code,start,end in parts:
        mask=(t-PRE['history_guard']>=start)&(t+PRE['future_guard']<end)
        split[mask]=code
        types=[]
        for k in FAULTS:
            s=frame[f'label__conservative__{k}__h{h}'].to_numpy().copy()
            # NEG의 근거인 다음 동일 유형 고장도 반드시 이 분할 안에 있어야 한다.
            s[(t+frame['ttf_'+k].to_numpy()>=end)&(s!=EXCLUDE)]=NA
            types.append(s)
        a=np.stack(types);y=np.full(len(t),NA,dtype=np.int8)
        y[(a==POS).any(axis=0)]=POS;y[(a==NEG).all(axis=0)]=NEG
        y[(a==EXCLUDE).any(axis=0)]=EXCLUDE
        bounded[mask]=y[mask]
    return split,bounded

def iter_batches(scheme,split,family='A',h=600,cohort='maximum',*,include_unlabeled=False,tool=None):
    """모델 입력 X에는 화이트리스트만 반환한다. metadata는 모델에 전달하지 않는다."""
    if scheme not in ['temporal','group'] or split not in SPLITS or family not in 'ABC' or h not in CFG['horizons'] or cohort not in ['maximum','common']:raise ValueError('후보 선택 오류')
    registry=json.loads((PRE_DIR/'manifest.json').read_text(encoding='utf-8'))
    config_hash=hashlib.sha256(json.dumps(PRE,sort_keys=True).encode()).hexdigest()
    if registry['config_hash']!=config_hash:raise ValueError('분할 설정이 변경되어 재생성 필요')
    for item in registry['files']:
        if item['scheme']!=scheme:continue
        if tool is not None and item['Tool']!=tool:continue
        from .query import manifest
        manifest(item['Tool'])  # Sensor/TTF/Fault 변경 시 오래된 정답으로 학습하지 않는다.
        source=ROOT/item['source']
        if [source.stat().st_size,source.stat().st_mtime_ns]!=item['source_signature']:raise ValueError('Anchor 캐시 변경: 재생성 필요')
        m=pd.read_parquet(ROOT/item['assignment'])
        mask=m.split.eq(SPLITS.index(split))
        if not include_unlabeled:mask &= m[f'y_h{h}'].isin([NEG,POS])
        if not mask.any():continue
        cols=list(dict.fromkeys(schema()[family]+['time','segment_id',family+'_available','common_available']))
        d=pd.read_parquet(source,columns=cols)
        if len(d)!=len(m) or not d.time.equals(m.time):raise ValueError('후보 목록과 특징 시각 불일치')
        mask &= d['common_available' if cohort=='common' else family+'_available']
        if not mask.any():continue
        x=d.loc[mask,schema()[family]].reset_index(drop=True).copy()
        if not np.isfinite(x).all().all():raise ValueError('유효 후보에 비유한 특징 존재')
        for c in PRE['categorical_features']:x[c]=x[c].astype('string')
        y=m.loc[mask,f'y_h{h}'].reset_index(drop=True)
        meta=pd.DataFrame({'Tool':item['Tool'],'time':d.loc[mask,'time'].to_numpy(),'segment_id':d.loc[mask,'segment_id'].to_numpy(),'source_row':np.flatnonzero(mask),'source_file':item['source']})
        yield x,y,meta

def iter_scoring_batches(scheme,split,family='A',h=600,cohort='maximum',*,tool=None):
    """NA/EXCLUDE도 유지하는 전체 가용 입력. audit_label은 평가 전용이다."""
    for x,y,meta in iter_batches(scheme,split,family,h,cohort,include_unlabeled=True,tool=tool):
        meta['audit_label']=y.to_numpy()
        yield x,meta
