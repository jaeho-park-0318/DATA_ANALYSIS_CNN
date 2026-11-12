"""20대의 라벨/Anchor/특징 가용성 및 고장 이벤트 보존 집계."""
from collections import defaultdict
import json
import time
import numpy as np
import pandas as pd
from .common import *
from .query import manifest
from .labels import make_labels,POS,NEG,NA,EXCLUDE
from .features import calculate,schema

ANALYSIS_VERSION=1

def analyze_tool(tool,events):
    source=CACHE/'source'/tool;dest=CACHE/'analysis'/tool;dest.mkdir(parents=True,exist_ok=True)
    marker=dest/'manifest.json';meta=manifest(tool)
    signature=hashlib.sha256(json.dumps(CFG,sort_keys=True).encode()).hexdigest()
    if marker.exists():
        m=json.loads(marker.read_text(encoding='utf-8'))
        if m['version']!=ANALYSIS_VERSION or m['config_hash']!=signature:raise ValueError('새 분석 버전 필요')
        print(f'{tool}: 분석 캐시 재사용',flush=True);return
    if sum(meta['ttf_mismatch'].values()):raise ValueError('원본 TTF 정합성 오류를 먼저 검토해야 합니다')
    seg=pd.read_parquet(source/'segments.parquet');segend=seg[seg.kind.eq('continuous')].set_index('segment_id').end
    own=events[events.Tool.eq(tool)].copy();event_stats={r.event_id:{'pre_'+str(w):0 for w in [60,300,600,900,1800]} for r in own.itertuples()}
    for r in own.itertuples():event_stats[r.event_id].update(last_sensor=np.nan,warning_sensor_300=0,warning_sensor_600=0,warning_sensor_1800=0)
    counts=defaultdict(lambda:np.zeros(4,dtype=np.int64));represented=defaultdict(set)
    feature_counts=defaultdict(lambda:np.zeros(6,dtype=np.int64));feature_events=defaultdict(set)
    event_counts=defaultdict(lambda:np.zeros(8,dtype=np.int64))
    process_counts=defaultdict(lambda:np.zeros(4,dtype=np.int64));null_counts=defaultdict(lambda:np.zeros(2,dtype=np.int64))
    available_counts=np.zeros(5,dtype=np.int64);index=[];carry=None
    next_anchor=int(np.ceil(meta['min_time']/CFG['anchor_stride'])*CFG['anchor_stride']);t0=time.monotonic();anchor_count=0
    for part,item in enumerate(meta['normalized_index']):
        n=pd.read_parquet(source/item['file']);t=n.time.to_numpy();end=n.segment_id.map(segend).to_numpy()
        lab,delta,ids,_,_=make_labels(t,events,tool,meta['max_time'],end)
        for key,s in lab.items():
            counts[('raw_rows',key)]+=np.bincount(s,weights=n.duplicate_n,minlength=4).astype(np.int64)
            policy,k,h=key.split('__')
            if k!='any':represented[('raw_rows',key)].update(ids[k][s==POS])
            else:
                for f in FAULTS:represented[('raw_rows',key)].update(ids[f][(s==POS)&(lab[f'{policy}__{f}__{h}']==POS)])
        for r in own.itertuples():
            e=event_stats[r.event_id];p=np.searchsorted(t,r.fault_time,side='left')
            if p:e['last_sensor']=float(t[p-1])
            for w in [60,300,600,900,1800]:e['pre_'+str(w)]+=int(p-np.searchsorted(t,r.fault_time-w,side='right'))
            for h in CFG['horizons']:e['warning_sensor_'+str(h)]+=int(np.searchsorted(t,r.fault_time-CFG['minimum_lead'],side='right')-np.searchsorted(t,r.fault_time-h,side='left'))
        a=np.arange(next_anchor,int(t[-1])+1,CFG['anchor_stride'],dtype=np.int64)
        if not len(a):carry=n.tail(1000);continue
        context=pd.concat([carry,n],ignore_index=True) if carry is not None else n
        f=calculate(context,a);next_anchor=int(a[-1]+CFG['anchor_stride']);anchor_count+=len(f)
        latest=f.segment_id.map(segend).fillna(-np.inf).to_numpy()
        al,ad,aid,tag,_=make_labels(a,events,tool,meta['max_time'],latest)
        for k in FAULTS:f['next_event_'+k]=aid[k];f['ttf_'+k]=ad[k];f['tag_conflict_'+k]=tag[k]
        f=pd.concat([f,pd.DataFrame({'label__'+key:s for key,s in al.items()})],axis=1)
        for key,s in al.items():
            counts[('anchor',key)]+=np.bincount(s,minlength=4)
            policy,k,h=key.split('__')
            types=list(FAULTS) if k=='any' else [k]
            for code in types:represented[('anchor',key)].update(aid[code][(s==POS)&(al[f'{policy}__{code}__{h}']==POS)])
            for cohort in ['maximum','common']:
                for family in 'ABC':
                    mask=f[family+'_available'].to_numpy() if cohort=='maximum' else f.common_available.to_numpy()
                    values=np.bincount(s[mask],minlength=4)
                    fk=(key,family,cohort);feature_counts[fk]+=np.r_[values,mask.sum(),((~f.A_available)&(~mask)).sum()]
                    for code in types:feature_events[fk].update(aid[code][mask&(s==POS)&(al[f'{policy}__{code}__{h}']==POS)])
        available_counts+=np.array([f.A_available.sum(),f.B_available.sum(),f.C_available.sum(),f.history_60.sum(),f.common_available.sum()])
        for (recipe,stage),group in f.groupby(['recipe','stage'],dropna=False):
            key=(str(recipe),str(stage));process_counts[key]+=np.array([len(group),group.A_available.sum(),group.B_available.sum(),group.C_available.sum()])
        for k in FAULTS:
            for s in [NEG,POS,NA,EXCLUDE]:
                for missing in [False,True]:
                    mask=(al[f'conservative__{k}__h600']==s)&(np.isnan(ad[k])==missing)
                    null_counts[(k,s,missing)]+=np.array([mask.sum(),len(a)])
        for r in own.itertuples():
            for h in CFG['horizons']:
                low=np.searchsorted(a,r.fault_time-h,side='left');high=np.searchsorted(a,r.fault_time-CFG['minimum_lead'],side='right')
                if low==high:continue
                match=aid[r.fault_type][low:high]==r.event_id
                for policy in CFG['policies']:
                    s=al[f'{policy}__{r.fault_type}__h{h}'][low:high]
                    positive=(s==POS)&match
                    for family in 'ABC':
                        mask=f[family+'_available'].to_numpy()[low:high]
                        key=(r.event_id,policy,h,family)
                        event_counts[key]+=np.array([high-low,positive.sum(),(positive&mask).sum(),f.A_available.iloc[low:high].sum(),f.history_60.iloc[low:high].sum(),f.history_300.iloc[low:high].sum(),f.history_900.iloc[low:high].sum(),(s==EXCLUDE).sum()])
        # 파생 특징만 float32. 원본/정규화 센서와 시간 정밀도는 유지한다.
        for c in schema()['C']:f[c]=f[c].astype('float32')
        name=f'anchors-{part:04d}.parquet';write_parquet(f,dest/name);index.append({'file':name,'start':int(a[0]),'end':int(a[-1]),'rows':len(f)})
        carry=context[context.time>t[-1]-max(CFG['history_windows'])].copy()
        if carry.empty:carry=context.tail(1).copy()
        if part%5==0:print(f'{tool}: Anchor {anchor_count:,} / {time.monotonic()-t0:.1f}s',flush=True)
    rows=[]
    for (level,key),values in counts.items():
        p,k,h=key.split('__');rows.append({'Tool':tool,'sample_level':level,'policy':p,'fault_type':k,'H':int(h[1:]),'L':CFG['minimum_lead'],'positive':values[POS],'negative':values[NEG],'NA':values[NA],'excluded':values[EXCLUDE],'denominator':int(values.sum()),'positive_events':len(represented[(level,key)]),'negative_assumption':p!='conservative'})
    write_parquet(pd.DataFrame(rows),dest/'labels_summary.parquet')
    rows=[]
    for (key,family,cohort),v in feature_counts.items():
        p,k,h=key.split('__');rows.append({'Tool':tool,'policy':p,'fault_type':k,'H':int(h[1:]),'feature_set':family,'cohort':cohort,'feature_count':len(schema()[family]),'available_anchors':v[4],'positive_anchors':v[POS],'negative_anchors':v[NEG],'NA_anchors':v[NA],'excluded_anchors':v[EXCLUDE],'positive_events':len(feature_events[(key,family,cohort)]),'current_unavailable_anchors':v[5],'total_anchors':anchor_count})
    write_parquet(pd.DataFrame(rows),dest/'feature_summary.parquet')
    rows=[]
    for r in own.itertuples():
        e=event_stats[r.event_id]
        for policy in CFG['policies']:
            for h in CFG['horizons']:
                for family in 'ABC':
                    v=event_counts[(r.event_id,policy,h,family)]
                    reason='보존됨' if v[2] else ('태그 이력 격리' if r.tagged else '예측 구간 Anchor 없음' if not v[0] else '다른 고장 임박/라벨 근거 불충족' if not v[1] else '현재 센서 없음/오래됨/충돌' if not v[3] else '300 raw 이력 부족' if family in 'BC' and not v[5] else '900 raw 이력 부족' if family=='C' and not v[6] else '유효 특징과 양성 시점 불일치')
                    rows.append({'event_id':r.event_id,'Tool':tool,'fault_type':r.fault_type,'fault_time':r.fault_time,'policy':policy,'H':h,'feature_set':family,'warning_anchors':v[0],'label_positive_anchors':v[1],'usable_positive_anchors':v[2],'current_available':v[3],'history_60':v[4],'history_300':v[5],'history_900':v[6],'imminent_excluded':v[7],'preserved':bool(v[2]),'reason':reason})
    exclusions=pd.DataFrame(rows);write_parquet(exclusions,dest/'event_exclusion.parquet')
    erows=[]
    for r in own.itertuples():
        e=event_stats[r.event_id];selected=exclusions[exclusions.event_id.eq(r.event_id)&exclusions.policy.eq('conservative')&exclusions.H.eq(600)&exclusions.feature_set.eq('C')]
        erows.append({**r._asdict(),**e,'sensor_pre1800':e['pre_1800']>0,**{'coverage_pre'+str(w):e['pre_'+str(w)]/(w/CFG['nominal_interval']) for w in [60,300,900]},'last_sensor_lag':r.fault_time-e['last_sensor'],'label_status':'POS 후보 존재' if len(selected) and selected.label_positive_anchors.iloc[0]>0 else '미확인/제외/후보 없음','exclusion_reason_C_h600':selected.reason.iloc[0]})
    write_parquet(pd.DataFrame(erows),dest/'event_summary.parquet')
    pc=[{'Tool':tool,'recipe':k[0],'stage':k[1],'anchors':v[0],'A_available':v[1],'B_available':v[2],'C_available':v[3]} for k,v in process_counts.items()]
    write_parquet(pd.DataFrame(pc),dest/'process_summary.parquet')
    nc=[{'Tool':tool,'fault_type':k[0],'status':int(k[1]),'TTF_null':bool(k[2]),'anchors':int(v[0])} for k,v in null_counts.items()]
    write_parquet(pd.DataFrame(nc),dest/'null_label_summary.parquet')
    dump_json(marker,{'Tool':tool,'version':ANALYSIS_VERSION,'config_hash':signature,'index':index,'anchors':anchor_count,'available_A':int(available_counts[0]),'available_B':int(available_counts[1]),'available_C':int(available_counts[2]),'history_60':int(available_counts[3]),'common':int(available_counts[4]),'elapsed_seconds':time.monotonic()-t0})
    print(f'{tool}: 라벨/특징 가용성 완료 / {time.monotonic()-t0:.1f}s',flush=True)

def event_input_availability(tool,events):
    """같은 Anchor에서 모든 조건을 충족하는 이벤트 수를 계산한다."""
    a=json.loads((CACHE/'analysis'/tool/'manifest.json').read_text(encoding='utf-8'))
    d=pd.concat([pd.read_parquet(CACHE/'analysis'/tool/x['file'],columns=['time','A_available','B_available','C_available']) for x in a['index']],ignore_index=True)
    times=d.time.to_numpy();counts={}
    for h in CFG['horizons']:
        ranges=[(np.searchsorted(times,t-h,'left'),np.searchsorted(times,t-CFG['minimum_lead'],'right')) for t in events.fault_time]
        for family in 'ABC':
            available=d[family+'_available'].to_numpy()
            counts[(h,family)]=sum(bool(available[lo:hi].any()) for lo,hi in ranges)
    return counts

def consolidate(events):
    tools=sorted(p.parent.name for p in (CACHE/'analysis').glob('*/manifest.json'))
    mapping={'labels_summary':'label_status_summary','feature_summary':'feature_availability_summary','event_exclusion':'event_exclusion_reasons','event_summary':'fault_event_summary','process_summary':'process_feature_summary','null_label_summary':'ttf_label_relationship'}
    for source,name in mapping.items():save_table(pd.concat([pd.read_parquet(CACHE/'analysis'/t/(source+'.parquet')) for t in tools],ignore_index=True),name)
    segments=pd.concat([pd.read_parquet(CACHE/'source'/t/'segments.parquet') for t in tools],ignore_index=True);save_table(segments,'time_segment_summary')
    exclusions=pd.read_parquet(TABLES/'event_exclusion_reasons.parquet');rows=[];preserve=[]
    for t in tools:
        m=manifest(t);a=json.loads((CACHE/'analysis'/t/'manifest.json').read_text(encoding='utf-8'));own=events[events.Tool.eq(t)]
        default=exclusions[exclusions.Tool.eq(t)&exclusions.policy.eq('conservative')&exclusions.H.eq(600)&exclusions.feature_set.eq('C')]
        rows.append({'Tool':t,'raw_rows':m['raw_rows'],'normalized_rows':m['normalized_rows'],'time_start':m['min_time'],'time_end':m['max_time'],'fault_events':len(own),'raw_fault_rows':int(own.source_row_count.sum()),'duplicate_rows':m['duplicate_rows'],'duplicate_times':m['duplicate_times'],'conflict_times':m['conflict_times'],'long_gaps':m['long_gaps'],'max_gap':m['max_gap'],'anchors':a['anchors'],'available_A':a['available_A'],'available_B':a['available_B'],'available_C':a['available_C'],'usable_events_C_h600_conservative':int(default.preserved.sum()),**{'fault_'+k:int(own.fault_type.eq(k).sum()) for k in FAULTS},**{'ttf_null_ratio_'+k:m['ttf_null'][k]/m['raw_rows'] for k in FAULTS}})
        fs=pd.read_parquet(CACHE/'analysis'/t/'feature_summary.parquet');ev=pd.read_parquet(CACHE/'analysis'/t/'event_summary.parquet')
        input_events=event_input_availability(t,own)
        for p in CFG['policies']:
            for h in CFG['horizons']:
                for family in 'ABC':
                    q=fs[fs.policy.eq(p)&fs.H.eq(h)&fs.feature_set.eq(family)&fs.fault_type.eq('any')&fs.cohort.eq('maximum')].iloc[0]
                    ex=exclusions[exclusions.Tool.eq(t)&exclusions.policy.eq(p)&exclusions.H.eq(h)&exclusions.feature_set.eq(family)]
                    stage_rows=[('원본 Sensor',m['raw_rows'],'원본 행'),('시간 정규화',m['normalized_rows'],'고유 시각'),('중복·충돌 처리',m['normalized_rows']-m['conflict_times'],'고유 시각'),('Anchor 생성',a['anchors'],'Anchor'),('현재 센서 가용성',a['available_A'],'Anchor')]
                    if family in 'BC':stage_rows.append(('단기 이력 가용성',a['available_B'],'Anchor'))
                    if family=='C':stage_rows.append(('장기 이력 가용성',a['available_C'],'Anchor'))
                    stage_rows += [('라벨 상태 검사',int(q.positive_anchors+q.negative_anchors),'Anchor'),('연구용 후보',int(q.positive_anchors+q.negative_anchors),'Anchor')]
                    previous=None;previous_unit=None
                    for order,(stage,n,unit) in enumerate(stage_rows):
                        comparable=previous is not None and unit==previous_unit
                        if stage in ['라벨 상태 검사','연구용 후보']:eventn=int(ex.preserved.sum())
                        elif stage=='장기 이력 가용성':eventn=input_events[(h,'C')]
                        elif stage=='단기 이력 가용성':eventn=input_events[(h,'B')]
                        elif stage=='현재 센서 가용성':eventn=input_events[(h,'A')]
                        else:eventn=int(ev['warning_sensor_'+str(h)].gt(0).sum())
                        reasons={'원본 Sensor':'원본 입력','시간 정규화':'동시각 평균으로 단위 변경; 원본 별도 보존','중복·충돌 처리':'동일 시각 공정 충돌을 입력 후보에서 제외; 원본 보존','Anchor 생성':'60 raw 격자로 단위 변경; 공백 Anchor 보존','현재 센서 가용성':'관측 없음/12 raw 초과/공정 충돌/센서 비유한','단기 이력 가용성':'300 raw 연속 이력·90% 관측률·유한 특징 미충족','장기 이력 가용성':'900 raw 연속 이력·90% 관측률·유한 특징 미충족','라벨 상태 검사':'선택 정책/H에서 NA 또는 EXCLUDE','연구용 후보':'추가 제외 없음; 운영용 확정 정답 아님'}
                        preserve.append({'Tool':t,'policy':p,'H':h,'feature_set':family,'stage_order':order,'stage':stage,'unit':unit,'input_rows':previous if previous is not None else n,'input_unit':previous_unit or unit,'output_rows':n,'excluded_rows':previous-n if comparable else np.nan,'exclusion_reason':reasons[stage],'denominator_changed':not comparable,'source_events':len(own),'events_with_sensor':int(ev['warning_sensor_'+str(h)].gt(0).sum()),'events_with_eligible_history':eventn,'positive_anchor_events':int(ex.preserved.sum()) if stage in ['라벨 상태 검사','연구용 후보'] else np.nan,'note':'단위 변경 단계는 행 제외율을 계산하지 않음; A/B/C 독립 분기'})
                        previous=n;previous_unit=unit
    save_table(pd.DataFrame(rows),'tool_summary');save_table(pd.DataFrame(preserve),'preprocessing_preservation')
    dump_json(ROOT/'outputs/feature_schema.json',schema())
    dump_json(ROOT/'outputs/source_snapshot_after.json',snapshot())

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--tools',nargs='*');a=p.parse_args()
    events=pd.read_parquet(TABLES/'events.parquet')
    for tool in a.tools or sorted(events.Tool.unique()):analyze_tool(tool,events)
    consolidate(events)

if __name__=='__main__':main()
