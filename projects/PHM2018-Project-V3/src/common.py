"""설정, 출처 기록, 원본 이벤트 식별. 외부 통신 없음."""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import importlib.metadata
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/.mplconfig'))
CFG = json.loads((ROOT/'config/preprocessing.json').read_text(encoding='utf-8'))
RAW = Path(CFG['raw_root'])
CACHE = ROOT/'data/cache'
TABLES = ROOT/'outputs/tables'
SENSORS = ['FLOWCOOLPRESSURE','FLOWCOOLFLOWRATE','IONGAUGEPRESSURE','ETCHBEAMCURRENT','ETCHBEAMVOLTAGE','ETCHSUPPRESSORCURRENT']
SENSOR_KO = ['Flowcool 압력','Flowcool 유량','챔버 압력','빔 전류','빔 전압','억제 전류']
FAULTS = {'low':'FlowCool Pressure Dropped Below Limit','high':'Flowcool Pressure Too High Check Flowcool Pump','leak':'Flowcool leak'}
FAULT_KO = {'low':'저압','high':'고압','leak':'누출','any':'통합'}
POLICY_KO = {'conservative':'보수적','research':'제한 연구용','sensitivity':'완전 이력 가정'}
COLORS = {'POS':'#d55e00','NEG':'#0072b2','NA':'#929aa4','EXCLUDE':'#7b4ab5'}
CONTEXT = ['recipe','stage','recipe_step','Lot','runnum']
pa.set_cpu_count(CFG['threads'])
pa.set_io_thread_count(CFG['threads'])

def dump_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    def convert(x):
        if isinstance(x,(np.integer,np.floating,np.bool_)): return x.item()
        if isinstance(x,Path): return str(x)
        raise TypeError(str(type(x)))
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=convert),encoding='utf-8')

def save_table(frame, name):
    TABLES.mkdir(parents=True,exist_ok=True)
    frame.to_csv(TABLES/(name+'.csv'),index=False,encoding='utf-8-sig')
    frame.to_parquet(TABLES/(name+'.parquet'),index=False,compression='zstd')

def write_parquet(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),path,compression='zstd',row_group_size=CFG['row_group_rows'])

def snapshot():
    return {str(p.relative_to(RAW)).replace('\\','/'):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sorted(RAW.rglob('*.csv'))}

def environment():
    packages={}
    for x in ['pandas','numpy','pyarrow','duckdb','matplotlib','seaborn','nbformat','nbclient','ipykernel']:
        try: packages[x]=importlib.metadata.version(x)
        except importlib.metadata.PackageNotFoundError: packages[x]=None
    return {'python':sys.executable,'python_version':platform.python_version(),'packages':packages,'seed':CFG['seed'],'threads':CFG['threads'],'time_zone':'Asia/Seoul'}

def events_from_raw():
    frames=[]
    for path in sorted((RAW/'train/train_faults').glob('*.csv')):
        d=pd.read_csv(path,dtype={'time':'float64','Tool':'string','fault_name':'string'})
        d['source_file']=path.name;d['source_row']=np.arange(2,len(d)+2)
        frames.append(d)
    raw=pd.concat(frames,ignore_index=True)
    records=[]
    for (tool,t,name),g in raw.groupby(['Tool','time','fault_name'],sort=True):
        canonical=name.replace(' [NoWaferID]','')
        code=next(k for k,v in FAULTS.items() if v==canonical)
        loc=';'.join(f'{r.source_file}:{r.source_row}' for r in g.itertuples())
        key=f'{tool}|{t:.12g}|{name}|{loc}'
        records.append({'event_id':f'{tool}-{code}-'+hashlib.sha256(key.encode()).hexdigest()[:12],
                        'Tool':tool,'fault_type':code,'fault_name':name,'fault_time':t,
                        'tagged':'[NoWaferID]' in name,'source_rows':loc,'source_row_count':len(g)})
    return pd.DataFrame(records)

def event_arrays(events, tool, include_tag=False):
    e=events[events.Tool.eq(tool)]
    if not include_tag:e=e[~e.tagged]
    return {k:e[e.fault_type.eq(k)].sort_values('fault_time') for k in FAULTS}

def next_events(times, table):
    a=table.fault_time.to_numpy(float)
    idx=np.searchsorted(a,times,side='left');valid=idx<len(a)
    delta=np.full(len(times),np.nan);ids=np.full(len(times),'',dtype=object)
    delta[valid]=a[idx[valid]]-times[valid]
    ids[valid]=table.event_id.to_numpy()[idx[valid]]
    return delta,ids

def source_signature(tool):
    stem=f'{tool[:2]}_{tool[2:]}_DC_train.csv'
    paths=[RAW/'train'/stem,RAW/'train/train_ttf'/stem,RAW/'train/train_faults'/f'{tool[:2]}_{tool[2:]}_train_fault_data.csv']
    return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in paths}
