"""저장된 Anchor를 필요한 열만 읽어 사유별 가용성을 보강한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from src.common import *
from src.query import manifest

def main():
    fs=pd.read_parquet(TABLES/'feature_availability_summary.parquet');rows=[]
    for tool in sorted(fs.Tool.unique()):
        m=manifest(tool);a=json.loads((CACHE/'analysis'/tool/'manifest.json').read_text(encoding='utf-8'))
        counts={f:dict(current_gap=0,history_gap=0,initial_history=0,sensor_or_conflict=0,other_history=0,available=0,total=0) for f in ['A','B','C','common']}
        cols=['time','current_lag','segment_start','A_available','B_available','C_available','common_available']
        for item in a['index']:
            d=pd.read_parquet(CACHE/'analysis'/tool/item['file'],columns=cols)
            for f,w in [('A',0),('B',300),('C',900),('common',900)]:
                available=d[f+'_available'];gap=d.current_lag.isna()|d.current_lag.gt(CFG['gap_threshold'])
                conflict=(~gap)&(~d.A_available);short=d.time-d.segment_start<w
                history=(~gap)&(~conflict)&(~available)&short&(d.segment_start>m['min_time'])
                initial=(~gap)&(~conflict)&(~available)&short&(d.segment_start<=m['min_time'])
                other=(~available)&~gap&~conflict&~history&~initial
                c=counts[f]
                for key,mask in [('current_gap',gap),('history_gap',history),('initial_history',initial),('sensor_or_conflict',conflict),('other_history',other),('available',available)]:c[key]+=int(mask.sum())
                c['total']+=len(d)
        for f,c in counts.items():
            if sum(v for k,v in c.items() if k!='total')!=c['total']:raise AssertionError('가용성 사유 분모 오류')
            rows.append(dict(Tool=tool,feature_set=f,**c))
    r=pd.DataFrame(rows);save_table(r,'feature_unavailability_reasons')
    fs=fs.drop(columns=['current_gap_anchors','history_gap_anchors'],errors='ignore')
    maximum=r[r.feature_set.ne('common')][['Tool','feature_set','current_gap','history_gap']].copy();maximum['cohort']='maximum'
    common=r[r.feature_set.eq('common')][['Tool','current_gap','history_gap']]
    common=pd.concat([common.assign(feature_set=f,cohort='common') for f in 'ABC'],ignore_index=True)
    fs=fs.merge(pd.concat([maximum,common],ignore_index=True),on=['Tool','feature_set','cohort'],how='left')
    fs=fs.rename(columns={'current_gap':'current_gap_anchors','history_gap':'history_gap_anchors'})
    save_table(fs,'feature_availability_summary')
    dump_json(ROOT/'outputs/code_provenance.json',{'files':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','scripts','dashboard','tests','config'] for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.js','.html','.css','.json']},'source_metadata_only':True,'raw_content_hashes_not_computed':True})
    print('20대 가용성 제외 사유 분모 검증 완료')

if __name__=='__main__':main()
