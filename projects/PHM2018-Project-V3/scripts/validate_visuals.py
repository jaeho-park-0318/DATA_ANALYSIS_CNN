"""그림 수치, 실제 구간, 이벤트 식별, Notebook 실행을 독립 검증."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from PIL import Image
from src.common import *
from src.detail import get_detail
from src.query import manifest

def main():
    checks=[]
    def check(name,value):
        checks.append({'check':name,'passed':bool(value)})
        if not value:raise AssertionError(name)
    figures=json.loads((ROOT/'outputs/figure_manifest.json').read_text(encoding='utf-8'));tools=pd.read_parquet(TABLES/'tool_summary.parquet');events=pd.read_parquet(TABLES/'events.parquet');labels=pd.read_parquet(TABLES/'label_status_summary.parquet');features=pd.read_parquet(TABLES/'feature_availability_summary.parquet');ex=pd.read_parquet(TABLES/'event_exclusion_reasons.parquet')
    check('원본에서 재생성한 이벤트 ID와 일치',events_from_raw().event_id.tolist()==events.event_id.tolist())
    for f in figures:
        path=ROOT/f['file'];check(f['file']+' 파일 존재',path.exists())
        with Image.open(path) as im:check(f['file']+' 유효 그림',im.width>=900 and im.height>=400)
        if f['plot_values'] and 'raw_rows' in f['plot_values']:check('원본 행 수 그림과 표 일치',f['plot_values']['raw_rows']==tools.raw_rows.tolist())
        if f['plot_values'] and 'fault_events' in f['plot_values']:check('고장 그림과 원본 고유 이벤트 일치',sum(f['plot_values']['fault_events'])==len(events))
        if f['plot_values'] and 'status_totals' in f['plot_values']:
            p=f['plot_values']['policy'];q=labels[labels.policy.eq(p)&labels.H.eq(600)&labels.fault_type.eq('any')&labels.sample_level.eq('raw_rows')]
            check(p+' 라벨 그림과 독립 표 합계 일치',all(int(q[c].sum())==v for c,v in f['plot_values']['status_totals'].items()))
    for (tool,p,h,family),g in ex.groupby(['Tool','policy','H','feature_set']):
        q=features[features.Tool.eq(tool)&features.policy.eq(p)&features.H.eq(h)&features.feature_set.eq(family)&features.fault_type.eq('any')&features.cohort.eq('maximum')]
        if int(q.positive_events.iloc[0])!=int(g.preserved.sum()):raise AssertionError('특징 표와 이벤트 ID별 보존 합계 불일치')
    check('모든 정책/H/특징의 이벤트 보존 수가 개별 ID 집계와 일치',True)
    for c in json.loads((ROOT/'outputs/case_catalog.json').read_text(encoding='utf-8')):
        d=get_detail(c['Tool'],c['start'],c['end'],c['event_id'] or None)
        for s,z in d['series'].items():
            for key in ['x','raw_x']:
                x=z[key];check(c['case_id']+s+key+' 공백 연결 없음',all(a is None or b is None or b-a<=CFG['gap_threshold'] for a,b in zip(x,x[1:])))
        if c['event_id']:
            e=events[events.event_id.eq(c['event_id'])].iloc[0]
            check(c['case_id']+' 이벤트 유형/시간 일치',d['event']['fault_type']==e.fault_type and d['event']['fault_time']==e.fault_time)
            check(c['case_id']+' 센서 변화 요약에 고장 이후 없음',all(r['last_time'] is None or r['last_time']<=e.fault_time-CFG['minimum_lead'] for r in d['sensor_summary']))
        check(c['case_id']+' 원본 반환 범위와 분모',d['raw_rows']==d['load_audit'][0]['rows_returned'] and d['load_audit'][0]['files_selected']<d['load_audit'][0]['files_total'])
    for tool in tools.Tool:
        a=json.loads((CACHE/'analysis'/tool/'manifest.json').read_text(encoding='utf-8'))
        for item in a['index']:
            d=pd.read_parquet(CACHE/'analysis'/tool/item['file'],columns=['time','last_observation','source_max_60','source_max_300','source_max_900'])
            for col in ['last_observation','source_max_60','source_max_300','source_max_900']:
                if not (d[col].isna()|d[col].le(d.time)).all():raise AssertionError('미래 센서 참조: '+tool)
        check(tool+' 전체 Anchor 특징 출처가 현재 이하',True)
    path=TABLES/'relative_event_profiles.parquet'
    if path.exists():
        r=pd.read_parquet(path);check('상대시간 bin 내 이벤트 중복 없음',not r.duplicated(['Tool','fault_type','recipe','stage','recipe_step','event_id','relative_bin']).any())
        check('상대시간에 고장 이후 입력 없음',r.relative_bin.le(-60).all())
    n=json.loads((ROOT/'outputs/validation/notebook_execution.json').read_text(encoding='utf-8'))
    check('Notebook5개 실제 실행 오류 없음',len(n)==5 and all(x['executed']==x['code_cells'] and x['errors']==0 for x in n))
    import nbformat
    for path in (ROOT/'notebooks').glob('*.ipynb'):
        book=nbformat.read(path,as_version=4)
        check(path.name+' 실제 그림 출력 포함',any('image/png' in o.get('data',{}) for c in book.cells if c.cell_type=='code' for o in c.get('outputs',[])))
    check('원본 메타데이터 유지',snapshot()==json.loads((ROOT/'outputs/source_snapshot_before.json').read_text(encoding='utf-8')))
    dump_json(ROOT/'outputs/validation/visual_checks.json',{'figure_count':len(figures),'case_count':12,'checks':checks})
    print(f'시각화·구간 검증 {len(checks)}개 통과')

if __name__=='__main__':main()
