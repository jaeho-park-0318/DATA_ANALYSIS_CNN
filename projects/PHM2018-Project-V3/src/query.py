"""시간 인덱스와 Parquet 통계로 선택 구간/열만 읽는 로컬 조회."""
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .common import *

def manifest(tool):
    path=CACHE/'source'/tool/'manifest.json'
    if not path.exists():raise ValueError('처리되지 않은 장비: '+str(tool))
    meta=json.loads(path.read_text(encoding='utf-8'))
    if meta['source_signature']!=source_signature(tool):raise ValueError('원본 Sensor/TTF/Fault가 변경되어 캐시 재생성이 필요합니다: '+tool)
    return meta

def read_window(tool,start,end,kind='normalized',columns=None,enforce_limit=True):
    if kind not in ('normalized','raw','anchors'):raise ValueError('조회 종류 오류')
    start=float(start);end=float(end)
    if not np.isfinite([start,end]).all() or end<start:raise ValueError('유효한 시작/종료 시각 필요')
    if enforce_limit and end-start>CFG['max_detail_width']:raise ValueError(f'상세 조회는 {CFG["max_detail_width"]:,} raw 이하로 확대하세요. 전체 범위는 집계 타임라인을 사용합니다.')
    m=manifest(tool)
    if kind=='anchors':
        directory=CACHE/'analysis'/tool
        index=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))['index']
    else:directory=CACHE/'source'/tool;index=m[kind+'_index']
    selected=[i for i in index if i['end']>=start and i['start']<=end]
    frames=[];groups=0;physical_rows=0
    for item in selected:
        path=directory/item['file'];pf=pq.ParquetFile(path)
        col=pf.schema_arrow.get_field_index('time')
        for j in range(pf.num_row_groups):
            rg=pf.metadata.row_group(j);st=rg.column(col).statistics
            if st is None or (st.max>=start and st.min<=end):groups+=1;physical_rows+=rg.num_rows
        # 원본/Anchor 시간은 int64다. float 리터럴의 Arrow 암묵 변환으로
        # 2**24 이상 시각이 float32 안전 범위 오류를 내지 않도록 정수 경계를 쓴다.
        lower=int(np.ceil(start));upper=int(np.floor(end))
        table=pq.read_table(path,columns=columns,filters=[('time','>=',lower),('time','<=',upper)])
        frames.append(table.to_pandas())
    if frames:frame=pd.concat(frames,ignore_index=True)
    else:
        path=directory/index[0]['file'] if index else None
        names=columns or (pq.ParquetFile(path).schema_arrow.names if path else ['time'])
        frame=pd.DataFrame(columns=names)
    audit={'Tool':tool,'kind':kind,'requested_start':start,'requested_end':end,'files_total':len(index),'files_selected':len(selected),'row_groups_selected':groups,'physical_row_group_rows':physical_rows,'rows_returned':len(frame),'filter':'start <= time <= end','note':'압축 row group 단위 읽기 후 요청 시각으로 필터. 관련 없는 파일/row group은 제외.'}
    return frame,audit

def broken_line(times,values,threshold=None):
    """공백 뒤에 None을 삽입하여 선 연결을 금지한다. 0은 보존한다."""
    threshold=CFG['gap_threshold'] if threshold is None else threshold
    x=[];y=[]
    for i,(t,v) in enumerate(zip(times,values)):
        if i and t-times[i-1]>threshold:x.append(None);y.append(None)
        x.append(float(t));y.append(float(v) if np.isfinite(v) else None)
    return x,y
