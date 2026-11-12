"""처리된 TTF의 시작·종료 표본을 이벤트 이력과 다시 대조한다."""
import numpy as np
import pyarrow.parquet as pq
from src.improvement_common import *

def run():
    rows=[];events=event_table();events=events[~events.tagged]
    for directory in sorted((DEST/'observations').iterdir()):
        if not directory.is_dir():continue
        files=sorted(directory.glob('*.parquet'))
        for location,path in [('first',files[0]),('last',files[-1])]:
            pf=pq.ParquetFile(path)
            table=pf.read_row_group(0 if location=='first' else pf.num_row_groups-1,columns=['time','ttf_f1','ttf_f2','ttf_f3'])
            f=table.to_pandas();f=f.head(4096) if location=='first' else f.tail(4096)
            for code in ['f1','f2','f3']:
                ft=np.sort(events.loc[(events.Tool==directory.name)&(events.fault_code==code),'time'].to_numpy())
                t=f.time.to_numpy();idx=np.searchsorted(ft,t,side='left');valid=idx<len(ft)
                expected=np.full(len(t),np.nan);expected[valid]=ft[idx[valid]]-t[valid]
                observed=f['ttf_'+code].to_numpy();mismatch=int((~np.isclose(observed,expected,rtol=0,atol=1e-6,equal_nan=True)).sum())
                rows.append({'Tool':directory.name,'location':location,'fault_code':code,'rows':len(t),'nulls':int(np.isnan(observed).sum()),'mismatches':mismatch})
    assert sum(r['mismatches'] for r in rows)==0
    save_json(OUT/'ttf_recheck.json',{'scope':'각 장비의 처음·마지막 최대 4096 정규화 시각. 전체 검증은 기존 3단계 증거를 재사용하며 이번 대조는 독립 표본 검사다.','rows':rows,'mismatches':0})

if __name__=='__main__':run()
