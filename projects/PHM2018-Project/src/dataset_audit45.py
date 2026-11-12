"""전체 라벨의 NA 원인을 집계하고, 준비된 과거 특징의 버전 독립 캐시를 만든다."""
import duckdb
import numpy as np
from src.improvement_common import *

def run():
    initialize();con=duckdb.connect();con.execute("SET threads=4");con.execute("SET memory_limit='3GB'")
    cache=OUT/'cache';cache.mkdir(exist_ok=True)
    events=event_table()[['Tool','time']].drop_duplicates();con.register('events',events)
    all_counts=[]
    for directory in sorted((DEST/'labels').iterdir()):
        if not directory.is_dir():continue
        tool=directory.name;path=cache/f'{tool}.parquet';audit=cache/f'{tool}_audit.json'
        if audit.exists() and path.exists():all_counts.extend(read_json(audit));continue
        labels=(directory/'*.parquet').as_posix();features=(DEST/'features'/tool/'*.parquet').as_posix()
        con.execute(f"CREATE OR REPLACE VIEW labels AS SELECT * FROM read_parquet('{labels}')")
        # 작은 이벤트 표와 결합해 NA 원인의 우선순위를 명시한다. 미래 정보는 감사에만 사용한다.
        rows=[]
        for code in ['f1','f2','f3','any']:
            label=f'label_{code}_h600raw'
            if code!='any':
                reason=f"""CASE WHEN l.{label} IS NOT NULL THEN 'known'
                WHEN l.state_ambiguous THEN 'ambiguous_state'
                WHEN e.time IS NOT NULL THEN 'at_fault_timestamp'
                WHEN l.next_event_{code} IS NULL THEN 'no_later_type_event'
                WHEN NOT l.ttf_verified_{code} THEN 'ttf_not_verified'
                WHEN CAST(split_part(l.next_event_{code},':',2) AS DOUBLE)-l.time<=600 THEN 'gap_before_event'
                ELSE 'incomplete_horizon' END"""
            else:
                reason="""CASE WHEN l.label_any_h600raw IS NOT NULL THEN 'known'
                WHEN l.state_ambiguous THEN 'ambiguous_state'
                WHEN e.time IS NOT NULL THEN 'at_fault_timestamp'
                WHEN l.next_event_f1 IS NULL OR l.next_event_f2 IS NULL OR l.next_event_f3 IS NULL THEN 'one_or_more_type_history_ended'
                ELSE 'gap_or_unverified' END"""
            query=f"""SELECT l.Tool,CASE WHEN l.stage=0 THEN 'Stage=0' ELSE 'Stage>0' END AS state,
            '{code}' AS fault_code,{reason} AS reason,count(*) AS rows,
            count(*) FILTER(WHERE l.{label}=1) AS positive,count(*) FILTER(WHERE l.{label}=0) AS negative,
            count(*) FILTER(WHERE l.{label} IS NULL) AS unknown
            FROM labels l LEFT JOIN events e ON l.Tool=e.Tool AND l.time=e.time GROUP BY ALL"""
            rows.extend(con.execute(query).df().to_dict('records'))
        cols=[f'CAST(f."{c}" AS FLOAT) AS "{c}"' for c in configs()['C']]
        meta=['f.Tool','f.time','f.row_id','f.segment_id','f.history_min_time','f.source_max_time',
              'l.label_any_h600raw']+[f'l.label_{c}_h600raw' for c in ['f1','f2','f3']]+[f'l.next_event_{c}' for c in ['f1','f2','f3']]
        if not path.exists():
            con.execute(f"COPY (SELECT {','.join(cols+meta)} FROM read_parquet('{features}') f JOIN labels l USING(row_id) WHERE f.feature_ready ORDER BY f.time) TO '{path.as_posix()}' (FORMAT PARQUET,COMPRESSION ZSTD)")
        save_json(audit,rows);all_counts.extend(rows);progress(f'NA 감사·float32 캐시 완료: {tool}')
    import pandas as pd
    frame=pd.DataFrame(all_counts);frame.to_parquet(OUT/'na_audit.parquet',index=False)
    sizes=con.execute(f"SELECT Tool,count(*) AS rows,count(*) FILTER(WHERE {TARGET}=1) AS positive,count(*) FILTER(WHERE {TARGET}=0) AS negative,count(*) FILTER(WHERE {TARGET} IS NULL) AS unknown FROM read_parquet('{cache.as_posix()}/*.parquet') GROUP BY Tool ORDER BY Tool").df()
    sizes.to_parquet(OUT/'cache_counts.parquet',index=False)
    # 라벨 완화는 완전한 고장 이력이라는 추가 가정이 필요하므로 실행하지 않는다.
    save_json(OUT/'label_policy.json',{'old':'ml_v1','labels_changed':0,'new_dataset_created':False,'reason':'TTF Null은 이후 동일 유형 기록이 없다는 뜻이며 고장 부재의 충분한 증거가 아니다. 공백을 건넌 양성·음성 확대도 근거 부족으로 보류한다.',
       'improvement':'라벨은 보존하고 NA 사유를 분리한다. 기존 training의 다른 타깃 알려짐 필터를 제거한 모든 feature_ready anchor를 추론 캐시에 보존한다. NA는 학습 제외, 경보량은 별도 측정한다.',
       'full_counts':frame.groupby('fault_code')[['rows','positive','negative','unknown']].sum().reset_index().to_dict('records'),'cache_rows':int(sizes.rows.sum())})
    con.close()

if __name__=='__main__':run()
