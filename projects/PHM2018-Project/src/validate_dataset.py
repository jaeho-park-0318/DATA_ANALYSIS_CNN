"""라벨·특징·분할을 검증하고 실제 학습용 Parquet 및 집계 보고서를 만든다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.dataset_common import *


def quote(value):return "'"+str(value).replace("'","''")+"'"
def ident(value):return '"'+str(value).replace('"','""')+'"'


def connection():
    conn=duckdb.connect()
    conn.execute("SET memory_limit='2GB'");conn.execute("SET threads=4")
    spill=REPORT/"duckdb_spill";spill.mkdir(parents=True,exist_ok=True)
    conn.execute("SET temp_directory="+quote(spill.as_posix()))
    return conn


def materialize_training(conn, tools):
    """각 target은 NA 제거 후 사용하며, 적어도 한 target이 알려진 준비 행만 저장한다."""
    for tool in tools:
        destination=DEST/"training"/tool/"dataset.parquet"
        if destination.exists():continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        segments=pd.read_parquet(DEST/"segments"/f"{tool}.parquet")
        low=float(segments.start.min());high=float(segments.end.max());c1=low+.6*(high-low);c2=low+.8*(high-low)
        labels=",".join("l."+ident(c) for c in label_columns())
        keys=",".join("l.next_event_"+code for code in ("f1","f2","f3"))
        known=" OR ".join("l."+ident(c)+" IS NOT NULL" for c in label_columns())
        split=f"""CASE WHEN l.split_time='validation' AND f.history_min_time<{c1} THEN 'purged'
                       WHEN l.split_time='evaluation' AND f.history_min_time<{c2} THEN 'purged'
                       ELSE l.split_time END"""
        query=f"""SELECT f.*,l.split_tool,{split} AS split_time,{labels},{keys}
            FROM read_parquet({quote((DEST/'features'/tool/'*.parquet').as_posix())}) f
            JOIN read_parquet({quote((DEST/'labels'/tool/'*.parquet').as_posix())}) l USING(Tool,row_id)
            WHERE f.feature_ready AND ({known})"""
        conn.execute(f"COPY ({query}) TO {quote(destination.as_posix())} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 100000)")
        print("학습용 Parquet 생성:",tool,flush=True)


def aggregate_counts(conn, source, group_columns):
    expressions=[]
    for col in label_columns():
        expressions += [f"count(*) FILTER(WHERE {ident(col)}=1) AS {ident(col+'__positive')}",
                        f"count(*) FILTER(WHERE {ident(col)}=0) AS {ident(col+'__negative')}",
                        f"count(*) FILTER(WHERE {ident(col)} IS NULL) AS {ident(col+'__unknown')}"]
    group=",".join(group_columns)
    frame=conn.execute(f"SELECT {group},count(*) AS rows,{','.join(expressions)} FROM {source} GROUP BY {group}").df()
    records=[]
    for row in frame.to_dict("records"):
        for code in ("f1","f2","f3","any"):
            for horizon in HORIZONS:
                col=f"label_{code}_h{horizon}raw"
                records.append({**{g:row[g] for g in group_columns},"fault_code":code,"horizon_raw":horizon,
                                "rows":row["rows"],"positive":row[col+"__positive"],
                                "negative":row[col+"__negative"],"unknown":row[col+"__unknown"]})
    return records


def validate(tools=None, materialize=True):
    ensure_compatible()
    conn=connection()
    available=sorted(p.name for p in (DEST/"labels").iterdir() if p.is_dir())
    tools=tools or available
    if not all((DEST/"features"/tool).exists() for tool in tools):raise ValueError("특징 구축을 먼저 완료하세요.")
    all_checks=[]
    label_rows=0;feature_rows=0
    for tool in tools:
        labelpath=quote((DEST/"labels"/tool/"*.parquet").as_posix())
        featurepath=quote((DEST/"features"/tool/"*.parquet").as_posix())
        conn.execute(f"CREATE OR REPLACE VIEW l AS SELECT * FROM read_parquet({labelpath})")
        conn.execute(f"CREATE OR REPLACE VIEW f AS SELECT * FROM read_parquet({featurepath})")
        n,distinct=conn.execute("SELECT count(*),count(DISTINCT row_id) FROM l").fetchone()
        nf,distinctf=conn.execute("SELECT count(*),count(DISTINCT row_id) FROM f").fetchone()
        assert n==distinct and nf==distinctf,(tool,"키 중복")
        label_rows+=n;feature_rows+=nf
        assert conn.execute("SELECT count(*) FROM f WHERE source_max_time>time OR history_min_time>time").fetchone()[0]==0
        invalid=0
        for code in ("f1","f2","f3"):
            event_time=f"try_cast(split_part(next_event_{code},':',2) AS DOUBLE)"
            predicates=[]
            for horizon in HORIZONS:
                col=f"label_{code}_h{horizon}raw"
                predicates.append(f"({col}=1 AND (NOT ttf_verified_{code} OR {event_time}<=time OR {event_time}>time+{horizon} OR {event_time}>segment_end_raw+{GAP}))")
                predicates.append(f"({col}=0 AND (NOT ttf_verified_{code} OR {event_time}<=time+{horizon} OR time+{horizon}>segment_end_raw))")
            invalid+=conn.execute(f"SELECT count(*) FROM l WHERE {' OR '.join(predicates)}").fetchone()[0]
        assert invalid==0,(tool,"라벨 근거 불일치")
        checks=[]
        for h in HORIZONS:
            expected=f"CASE WHEN label_f1_h{h}raw=1 OR label_f2_h{h}raw=1 OR label_f3_h{h}raw=1 THEN 1 WHEN label_f1_h{h}raw=0 AND label_f2_h{h}raw=0 AND label_f3_h{h}raw=0 THEN 0 ELSE NULL END"
            checks.append(f"label_any_h{h}raw IS DISTINCT FROM ({expected})")
        assert conn.execute(f"SELECT count(*) FROM l WHERE {' OR '.join(checks)}").fetchone()[0]==0
        # 라벨/TTF/고장 ID를 입력 특징으로 허용하지 않는 명시적 화이트리스트.
        allowed=feature_columns()
        assert not any(any(token in c.lower() for token in ("ttf","label","fault","event","next_","split","segment_end")) for c in allowed)
        assert set(allowed).issubset(set(pq.ParquetFile(next((DEST/"features"/tool).glob("*.parquet"))).schema.names))
        finite=" AND ".join(f"isfinite({ident(c)})" for c in allowed)
        assert conn.execute(f"SELECT count(*) FROM f WHERE feature_ready AND NOT ({finite})").fetchone()[0]==0
        assert conn.execute("SELECT count(*) FROM f LEFT JOIN l USING(Tool,row_id) WHERE l.time IS NULL OR f.time<>l.time").fetchone()[0]==0
        all_checks.append({"Tool":tool,"label_rows":n,"anchor_rows":nf,"passed":True})
        print("라벨·특징 검증 통과:",tool,flush=True)
    if materialize:materialize_training(conn,tools)
    # Parquet를 한 번에 Pandas로 읽지 않고 집계 결과만 DataFrame으로 반환한다.
    paths=[str(DEST/"labels"/tool/"*.parquet") for tool in tools]
    feature_paths=[str(DEST/"features"/tool/"*.parquet") for tool in tools]
    training_paths=[str(DEST/"training"/tool/"dataset.parquet") for tool in tools]
    def path_list(values):return '['+','.join(quote(p.replace('\\','/')) for p in values)+']'
    conn.execute(f"CREATE OR REPLACE VIEW labels AS SELECT *,CASE WHEN state_ambiguous THEN 'ambiguous' WHEN stage=0 THEN 'stage0' WHEN stage>0 THEN 'stage_positive' ELSE 'unknown' END AS state_group FROM read_parquet({path_list(paths)})")
    conn.execute(f"CREATE OR REPLACE VIEW features AS SELECT * FROM read_parquet({path_list(feature_paths)})")
    conn.execute(f"CREATE OR REPLACE VIEW training AS SELECT *,CASE WHEN stage=0 THEN 'stage0' WHEN stage>0 THEN 'stage_positive' ELSE 'unknown' END AS state_group FROM read_parquet({path_list(training_paths)})")
    full_counts=aggregate_counts(conn,"labels",["Tool","state_group"])
    ready_counts=aggregate_counts(conn,"training",["Tool","state_group","split_tool","split_time"])
    policy_impact=[]
    for h in HORIZONS:
        base=f"label_f2_h{h}raw";alt=f"alt_label_f2_h{h}raw"
        row=conn.execute(f"""SELECT count(*) AS rows,
             count(*) FILTER(WHERE {base} IS DISTINCT FROM {alt}) AS changed_rows,
             count(*) FILTER(WHERE alt_include_tag_ttf_conflict) AS conflict_rows,
             count(*) FILTER(WHERE {alt}=1) AS include_positive,
             count(*) FILTER(WHERE {alt}=0) AS include_negative,
             count(*) FILTER(WHERE {alt} IS NULL) AS include_unknown,
             count(*) FILTER(WHERE alt_include_tag_ttf_conflict AND {base} IS NOT NULL) AS quarantine_extra_unknown
             FROM labels""").df().iloc[0].to_dict()
        policy_impact.append({"horizon_raw":h,**row})
    # 동일 이벤트에 대한 양성 anchor가 서로 다른 시간 분할에 걸치는지 확인한다.
    event_groups=[];shared=[]
    for code in ("f1","f2","f3"):
        for scheme in ("split_tool","split_time"):
            for h in HORIZONS:
                col=f"label_{code}_h{h}raw"
                frame=conn.execute(f"SELECT next_event_{code} AS event_key,{scheme} AS split,count(*) AS positive_anchors FROM training WHERE {col}=1 AND {scheme}<>'purged' GROUP BY event_key,{scheme}").df()
                frame["fault_code"]=code;frame["scheme"]=scheme;frame["horizon_raw"]=h
                event_groups+=frame.to_dict("records")
                if len(frame):
                    bad=frame.groupby("event_key")["split"].nunique()
                    shared += [{"scheme":scheme,"fault_code":code,"horizon_raw":h,"event_key":k} for k in bad[bad>1].index]
    assert not shared,"동일 이벤트가 여러 분할의 양성으로 공유됨"
    if event_groups:
        cross_horizon=pd.DataFrame(event_groups).groupby(["scheme","event_key"])["split"].nunique()
        assert (cross_horizon<=1).all(),"예측 구간을 통틀어 동일 이벤트가 분할 간 공유됨"
    intervals=conn.execute("SELECT Tool,split_time,min(history_min_time) AS history_min,max(time+1800) AS target_max,count(*) AS rows FROM training WHERE split_time<>'purged' GROUP BY Tool,split_time").df()
    overlap=[]
    for tool,group in intervals.groupby("Tool"):
        group=group.set_index("split_time")
        for left,right in (("train","validation"),("validation","evaluation"),("train","evaluation")):
            if left in group.index and right in group.index and group.loc[left,"target_max"]>=group.loc[right,"history_min"]:
                overlap.append({"Tool":tool,"left":left,"right":right})
    assert not overlap,"서로 다른 시간 분할의 특징/타깃 시간 범위가 겹침"
    events=event_table();event_state=[]
    for tool in tools:
        summary=json.loads((REPORT/f"labels_{tool}.json").read_text(encoding="utf-8"))
        event_state+=summary["events"]
    state=pd.DataFrame(event_state)
    state["state_group"]=np.where(~state.state_known,"unknown",np.where(state.stage==0,"stage0","stage_positive"))
    event_counts=[]
    overview=read_json("overview.json")
    bounds={r["Tool"]:(r["min_time"],r["max_time"]) for r in overview["tools"] if r["split"]=="train"}
    for row in state.itertuples():
        low,high=bounds[row.Tool];t=row.fault_time
        temporal="train" if t<low+.6*(high-low) else "validation" if t<low+.8*(high-low) else "evaluation"
        event_counts.append({"Tool":row.Tool,"fault_code":row.fault_code,"tagged":row.tagged,"state_group":row.state_group,
                             "event_key":row.event_key,"split_tool":tool_split(row.Tool),"event_time_split":temporal,
                             "inside_sensor_range":bool(low<=t<=high)})
    # 정답 없는 공식 test는 모델 구축과 내부 평가 데이터에 포함하지 않는다.
    original=json.loads((ROOT/"reports/local/inspection.json").read_text(encoding="utf-8-sig"))
    unchanged=snapshot()=={r["path"]:[r["bytes"],r["mtime_ns"]] for r in original["inventory"]}
    assert unchanged,"초기 점검 대비 원본 메타데이터 변경"
    distribution=conn.execute("SELECT Tool,stage,recipe,count(*) AS rows FROM labels GROUP BY Tool,stage,recipe").df()
    sizes={kind:sum(p.stat().st_size for p in (DEST/kind).rglob("*.parquet")) for kind in ("observations","segments","labels","features","training")}
    result={"policy":POLICY,"tools":tools,"checks":all_checks,"label_rows":label_rows,"anchor_rows":feature_rows,
            "feature_ready_rows":int(conn.execute("SELECT count(*) FROM features WHERE feature_ready").fetchone()[0]),
            "training_rows":int(conn.execute("SELECT count(*) FROM training").fetchone()[0]),
            "feature_count":len(feature_columns()),"feature_columns":feature_columns(),"full_counts":full_counts,"ready_counts":ready_counts,
            "policy_impact":policy_impact,"observed_positive_events":event_groups,"fault_events":event_counts,
            "process_distribution":distribution.to_dict("records"),"temporal_intervals":intervals.to_dict("records"),
            "shared_positive_events":shared,"overlapping_time_splits":overlap,"source_metadata_unchanged":unchanged,
            "sizes_bytes":sizes,"all_checks_passed":True,
            "official_test_has_local_labels":False,"physical_minutes_verified":False}
    save_json(REPORT/"dataset_summary.json",result)
    save_json(REPORT/"provenance.json",{"pipeline_fingerprint":fingerprint(),"source_snapshot":snapshot(),
                                       "python":sys.executable,"versions":{p:__import__('importlib.metadata',fromlist=['version']).version(p) for p in ("pandas","numpy","pyarrow","duckdb","nbformat","nbclient","ipykernel")}})
    save_json(DEST/"training_schema.json",{"input_features":feature_columns(),"targets":label_columns(),
                                          "metadata_only":["row_id","Tool","time","segment_id","history_min_time","source_max_time","feature_ready","split_tool","split_time","next_event_f1","next_event_f2","next_event_f3"],
                                          "selection":"target IS NOT NULL; chosen split; optionally stage>0"})
    print(f"검증 완료: label {label_rows:,}, anchor {feature_rows:,}, training {result['training_rows']:,}행",flush=True)
    conn.close()
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--tools",nargs="*");args=parser.parse_args()
    validate(args.tools)
