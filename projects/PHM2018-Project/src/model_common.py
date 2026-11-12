"""4단계 모델 실험의 고정 설정·읽기 전용 데이터 로딩·사전 검증."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from src.dataset_common import (ROOT,DATA,DEST,REPORT,SENSORS,FAULTS,feature_columns,
                                event_table,ensure_compatible,snapshot)

OUT=ROOT/"outputs/stage4"
SEED=42
SCHEMES=("split_tool","split_time")
META=["Tool","time","row_id","segment_id","history_min_time","source_max_time","stage"]
EVENT_COLS=["next_event_"+c for c in ("f1","f2","f3")]


def save_json(path,value):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,np.generic):return clean(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2),encoding="utf-8")


def read_json(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def configs():
    all_cols=feature_columns()
    return {"A":["recipe","stage"]+[s+"__current" for s in SENSORS],
            "B":[c for c in all_cols if c not in ("stage_zero_fraction_300raw","stage_change_count_300raw","recipe_change_count_300raw")],
            "C":all_cols}


def load_split(scheme,horizon,split,columns=None,with_na=False):
    """원본이 아닌 해당 실험의 처리된 후보만 배치로 읽는다. NA를 0으로 채우지 않는다."""
    assert scheme in SCHEMES and horizon in (300,600,1800)
    target=f"label_any_h{horizon}raw"
    cols=columns or feature_columns()
    names=list(dict.fromkeys(cols+META+EVENT_COLS+[target]+[f"label_{c}_h{horizon}raw" for c in ("f1","f2","f3")]))
    files=[str(p) for p in sorted((DEST/"training").rglob("*.parquet"))]
    condition=ds.field(scheme)==split
    if not with_na:condition=condition & ds.field(target).is_valid()
    dataset=ds.dataset(files,format="parquet")
    frames=[batch.to_pandas() for batch in dataset.to_batches(columns=names,filter=condition,batch_size=65536) if batch.num_rows]
    frame=pd.concat(frames,ignore_index=True).sort_values(["Tool","time"],kind="stable").reset_index(drop=True)
    assert (frame.source_max_time<=frame.time).all()
    return frame


def matrix(frame,columns):
    assert set(columns).issubset(set(feature_columns()))
    assert not any(any(v in c.lower() for v in ("ttf","label","event","split","time","lot","usage")) for c in columns)
    x=frame[columns].to_numpy(dtype=np.float32)
    assert np.isfinite(x).all()
    return x


def preflight():
    OUT.mkdir(parents=True,exist_ok=True)
    # 이전 산출물의 내용 해시를 보존하여 이번 작업의 비변경 여부를 검사한다.
    old=[]
    for folder in (ROOT/"src",ROOT/"scripts",ROOT/"notebooks",ROOT/"reports"):
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix in (".py",".ipynb",".md",".json",".txt") and "local" not in p.parts:
                if p.name in ("model_common.py","train_baseline.py","train_xgboost.py","evaluate_models.py","model_plots.py","model_audit.py","build_model_notebooks.py","run_model_experiments.py","write_model_report.py"):continue
                if p.name.startswith(("06_","07_","08_","04_model_training_summary")):continue
                old.append({"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    if not (OUT/"preexisting_files.json").exists():save_json(OUT/"preexisting_files.json",old)
    ensure_compatible()
    summary=read_json(REPORT/"dataset_summary.json")
    assert summary["all_checks_passed"] and not summary["shared_positive_events"] and not summary["overlapping_time_splits"]
    time_checks=[]
    for path in sorted((DATA/"train").glob("*.csv")):
        sensor=pd.read_csv(path,usecols=["time"],nrows=10000)
        ttf=pd.read_csv(DATA/"train/train_ttf"/path.name,nrows=10000)
        assert np.array_equal(sensor.time,ttf.time)
        dt=np.diff(sensor.time)
        for f in FAULTS:
            x=ttf["TTF_"+f].to_numpy()
            same=np.isfinite(x[1:])&np.isfinite(x[:-1])&np.isclose(x[1:]+sensor.time.to_numpy()[1:],x[:-1]+sensor.time.to_numpy()[:-1],rtol=0,atol=1e-6)
            time_checks.append({"file":path.name,"fault":f,"rows":len(sensor),"mode_dt":float(pd.Series(dt[dt>0]).mode().iloc[0]),
                                "same_event_pairs":int(same.sum()),"ttf_decrease_matches_dt":int(np.isclose(x[:-1][same]-x[1:][same],dt[same],rtol=0,atol=1e-6).sum())})
    conn=duckdb.connect();conn.execute("SET memory_limit='1GB'")
    glob=(DEST/"training/*/*.parquet").as_posix()
    conn.execute("CREATE VIEW candidates AS SELECT * FROM read_parquet('"+glob.replace("'","''")+"')")
    distributions=[];events=[];bad_intervals=[]
    for scheme in SCHEMES:
        for h in (300,600,1800):
            label=f"label_any_h{h}raw"
            d=conn.execute(f"SELECT {scheme} AS split,count(*) AS candidates,count(*) FILTER(WHERE {label}=1) AS positive,count(*) FILTER(WHERE {label}=0) AS negative,count(*) FILTER(WHERE {label} IS NULL) AS unknown FROM candidates GROUP BY {scheme}").df()
            distributions += [dict(row,scheme=scheme,horizon=h) for row in d.to_dict("records")]
            for c in ("f1","f2","f3"):
                e=conn.execute(f"SELECT {scheme} AS split,next_event_{c} AS event_key,count(*) AS anchors FROM candidates WHERE label_{c}_h{h}raw=1 AND {scheme}<>'purged' GROUP BY {scheme},next_event_{c}").df()
                assert not len(e) or e.groupby("event_key")["split"].nunique().max()==1
                events += [dict(row,scheme=scheme,horizon=h,fault_code=c) for row in e.to_dict("records")]
        intervals=conn.execute(f"SELECT Tool,{scheme} AS split,min(history_min_time) AS history_min,max(time+1800) AS forecast_max FROM candidates WHERE {scheme}<>'purged' GROUP BY Tool,{scheme}").df()
        for tool,g in intervals.groupby("Tool"):
            g=g.set_index("split")
            for a,b in (("train","validation"),("validation","evaluation"),("train","evaluation")):
                if a in g.index and b in g.index and g.loc[a,"forecast_max"]>=g.loc[b,"history_min"]:bad_intervals.append((scheme,tool,a,b))
    assert not bad_intervals
    result={"time_unit":"seconds_supported_by_primary_research_and_data", "raw_names_retained":True,
            "time_evidence":time_checks,"source_primary_paper":"https://papers.phmsociety.org/index.php/phmconf/article/download/590/phmc_18_590",
            "official_source":"https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/",
            "time_interpretation":"공식 초 예시·동일 데이터 선행 연구의 약 4초 기록·원본 차분/TTF 감소 일치를 종합해 초 해석을 지지한다. 외부 시계 보정은 미검증이며 기존 raw 컬럼명과 정책 파일은 유지한다.",
            "distributions":distributions,"events":events,"overlapping_intervals":bad_intervals,
            "candidate_rows":summary["training_rows"],"normalized_rows":summary["label_rows"],
            "anchors":summary["anchor_rows"],"feature_ready":summary["feature_ready_rows"],
            "representativeness_warning":"일정 주기 anchor·과거 이력·TTF·미래 연속 관측으로 조건화된 표본이다. 전체 가동률/고장률의 무편향 표본이 아니며 공백 내 이벤트를 배제하는 선택 편향이 있다.",
            "sampling":"클래스별 재표본화 없음; 모든 상태에 동일한 60 원시단위 bin의 첫 시각",
            "preprocessing":"주 실험은 유한 원본 특징만 사용, 대치·스케일링 없음; recipe/stage ID는 트리의 숫자 범주로 취급하는 한계",
            "feature_configs":configs(),"versions":{p:importlib.metadata.version(p) for p in ("numpy","pandas","scikit-learn","xgboost","pyarrow","duckdb","joblib")},
            "source_snapshot":snapshot(),"passed":True}
    save_json(OUT/"preflight.json",result);conn.close()
    print("사전 검증 완료: 분할 누수 없음 / 조건부 표본의 대표성 제한 기록",flush=True)
    return result
