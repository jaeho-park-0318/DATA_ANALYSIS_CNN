"""3단계 데이터셋의 명시적 정책과 공통 입출력. 기존 EDA 결과는 변경하지 않는다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.eda_common import ROOT, DATA, SENSORS, FAULTS, load_faults, snapshot, read_json

DEST = ROOT / "data/processed/ml_v1"
REPORT = ROOT / "reports/local/dataset_v1"
HORIZONS = (300, 600, 1800)
WINDOWS = (60, 300)
LOOKBACK = 600
GAP = 12
STRIDE = 60
CHUNK = 200_000
FAULT_CODES = dict(zip(FAULTS, ("f1", "f2", "f3")))
POLICY = {
    "time_unit": "raw_unverified", "physical_minutes_verified": False,
    "horizons_raw": list(HORIZONS), "feature_windows_raw": list(WINDOWS),
    "max_feature_lookback_raw": LOOKBACK, "max_continuity_gap_raw": GAP,
    "feature_anchor_stride_raw": STRIDE,
    "default_event_policy": "deduplicate_Tool_time_raw_type_then_exclude_NoWaferID",
    "negative_evidence": "finite_TTF_matches_next_same_type_event_and_full_horizon_continuous",
    "missing_ttf": "NA", "at_any_fault_timestamp": "NA",
    "duplicate_sensor_time": "same_time_batch_sensor_mean; ambiguous_recipe_stage_quarantined",
    "combined_label": "any_positive=1; all_three_negative=0; otherwise_NA",
    "tool_split": "prefix01-06=train,07-08=validation,09-10=evaluation",
    "time_split": "per_tool_raw_time_60/20/20; purge_full_history_and_1800_future_crossing_boundary",
    "official_test": "local_unlabelled; never_used_for_label_or_feature_selection",
}


def save_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,
                               default=lambda v: v.item() if hasattr(v,"item") else str(v)),encoding="utf-8")


def read_batches(path, columns=None):
    for file in sorted(Path(path).glob("*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=CHUNK,columns=columns):
            yield batch.to_pandas()


def write_part(frame, directory, part):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    table=pa.Table.from_pandas(frame,preserve_index=False)
    pq.write_table(table,directory/f"part-{part:05d}.parquet",compression="zstd",row_group_size=100_000)


def event_table():
    frame=load_faults().drop_duplicates(["Tool","time","fault_raw"]).copy()
    frame["tagged"]=frame.fault_raw.str.contains("[NoWaferID]",regex=False)
    frame["fault_code"]=frame.fault_name.map(FAULT_CODES)
    frame["event_key"]=frame.apply(lambda r:f"{r.Tool}:{int(r.time)}:{r.fault_code}",axis=1)
    return frame


def feature_columns():
    names=["recipe","stage"]
    for s in SENSORS:
        names.append(s+"__current")
        for w in WINDOWS:
            names += [f"{s}__{kind}_{w}raw" for kind in ("mean","std","min","max","delta_mean","rate_mean")]
    return names+["stage_zero_fraction_300raw","stage_change_count_300raw","recipe_change_count_300raw"]


def label_columns():
    return [f"label_{f}_h{h}raw" for f in ("f1","f2","f3","any") for h in HORIZONS]


def tool_split(tool):
    number=int(tool[:2])
    return "train" if number<=6 else "validation" if number<=8 else "evaluation"


def temporal_split(t, low, high):
    cuts=np.array([low+.6*(high-low),low+.8*(high-low)])
    out=np.full(len(t),"purged",dtype=object)
    out[(t-LOOKBACK>=low)&(t+max(HORIZONS)<cuts[0])]="train"
    out[(t-LOOKBACK>=cuts[0])&(t+max(HORIZONS)<cuts[1])]="validation"
    out[(t-LOOKBACK>=cuts[1])&(t+max(HORIZONS)<=high)]="evaluation"
    return out


def fingerprint():
    sources=[Path(__file__),ROOT/"src/build_labels.py",ROOT/"src/build_features.py"]
    digest=hashlib.sha256(json.dumps(snapshot(),sort_keys=True).encode())
    for source in sources:
        if source.exists():digest.update(source.read_bytes())
    return digest.hexdigest()


def ensure_compatible():
    """검증 완료 후 입력·코드가 달라지면 기존 캐시의 조용한 재사용을 막는다."""
    path=REPORT/"provenance.json"
    if path.exists():
        previous=json.loads(path.read_text(encoding="utf-8"))
        if previous["pipeline_fingerprint"]!=fingerprint():
            raise ValueError("원본 메타데이터 또는 파이프라인 코드가 바뀌었습니다. 새 버전 출력 경로에서 재구축하세요.")


def feature_is_ready(frame):
    return frame.feature_ready & np.isfinite(frame[feature_columns()].to_numpy()).all(axis=1)
