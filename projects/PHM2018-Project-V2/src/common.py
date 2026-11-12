from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/preprocessing.json").read_text(encoding="utf-8"))
os.environ.setdefault("POLARS_MAX_THREADS", str(CONFIG["threads"]))
import numpy as np
import polars as pl

RAW = Path(CONFIG["raw_root"])
OUT = ROOT / "outputs"
DEST = ROOT / "data/processed/diagnostic_pilot"
SENSORS = ["FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE", "IONGAUGEPRESSURE",
           "ETCHBEAMCURRENT", "ETCHBEAMVOLTAGE", "ETCHSUPPRESSORCURRENT"]
FAULTS = {"f1": "FlowCool Pressure Dropped Below Limit",
          "f2": "Flowcool Pressure Too High Check Flowcool Pump", "f3": "Flowcool leak"}
SEVEN = ["02M02", "03M01", "04M02", "07M02", "08M02", "09M02", "10M02"]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def source_signature(paths):
    return {str(p): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in paths}


def tool_path(tool):
    return RAW / "train" / f"{tool[:2]}_{tool[2:]}_DC_train.csv"


def fault_table(tool):
    p = RAW / "train/train_faults" / f"{tool[:2]}_{tool[2:]}_train_fault_data.csv"
    frame = pl.read_csv(p).with_row_index("fault_source_row")
    if frame["Tool"].unique().to_list() != [tool]:
        raise ValueError(f"Fault Tool 오류: {tool}")
    return frame.with_columns(
        pl.col("fault_name").str.contains(r"\[NoWaferID\]").alias("tagged"),
        pl.col("fault_name").str.replace(" [NoWaferID]", "", literal=True).alias("canonical_name"),
    )


def event_times(events, code, include_tag=False):
    e = events.filter(pl.col("canonical_name") == FAULTS[code])
    if not include_tag:
        e = e.filter(~pl.col("tagged"))
    return np.sort(e["time"].unique().to_numpy())


def next_delta(t, ft):
    idx = np.searchsorted(ft, t, side="left")
    d = np.full(len(t), np.nan)
    ok = idx < len(ft)
    d[ok] = ft[idx[ok]] - t[ok]
    return d


def candidates(t, delta, verified, conflict, horizon, lead=60):
    """진단값 -1=NA,-2=60 미만 제외. 미래 센서의 연속성을 라벨 근거로 쓰지 않는다."""
    y = np.full(len(t), -1, dtype=np.int8)
    y[verified & (delta >= lead) & (delta <= horizon)] = 1
    y[verified & (delta > horizon)] = 0
    y[np.isfinite(delta) & (delta < lead)] = -2
    y[conflict] = -1
    return y


def counts(y):
    return {"positive": int((y == 1).sum()), "negative": int((y == 0).sum()),
            "NA": int((y == -1).sum()), "excluded_lt60": int((y == -2).sum())}


def release_blockers():
    reasons = []
    if CONFIG["seconds_per_raw_unit"] != 1:
        reasons.append("물리 초 단위/변환 미검증; 현재 창은 원시시간 단위 진단용")
    if not CONFIG["history_completeness_verified"]:
        reasons.append("Fault 이력의 유효 범위와 TTF Null 검열 한계 미확정")
    if not CONFIG["maintenance_intervals_verified"]:
        reasons.append("고장 후 정비 종료 시각/구간 미확정")
    return reasons
