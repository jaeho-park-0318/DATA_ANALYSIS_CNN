"""대용량 CSV의 읽기 전용 청크 집계 및 로컬 EDA. 외부 통신 없음."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "phm_data_challenge_2018"
OUT = ROOT / "reports/local/eda"
SENSORS = ["FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE", "IONGAUGEPRESSURE",
           "ETCHBEAMCURRENT", "ETCHBEAMVOLTAGE", "ETCHSUPPRESSORCURRENT"]
FAULTS = ["FlowCool Pressure Dropped Below Limit",
          "Flowcool Pressure Too High Check Flowcool Pump", "Flowcool leak"]
CHUNK = 200_000
SOURCE_URL = "https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/"


def write_json(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")


def read_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def snapshot():
    return {p.relative_to(DATA).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(DATA.rglob("*")) if p.is_file()}


def signature():
    # 코드/입력 메타데이터 변경 시 이전 캐시를 사용하지 않는다. 전체 파일 해시는 읽지 않는다.
    payload = json.dumps(snapshot(), sort_keys=True).encode() + Path(__file__).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def inventory():
    records = []
    for path in sorted(DATA.rglob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as f:
            columns = next(csv.reader(f))
        records.append({"path": path.relative_to(DATA).as_posix(), "bytes": path.stat().st_size,
                        "columns": columns})
    return records


def load_faults():
    # 전체 fault CSV 합계 약 64 KiB이므로 이 소형 라벨 테이블만 메모리에 보관한다.
    frames = []
    for path in sorted((DATA / "train/train_faults").glob("*.csv")):
        frame = pd.read_csv(path, dtype={"time": "float64", "Tool": "str", "fault_name": "str"})
        frame["source"] = path.name
        frames.append(frame)
    faults = pd.concat(frames, ignore_index=True)
    faults["fault_raw"] = faults["fault_name"]
    # 별도 상태 태그 한 종류만 정규화하며 원문을 보존한다.
    faults["fault_name"] = faults["fault_name"].str.replace(" [NoWaferID]", "", regex=False)
    faults["event_id"] = np.arange(len(faults))
    return faults


def csv_chunks(path, sensors=True):
    cols = ["time", "Tool", "stage", "recipe"] + (SENSORS if sensors else [])
    dtype = {c: "float64" for c in cols if c != "Tool"}
    dtype["Tool"] = "str"
    yield from pd.read_csv(path, usecols=cols, dtype=dtype, chunksize=CHUNK)


class Moments:
    """누락값을 제외한 센서별 요약. 범위/평균은 전체 청크의 정확 집계."""
    def __init__(self):
        self.rows = 0
        self.n = np.zeros(6, dtype=np.int64)
        self.missing = np.zeros(6, dtype=np.int64)
        self.infinite = np.zeros(6, dtype=np.int64)
        self.low = np.full(6, np.inf)
        self.high = np.full(6, -np.inf)
        self.total = np.zeros(6)
        self.total2 = np.zeros(6)

    def add(self, frame):
        x = frame[SENSORS].to_numpy(dtype=float)
        self.rows += len(x)
        self.missing += np.isnan(x).sum(0)
        self.infinite += np.isinf(x).sum(0)
        valid = np.isfinite(x)
        self.n += valid.sum(0)
        z = np.where(valid, x, 0)
        self.total += z.sum(0)
        self.total2 += (z*z).sum(0)
        self.low = np.minimum(self.low, np.where(valid, x, np.inf).min(0))
        self.high = np.maximum(self.high, np.where(valid, x, -np.inf).max(0))

    def records(self):
        result = []
        for j, sensor in enumerate(SENSORS):
            n = int(self.n[j])
            mean = self.total[j]/n if n else np.nan
            variance = max(0, self.total2[j]/n-mean*mean) if n else np.nan
            result.append({"sensor": sensor, "rows": self.rows, "finite": n,
                           "missing": int(self.missing[j]), "missing_ratio": float(self.missing[j]/self.rows),
                           "infinite": int(self.infinite[j]), "min": float(self.low[j]),
                           "max": float(self.high[j]), "mean": float(mean), "std": float(np.sqrt(variance)),
                           "read_dtype": "float64"})
        return result


def overview(force=False):
    """전체 센서 25개와 TTF 20개를 한 번씩 청크 순회한다."""
    sig = signature()
    cache = OUT / "overview.json"
    if not force and cache.exists() and read_json("overview.json").get("signature") == sig:
        print("검증된 입력/코드 메타데이터와 일치하는 로컬 집계 캐시 사용")
        return read_json("overview.json")
    before = snapshot()
    faults = load_faults()
    tool_rows, sensor_rows, groups, intervals, alignment = [], [], [], [], []
    schemas = inventory()
    global_moments = Moments()
    for split in ("train", "test"):
        for path in sorted((DATA / split).glob("*.csv")):
            start = time.monotonic()
            states = {}
            file_faults = faults[faults["source"].str[:6] == path.name[:6]] if split == "train" else faults.iloc[:0]
            align_state = {int(row.event_id): {"event_id": int(row.event_id), "Tool": row.Tool,
                           "fault_name": row.fault_name, "fault_time": float(row.time),
                           "previous_time": -np.inf, "next_time": np.inf, "exact_rows": 0}
                           for row in file_faults.itertuples()}
            for chunk in csv_chunks(path, sensors=split == "train"):
                for tool, part in chunk.groupby("Tool", dropna=False, sort=False):
                    key = str(tool)
                    if key not in states:
                        states[key] = {"rows": 0, "min_time": np.inf, "max_time": -np.inf,
                                       "last": None, "dt": Counter(), "null_time": 0,
                                       "groups": {}, "moments": Moments(), "bad_order": 0}
                    st = states[key]
                    st["rows"] += len(part)
                    t = part["time"].to_numpy()
                    st["null_time"] += int((~np.isfinite(t)).sum())
                    if not np.isfinite(t).all():
                        raise ValueError(f"시간 결측/무한대: {path.name}/{tool}; 순서 분석 중단")
                    st["min_time"] = min(st["min_time"], float(t.min()))
                    st["max_time"] = max(st["max_time"], float(t.max()))
                    delta = np.diff(np.r_[st["last"], t] if st["last"] is not None else t)
                    st["last"] = float(t[-1])
                    st["bad_order"] += int((delta < 0).sum())
                    vals, counts = np.unique(delta, return_counts=True)
                    st["dt"].update({float(v): int(n) for v, n in zip(vals, counts)})
                    if split == "train":
                        st["moments"].add(part)
                        global_moments.add(part)
                    for (recipe, stage), group in part.groupby(["recipe", "stage"], dropna=False, sort=False):
                        gkey = (str(recipe), str(stage))
                        acc = st["groups"].setdefault(gkey, {"n": 0, "sum": np.zeros(6), "valid": np.zeros(6, dtype=int)})
                        acc["n"] += len(group)
                        if split == "train":
                            x = group[SENSORS].to_numpy()
                            valid = np.isfinite(x)
                            acc["sum"] += np.where(valid, x, 0).sum(0)
                            acc["valid"] += valid.sum(0)
                    # 센서 시각의 직전/직후를 모든 이벤트에 대해 추적한다. 행 번호로 결합하지 않는다.
                    ordered = np.sort(t) if np.any(np.diff(t) < 0) else t
                    for a in align_state.values():
                        if a["Tool"] != key:
                            continue
                        target = a["fault_time"]
                        left = int(np.searchsorted(ordered, target, side="left"))
                        right = int(np.searchsorted(ordered, target, side="right"))
                        a["exact_rows"] += right-left
                        if left:
                            a["previous_time"] = max(a["previous_time"], float(ordered[left-1]))
                        if right < len(ordered):
                            a["next_time"] = min(a["next_time"], float(ordered[right]))
            for tool, st in states.items():
                positive = [(v,n) for v,n in st["dt"].items() if v>0]
                mode = max(positive, key=lambda x:x[1])[0] if positive else np.nan
                tool_rows.append({"split": split, "file": path.name, "Tool": tool,
                                  "rows": st["rows"], "min_time": st["min_time"], "max_time": st["max_time"],
                                  "duration_raw": st["max_time"]-st["min_time"], "mode_dt": mode,
                                  "duplicate_adjacent_time": st["dt"].get(0,0),
                                  "time_reversals": st["bad_order"], "null_time": st["null_time"],
                                  "gaps_gt_mode": sum(n for v,n in positive if v>mode)})
                if split == "train":
                    sensor_rows += [dict(row, Tool=tool) for row in st["moments"].records()]
                intervals += [{"split": split, "Tool": tool, "delta_raw": v, "count": n}
                              for v,n in sorted(st["dt"].items())]
                for (recipe,stage), acc in st["groups"].items():
                    row = {"split": split, "Tool": tool, "recipe": recipe, "stage": stage, "rows": acc["n"]}
                    if split == "train":
                        row.update({s: float(acc["sum"][j]/acc["valid"][j]) if acc["valid"][j] else None
                                    for j,s in enumerate(SENSORS)})
                    groups.append(row)
            for a in align_state.values():
                a["previous_gap_raw"] = a["fault_time"]-a.pop("previous_time")
                a["next_gap_raw"] = a.pop("next_time")-a["fault_time"]
                alignment.append(a)
            print(f"{split}/{path.name}: {sum(s['rows'] for s in states.values()):,}행, {time.monotonic()-start:.1f}초", flush=True)
    ttf = scan_ttf(faults)
    result = {"signature": sig, "scope": "전체 train/test 시간·공정 집계; 센서 통계는 train만; 전체 train TTF",
              "chunk_size": CHUNK, "schemas": schemas, "tools": tool_rows,
              "sensor_by_tool": sensor_rows, "sensor_global": global_moments.records(),
              "groups": groups, "intervals": intervals, "fault_alignment": alignment,
              "fault_rows": len(faults), "fault_duplicates": int(faults.duplicated(["Tool","time","fault_raw"]).sum()),
              "fault_raw_counts": faults["fault_raw"].value_counts().to_dict(),
              "fault_counts": faults["fault_name"].value_counts().to_dict(), "ttf": ttf,
              "source_metadata_unchanged": before == snapshot(),
              "versions": {p: importlib.metadata.version(p) for p in ["pandas","numpy","matplotlib","nbformat","nbclient","ipykernel"]}}
    write_json("overview.json", result)
    return result


def scan_ttf(faults):
    """TTF를 다음 동일 유형 fault까지의 시간차와 비교하고 시간 순서를 검사한다."""
    records = []
    for path in sorted((DATA / "train/train_ttf").glob("*.csv")):
        file_faults = faults[faults["source"].str[:6] == path.name[:6]]
        tools = file_faults["Tool"].unique()
        tool = str(tools[0]) if len(tools) == 1 else path.stem
        times = {f: np.sort(file_faults.loc[file_faults.fault_name == f,"time"].unique()) for f in FAULTS}
        acc = {f: {"Tool":tool,"column":"TTF_"+f,"rows":0,"missing":0,"finite":0,"negative":0,
                   "expected_available":0,"matches_next_fault":0,"mismatch":0,"no_future_fault_finite":0,
                   "missing_with_future_fault":0,"missing_without_future_fault":0,
                   "min":np.inf,"max":-np.inf,"max_abs_error":0,
                   "positive_300":0,"positive_600":0,"positive_1800":0} for f in FAULTS}
        row_count, min_time, max_time, last, reverse, duplicates = 0,np.inf,-np.inf,None,0,0
        for chunk in pd.read_csv(path, dtype="float64", chunksize=CHUNK):
            t = chunk.time.to_numpy()
            if not np.isfinite(t).all():
                raise ValueError(f"TTF 시간 결측/무한대: {path}")
            dt=np.diff(np.r_[last,t] if last is not None else t)
            reverse+=int((dt<0).sum()); duplicates+=int((dt==0).sum()); last=float(t[-1])
            row_count+=len(t); min_time=min(min_time,float(t.min())); max_time=max(max_time,float(t.max()))
            for fault, a in acc.items():
                x = chunk["TTF_"+fault].to_numpy()
                a["rows"]+=len(x); valid=np.isfinite(x)
                a["missing"]+=int(np.isnan(x).sum()); a["finite"]+=int(valid.sum())
                a["negative"]+=int((x<0).sum())
                if valid.any():
                    a["min"]=min(a["min"],float(x[valid].min())); a["max"]=max(a["max"],float(x[valid].max()))
                ft=times[fault]; idx=np.searchsorted(ft,t,side="left"); available=idx<len(ft)
                expected=np.full(len(t),np.nan)
                expected[available]=ft[idx[available]]-t[available]
                both=valid & available
                error=np.abs(x[both]-expected[both]); matching=np.isclose(x[both],expected[both],atol=1e-6,rtol=0)
                a["expected_available"]+=int(available.sum()); a["matches_next_fault"]+=int(matching.sum())
                a["mismatch"]+=int((~matching).sum()); a["no_future_fault_finite"]+=int((valid & ~available).sum())
                a["missing_with_future_fault"]+=int((np.isnan(x) & available).sum())
                a["missing_without_future_fault"]+=int((np.isnan(x) & ~available).sum())
                if len(error): a["max_abs_error"]=max(a["max_abs_error"],float(error.max()))
                for horizon in (300,600,1800): a[f"positive_{horizon}"]+=int(((x>0)&(x<=horizon)).sum())
        records += [dict(a, time_min=min_time,time_max=max_time, time_reversals=reverse,
                         adjacent_duplicates=duplicates) for a in acc.values()]
        print(f"TTF/{path.name}: {row_count:,}행",flush=True)
    return records
