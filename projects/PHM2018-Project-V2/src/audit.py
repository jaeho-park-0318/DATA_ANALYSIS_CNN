"""20대 원본만 읽는 감사. 장비별 체크포인트, SHA256, TTF/정책 교차검증."""
from __future__ import annotations

import gc
import json
import time
from collections import Counter

from src.common import *


def audit_tool(tool):
    started = time.monotonic()
    p = tool_path(tool)
    paths = [p, RAW / "train/train_ttf" / p.name,
             RAW / "train/train_faults" / f"{tool[:2]}_{tool[2:]}_train_fault_data.csv"]
    before = source_signature(paths)
    # 1장비만 적재, 청크 엔진으로 CSV를 파싱한다. 타 장비와 동시 실행하지 않는다.
    df = pl.scan_csv(p, infer_schema_length=10000, null_values=["null", "NaN"]).collect(engine="streaming")
    t = df["time"].to_numpy()
    if not np.isfinite(t).all():
        raise ValueError(f"시간 결측: {tool}")
    if df["Tool"].unique().to_list() != [tool]:
        raise ValueError(f"센서 장비 키 불일치: {tool}")
    dt = np.diff(t)
    if (dt < 0).any():
        raise ValueError(f"시간 역전: {tool}; 정렬 전 원본 감사 필요")
    unique_t, n_at_t = np.unique(t, return_counts=True)
    duplicate_times = unique_t[n_at_t > 1]
    repeated = df.filter(pl.col("time").is_in(pl.Series(duplicate_times).implode()))
    dup_rows = repeated.height - repeated["time"].n_unique()
    duplicate_exact_extra = repeated.height - repeated.unique().height
    conflicting_times = repeated.group_by("time").agg(
        pl.struct(SENSORS + ["stage", "recipe"]).n_unique().alias("variants")
    ).filter(pl.col("variants") > 1).height
    du = np.diff(unique_t)
    gap_i = np.flatnonzero(du > CONFIG["gap_threshold"])
    gaps = [{"start": int(unique_t[i]), "end": int(unique_t[i + 1]), "gap": int(du[i])}
            for i in gap_i[np.argsort(du[gap_i])[-20:]]]
    # V1의 연속 세그먼트 종단을 비교용으로만 만든다. 특징에는 전달하지 않는다.
    starts = np.r_[0, np.flatnonzero(dt > CONFIG["gap_threshold"]) + 1]
    ends = np.r_[starts[1:] - 1, len(t) - 1]
    segment_end = np.repeat(t[ends], ends - starts + 1)
    events = fault_table(tool)
    unknown_types = events.filter(~pl.col("canonical_name").is_in(list(FAULTS.values())))
    if unknown_types.height:
        raise ValueError(f"미확인 고장 유형: {tool}")
    unique_events = events.unique(["Tool", "time", "fault_name"])
    unique_events.write_parquet(OUT / "audit" / f"{tool}_events.parquet")
    tt = pl.read_csv(paths[1], null_values=["null", "NaN"], infer_schema_length=10000)
    if tt.height != df.height or not np.array_equal(tt["time"].to_numpy(), t):
        raise ValueError(f"セン​​サー/TTF行・時刻不整合: {tool}")
    r = {
        "tool": tool, "raw_rows": len(t), "unique_times": len(unique_t),
        "time_min": int(t[0]), "time_max": int(t[-1]), "time_reversals": int((dt < 0).sum()),
        "duplicate_extra_rows": int(dup_rows), "exact_duplicate_extra_rows": duplicate_exact_extra,
        "duplicate_sensor_or_state_conflict_times": conflicting_times,
        "interval_counts": {str(k): int(v) for k, v in sorted(Counter(dt.tolist()).items()) if v > 10},
        "interval_quantiles": np.quantile(du, [0, .5, .9, .99, 1]).tolist(),
        "long_gap_count": len(gap_i), "long_gap_total_raw": int(du[gap_i].sum()), "largest_gaps": gaps,
        "null_counts_all_columns": df.null_count().to_dicts()[0],
        "nonfinite_sensor_counts": {c: int((~np.isfinite(df[c].to_numpy())).sum()) for c in SENSORS},
        "fault_raw_rows": events.height, "fault_unique_events": unique_events.height,
        "fault_duplicate_extra_rows": events.height - unique_events.height,
        "tagged_unique_events": unique_events.filter(pl.col("tagged")).height,
        "event_counts": {c: len(event_times(events, c)) for c in FAULTS},
        "ttf": {}, "distributions_raw": {}, "sensitivity_any": {}, "v1_policy_recomputed": {},
    }
    for state in ["stage", "recipe"]:
        df.group_by(state).len().sort(state).write_parquet(OUT / "audit" / f"{tool}_{state}.parquet")
    sensor_conflict_state = repeated.group_by("time").agg(
        pl.col("stage").n_unique().alias("s"), pl.col("recipe").n_unique().alias("r")
    ).filter((pl.col("s") > 1) | (pl.col("r") > 1)).height
    r["duplicate_state_conflict_times"] = sensor_conflict_state
    v = {}; deltas = {}; conflicts = {}; old_y = {}
    for code, fault in FAULTS.items():
        x = tt["TTF_" + fault].cast(pl.Float64).to_numpy()
        d = next_delta(t, event_times(events, code))
        di = next_delta(t, event_times(events, code, True))
        match = np.isclose(x, d, rtol=0, atol=1e-6, equal_nan=True)
        conflict = ~np.isclose(d, di, rtol=0, atol=1e-6, equal_nan=True)
        verified = np.isfinite(x) & match
        v[code] = verified; deltas[code] = d; conflicts[code] = conflict
        r["ttf"][code] = {
            "null": int(np.isnan(x).sum()), "finite": int(np.isfinite(x).sum()),
            "negative": int((x < 0).sum()), "exclude_tag_mismatch": int((~match).sum()),
            "include_tag_mismatch": int((~np.isclose(x, di, rtol=0, atol=1e-6, equal_nan=True)).sum()),
            "missing_with_future_event": int((np.isnan(x) & np.isfinite(d)).sum()),
            "tag_policy_conflict_rows": int(conflict.sum()),
        }
        for h in CONFIG["horizons"]:
            r["distributions_raw"][f"{code}_h{h}"] = counts(candidates(t, d, verified, conflict, h))
            old = np.full(len(t), -1, dtype=np.int8)
            old[verified & (d > 0) & (d <= h) & (t + d <= segment_end + CONFIG["gap_threshold"])] = 1
            old[verified & (d > h) & (t + h <= segment_end)] = 0
            old_y[code, h] = old
    all_d = np.fmin.reduce(list(deltas.values()))
    all_conflict = np.logical_or.reduce(list(conflicts.values()))
    for h in CONFIG["horizons"]:
        ys = np.column_stack([candidates(t, deltas[c], v[c], conflicts[c], h) for c in FAULTS])
        any_y = np.where((ys == 1).any(axis=1), 1, np.where((ys == 0).all(axis=1), 0, -1)).astype(np.int8)
        any_y[np.isfinite(all_d) & (all_d < CONFIG["minimum_lead"])] = -2
        any_y[all_conflict] = -1
        r["distributions_raw"][f"any_h{h}"] = counts(any_y)
        # 민감도 분석: Fault 이력이 센서 종단까지 완전하다는 미검증 가정.
        # TTF Null이 아닌, 명시한 이력 범위 가설을 근거로 음성 후보를 만든다.
        hypothetical = np.full(len(t), -1, dtype=np.int8)
        hypothetical[(t + h <= t[-1]) & (np.isnan(all_d) | (all_d > h))] = 0
        hypothetical[np.isfinite(all_d) & (all_d >= 60) & (all_d <= h)] = 1
        hypothetical[np.isfinite(all_d) & (all_d < 60)] = -2
        hypothetical[all_conflict] = -1
        r["sensitivity_any"][f"h{h}"] = counts(hypothetical)
        old = np.column_stack([old_y[c, h] for c in FAULTS])
        old = np.where((old == 1).any(axis=1), 1, np.where((old == 0).all(axis=1), 0, -1))
        old[np.isin(t, events["time"].unique().to_numpy())] = -1
        r["v1_policy_recomputed"][f"h{h}"] = counts(old)
        r["v1_policy_recomputed"][f"h{h}"]["changed_raw_rows"] = int((old != any_y).sum())
    r["sha256"] = {str(p): digest(p) for p in paths}
    if source_signature(paths) != before:
        raise RuntimeError("감사 도중 원본 메타데이터 변경")
    r["source_signature"] = before
    r["elapsed_seconds"] = round(time.monotonic() - started, 3)
    del df, tt, old_y, v, deltas, conflicts, segment_end
    gc.collect()
    return r


def run(tools=None):
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    tools = tools or [p.stem[:6].replace("_", "") for p in sorted((RAW / "train").glob("*.csv"))]
    cfg_hash = digest(ROOT / "config/preprocessing.json")
    code_hash = digest(Path(__file__)) + digest(ROOT / "src/common.py")
    result = []
    for tool in tools:
        cache = OUT / "audit" / f"{tool}.json"
        if cache.exists():
            r = json.loads(cache.read_text(encoding="utf-8"))
            if (r.get("config_hash") == cfg_hash and r.get("code_hash") == code_hash
                    and source_signature([Path(p) for p in r["source_signature"]]) == r["source_signature"]):
                result.append(r)
                print(f"감사 캐시 재사용 {tool}", flush=True)
                continue
        r = audit_tool(tool)
        r["config_hash"] = cfg_hash; r["code_hash"] = code_hash
        save(cache, r); result.append(r)
        print(f"감사 {tool}: {r['raw_rows']:,}행, {r['elapsed_seconds']}초", flush=True)
    save(OUT / "audit_summary.json", result)
    return result
