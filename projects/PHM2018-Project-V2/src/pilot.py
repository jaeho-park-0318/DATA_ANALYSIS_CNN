"""2대만 진단용 특징/후보 라벨 생성. 승인된 학습 target은 생성하지 않는다."""
from __future__ import annotations

import gc
import json

from src.common import *
from src.features import normalize, make_features_batched, feature_names
from src.splits import time_roles, tool_roles


def build_tool(tool):
    p = tool_path(tool)
    original = pl.scan_csv(p, null_values=["null", "NaN"]).select(["time", "stage", "recipe"] + SENSORS).collect(engine="streaming")
    original = original.with_columns(*[pl.col(s).fill_nan(None) for s in SENSORS])
    raw = normalize(original)
    assert raw["raw_count"].sum() == original.height
    original_rows = original.height
    del original
    gc.collect()
    out = make_features_batched(raw)
    events = fault_table(tool)
    tt = pl.read_csv(RAW / "train/train_ttf" / p.name, null_values=["null", "NaN"])
    # 동일 시각 TTF 값이 일치하는지 원본 감사에서 먼저 검증해야 한다.
    tt = tt.unique("time", keep="first").sort("time").rename({"time": "ttf_source_time"})
    evidence = out.select("time").join_asof(tt, left_on="time", right_on="ttf_source_time", strategy="backward")
    t = out["time"].to_numpy()
    age = t - evidence["ttf_source_time"].to_numpy()
    ys = {}; ds = {}; conflicts = {}; additions = []
    summaries = {}
    for code, fault in FAULTS.items():
        d = next_delta(t, event_times(events, code)); ds[code] = d
        di = next_delta(t, event_times(events, code, True))
        conflict = ~np.isclose(d, di, rtol=0, atol=1e-6, equal_nan=True)
        conflicts[code] = conflict
        x = evidence["TTF_" + fault].cast(pl.Float64).to_numpy() - age
        source_verified = np.isfinite(x) & np.isclose(x, d, rtol=0, atol=1e-6)
        # 원본 전체 TTF/Fault 정합성 통과를 전제한다. 알려진 미래 이벤트와
        # 입력 센서 가용성은 독립이며, 관측 공백만으로 미래 이벤트를 지우지 않는다.
        verified = np.isfinite(d)
        additions.extend([pl.Series(f"next_event_time_{code}", np.where(np.isfinite(d), t + d, np.nan)).fill_nan(None),
                          pl.Series(f"ttf_verified_source_{code}", source_verified), pl.Series(f"tag_conflict_{code}", conflict)])
        for h in CONFIG["horizons"]:
            y = candidates(t, d, verified, conflict, h)
            ys[code, h] = y
            name = f"candidate_{code}_h{h}raw"
            additions.extend([pl.Series(name, y).cast(pl.Int8).set(pl.Series(y < 0), None),
                              pl.Series(name + "_excluded_lt60", y == -2)])
            summaries[name] = counts(y)
    any_delta = np.fmin.reduce(list(ds.values()))
    for h in CONFIG["horizons"]:
        matrix = np.column_stack([ys[c, h] for c in FAULTS])
        y = np.where((matrix == 1).any(axis=1), 1, np.where((matrix == 0).all(axis=1), 0, -1)).astype(np.int8)
        y[np.isfinite(any_delta) & (any_delta < 60)] = -2
        y[np.logical_or.reduce(list(conflicts.values()))] = -1
        name = f"candidate_any_h{h}raw"
        additions.extend([pl.Series(name, y).cast(pl.Int8).set(pl.Series(y < 0), None),
                          pl.Series(name + "_excluded_lt60", y == -2)])
        summaries[name] = counts(y)
    additions.extend([pl.Series("time_split", time_roles(t, raw["time"].min(), raw["time"].max())),
                      pl.lit(tool).alias("tool"), pl.lit(False).alias("approved_for_training"),
                      pl.Series("maintenance_status", np.where(t >= events["time"].min(), "post_fault_end_unknown", "before_first_recorded_fault"))])
    additions += [pl.lit(role).alias(fold) for fold, role in tool_roles(tool).items()]
    out = out.with_columns(*additions)
    DEST.mkdir(parents=True, exist_ok=True)
    temp = DEST / f"{tool}.parquet.tmp"
    out.write_parquet(temp, compression="zstd")
    temp.replace(DEST / f"{tool}.parquet")
    # 고장별 분모는 태그 포함/제외를 모두 원문으로 남긴다.
    preservation = []
    for e in events.unique(["time", "fault_name"]).iter_rows(named=True):
        code = next(c for c, f in FAULTS.items() if f == e["canonical_name"])
        raw_n = raw.filter((pl.col("time") >= e["time"] - 600) & (pl.col("time") <= e["time"] - 60)).height
        grid_window = out.filter((pl.col("time") >= e["time"] - 600) & (pl.col("time") <= e["time"] - 60))
        pos = grid_window.filter(pl.col(f"candidate_{code}_h600raw") == 1)
        preservation.append({"tool": tool, "event_time": e["time"], "fault": code, "tagged": e["tagged"],
                             "raw_eligible_times": raw_n, "grid_anchors": grid_window.height,
                             "observed_grid_anchors": grid_window.filter(pl.col("current_observed")).height,
                             "positive_candidate_anchors": pos.height,
                             "history900_positive_anchors": pos.filter(pl.col("feature_status") == "history_span_available").height})
    pl.DataFrame(preservation).write_parquet(DEST / f"{tool}_event_preservation.parquet")
    r = {"tool": tool, "raw_rows": original_rows, "normalized_times": raw.height, "anchors": out.height,
         "feature_status": out.group_by("feature_status").len().to_dicts(), "candidate_counts": summaries,
         "total_unique_events": len(preservation),
         "events_with_raw_window": sum(e["raw_eligible_times"] > 0 for e in preservation),
         "events_with_positive_anchor": sum(e["positive_candidate_anchors"] > 0 for e in preservation),
         "approved_training_rows": 0, "parquet_sha256": digest(DEST / f"{tool}.parquet")}
    save(DEST / f"{tool}_manifest.json", r)
    del raw, out, tt, evidence
    gc.collect()
    print(f"진단 특징 완료 {tool}: {r['anchors']:,} anchors", flush=True)
    return r


def run():
    output = []
    for tool in CONFIG["pilot_tools"]:
        audit = json.loads((OUT / "audit" / f"{tool}.json").read_text(encoding="utf-8"))
        if any(v["exclude_tag_mismatch"] for v in audit["ttf"].values()):
            raise RuntimeError(f"기본 TTF 불일치로 진단 중단: {tool}")
        output.append(build_tool(tool))
    save(DEST / "feature_schema.json", {"sets": feature_names(), "metadata_is_not_input": True,
        "status": "DIAGNOSTIC_ONLY", "sentinels": "후보 라벨은 0/1/null, 60미만 제외는 별도 bool", "blockers": release_blockers()})
    save(OUT / "pilot_summary.json", output)
    return output
