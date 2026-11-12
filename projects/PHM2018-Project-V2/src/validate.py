"""저장된 진단 Parquet의 시점/창/분할/이벤트 불변조건 검사."""
from src.common import *
from src.features import feature_names
from src.splits import ROLES


def run():
    findings = []
    for tool in CONFIG["pilot_tools"]:
        df = pl.read_parquet(DEST / f"{tool}.parquet")
        issues = []
        if df["time"].n_unique() != df.height or df.filter(pl.col("time") % 60 != 0).height:
            issues.append("Anchor 고정 격자/고유성 오류")
        if df.filter(pl.col("sensor_time") > pl.col("time")).height:
            issues.append("미래 현재값")
        for w in [300, 900]:
            for s in SENSORS:
                if df.filter((pl.col(f"{s}_w{w}_source_max") > pl.col("time")) |
                             (pl.col(f"{s}_w{w}_source_min") <= pl.col("time") - w)).height:
                    issues.append(f"Rolling 시점 위반: {s}/{w}")
                if df.filter(pl.col(f"{s}_w{w}_source_min") < pl.col("segment_start")).height:
                    issues.append(f"공백 횡단: {s}/{w}")
        event_overlaps = 0
        for code in FAULTS:
            for h in CONFIG["horizons"]:
                pos = df.filter((pl.col(f"candidate_{code}_h{h}raw") == 1) & (pl.col("time_split") != "purged"))
                event_overlaps += pos.group_by(f"next_event_time_{code}").agg(pl.col("time_split").n_unique().alias("n")).filter(pl.col("n") > 1).height
        if event_overlaps:
            issues.append("동일 이벤트 시간 분할 중복")
        bounds = []
        for role in ROLES:
            part = df.filter(pl.col("time_split") == role)
            if part.height:
                bounds.append((role, int(part["time"].min()) - 900, int(part["time"].max()) + 1800))
        if any(a[2] >= b[1] for a, b in zip(bounds, bounds[1:])):
            issues.append("시간 분할 지지 구간 겹침")
        for col in feature_names()["C"]:
            if col in SENSORS or col not in ["recipe", "stage"]:
                if df.filter(pl.col(col).is_infinite() | pl.col(col).is_nan()).height:
                    issues.append(f"비유한 특징: {col}")
        # Fold별/시간별 클래스와 대표 고유 이벤트 수는 무작위 재표본화 없이 집계한다.
        summaries = []
        for scheme in ["time_split"] + [f"fold_{f}" for f in range(5)]:
            for role in ROLES + (["purged"] if scheme == "time_split" else []):
                group = df.filter(pl.col(scheme) == role)
                y = group["candidate_any_h600raw"]
                events = set()
                for code in FAULTS:
                    events.update((code, t) for t in group.filter(pl.col(f"candidate_{code}_h600raw") == 1)[f"next_event_time_{code}"].drop_nulls().unique().to_list())
                summaries.append({"tool": tool, "scheme": scheme, "role": role, "anchors": group.height,
                                  "positive": int((y == 1).sum()), "negative": int((y == 0).sum()),
                                  "null_including_exclusions": y.null_count(), "positive_unique_events": len(events)})
        findings.append({"tool": tool, "issues": issues, "event_split_overlaps": event_overlaps,
                         "time_support_bounds": bounds, "anchor_rows": df.height,
                         "approved_training_rows": int(df["approved_for_training"].sum()), "splits": summaries})
    save(OUT / "validation.json", findings)
    if any(r["issues"] for r in findings):
        raise RuntimeError("진단 검증 실패: outputs/validation.json 확인")
    print("2대 진단 시점/창/이벤트/시간 분할 검증 통과", flush=True)
    return findings
