"""원시 관측 전체에서 과거 특징 계산 후 고정 격자의 Anchor만 선택한다."""
from __future__ import annotations

from src.common import *


def normalize(df):
    """진단용 동시각 평균. 원본 행 수와 상충을 보존하고 상태 상충은 격리한다."""
    out = df.group_by("time").agg(
        *[pl.col(c).mean().alias(c) for c in SENSORS],
        pl.col("stage").first(), pl.col("recipe").first(), pl.len().alias("raw_count"),
        (pl.col("stage").n_unique() > 1).alias("stage_conflict"),
        (pl.col("recipe").n_unique() > 1).alias("recipe_conflict"),
    ).sort("time")
    out = out.with_columns((pl.col("stage_conflict") | pl.col("recipe_conflict") |
                           pl.col("stage").is_null() | pl.col("recipe").is_null()).alias("state_conflict"))
    out = out.with_columns(
        ((pl.col("time").diff().fill_null(CONFIG["gap_threshold"] + 1) > CONFIG["gap_threshold"]) |
         pl.col("state_conflict") | pl.col("state_conflict").shift(1).fill_null(False)
         ).cast(pl.Int64).cum_sum().alias("segment_id"))
    same_segment = pl.col("segment_id") == pl.col("segment_id").shift(1)
    return out.with_columns(
        ((pl.col("stage") != pl.col("stage").shift(1)) & same_segment).fill_null(False).alias("observed_stage_transition"),
        ((pl.col("recipe") != pl.col("recipe").shift(1)) & same_segment).fill_null(False).alias("observed_recipe_transition"),
    )


def feature_names():
    a = SENSORS + ["recipe", "stage"]
    b = a + [f"{s}_w300_{stat}" for s in SENSORS for stat in ["mean", "std", "min", "max"]]
    c = b + [f"{s}_w900_{stat}" for s in SENSORS for stat in ["slope", "delta"]]
    return {"A": a, "B": b, "C": c}


def make_features(raw, stride=None, grid_bounds=None):
    """창은 (t-w,t]. 시점까지의 최신 관측만 asof 연결하고 보간하지 않는다."""
    stride = stride or CONFIG["anchor_stride"]
    lo, hi = grid_bounds or (int(raw["time"].min()), int(raw["time"].max()))
    grid = pl.DataFrame({"time": np.arange(((lo + stride - 1) // stride) * stride, hi + 1, stride, dtype=np.int64)})
    current = grid.join_asof(raw.rename({"time": "sensor_time"}), left_on="time", right_on="sensor_time", strategy="backward")
    current = current.with_columns((pl.col("time") - pl.col("sensor_time")).alias("sensor_age"))
    available = pl.col("sensor_age") <= CONFIG["gap_threshold"]
    current = current.with_columns(
        *[pl.when(available).then(pl.col(c)).otherwise(None).alias(c) for c in SENSORS + ["recipe", "stage"]],
        pl.when(available).then(pl.col("segment_id")).otherwise(-1).alias("segment_id"),
        available.alias("current_observed"),
    )
    # Anchor 행에는 센서값을 넣지 않는다. 전체 원본 관측으로 통계를 계산한다.
    rows = raw.select(["time", "segment_id"] + SENSORS).with_columns(pl.lit(False).alias("is_anchor"))
    anchors = current.select("time", "segment_id").with_columns(pl.lit(True).alias("is_anchor"))
    timeline = pl.concat([rows, anchors], how="diagonal_relaxed").sort(["segment_id", "time", "is_anchor"])
    timeline = timeline.with_columns((pl.col("time") - pl.col("time").min().over("segment_id")).cast(pl.Float64).alias("x"))
    output = current
    for w in [300, 900]:
        expressions = []
        for s in SENSORS:
            value = pl.col(s)
            x = pl.when(value.is_not_null()).then(pl.col("x"))
            expressions.extend([value.count().alias(f"{s}_w{w}_n"),
                                pl.col("time").filter(value.is_not_null()).min().alias(f"{s}_w{w}_source_min"),
                                pl.col("time").filter(value.is_not_null()).max().alias(f"{s}_w{w}_source_max")])
            if w == 300:
                expressions.extend([value.mean().alias(f"{s}_w300_mean"), value.std(ddof=0).alias(f"{s}_w300_std"),
                                    value.min().alias(f"{s}_w300_min"), value.max().alias(f"{s}_w300_max")])
            else:
                n = value.count()
                den = (x * x).sum() - x.sum() * x.sum() / n
                num = (x * value).sum() - x.sum() * value.sum() / n
                expressions.extend([
                    pl.when((n >= 2) & (den > 0)).then(num / den).otherwise(None).alias(f"{s}_w900_slope"),
                    (value.drop_nulls().last() - value.drop_nulls().first()).alias(f"{s}_w900_delta"),
                ])
        roll = timeline.rolling("time", period=f"{w}i", group_by="segment_id", closed="right").agg(*expressions)
        # 입력 행마다 창 하나를 반환한다. 동시각 원본/Anchor의 창은 같다.
        roll = roll.filter(timeline["is_anchor"]).drop("segment_id")
        output = output.join(roll, on="time", how="left", validate="1:1")
    starts = raw.group_by("segment_id").agg(pl.col("time").min().alias("segment_start"))
    output = output.join(starts, on="segment_id", how="left").with_columns(
        (pl.col("time") - pl.col("segment_start")).alias("history_span_raw"),
        pl.when(~pl.col("current_observed")).then(pl.lit("sensor_stale_or_gap"))
        .when(pl.col("state_conflict")).then(pl.lit("state_conflict"))
        .when(pl.col("time") - pl.col("segment_start") < 900).then(pl.lit("history_lt900"))
        .otherwise(pl.lit("history_span_available")).alias("feature_status"),
    )
    # Stage=0의 가동/정지 의미는 확정하지 않는다.
    output = output.with_columns(
        pl.when(pl.col("observed_stage_transition") | pl.col("observed_recipe_transition")).then(pl.lit("observed_process_transition"))
        .when(pl.col("stage") == 0).then(pl.lit("stage_zero_semantics_unknown"))
        .when(pl.col("stage").is_null()).then(pl.lit("unobserved"))
        .otherwise(pl.lit("stage_nonzero")).alias("state_group")
    )
    return output.sort("time")


def make_features_batched(raw, span=600000):
    """각 블록의 900 과거 단위와 직전 관측을 겹쳐 계산한다. 창 경계는 동일하다."""
    times = raw["time"].to_numpy()
    low, high = int(times[0]), int(times[-1])
    parts = []
    for left in range((low // 60) * 60, high + 1, span):
        right = min(left + span - 1, high)
        begin = max(0, int(np.searchsorted(times, left - 900, side="left")) - 1)
        end = int(np.searchsorted(times, right, side="right"))
        if end <= begin:
            continue
        f = make_features(raw.slice(begin, end - begin), grid_bounds=(max(left, low), right))
        parts.append(f.drop("segment_start", "history_span_raw", "feature_status"))
    output = pl.concat(parts)
    starts = raw.group_by("segment_id").agg(pl.col("time").min().alias("segment_start"))
    return output.join(starts, on="segment_id", how="left").with_columns(
        (pl.col("time") - pl.col("segment_start")).alias("history_span_raw"),
        pl.when(~pl.col("current_observed")).then(pl.lit("sensor_stale_or_gap"))
        .when(pl.col("state_conflict")).then(pl.lit("state_conflict"))
        .when(pl.col("time") - pl.col("segment_start") < 900).then(pl.lit("history_lt900"))
        .otherwise(pl.lit("history_span_available")).alias("feature_status"),
    ).sort("time")
