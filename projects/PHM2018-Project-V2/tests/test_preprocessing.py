import unittest

from src.common import *
from src.features import normalize, make_features, make_features_batched, feature_names
from src.splits import time_roles, tool_roles


def series_frame(times):
    return pl.DataFrame({"time": times, "stage": [0] * len(times), "recipe": [1] * len(times),
                         **{c: np.array(times, dtype=float) for c in SENSORS}})


class PreprocessingTests(unittest.TestCase):
    def test_label_boundaries_and_missing(self):
        d = np.array([0, 59, 60, 600, 601, np.nan])
        y = candidates(np.arange(6), d, np.isfinite(d), np.zeros(6, bool), 600)
        np.testing.assert_array_equal(y, [-2, -2, 1, 1, 0, -1])
        y = candidates(np.array([0]), np.array([100]), np.array([True]), np.array([True]), 600)
        self.assertEqual(y[0], -1)

    def test_event_inclusive_and_tail(self):
        np.testing.assert_allclose(next_delta(np.array([0, 60, 61]), np.array([60])), [60, 0, np.nan], equal_nan=True)

    def test_future_invariance_and_exact_grid_window(self):
        times = np.arange(0, 1204, 4)
        a = series_frame(times)
        b = a.with_columns(*[pl.when(pl.col("time") > 600).then(1e8).otherwise(pl.col(s)).alias(s) for s in SENSORS])
        fa = make_features(normalize(a)); fb = make_features(normalize(b))
        self.assertTrue(fa.filter(pl.col("time") <= 600).equals(fb.filter(pl.col("time") <= 600)))
        row = fa.filter(pl.col("time") == 600).to_dicts()[0]
        self.assertAlmostEqual(row[SENSORS[0] + "_w300_mean"], np.mean(np.arange(304, 601, 4)))
        self.assertAlmostEqual(row[SENSORS[0] + "_w900_slope"], 1.0)
        self.assertEqual(row["stage"], 0)
        self.assertEqual(row[SENSORS[0] + "_w300_n"], 75)
        self.assertTrue(fa.equals(make_features(normalize(a))))

    def test_gaps_reset_history_and_no_interpolation(self):
        raw = normalize(series_frame(np.r_[np.arange(0, 304, 4), np.arange(1200, 1504, 4)]))
        f = make_features(raw)
        self.assertIsNone(f.filter(pl.col("time") == 600)[SENSORS[0]][0])
        self.assertEqual(f.filter(pl.col("time") == 1200)[SENSORS[0] + "_w300_n"][0], 1)
        self.assertEqual(f.filter(pl.col("time") == 1200)["history_span_raw"][0], 0)

    def test_duplicate_boundary_and_state_conflict(self):
        df = series_frame([0, 0, 4, 8])
        n = normalize(df)
        self.assertEqual(n["raw_count"].sum(), 4)
        self.assertEqual(n.height, 3)
        conflicting = df.with_columns(pl.Series("stage", [0, 1, 1, 1]))
        self.assertTrue(normalize(conflicting)["state_conflict"][0])

    def test_whitelist_and_group_folds(self):
        self.assertEqual(len(feature_names()["C"]), 44)
        self.assertFalse(set(feature_names()["C"]) & {"time", "tool", "TTF", "Lot", "next_event_time_f1"})
        for number in range(1, 11):
            self.assertEqual(tool_roles(f"{number:02}M01"), tool_roles(f"{number:02}M02"))
        for fold in range(5):
            roles = [tool_roles(f"{i:02}M01")[f"fold_{fold}"] for i in range(1, 11)]
            self.assertEqual(roles.count("evaluation"), 2)
            self.assertEqual(roles.count("early_stopping"), 2)
            self.assertEqual(roles.count("threshold_selection"), 2)

    def test_time_support_purge(self):
        t = np.arange(0, 100001, 60)
        roles = time_roles(t, 0, 100000)
        previous_end = None
        for role in ["train", "early_stopping", "threshold_selection", "evaluation"]:
            x = t[roles == role]
            if previous_end is not None:
                self.assertGreater(x.min() - 900, previous_end)
            previous_end = x.max() + 1800

    def test_release_gate(self):
        self.assertTrue(release_blockers())

    def test_batch_overlap_matches_full(self):
        raw = normalize(series_frame(np.r_[np.arange(2, 2000, 4), np.arange(10002, 13002, 4)]))
        full = make_features(raw)
        batch = make_features_batched(raw, span=1200).select(full.columns)
        self.assertEqual(full.height, batch.height)
        for col in full.columns:
            if full[col].dtype.is_float():
                np.testing.assert_allclose(full[col].to_numpy(), batch[col].to_numpy(), atol=1e-7, equal_nan=True)
            else:
                self.assertTrue(full[col].equals(batch[col]), col)


if __name__ == "__main__":
    unittest.main()
