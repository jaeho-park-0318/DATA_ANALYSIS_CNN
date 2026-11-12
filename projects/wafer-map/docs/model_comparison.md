# Model comparison and evaluation notes

## Evaluation scope

All headline comparisons refer to the original WM-811K Test partition with 7,894 eight-class defect wafers. Earlier CNN/transfer-learning numbers are reported values from the original presentation; the new values come from the saved evaluation JSON and per-wafer predictions. Historical CNNs have not been rerun for this publication.

| Model | Test accuracy | Test Macro F1 |
|---|---:|---:|
| WaferGuard zero-shot | 30.63% | 24.91% |
| Transfer Stage 1 | 59.18% | Not reported |
| Transfer Stage 2 | 63.90% | 64.76% |
| WM-811K CNN baseline | 70.09% | 67.02% |
| Spatial features + boosting | 77.24% | 77.67% |

New minus CNN: accuracy **+7.15 percentage points**, Macro F1 **+10.65 percentage points**. New minus transfer Stage 2: accuracy **+13.34 points**, Macro F1 **+12.91 points**. WaferGuard metrics on its separate source task are excluded from these comparisons.

## New-model per-class results

| Class | Precision | Recall | F1 | Test count |
|---|---:|---:|---:|---:|
| Center | 83.97% | 76.20% | 79.90% | 832 |
| Donut | 62.43% | 80.82% | 70.45% | 146 |
| Edge-Loc | 81.99% | 87.70% | 84.75% | 2,772 |
| Edge-Ring | 91.15% | 81.44% | 86.02% | 1,126 |
| Loc | 73.21% | 63.30% | 67.90% | 1,973 |
| Near-full | 94.79% | 95.79% | 95.29% | 95 |
| Random | 85.34% | 77.04% | 80.98% | 257 |
| Scratch | 48.57% | 66.23% | 56.04% | 693 |

Weighted F1 is 77.43%; mean class recall is 78.57%.

## Recall against earlier models

Earlier values below are rounded values from the original presentation. Donut is shown at the finer precision available in its dedicated comparison. Percentage-point differences computed from rounded numbers are approximate.

| Class | Earlier CNN | Transfer Stage 2 | Spatial boosting |
|---|---:|---:|---:|
| Center | 70.40% | 50.20% | 76.20% |
| Donut | 26.03% | 68.49% | 80.82% |
| Edge-Loc | 84.00% | 81.60% | 87.70% |
| Edge-Ring | 73.80% | 71.30% | 81.44% |
| Loc | 59.20% | 47.60% | 63.30% |
| Near-full | 94.70% | 98.90% | 95.79% |
| Random | 68.90% | 72.80% | 77.04% |
| Scratch | 45.50% | 34.90% | 66.23% |

Donut recall improves substantially, while precision is 62.43%, below the reported CNN value of 71.7%. Scratch recall improves, but its precision remains 48.57%. Transfer Stage 2 has higher Near-full recall than the new model.

## Generalization and duplicate audit

| Model | Reported validation accuracy | Test accuracy | Gap |
|---|---:|---:|---:|
| CNN baseline | 95.29% | 70.09% | 25.20 pp |
| Transfer Stage 2 | 90.70% | 63.90% | 26.80 pp |
| Spatial boosting | 96.96% | 77.24% | 19.73 pp |

The new gap is calculated before rounding. Validation figures for the earlier models refer to their earlier reported validation results. Matching sample identities, selection rules, and final-fit protocols have not been reproduced, so gap reduction cannot be attributed to a single design choice.

- Training/Test lot-name overlap: **0**.
- Test entries with an exact Training-map match: **92**.
- Excluding those entries: **7,802** Test wafers, **77.01%** accuracy and **77.35%** Macro F1.
- Exact-map checks hash the array shape and uint8 pixels; rotated, near-duplicate, and related process instances are not exhaustively excluded.
- Candidate fit/validation is lot-disjoint; two exact-map copies were removed from the candidate fit subset.
- Only one validation fold was used. No repeated-seed confidence interval or five-fold average is reported.

## Remaining errors

| True class → prediction | Count |
|---|---:|
| Loc → Scratch | 311 |
| Loc → Edge-Loc | 251 |
| Edge-Ring → Edge-Loc | 185 |
| Scratch → Loc | 181 |

These counts come from the saved confusion matrix. They indicate opportunities to improve local-cluster versus line-shape discrimination and partial-edge versus full-ring discrimination; they do not identify manufacturing causes.

## Interpreting the improvement

| Observed design change | Plausible benefit | What remains unverified |
|---|---|---|
| Explicit radial, angular, grid, and morphology features | Supplies pattern summaries directly rather than learning all geometry from pixels | Individual feature contributions and interactions |
| Relative coordinates and regional failure rates | May reduce some map-size and die-count sensitivity | Robustness under controlled size, padding, and density changes |
| Sorted density summaries | May reduce some orientation sensitivity | Full rotation invariance is not guaranteed |
| Regularized boosted trees | May suit this feature representation and sample size | Classifier contribution with features and fit samples held fixed |
| Final refit on all original Training | Uses additional fit samples after selection | Benefit relative to the historical checkpoint protocol |

The Donut result is consistent with radial-density information being useful. It does not establish that radial features caused a particular improvement. No feature ablation, calibrated feature-attribution study, or rotation stress test was performed.

## Additional evidence from the GitHub notebook review

`Jupyter_notebook/Untitled9.ipynb` contains the historical lot-disjoint split and CNN/transfer-learning code. Its relevant split uses the first fold of `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`. The CNN fit cell trains on the group-training subset, selects a checkpoint by validation loss, and its Test cell loads that checkpoint. The spatial model instead refits on all 17,625 original Training entries after validation-based selection. The initial splitter configuration is therefore similar, but equal sample membership and equal final-fit conditions have not been independently verified.

This notebook review was performed after the comparison slides were authored. Slide statements about the earlier notebook being unavailable describe the earlier review context. The source is now available, while the earlier checkpoints and per-wafer predictions remain unavailable. No historical result has been replaced by an invented or newly claimed CNN measurement.

## Follow-up experiments

1. Compare CNN and spatial pipelines using identical fit, validation, and Test rows, equivalent final refits, and comparable selection budgets.
2. Repeat across seeds, evaluate all group folds, and report uncertainty.
3. Remove radial, morphology, angular/grid, and sorting components separately; refit each variant.
4. Hold the feature matrix fixed while comparing classifiers.
5. Stress-test rotation, resolution, padding, and defect-density changes; evaluate independent lots.
6. Add a normal/defect detection stage if the intended use includes normal wafers.

These are proposed experiments, not completed results.

## Sources

- [Original project presentation](../data/Wafer%20Map%20Defect%20Analysis%20%26%20Process%20Root%20Cause%20Study.pdf).
- [Historical notebook index](../Jupyter_notebook/README.md).
- [Saved evaluation](../spatial_boosting/results/evaluation.json), [validation candidates](../spatial_boosting/results/validation.json), and [Test predictions](../spatial_boosting/results/test_predictions.csv).
- [New model card](../spatial_boosting/README.md).
