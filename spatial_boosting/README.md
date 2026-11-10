# Spatial-feature gradient boosting: model card

## Purpose and scope

This model assigns a wafer map to one of eight WM-811K defect classes. It is an alternative to the project's image-based CNN and transfer-learning pipelines. It uses handcrafted spatial features and `HistGradientBoostingClassifier`, without external pretraining or data augmentation.

Class order: `Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Near-full`, `Random`, `Scratch`.

For an executed, step-by-step English walkthrough with feature illustrations and evaluation plots, open [Spatial_Feature_Boosting.ipynb](Spatial_Feature_Boosting.ipynb). Install notebook dependencies using `python -m pip install -r spatial_boosting/requirements-notebook.txt` from the repository root. Its default execution reads the published artifacts and leaves training disabled.

This order is stored in the model bundle. It differs from the WaferGuard order used in the historical project, where Scratch and Random occupy the opposite final positions. Always map predictions by class name.

## Included artifacts

| Path | Purpose |
|---|---|
| `results/spatial_model.joblib` | Fitted estimator, class names, and feature-version identifier |
| `results/model_metadata.json` | Actual fitted parameters, feature count, iteration count, and tree count |
| `results/evaluation.json` | Full classification report, confusion matrix, split counts, and duplicate audit |
| `results/validation.json` | Three candidate validation results |
| `results/test_predictions.csv` | True and predicted class, lot, and source row index for 7,894 Test wafers |
| `results/split_indices.npz` | Fit, validation, Training, and Test positions in the filtered feature matrix |
| `results/verification.txt` | Recorded saved-model consistency and feature-edge-case checks |
| `results/artifact_manifest.json` | SHA-256 hashes for the published model and result artifacts |

The source indices in the CSV refer to the source DataFrame index. The NPZ split positions refer to the 25,519-row filtered matrix, so these two kinds of indices are not interchangeable. Raw wafer maps and the regenerable feature cache are excluded from Git.

## Input and feature extraction

Input is a non-empty 2D array with `0 = outside wafer`, `1 = good die`, and `2 = defective die`. Inference uses an `.npy` file with pickle loading disabled.

| Feature family | Count | Computation |
|---|---:|---|
| Overall failure rate | 1 | Defective die / valid die |
| Radial density | 16 | Local failure rate in normalized radial bins |
| Angular and radial-angular density | 112 | 4/8/16 sectors over the whole wafer and three radial bands; sorted within each group |
| Grid density | 80 | Sorted rates from 4×4 and 8×8 grids |
| Coordinates, radius, covariance | 13 | Center displacement, x/y spread, radial quantiles, and covariance eigenvalues/ratio |
| Connected components and morphology | 28 | Four mask scales × component count, total area, and five largest areas |
| Multiscale smoothed density | 24 | Four Gaussian scales × six quantiles |
| **Total** | **274** | Fixed-length vector |

Coordinate normalization uses the map dimensions, not calibrated physical wafer radius. The last radial bin also collects radii beyond its lower boundary. Regional failure rates exclude outside-wafer cells from their denominators. Empty regions receive a rate of zero.

Morphology uses a nearest-resampled 64×64 binary defect mask. The four versions are the original mask and square openings of sizes 2, 3, and 5; connected components use eight-neighbor connectivity. Gaussian smoothing uses sigma 1, 2, 4, and 8, summarized at quantiles 0.5, 0.75, 0.9, 0.95, 0.99, and 1. These mask-based features treat outside-wafer pixels as zero rather than dividing by valid-die counts.

Sorting removes location order from density lists and may reduce some directional sensitivity, but also loses information. The complete feature vector is **not rotation invariant**. Resampling and opening may remove fine scratches. Lot name, wafer index, split label, and map dimensions as explicit columns are not classifier inputs.

## Training and model selection

1. Retain only the eight defect classes, preserving source row order.
2. Keep the original Training/Test boundary: 17,625 / 7,894.
3. Within Training, take the first split from `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, grouped by lot name.
4. The initial fit/validation counts are 14,100 / 3,525. Remove two fit entries whose exact wafer-map hashes occur in validation, leaving **14,098 fit entries**.
5. Fit Extra Trees, Random Forest, and histogram gradient boosting on that fit set. Select by validation accuracy, using Macro F1 to break a tie.
6. Refit the winning configuration on **all 17,625 original Training entries** and evaluate the original Test entries.

The two removed entries are excluded only during candidate selection. They are part of the final full-Training refit. No Test score is used by the selection code. The five-fold splitter supplies one validation split; a complete five-fold experiment was not performed.

| Candidate | Validation accuracy | Validation Macro F1 |
|---|---:|---:|
| Extra Trees | 96.79% | 93.88% |
| Random Forest | 95.66% | 91.32% |
| **Histogram gradient boosting** | **96.96%** | **93.93%** |

The candidate Random Forest and Extra Trees use class balancing. The selected final gradient boosting model has `class_weight=None`.

## Learning principle and fitted parameters

Each boosting step adds a small tree to correct the current class scores. The loss is multiclass log loss, using gradient/Hessian information to fit updates. Histogram binning groups feature values to search for split candidates. Softmax converts the final class scores to probabilities, and the largest score determines the predicted class.

| Setting | Saved value |
|---|---|
| Estimator | `HistGradientBoostingClassifier` |
| Input dimensions / classes | 274 / 8 |
| Actual boosting iterations | 220 |
| Trees per iteration / total | 8 / 1,760 |
| `learning_rate` | 0.07 |
| `max_leaf_nodes` | 23 |
| `min_samples_leaf` | 20 |
| `l2_regularization` | 3 |
| `max_bins` | 255 |
| `early_stopping` | `False` |
| `class_weight` | `None` |
| `random_state` | 42 |

Leaf-size and L2 constraints limit fitting flexibility; they do not guarantee generalization. The saved scores have not undergone a probability-calibration experiment.

## Reproduce or use the model

The recorded run used **Python 3.12.14 on Windows**. From the repository root, install the exact dependencies in a separate virtual environment:

```bash
python -m pip install -r spatial_boosting/requirements.txt
python spatial_boosting/predict.py path/to/wafer_map.npy
```

To build features and retrain:

```bash
python spatial_boosting/train_spatial.py --dataset data/WM811K.pkl
python spatial_boosting/verify_model.py
```

The pickle must contain `waferMap`, `failureType`, `trainTestLabel`, and `lotName` columns. String and nested singleton labels are supported. Counts are checked against the recorded eight-class Training/Test counts. No GPU or TensorFlow is needed for the spatial model.

The cache at `results/features.npz` is created on first training and reused on later runs. It is not automatically fingerprinted against a new dataset: use `--rebuild-cache` whenever changing the dataset, row order, or feature code. Training overwrites the saved model, evaluation, predictions, and split files. The published manifest and metadata describe the published artifact; regenerate those records if you replace it.

`verify_model.py` requires the feature cache generated by training. It checks that the loaded model reproduces every recorded Test prediction, that fit/validation lots are disjoint, and that 12 constant-map edge cases produce finite, fixed-length features. It does not rerun the historical CNN, establish out-of-distribution performance, or constitute a causal feature study.

The release packaging adds a configurable dataset path and normalizes nested labels; the published model weights and metric files were copied from the completed training run, not retrained during repository publication. Runtime information and checksums are in `results/artifact_manifest.json`. Use compatible pinned dependency versions when loading the serialized estimator. Only load pickle/joblib artifacts from trusted sources.

## Results and limitations

Test accuracy is **77.24%**, Macro F1 **77.67%**, weighted F1 **77.43%**, and mean class recall **78.57%**. See [the full comparison](../docs/model_comparison.md) for per-class results and earlier model scores.

There are no Training/Test lot-name overlaps, but 92 Test entries exactly duplicate a Training map. Removing those Test entries leaves 7,802 entries with accuracy **77.01%** and Macro F1 **77.35%**. Hashes cover shape and pixels; they do not detect rotated or approximately matching maps.

The validation-to-Test accuracy gap is about **19.73 percentage points**. Loc/Scratch confusion remains substantial. Unknown, normal, mixed-defect, cross-factory, and future-lot performance is unmeasured. A map-pattern label cannot establish a process root cause.

## Why it may outperform the earlier CNN

Spatial summaries directly expose center/ring/edge concentration, clustering, and elongation. This may make the current task easier to learn with the available sample size. Relative coordinates and density normalization may also reduce some variations in map size and die count. However, feature importance, feature removal, rotation robustness, and equal-training-budget comparisons have not been measured.

The historical notebook's CNN is fit on an internal Training subset and evaluated from its selected checkpoint. This model is refit on all original Training entries. Thus the observed improvement combines representation, classifier, regularization, selection, and final-fit differences. It should not be read as evidence that gradient boosting is universally better than CNNs.

Algorithm references: [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) and [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
