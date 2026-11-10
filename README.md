# Wafer Map Defect Analysis: CNN, Transfer Learning, and Spatial Boosting

This project studies eight wafer-map defect patterns using WM-811K. It combines exploratory data analysis, CNN classification, WaferGuard-to-WM-811K transfer learning, and a spatial-feature-based gradient boosting model. The accompanying presentations discuss possible semiconductor process causes and the evidence needed to investigate them.

The latest model achieves **77.24% Test accuracy** and **77.67% Test Macro F1** on 7,894 labeled defect wafers. Compared with the CNN baseline reported in the original presentation, these are improvements of **7.15 and 10.65 percentage points**, respectively. This is a comparison of complete pipelines, including their training procedures; it does not isolate the effect of replacing CNNs with trees.

## Results at a glance

| Model | Test accuracy | Test Macro F1 | Evidence |
|---|---:|---:|---|
| WaferGuard zero-shot | 30.63% | 24.91% | Original presentation |
| Transfer learning, Stage 1 | 59.18% | Not reported | Original presentation |
| Transfer learning, Stage 2 | 63.90% | 64.76% | Original presentation |
| WM-811K CNN baseline | 70.09% | 67.02% | Original presentation |
| **Spatial features + HistGradientBoosting** | **77.24%** | **77.67%** | Saved evaluation and per-wafer predictions |

The earlier metrics are reported results, not a fresh execution of the CNN notebooks. WaferGuard's separate task and its metrics are not directly comparable with this eight-class WM-811K evaluation.

## Repository guide

```text
.
├── Jupyter_notebook/          # Original experiments and historical notes
│   └── README.md             # Index identifying wafer, PCB, and practice notebooks
├── data/                     # Original project presentations; raw dataset is local only
├── docs/
│   └── model_comparison.md   # Detailed results, methodology, and limitations
├── reports/
│   ├── README.md
│   ├── model_comparison_learning_principles.pptx
│   ├── process_root_cause_expanded.pptx
│   └── generalization_report.pdf
└── spatial_boosting/
    ├── README.md             # Model card and reproducibility instructions
    ├── Spatial_Feature_Boosting.ipynb  # Executed walkthrough with plots and metrics
    ├── requirements.txt
    ├── wafer_features.py     # 274 handcrafted spatial features
    ├── train_spatial.py      # Validation selection, refit, and Test evaluation
    ├── predict.py            # Inference on one .npy wafer map
    ├── verify_model.py       # Artifact consistency and feature checks
    └── results/              # Trained model, metrics, predictions, and split indices
```

Existing notebooks and their filenames have been preserved so that earlier references remain usable. New model code and final reports have separate locations; temporary slide builds, feature caches, Python environments, and duplicate presentation revisions are excluded.

## Dataset and task

Only **25,519 wafers with one of eight defect labels** are used:

`Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Near-full`, `Random`, `Scratch`.

The original split contains **17,625 Training** and **7,894 Test** wafers. Wafer-map values are `0 = outside wafer`, `1 = good die`, and `2 = defective die`. The model does not use `lotName`, `waferIndex`, or the split label as prediction features.

Normal (`none`) and unlabeled wafers are excluded. Consequently, these results measure classification **among known defect types**, not normal-versus-defective detection or recognition of mixed defects. The full dataset is not redistributed here; obtain WM-811K separately and follow its usage terms. Put the pandas pickle at `data/WM811K.pkl` or provide an explicit path.

## Run the new model

Create a separate environment for spatial boosting. The historical CNN notebooks need their own TensorFlow/Keras environment.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
python -m pip install -r spatial_boosting/requirements.txt
```

Use a compatible Python version with the pinned packages; the recorded runtime is listed in the [model card](spatial_boosting/README.md). The trained artifact is included, so prediction does not require retraining or the full dataset:

```bash
python spatial_boosting/predict.py path/to/wafer_map.npy
```

The input must be a two-dimensional NumPy array containing only `0`, `1`, and `2`. Output is a predicted class plus eight probability scores. These scores have not been calibrated.

To reproduce training and evaluation:

```bash
python spatial_boosting/train_spatial.py --dataset data/WM811K.pkl
python spatial_boosting/verify_model.py
```

Training writes to `spatial_boosting/results/` and replaces the existing model and result files. The first run builds an ignored feature cache. Use `--rebuild-cache` when changing the dataset or feature implementation. The code accepts string labels and the nested singleton label arrays used by WM-811K.

### Jupyter Notebook walkthrough

Open [Spatial_Feature_Boosting.ipynb](spatial_boosting/Spatial_Feature_Boosting.ipynb) for the complete English walkthrough, including synthetic feature illustrations, fitted parameters, validation selection, Test metrics recalculated from the prediction CSV, a confusion matrix, and a comparison chart.

```bash
python -m pip install -r spatial_boosting/requirements-notebook.txt
python -m jupyterlab spatial_boosting/Spatial_Feature_Boosting.ipynb
```

The published notebook includes executed outputs. All default cells were run successfully with retraining disabled. Set `RUN_TRAINING=True` only when intentionally rerunning the optional training section with the raw dataset available.

## How spatial boosting learns

1. Convert each map into **274 features** describing failure rates, radial and angular densities, grid densities, coordinate statistics, connected components, and smoothed defect density.
2. Compare Extra Trees, Random Forest, and histogram gradient boosting using **one lot-disjoint validation fold** from the original Training partition.
3. Select the candidate by validation accuracy, with Macro F1 as the tie-breaker.
4. Refit the selected model on all 17,625 original Training wafers and evaluate the held-out original Test partition.

The saved model is `HistGradientBoostingClassifier` with 220 boosting iterations, a learning rate of 0.07, and up to 23 leaves per tree. With eight classes, this produces **1,760 trees**. Trees sequentially adjust class scores to reduce multiclass log loss; this differs from learning convolution filters directly from pixels.

Explicit spatial summaries may reduce the burden of learning wafer geometry from pixels. Normalized coordinates and sorted density summaries may also reduce sensitivity to some map-size and orientation changes. These are explanations consistent with the design, **not measured causal contributions**: no feature ablation or rotation-robustness experiment has been run. The final refit also uses more fit samples than the historical CNN training shown in the notebook.

## Important evaluation findings

- New-model validation accuracy is 96.96%, but Test accuracy is 77.24%: a generalization gap of approximately **19.73 percentage points** remains.
- Training and Test contain no overlapping lot names, but **92 Test maps exactly match a Training map**. After excluding those 92 Test entries, accuracy is **77.01%** and Macro F1 is **77.35%**.
- Donut recall improves from the reported CNN value of 26.03% to **80.82%**. Scratch recall improves to **66.23%**, but Scratch precision remains **48.57%**.
- Recall is higher than the CNN baseline for all eight classes; transfer learning still has higher Near-full recall.
- Validation uses the **first fold only** of a five-fold splitter. These are not five-fold cross-validation averages.

See the [detailed comparison](docs/model_comparison.md) and [model card](spatial_boosting/README.md) for per-class scores, duplicate checks, reproducibility details, and limitations.

## Presentations and supporting material

- [Model comparison and learning principles — 24 slides](reports/model_comparison_learning_principles.pptx): overall and per-class metrics, feature extraction, boosting, possible explanations, and follow-up experiments.
- [Expanded process root-cause study](reports/process_root_cause_expanded.pptx): process-specific mechanisms, literature cases, figures, and source references. Wafer patterns suggest investigation hypotheses; they do not by themselves prove a manufacturing root cause.
- [Generalization and model evaluation report](reports/generalization_report.pdf).
- [Original project presentation](data/Wafer%20Map%20Defect%20Analysis%20%26%20Process%20Root%20Cause%20Study.pdf).

The presentations retain their Korean content. This README and the newly added Markdown documentation are written in English. Historical Korean notes are preserved as original project records.

## Next validation steps

Evaluate CNNs and spatial models on identical fit/validation samples and equal selection budgets, repeat across seeds and all folds, remove feature groups one at a time, and test rotation/resolution changes and independent lots. Improving Loc–Scratch separation and adding a separate normal/defect stage are useful next directions.

## References and attribution

- The original project notebook and presentation are the sources for the earlier CNN and transfer-learning results.
- [scikit-learn: HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
- [scikit-learn: StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- Research figures in the process presentation retain their individual source attributions. Their inclusion does not grant a blanket license to reuse third-party figures or datasets.
