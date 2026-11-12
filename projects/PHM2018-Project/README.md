# PHM 2018 V1: Baselines and Group Validation

This historical project progressed from source-data auditing and label construction to baseline models, XGBoost experiments, threshold analysis, and four-fold grouped validation. The original README described an earlier EDA-only milestone; the publication guide reflects the completed notebooks. Unknown labels must not be treated as healthy observations. The retrospective group experiments did not establish reliable equipment generalization or deployment readiness.

## Start here

- [Recommended notebook](notebooks/08_model_comparison.ipynb)
- [Complete notebook index](notebooks/README.md)
- [Reports and historical evidence](reports/README.md)
- [Repository research overview](../../docs/RESEARCH_OVERVIEW.md)
- [Data and reproducibility policy](../../docs/DATA_AND_REPRODUCIBILITY.md)

## Publication contents

Source code, experiment configuration, executed notebooks, figures, and bounded aggregate results are included. Original project directory names are retained so sibling-project references remain recognizable. Raw datasets, Parquet/NumPy caches, Python environments, new model binaries, and downloaded TimesFM weights are not redistributed. Historical wafer model artifacts already in the original repository are retained.

The research originals on the local machine were not moved or modified. Historical Korean Markdown records are preserved as UTF-8 `.txt` files under `docs/historical-records/`; the published Markdown navigation is in English. Notebook text and original execution outputs retain their original language.

## Reproduction

Use a separate environment for this project and its recorded `requirements*.txt`. Obtain the source data separately and review the experiment configuration before running scripts. Saved notebook outputs can be inspected without retraining. Original notebooks are not all standalone: missing local source datasets or model artifacts must be restored before rerunning their data/model cells.

This publication does not claim that every historical notebook was freshly executed. See the repository notebook catalog for existing error outputs and unexecuted code cells.
