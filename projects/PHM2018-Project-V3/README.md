# PHM 2018 V3: Alarm Evaluation and Final Time Holdout

V3 separates label uncertainty, input availability, alarm burden, equipment-group validation, and final time holdout evaluation. The frozen B32 XGBoost model and thresholds were selected before accessing the final time holdout. At the validation-budget-10 operating point it detected 33 of 194 recorded faults (17.01%), or 33 of 91 input-available faults (36.26%). The 103 unavailable events remain in the recorded-event denominator. Time is expressed in unverified raw units. The later equipment-specific pilot reused already-seen late periods and did not demonstrate improvement.

## Start here

- [Recommended notebook](notebooks/00_PHM2018_종합_수행결과.ipynb)
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
