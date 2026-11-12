# TEP: Fault Diagnosis and Process-Variable Forecasting

The TEP project covers data auditing, fault detection and diagnosis, CNN/LSTM/GRU/XGBoost comparisons, diagnosis timing, and input-availability studies. A separate regression LSTM predicts the next 3-minute observation from 20 past observations of 52 process variables. The new TimesFM experiment compares zero-shot forecasts with this saved LSTM and Persistence on identical Test targets. Classification Accuracy and regression RMSE/MAE measure different tasks and are never ranked together.

## Start here

- [Recommended notebook](notebooks/19_TimesFM_LSTM_공정예측_비교.ipynb)
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
