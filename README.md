# Manufacturing AI Research

Research projects on semiconductor manufacturing data: wafer-map defect classification, PHM 2018 equipment fault prediction, and Tennessee Eastman Process (TEP) diagnosis and forecasting.

This repository expands the former `DATA_ANALYSIS_CNN` project. Its Git history and wafer-map notebooks, presentations, and trained spatial model are preserved under `projects/wafer-map/`. PHM and TEP are added as separate research projects. New Markdown documentation is in English; historical notebook text and outputs retain their original language.

## Project guide

| Project | Research question | Recommended entry |
|---|---|---|
| [Wafer-map classification](projects/wafer-map/README.md) | Which of eight known defect patterns is present? | [Spatial-feature boosting notebook](projects/wafer-map/spatial_boosting/Spatial_Feature_Boosting.ipynb) |
| [PHM V1](projects/PHM2018-Project/README.md) | How reliable are baseline fault predictors, labels, and grouped validation? | [Model comparison](projects/PHM2018-Project/notebooks/08_model_comparison.ipynb) |
| [PHM V2](projects/PHM2018-Project-V2/README.md) | Are time units, labels, and feature candidates ready for training? | [Data validation](projects/PHM2018-Project-V2/notebooks/04_dataset_validation.ipynb) |
| [PHM V3](projects/PHM2018-Project-V3/README.md) | Can a frozen model give useful early alarms under input and label uncertainty? | [Consolidated PHM notebook](projects/PHM2018-Project-V3/notebooks/00_PHM2018_종합_수행결과.ipynb) |
| [TEP](projects/TEP/README.md) | How well can models diagnose faults and forecast process variables? | [TimesFM versus LSTM](projects/TEP/notebooks/19_TimesFM_LSTM_공정예측_비교.ipynb) |

## Selected results and their scope

| Study | Result | Evaluation scope |
|---|---|---|
| Wafer spatial boosting | 77.24% Accuracy; 77.67% Macro F1 | 7,894 labeled defect wafers; eight known defect classes; healthy/unlabeled wafers excluded |
| PHM V3 frozen B XGBoost | 33/194 recorded faults detected; 33/91 input-available faults detected | Final time holdout at a fixed validation-budget-10 operating point; raw time units |
| TEP TimesFM versus LSTM | RMSE 0.963627 versus 1.468793; MAE 0.596248 versus 0.733885 | Same 9,870 representative targets; 52 variables; 20 observations to one step ahead; Train-SD units |

These rows describe different tasks and denominators. They are not a cross-project leaderboard. TEP sample results do not establish full-Test superiority, and PHM results do not demonstrate deployment readiness. PHM V2 did not train a model.

## Read next

- [Research overview and interpretation](docs/RESEARCH_OVERVIEW.md)
- [All notebooks, execution state, and recommended reading order](docs/NOTEBOOKS.md)
- [Data, artifacts, environments, and reproducibility](docs/DATA_AND_REPRODUCIBILITY.md)
- [Publication validation and exclusions](docs/PUBLICATION_STATUS.md)

Raw datasets, environments, large caches, downloaded foundation-model weights, and newly generated model binaries are excluded. Original research files remain on the local machine. Some historical notebooks depend on these local artifacts; a saved output is not evidence of a fresh publication-time execution. The approximately 48-MiB consolidated PHM notebook may need to be downloaded and opened locally if GitHub declines to render it.

## Repository layout

```text
projects/
  wafer-map/             # Existing wafer/PCB history and spatial boosting
  PHM2018-Project/        # V1 baselines and grouped validation
  PHM2018-Project-V2/     # Readiness audit; no trained model
  PHM2018-Project-V3/     # Alarm studies, frozen holdout, specialist pilot
  TEP/                   # Diagnosis, availability, LSTM, and TimesFM
docs/                    # English navigation, policies, and publication checks
```

Project directories retain their historical names to preserve sibling-project references. Third-party datasets, figures, and pretrained models retain their respective terms; no blanket license is granted for those materials.
