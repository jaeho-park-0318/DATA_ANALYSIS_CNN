# TimesFM versus LSTM: TEP one-step forecasting

This completed experiment uses the same 20 historical observations and 52 variables to predict one observation (3 minutes) ahead. TimesFM is used zero-shot, with no fine-tuning. The saved seed-42 LSTM was rerun on exactly the same sample targets.

## Results on the representative sample

| Model | RMSE (Train SD) | MAE (Train SD) |
|---|---:|---:|
| Persistence | 1.606110 | 0.688398 |
| LSTM | 1.468793 | 0.733885 |
| TimesFM | 0.963627 | 0.596248 |

TimesFM reduced pooled RMSE by 34.39% and MAE by 18.75% relative to LSTM. The advantage is not universal: LSTM performed better in healthy context, while TimesFM performed better after fault injection. TimesFM had lower RMSE for 42 of 52 variables and lower MAE for 42 of 52 variables.

| Region | LSTM RMSE | TimesFM RMSE | LSTM MAE | TimesFM MAE |
|---|---:|---:|---:|---:|
| Healthy context | 0.757884 | 0.824791 | 0.534474 | 0.571842 |
| Post-fault | 1.589771 | 0.993285 | 0.780497 | 0.601953 |

## Evaluation scope

The sample comprises 5 official Test run families (50, 214, 435, 489, 497), all 21 scenarios, and 94 time-stratified targets per trajectory: 105 trajectories and 9,870 targets, or 0.1% of the full dense Test evaluation. Seed 20261112 and per-trajectory time-bin sampling were fixed independently of observed errors.

**The full 9,870,000-target TimesFM evaluation has not been run.** Sample results must not be compared directly with historical full-Test LSTM metrics. No statistical significance or deployment readiness is claimed.

## Configuration and limitations

The existing StandardScaler was fitted only on 200,000 healthy Training rows. Train/Validation families are split 400/100; official Test uses a distinct RNG family. Windows never cross simulation runs or include their future target. Healthy context includes the normal scenario and pre-injection targets; fault scenarios with target sample greater than 160 are post-fault.

The recorded package is `timesfm==3.0.2`, using `google/timesfm-3.0-pytorch` revision `43046b85ec22d584a13f8098c2ed39c889e129c2`. The official Evaluator receives `(52,20)` contexts and processes channels as 32 + 20, with official repeated padding of the last chunk. All 52 variables are predicted, but not through one joint 52-channel attention pass. The point forecast is the official median. Symmetric averaging, positivity clipping, extra z-normalization, future covariates, XReg, and fine-tuning are disabled.

Experiments ran on an RTX 3060 Laptop GPU (6 GB) in an isolated environment. FP32 was used for the comparison; BF16 was explored only for throughput. The downloaded 3.0 weights carry a separate non-commercial/non-production license and are not redistributed here. See the [official model card](https://huggingface.co/google/timesfm-3.0-pytorch).

## Files and execution

- [Narrative comparison notebook](../notebooks/19_TimesFM_LSTM_공정예측_비교.ipynb): portable embedded results and six figures.
- [Original executed experiment notebook](TimesFM_TEP_실험.ipynb): requires local data/model artifacts to rerun inference.
- [Pooled and regional metrics](model_comparison.csv), [variable metrics](variable_metrics.csv), and [figures](figures/).
- `requirements.txt`, `experiment.py`, `download_model.py`, and `run_experiment.ps1` retain the experiment implementation.
- Small validation JSON records describe checks executed during the local experiment. Large prediction archives, source data, and pretrained weights are local-only.

Run the PowerShell wrapper only after obtaining data and restoring the saved LSTM/scaler locally. `run_experiment.ps1 -Full` requests the separate full evaluation; that command has not been run for this publication.
