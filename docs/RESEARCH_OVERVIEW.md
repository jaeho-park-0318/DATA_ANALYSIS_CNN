# Research Overview

## Wafer maps: classify known spatial defect types

The WM-811K study covers exploration, CNN classification, WaferGuard transfer learning, and handcrafted spatial-feature boosting. The selected histogram gradient boosting pipeline uses 274 spatial features and achieved 77.24% Test Accuracy and 77.67% Macro F1. Earlier presentation-reported CNN results were 70.09% and 67.02%. These are pipeline comparisons, not controlled estimates of architecture alone. Healthy and unlabeled wafers are excluded. Exact map duplicates and a substantial validation-to-Test gap remain documented in the project model card.

## PHM: progress from data inspection to qualified alarm evaluation

V1 includes EDA, label/feature construction, baseline classifiers, XGBoost, four-fold grouped validation, and threshold policies. It did not establish reliable equipment generalization. V2 then audits data and candidate-label readiness, with training-data release blocked by unverified time units, event-history coverage, and maintenance boundaries; it has no trained predictive model.

V3 explicitly separates unknown labels, unavailable inputs, event recall, alarm burden, and lead time. The final B32 XGBoost artifact and two thresholds were frozen before a single final time-holdout inference. At the budget-5 point it detected 7/194 recorded and 7/91 available events; at budget-10 it detected 33/194 and 33/91. The latter corresponds to 17.01% recorded-event recall and 36.26% input-available recall. Alarm burden was 9.8986 episodes per 10,000 raw time units. These are fixed operating-point results, not physical-hours rates or an approved deployment policy.

The equipment-specific follow-up trained one eligible tool (03M02). It detected none of the three late-period input-available events; 24 of 27 recorded events lacked prediction-window inputs. The reused late period was a follow-up comparison, not an independent new holdout. There is no demonstrated improvement from that specialist pilot.

## TEP: distinguish diagnosis from forecasting

TEP classification uses 41 measured and 11 manipulated variables at 3-minute intervals. The study includes temporal CNN/LSTM/GRU and feature-based XGBoost baselines, fault-type diagnosis, detection/diagnosis delay, unknown-fault experiments, and sensor-availability stress tests. The selected CNN is a single seed-42 model, not a three-seed ensemble.

The independent regression LSTM predicts 52 next-step values from 20 observations. Its historical dense-Test RMSE/MAE are 1.494316/0.737834; Persistence is 1.630137/0.690530. Those full-Test numbers must not be ranked directly against the new sample-only TimesFM result.

The paired TimesFM study reuses the healthy-Train StandardScaler and saved LSTM weights on 9,870 representative Test targets. TimesFM reduced pooled RMSE by 34.39% and MAE by 18.75%. Healthy-context RMSE was worse (0.824791 versus 0.757884); post-fault RMSE was better (0.993285 versus 1.589771). All 52 variables are used through the official 32+20-channel Evaluator chunks. No fine-tuning is performed. The full 9,870,000-target TimesFM evaluation has not been run.

## Interpretation boundaries

- Wafer Accuracy/Macro F1, PHM event recall, and TEP regression error have different tasks and denominators.
- Label uncertainty and missing prediction-window inputs are not equivalent to model false negatives on valid inputs.
- Raw PHM time units are not independently verified as seconds or minutes.
- Reused late periods, retrospective development folds, and independent final tests must remain distinct.
- The TEP sample comprises five Test families; it does not establish statistical significance or full-population superiority.
- Pretraining-corpus overlap with TEP has not been independently excluded.
