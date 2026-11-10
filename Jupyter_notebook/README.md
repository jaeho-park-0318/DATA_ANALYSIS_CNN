# Historical notebook index

These notebooks are preserved with their original names, contents, and execution outputs. The descriptions below are based on source inspection, not a fresh execution of every notebook. Some cells use local paths or refer to datasets, images, and model checkpoints that are not included in this repository.

| Notebook | Main contents |
|---|---|
| [Untitled7.ipynb](Untitled7.ipynb) | WM-811K loading, exploratory summaries, and wafer-map visualization |
| [Untitled8.ipynb](Untitled8.ipynb) | WM-811K CNN experiments, lot-group splits, saved models, and prediction |
| [Untitled9.ipynb](Untitled9.ipynb) | WaferGuard learning, WM-811K transfer learning, and lot-disjoint CNN baseline comparisons |
| [Untitled11.ipynb](Untitled11.ipynb) | PCB image analysis using morphology and skeletonization |
| [Untitled12.ipynb](Untitled12.ipynb) | Further PCB image and morphology experiments |
| [Untitled13.ipynb](Untitled13.ipynb) | Gerber/PCB geometry analysis using Gerbonara and Shapely |
| Untitled1–Untitled6.ipynb | Earlier Python, CSV, plotting, and exploratory data-analysis exercises |

The earlier Korean project notes are [Wafer_Guard&WM_811K.md](Wafer_Guard%26WM_811K.md) and [Wafer_Map_defect_analys.md](Wafer_Map_defect_analys.md).

Start with `Untitled9.ipynb` for the historical transfer-learning versus CNN comparison. Its later cells include an eight-class WM-811K lot-disjoint baseline using the first fold of `StratifiedGroupKFold`, followed by Test evaluation of a saved `.keras` checkpoint. It also contains earlier experiments with other splits; those results should not be mixed with the final comparison.

The new executable spatial-feature model is maintained separately in [spatial_boosting](../spatial_boosting/README.md), with an [executed English notebook](../spatial_boosting/Spatial_Feature_Boosting.ipynb). It does not require executing the historical notebooks or installing TensorFlow.
