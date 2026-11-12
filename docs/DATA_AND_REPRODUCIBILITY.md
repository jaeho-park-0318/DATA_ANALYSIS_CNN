# Data and Reproducibility

## Included

Research code, configuration, executed notebook records, English Markdown guides, preserved historical text records, figures, bounded aggregate tables, and selected validation metadata. Existing published wafer-model artifacts are retained unchanged. Source-file hashes and publication checks are stored in this directory.

## Local-only artifacts

WM-811K pickle data, PHM source CSV files, TEP RData/Excel sources, Parquet/NumPy training caches, row-level prediction/episode archives, Python environments, downloaded TimesFM weights, and newly generated PHM/TEP model binaries are excluded. An omitted artifact is not a missing research result: aggregate results and saved notebook displays remain available. Do not infer that raw data or model binaries are licensed for redistribution.

## Separate environments

- Historical wafer CNN notebooks require TensorFlow/Keras; spatial boosting has its own recorded requirements.
- PHM uses the historical `phm2018` environment and version-specific requirements.
- TEP has `requirements.txt` and `requirements-lock.txt`.
- TimesFM uses a separate environment with CUDA12.1 PyTorch2.5.1 and its own pinned requirements. Do not upgrade an existing environment just to view saved results.

Obtain source datasets separately, update local configuration paths, and review whether a command trains models or only reads results before running it. Historic machine-specific paths are retained as provenance, not as portable installation instructions. Download TimesFM weights from its official model repository subject to its license.

## Notebook behavior

Saved outputs are historical records. The TEP narrative comparison and consolidated PHM notebook embed substantial evidence. Other notebooks may depend on local data or model artifacts that are intentionally omitted. Publication validates notebook structure and records execution state without rerunning expensive or data-dependent training cells. Existing error outputs and unexecuted code cells are disclosed in the notebook catalog.

Large embedded notebooks may exceed GitHub preview limits; download them and open in Jupyter. Historical Korean notebook Markdown/cell outputs are retained. New repository Markdown navigation is English. Korean `.md` records are preserved as `.txt` with English guide pages at their original Markdown locations; these guide pages are summaries, not full translations.

## Credentials and source preservation

No credentials, environment directories, or source datasets are intentionally included. The upload is assembled in a separate checkout; local research originals are not moved or edited. The original wafer repository is reorganized through a new commit with its ancestry retained, without force-pushing or rewriting history.
