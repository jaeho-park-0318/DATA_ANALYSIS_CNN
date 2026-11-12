# Publication Status

This publication adds PHM and TEP research to the existing wafer-map repository while preserving its commit ancestry. Local research projects remain unchanged.

## Completed validation

- 61 notebooks parsed successfully; no invalid notebook JSON.
- 5 historical saved-error cells across 3 notebooks are disclosed in the [catalog](NOTEBOOKS.md); they are not presented as fresh successful runs.
- 40 original wafer files remain at corresponding paths under `projects/wafer-map/`.
- No file is 100 MiB or larger; the largest file is 48.20 MiB.
- No high-confidence credential pattern was found in the assembled text payload. This is a focused scan, not a guarantee that every possible sensitive string can be detected.
- Raw datasets, new PHM/TEP model binaries, environments, and large data caches are excluded.
- No historical model was retrained and no final Test was reselected for publication.
- Full-Test TimesFM inference remains unexecuted; only the completed representative-sample results are reported.

## Evidence

- [Source-file manifest](source_manifest.json): original relative paths and hashes before publication document adaptation.
- [Notebook metadata catalog](notebook_catalog.json).
- [Validation metadata](publication_validation.json).
- [Exclusion summary](exclusion_summary.json).

Historical Korean Markdown is retained as UTF-8 text records, with English summary/navigation pages. Notebook text and saved outputs keep their original language. The source manifest hashes describe the local inputs; adapted English Markdown intentionally differs from those inputs.
