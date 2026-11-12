"""Read-only file inventory; inspect at most five rows of ONE sensor CSV.

Other representative CSVs are opened only to read their header. No training,
full-file scan, extraction, upload, or modification of source files occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def category(path: Path, root: Path) -> str:
    parts = {part.lower() for part in path.relative_to(root).parts[:-1]}
    if "train_faults" in parts or "fault" in path.stem.lower():
        return "fault"
    if "train_ttf" in parts or "ttf" in path.stem.lower():
        return "ttf"
    if path.suffix.lower() == ".csv" and "test" in parts:
        return "sensor_test"
    if path.suffix.lower() == ".csv" and "train" in parts:
        return "sensor_train"
    return "unclassified"


def choose_root(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.resolve()
        if not root.is_dir():
            raise ValueError(f"Data directory does not exist: {root}")
        return root
    candidates = (PROJECT_ROOT / "data/raw", PROJECT_ROOT / "phm_data_challenge_2018")
    for root in candidates:
        if root.is_dir() and any(root.rglob("*.csv")):
            return root
    raise ValueError("No CSV data found. Specify --data-root PATH.")


def inspect(root: Path, requested_sample: str | None = None) -> dict:
    # Only filesystem metadata is collected for the full dataset.
    paths = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix())
    inventory = []
    representatives = {}
    for path in paths:
        kind = category(path, root)
        match = re.match(r"(\d+)_(M\d+)_", path.name)
        stat = path.stat()
        inventory.append({
            "path": path.relative_to(root).as_posix(),
            "format": path.suffix.lower(),
            "bytes": stat.st_size,
            "mib": round(stat.st_size / 1024**2, 3),
            "mtime_ns": stat.st_mtime_ns,
            "category": kind,
            "file_prefix": match.group(1) if match else None,
            "machine_token": match.group(2) if match else None,
        })
        if path.suffix.lower() == ".csv":
            representatives.setdefault(kind, path)

    headers = {}
    for kind, path in representatives.items():
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            headers[kind] = {"path": path.relative_to(root).as_posix(),
                             "columns": next(csv.reader(stream), [])}

    sample = (root / requested_sample).resolve() if requested_sample else representatives.get("sensor_train")
    if sample is None:
        raise ValueError("No sensor training CSV found; specify --sample RELATIVE_PATH.")
    if not sample.is_relative_to(root) or sample.suffix.lower() != ".csv" or not sample.is_file():
        raise ValueError("Sample must be an existing CSV inside the data root.")
    import pandas as pd

    frame = pd.read_csv(sample, nrows=5, encoding="utf-8-sig")
    return {
        "data_root": str(root),
        "method": "Filesystem metadata + one header per category + one CSV with nrows=5",
        "file_count": len(inventory),
        "total_bytes": sum(item["bytes"] for item in inventory),
        "category_counts": dict(Counter(item["category"] for item in inventory)),
        "inventory": inventory,
        "representative_headers": headers,
        "sample": {
            "path": sample.relative_to(root).as_posix(),
            "rows_read": len(frame),
            "columns": list(frame.columns),
            "dtypes_inferred_from_five_rows": {col: str(dtype) for col, dtype in frame.dtypes.items()},
            "missing_in_five_rows": {col: int(n) for col, n in frame.isna().sum().items()},
            "first_five_rows": json.loads(frame.to_json(orient="records")),
        },
        "limitations": [
            "Five-row dtypes and missing counts do not describe the full dataset.",
            "Categories use local paths and representative headers; other headers are unverified.",
            "Filename prefixes, time/TTF units and label alignment require later validation.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Default: data/raw, then existing dataset directory")
    parser.add_argument("--sample", help="One CSV path relative to data root; default: first train sensor CSV")
    args = parser.parse_args()
    try:
        report = inspect(choose_root(args.data_root), args.sample)
    except (ValueError, OSError, ImportError, csv.Error) as exc:
        parser.exit(1, f"Inspection failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
