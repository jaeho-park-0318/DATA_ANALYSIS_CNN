"""Report the active Python environment without installing or changing anything."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys


def main() -> int:
    packages = {}
    for name in ("pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "jupyter"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    checks = {
        "python_3_11": sys.version_info[:2] == (3, 11),
        "environment_name_phm2018": os.path.basename(sys.prefix).lower() == "phm2018",
        "pandas_available": packages["pandas"] is not None,
        "numpy_available": packages["numpy"] is not None,
    }
    print(json.dumps({
        "executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "prefix": sys.prefix,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": packages,
        "checks": checks,
        "ready_for_inspection": all(checks.values()),
    }, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
