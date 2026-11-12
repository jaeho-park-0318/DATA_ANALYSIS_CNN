"""같은 원본·설정으로 대표2대를 재생성하고 Parquet SHA256을 대조한다."""
import subprocess
import sys

from src.common import *


def main():
    before = {tool: digest(DEST / f"{tool}.parquet") for tool in CONFIG["pilot_tools"]}
    from src.pilot import run
    run()
    checks = {tool: {"before_sha256": before[tool], "after_sha256": digest(DEST / f"{tool}.parquet"),
                    "identical": before[tool] == digest(DEST / f"{tool}.parquet")}
              for tool in CONFIG["pilot_tools"]}
    save(OUT / "reproducibility.json", checks)
    if not all(c["identical"] for c in checks.values()):
        raise RuntimeError("재생성 Parquet SHA256이 다릅니다. 수치/정렬 추가 확인 필요")
    result = subprocess.run([sys.executable, "-m", "src.run_pipeline", "--release"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    gate = {"command": "python -m src.run_pipeline --release", "returncode": result.returncode,
            "blocked": result.returncode != 0 and read_release_status() == "BLOCKED"}
    save(OUT / "release_gate_test.json", gate)
    assert gate["blocked"]
    print("대표2대 재생성 SHA256 동일 및 배포 차단 확인", flush=True)


def read_release_status():
    import json
    return json.loads((OUT / "release_status.json").read_text(encoding="utf-8"))["status"]


if __name__ == "__main__":
    main()
