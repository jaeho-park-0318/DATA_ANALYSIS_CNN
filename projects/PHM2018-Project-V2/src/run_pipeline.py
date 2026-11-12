"""python -m src.run_pipeline [--audit-only | --pilot-only | --release]"""
from __future__ import annotations

import argparse
import importlib.metadata
import platform
import subprocess
import sys
import threading
import time

from src.common import *
import psutil


class ResourceMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.violation = None

    def loop(self):
        process = psutil.Process()
        while not self.stop.is_set():
            mem = psutil.virtual_memory()
            rss = process.memory_info().rss / 1024**3
            sample = {"unix_time": time.time(), "rss_gib": rss, "available_ram_gib": mem.available / 1024**3}
            try:
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                sample["gpu_memory_mib_and_util_percent"] = gpu.stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                sample["gpu_memory_mib_and_util_percent"] = "unavailable"
            if rss > CONFIG["max_rss_gib"] or mem.available / 1024**3 < CONFIG["min_available_ram_gib"]:
                self.violation = "RAM 예산 초과; 다음 장비 실행 차단"
            self.samples.append(sample)
            self.stop.wait(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("작업 디렉터리를 V2 루트로 지정해야 합니다.")
    # 단위 변환/정비표/완전성 입력을 처리하는 정식 구현은 아직 없으므로 플래그만 바꿔 해제할 수 없다.
    if args.release:
        save(OUT / "release_status.json", {"status": "BLOCKED", "approved_training_rows": 0,
             "blockers": release_blockers() + ["정식 단위 변환·정비 구간 검증 및 20대 배포 경로 미승인"]})
        raise SystemExit("확정 데이터 생성 차단: outputs/release_status.json")
    all_v1_files = sorted(p for p in RAW.parent.rglob("*") if p.is_file() and ".git" not in p.relative_to(RAW.parent).parts)
    before = source_signature(all_v1_files)
    save(OUT / "v1_files_before.json", before)
    monitor = ResourceMonitor()
    worker = threading.Thread(target=monitor.loop, daemon=True)
    worker.start()
    status = "FAILED"
    try:
        from src.audit import run as audit
        if not args.audit_only:
            audit(CONFIG["pilot_tools"])
            from src.pilot import run as pilot
            from src.validate import run as validate
            pilot(); validate()
        if not args.pilot_only:
            if monitor.violation:
                raise RuntimeError(monitor.violation)
            audit()
        status = "DIAGNOSTIC_COMPLETE_RELEASE_BLOCKED"
    finally:
        monitor.stop.set(); worker.join(timeout=7)
        after = source_signature(all_v1_files)
        changed = [p for p in before if before[p] != after[p]]
        save(OUT / "resources.json", {"samples": monitor.samples, "violation": monitor.violation,
             "peak_rss_gib": max((s["rss_gib"] for s in monitor.samples), default=0),
             "min_available_ram_gib": min((s["available_ram_gib"] for s in monitor.samples), default=0)})
        packages = ["numpy", "pandas", "polars", "duckdb", "pyarrow", "psutil", "xgboost", "nbformat", "matplotlib"]
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        save(OUT / "environment.json", {"python": sys.executable, "python_version": sys.version,
             "os": platform.platform(), "cpu": cpu, "logical_cores": psutil.cpu_count(),
             "physical_cores": psutil.cpu_count(logical=False), "ram_gib": psutil.virtual_memory().total / 1024**3,
             "gpu": gpu, "packages": {p: importlib.metadata.version(p) for p in packages},
             "polars_threads": pl.thread_pool_size(), "models_trained": 0})
        save(OUT / "run_status.json", {"status": status, "v1_files_checked": len(before), "v1_metadata_changes": changed,
             "v1_integrity_scope": "실행 전후 전체 기존 파일 크기/mtime; 감사 원본60파일 SHA256 별도. V1 전체 전후 해시 검증은 아님",
             "blockers": release_blockers(), "approved_training_rows": 0})
        if changed:
            raise RuntimeError("V1 파일 메타데이터 변경 감지")


if __name__ == "__main__":
    main()
