"""V2 내부 커널 설정으로 한국어 Notebook 4개를 실행하고 결과를 저장한다."""
from __future__ import annotations

import os

from src.common import *


def main():
    for env, path in {
        "JUPYTER_PATH": ROOT / "outputs/jupyter",
        "JUPYTER_CONFIG_DIR": ROOT / "outputs/jupyter_config",
        "JUPYTER_RUNTIME_DIR": ROOT / "outputs/jupyter_runtime",
        "IPYTHONDIR": ROOT / "outputs/ipython",
        "MPLCONFIGDIR": ROOT / "outputs/matplotlib",
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[env] = str(path)
    save(ROOT / "outputs/jupyter/kernels/phm2018-v2/kernel.json", {
        "argv": [CONFIG["python"], "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "PHM2018 V2 (local Conda)", "language": "python",
    })
    import nbformat
    from nbclient import NotebookClient
    results = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(nb, timeout=180, kernel_name="phm2018-v2", resources={"metadata": {"path": str(ROOT)}})
        client.execute()
        nbformat.validate(nb)
        nbformat.write(nb, path)
        errors = [o for c in nb.cells if c.cell_type == 'code' for o in c.get('outputs', []) if o.output_type == 'error']
        results.append({"notebook": str(path.relative_to(ROOT)), "errors": len(errors),
                        "executed_code_cells": sum(c.cell_type == 'code' and c.execution_count is not None for c in nb.cells)})
        print(f"Notebook 실행 완료: {path.name}", flush=True)
    save(OUT / "notebook_validation.json", results)


if __name__ == "__main__":
    main()
