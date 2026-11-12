"""감사 원본60파일의 내용 해시, V1 메타데이터, 산출물 행수/실행 결과 최종 검사."""
import json

from src.common import *


def main():
    audits = json.loads((OUT / "audit_summary.json").read_text(encoding="utf-8"))
    assert len(audits) == 20
    expected = {p: h for a in audits for p, h in a["sha256"].items()}
    mismatches = [p for p, h in expected.items() if digest(p) != h]
    baseline = json.loads((OUT / "v1_files_before.json").read_text(encoding="utf-8"))
    after = source_signature([Path(p) for p in baseline])
    changed = [p for p in baseline if after[p] != baseline[p]]
    assert not mismatches and not changed
    for a in audits:
        assert all(sum(v.values()) == a['raw_rows'] for v in a['distributions_raw'].values())
    import nbformat
    notebooks = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        cells = [c for c in nb.cells if c.cell_type == 'code']
        assert all(c.execution_count is not None for c in cells)
        assert not [o for c in cells for o in c.get('outputs', []) if o.output_type == 'error']
        notebooks.append(path.name)
    assert len(notebooks) == 4
    for tool in CONFIG['pilot_tools']:
        manifest = json.loads((DEST / f"{tool}_manifest.json").read_text(encoding='utf-8'))
        assert all(sum(v.values()) == manifest['anchors'] for v in manifest['candidate_counts'].values())
        assert digest(DEST / f"{tool}.parquet") == manifest['parquet_sha256']
    save(OUT / "final_integrity.json", {
        "status": "PASS", "raw_sha256_rechecked_files": len(expected), "raw_content_changes": mismatches,
        "v1_metadata_checked_files": len(baseline), "v1_metadata_changes": changed,
        "executed_notebooks": notebooks, "audit_tools": len(audits), "raw_rows": sum(a['raw_rows'] for a in audits),
        "approved_training_rows": 0,
    })
    print(f"최종 검사 통과: 원본 {len(expected)}파일 SHA256 동일, Notebook4개, 행수 정합", flush=True)


if __name__ == "__main__":
    main()
