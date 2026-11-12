"""공개 GitHub 코드를 정적으로 내려받아 검토용 텍스트만 저장한다. 외부 코드는 실행하지 않는다."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from src.common import OUT, save

REPOS = ["YoungJang-KR/semiconductor-fault-detection",
         "ki-seoultech/PHM2018-Ion-Mill-Etch-3Stage-Predictive-Maintenance"]


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "PHM2018-V2-static-review", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main():
    base = OUT / "github_review"
    manifest = []
    for repo in REPOS:
        commit = json.loads(get(f"https://api.github.com/repos/{repo}/commits/main"))["sha"]
        tree = json.loads(get(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"))
        own = base / repo.split('/')[0]
        save(own / "tree.json", tree)
        files = []
        for item in tree['tree']:
            p = item['path']
            if item['type'] != 'blob' or not (p.endswith(('.py', '.ipynb')) or p == 'README.md'):
                continue
            record = {"path": p, "blob_sha": item['sha'], "size": item.get('size'),
                      "url": f"https://github.com/{repo}/blob/{commit}/{p}"}
            if item.get('size', 0) > 25 * 1024**2:
                record['status'] = 'size_limit_not_read'
                files.append(record)
                continue
            try:
                data = get(f"https://raw.githubusercontent.com/{repo}/{commit}/{urllib.parse.quote(p)}")
                if p.endswith('.ipynb'):
                    nb = json.loads(data)
                    parts = []
                    for i, c in enumerate(nb.get('cells', [])):
                        source = c.get('source', [])
                        source = ''.join(source) if isinstance(source, list) else source
                        if c.get('cell_type') in ['code', 'markdown']:
                            parts.append(f"\n# ===== CELL {i} ({c['cell_type']}) =====\n" + source)
                    text = '\n'.join(parts)
                else:
                    text = data.decode('utf-8-sig')
                dest = own / (p + '.txt')
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(text, encoding='utf-8')
                record['status'] = 'read_static_only'
                record['snapshot'] = str(dest)
            except Exception as error:
                record['status'] = 'access_failed'
                record['error'] = str(error)
            files.append(record)
        manifest.append({"repository": repo, "commit": commit, "tree_truncated": tree.get('truncated'),
                         "files": files, "external_code_executed": False})
        print(repo, commit, len(files), 'static files', flush=True)
    save(base / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
