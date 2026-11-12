"""Resolve local Git LFS pointers without overwriting the supplied checkout."""
from pathlib import Path
import concurrent.futures, hashlib, json, urllib.request, time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tennessee-eastman-dataset-main'
DEST = ROOT / 'data/raw/github'

def fetch(p):
    relative = p.relative_to(SOURCE)
    text = p.read_text().splitlines()
    oid = text[1].split(':', 1)[1]
    size = int(text[2].split()[1])
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            if not target.exists() or target.stat().st_size != size:
                url = 'https://media.githubusercontent.com/media/Blound04/tennessee-eastman-dataset/main/' + relative.as_posix()
                with urllib.request.urlopen(url, timeout=120) as response:
                    target.write_bytes(response.read())
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != oid: raise ValueError('LFS SHA256 mismatch')
            return {'path': str(relative), 'status': 'verified', 'bytes': size, 'sha256': oid}
        except Exception as e:
            if attempt == 2: return {'path': str(relative), 'status': 'failed', 'error': str(e)}
            time.sleep(2)

if __name__ == '__main__':
    paths = sorted(SOURCE.glob('simulations/**/*.xlsx'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = []
        for result in pool.map(fetch, paths):
            results.append(result)
            print(len(results), len(paths), result['path'], result['status'], flush=True)
    (ROOT/'reports/sources/github_download.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
