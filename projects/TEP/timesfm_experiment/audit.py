"""Read-only audit of the original project; write only inside this experiment."""
from pathlib import Path
import os, sys, json, hashlib, time
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def inventory():
    rows = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {'.git','.deps','.venv','__pycache__','timesfm_experiment','.matplotlib','.jupyter'}]
        for name in files:
            p = Path(base)/name
            rows.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size})
    return rows

def run():
    rows = inventory()
    (HERE/'project_inventory.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    before = HERE/'protected_before.json'
    if not before.exists():
        hashes = {}
        for r in rows:
            p = ROOT/r['path']
            if p.suffix.lower() in {'.py','.ipynb','.csv','.md','.json','.pt','.joblib','.txt','.ps1','.rdata'}:
                hashes[r['path']] = sha(p)
        before.write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    extracts = []
    for n in ['17_CNN_기능확장_종합안내.ipynb','18_TEP_PHM_전체모델_종합비교.ipynb']:
        p = ROOT/'notebooks'/n
        book = json.loads(p.read_text(encoding='utf-8'))
        extracts.append({'path':str(p.relative_to(ROOT)), 'cells':[
            {'cell_index':i,'cell_type':c['cell_type'],'source':''.join(c['source'])}
            for i,c in enumerate(book['cells']) if any(k in ''.join(c['source']).lower() for k in ['lstm','forecast','예측','분할','정규'])]})
    (HERE/'notebook_analysis.json').write_text(json.dumps(extracts,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Original inventory and hashes saved:',len(rows),'files',flush=True)

def verify():
    hashes=json.loads((HERE/'protected_before.json').read_text(encoding='utf-8'))
    changed=[p for p,h in hashes.items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
    result={'checked_files':len(hashes),'changed':changed,'passed':not changed}
    (HERE/'preservation_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert not changed, changed
    print('Preservation check PASSED:',len(hashes),'files')

if __name__=='__main__':
    verify() if '--verify' in sys.argv else run()
