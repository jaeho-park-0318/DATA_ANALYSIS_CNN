"""Execute every delivered notebook with a project-local kernelspec."""
from pathlib import Path
import json, os, sys
import nbformat
from nbclient import NotebookClient
if sys.platform=='win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ROOT=Path(__file__).resolve().parents[1]
kernel=ROOT/'.jupyter/kernels/tep-phase1'
kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'TEP Phase 1','language':'python','env':{'PYTHONPATH':str(ROOT/'.deps')}}),encoding='utf-8')
os.environ['JUPYTER_PATH']=str(ROOT/'.jupyter')
os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.jupyter/runtime')
Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(parents=True,exist_ok=True)
results=[]
paths=sorted((ROOT/'notebooks').glob('*.ipynb'))
if '--phase2-only' in sys.argv:paths=[p for p in paths if int(p.name.split('_')[0])>=6]
for path in paths:
    nb=nbformat.read(path,as_version=4)
    nbformat.validate(nb)
    NotebookClient(nb,timeout=180,kernel_name='tep-phase1',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,path)
    results.append({'notebook':path.name,'executed':True,'error_outputs':sum(len([o for o in c.get('outputs',[]) if o.output_type=='error']) for c in nb.cells),'code_cells':sum(c.cell_type=='code' for c in nb.cells)})
    print('Executed',path.name,flush=True)
validation_file=ROOT/'reports/sources'/('phase2/notebook_validation.json' if '--phase2-only' in sys.argv else 'notebook_validation.json')
validation_file.write_text(json.dumps(results,indent=2),encoding='utf-8')
