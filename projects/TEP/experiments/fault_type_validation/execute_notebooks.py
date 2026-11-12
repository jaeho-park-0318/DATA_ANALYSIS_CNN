"""Execute only the new diagnosis and forecasting notebooks in an isolated kernel."""
from pathlib import Path
import sys,os,json,asyncio
sys.dont_write_bytecode=True
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
if sys.platform=='win32':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
cache=HERE/'cache/notebook_runtime';kernel=cache/'.jupyter/kernels/tep-new-validation';kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'TEP new validation','language':'python','env':{'PYTHONPATH':str(ROOT/'.deps'),'PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(cache/'.matplotlib'),'IPYTHONDIR':str(cache/'.ipython')}}),encoding='utf-8')
os.environ['JUPYTER_PATH']=str(cache/'.jupyter');os.environ['JUPYTER_RUNTIME_DIR']=str(cache/'.runtime')
Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(exist_ok=True)
paths=[HERE/'15_fault_type_validation.ipynb',ROOT/'experiments/next_step_forecasting/16_next_step_forecasting.ipynb']
results=[]
for path in paths:
    if not path.exists():continue
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    NotebookClient(nb,timeout=240,kernel_name='tep-new-validation',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,path)
    outputs=[o for c in nb.cells for o in c.get('outputs',[])]
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
    assert not any(o.output_type=='error' for o in outputs)
    results.append({'notebook':str(path.relative_to(ROOT)),'executed_code_cells':sum(c.cell_type=='code' for c in nb.cells),'images':sum('image/png' in o.get('data',{}) for o in outputs),'errors':0})
    print('Executed',path.name,flush=True)
(ROOT/'reports/fault_type_validation/notebook_validation.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
