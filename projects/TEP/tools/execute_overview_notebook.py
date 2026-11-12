"""Execute only the consolidated notebook, retaining all tables and figures."""
from pathlib import Path
import json,os,sys,asyncio
import nbformat
from nbclient import NotebookClient
if sys.platform=='win32':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
ROOT=Path(__file__).resolve().parents[1]
os.environ['JUPYTER_PATH']=str(ROOT/'.jupyter')
os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.jupyter/runtime')
os.environ['MPLCONFIGDIR']=str(ROOT/'.matplotlib')
path=ROOT/'notebooks/00_TEP_전체_연구_과정과_결과.ipynb'
nb=nbformat.read(path,as_version=4)
client=NotebookClient(nb,timeout=240,kernel_name='tep-phase1',resources={'metadata':{'path':str(ROOT)}})
client.execute()
nbformat.validate(nb)
nbformat.write(nb,path)
outputs=[out for cell in nb.cells for out in cell.get('outputs',[])]
errors=[o for o in outputs if o.output_type=='error']
tables=sum('text/html' in o.get('data',{}) for o in outputs)
rendered_tables=sum('<table' in o.get('data',{}).get('text/html','') for o in outputs)
images=sum('image/png' in o.get('data',{}) for o in outputs)
code=[c for c in nb.cells if c.cell_type=='code']
assert not errors
assert all(c.execution_count is not None for c in code)
assert tables>=15 and images>=15,(tables,images)
checks={'notebook':path.name,'cells':len(nb.cells),'executed_code_cells':len(code),'html_outputs':tables,'rendered_tables':rendered_tables,'embedded_images':images,'error_outputs':len(errors),'bytes':path.stat().st_size}
(ROOT/'reports/sources/overview_notebook_validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False),flush=True)
