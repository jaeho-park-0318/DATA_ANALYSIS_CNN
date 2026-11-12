"""Finish artifacts only after successful paired evaluation."""
from pathlib import Path
import os,sys,json,platform,subprocess,importlib.metadata,time,asyncio
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
os.environ['PYTHONIOENCODING']='utf-8'
os.environ['MPLCONFIGDIR']=str(HERE/'mpl_cache')
os.environ['JUPYTER_PATH']=str(HERE/'jupyter')
os.environ['IPYTHONDIR']=str(HERE/'ipython')
os.environ['JUPYTER_RUNTIME_DIR']=str(HERE/'jupyter_runtime')
if sys.platform=='win32':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from audit import verify,sha
from report import main
import nbformat
from nbclient import NotebookClient

def finalize():
    # A first run started before the canonical output directory was finalized.
    old=HERE/'representative'
    if old.exists():
        meta=json.loads((old/'progress.json').read_text())
        assert meta['status']=='sample_complete','Do not move an active experiment'
        for p in old.iterdir():
            target=HERE/p.name
            if target.exists():raise RuntimeError(f'Refuse to overwrite {target}')
            p.rename(target)
        old.rmdir()
    progress=json.loads((HERE/'progress.json').read_text())
    assert progress['status']=='sample_complete' and progress['completed_trajectories']==105
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True,encoding='utf-8')
    (HERE/'requirements.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cu121\n'+freeze,encoding='utf-8')
    import torch
    env={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu','torch_cuda':torch.version.cuda,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'packages':{k:importlib.metadata.version(k) for k in ['timesfm','torch','numpy','pandas','scikit-learn','joblib','matplotlib','huggingface_hub','safetensors','nbformat','nbclient']},
        'inherited_PYTHONPATH':os.environ.get('PYTHONPATH',''),'original_environment_changed':False}
    (HERE/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
    main()
    kernel=HERE/'jupyter/kernels/timesfm-venv';kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'TimesFM isolated Python','language':'python'},indent=2),encoding='utf-8')
    path=HERE/'TimesFM_TEP_실험.ipynb'
    nb=nbformat.read(path,as_version=4);nb.metadata.kernelspec={'display_name':'TimesFM isolated Python','language':'python','name':'timesfm-venv'}
    client=NotebookClient(nb,timeout=240,kernel_name='timesfm-venv',resources={'metadata':{'path':str(HERE)}})
    client.execute();nbformat.write(nb,path)
    failures=[i for i,c in enumerate(nb.cells) if c.cell_type=='code' and any(o.output_type=='error' for o in c.outputs)]
    assert not failures
    (HERE/'notebook_execution.json').write_text(json.dumps({'passed':True,'executed_code_cells':sum(c.cell_type=='code' for c in nb.cells),'error_cells':failures,'kernel_python':sys.executable},indent=2),encoding='utf-8')
    verify()
    names=['experiment.py','audit.py','report.py','finalize.py','download_model.py','README.md','requirements.txt','TimesFM_TEP_실험.ipynb','experiment_summary.md','model_comparison.csv','variable_metrics.csv','sampling_manifest.json','environment.json','result_validation.json','preservation_check.json','notebook_execution.json','checkpoint/config.json','checkpoint/model.safetensors']
    names += [p.relative_to(HERE).as_posix() for p in (HERE/'figures').glob('*.png')]
    manifest={name:{'sha256':sha(HERE/name),'bytes':(HERE/name).stat().st_size} for name in names}
    (HERE/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Artifacts verified; notebook executed; original files preserved.',flush=True)

if __name__=='__main__':finalize()
