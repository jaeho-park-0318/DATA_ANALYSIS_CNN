"""Download public official artifacts into this experiment only; no user data uploaded."""
from pathlib import Path
import os, json, urllib.request, sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(HERE/'hf_cache')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import HfApi, snapshot_download

def main():
    sources=HERE/'sources';sources.mkdir(exist_ok=True)
    urls={
      'official_README.md':'https://raw.githubusercontent.com/google-research/timesfm/master/README.md',
      'official_pyproject.toml':'https://raw.githubusercontent.com/google-research/timesfm/master/pyproject.toml',
      'pypi.json':'https://pypi.org/pypi/timesfm/json',
      'github_commit.json':'https://api.github.com/repos/google-research/timesfm/commits/master',
    }
    for name,url in urls.items():
        with urllib.request.urlopen(url,timeout=60) as r: (sources/name).write_bytes(r.read())
    api=HfApi();info=api.model_info('google/timesfm-3.0-pytorch')
    (sources/'checkpoint_info.json').write_text(json.dumps({'id':info.id,'sha':info.sha,'license':(info.card_data or {}).get('license'),'files':[s.rfilename for s in info.siblings]},indent=2),encoding='utf-8')
    print('Checkpoint revision:',info.sha,flush=True)
    path=snapshot_download('google/timesfm-3.0-pytorch',revision=info.sha,local_dir=str(HERE/'checkpoint'),allow_patterns=['config.json','model.safetensors','README.md','LICENSE*'])
    print('Downloaded:',path,flush=True)

if __name__=='__main__':main()
