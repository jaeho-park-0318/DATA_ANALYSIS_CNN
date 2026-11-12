"""4.5단계 전용 경로, 자원 감시, 기존 결과 보존 검사."""
from pathlib import Path
import hashlib, json, os, sys, time, threading, subprocess
import psutil
from src.model_common import save_json, read_json, configs, matrix
from src.dataset_common import ROOT, DEST, event_table, snapshot, ensure_compatible

OUT=ROOT/'outputs/stage4_5'
TARGET='label_any_h600raw'
THREADS=4

def initialize():
    assert Path(sys.prefix).name.lower()=='phm2018'
    OUT.mkdir(parents=True,exist_ok=True)
    ensure_compatible()
    if not (OUT/'preservation.json').exists():
        files=[]
        for folder in ['src','scripts','notebooks','reports','outputs/stage4']:
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.ipynb','.md','.ubj','.joblib','.json','.parquet']:
                    files.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        save_json(OUT/'preservation.json',{'files':files,'raw':snapshot()})

class Resources:
    """동일 프로세스 RSS와 시스템 RAM을 주기적으로 기록한다."""
    def __enter__(self):
        self.start=time.perf_counter();self.stop=threading.Event();self.samples=[]
        def poll():
            while not self.stop.is_set():
                self.samples.append({'elapsed':time.perf_counter()-self.start,'rss':psutil.Process().memory_info().rss,'available':psutil.virtual_memory().available})
                self.stop.wait(.5)
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join()
        self.result={'seconds':time.perf_counter()-self.start,'peak_rss_bytes':max(x['rss'] for x in self.samples),'min_available_bytes':min(x['available'] for x in self.samples),'samples':self.samples}

def progress(message):
    print(message,flush=True)
    with (OUT/'progress.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'message':message,'rss':psutil.Process().memory_info().rss,'available':psutil.virtual_memory().available},ensure_ascii=False)+'\n')
