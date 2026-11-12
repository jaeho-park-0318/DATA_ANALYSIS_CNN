"""교차검증을 순차 실행하면서 시스템 GPU·메모리 사용을 기록한다."""
from pathlib import Path
import sys,subprocess,time,json,psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.improvement_common import OUT,save_json

if __name__=='__main__':
    samples=[];start=time.monotonic()
    with (OUT/'cv_log.txt').open('a',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-X','utf8','-m','src.group_validation45'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            query=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,encoding='utf-8',errors='replace')
            samples.append({'elapsed':time.monotonic()-start,'gpu':query.stdout.strip(),'available_ram':psutil.virtual_memory().available})
            save_json(OUT/'resource_trace.json',samples)
            time.sleep(1)
    save_json(OUT/'cv_execution.json',{'returncode':process.returncode,'elapsed_seconds':time.monotonic()-start,'python':sys.executable,'samples':len(samples)})
    raise SystemExit(process.returncode)
