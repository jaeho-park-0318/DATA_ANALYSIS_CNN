"""동일한 로컬 학습 행렬로 CPU/CUDA 동작과 소요시간을 비교한다."""
import json, sys, os, subprocess, warnings
import numpy as np
import psutil
from xgboost import XGBClassifier, build_info, __version__
from jupyter_client.kernelspec import KernelSpecManager
from src.improvement_common import *
from src.model_common import load_split

def run():
    initialize()
    path=OUT/'hardware.json'
    if path.exists():return read_json(path)
    result={'python':sys.executable,'prefix':sys.prefix,'python_version':sys.version,'execution':'로컬 Windows 프로세스와 로컬 파일시스템에서 실행. Codex UI의 Local 표시 자체는 별도 확인하지 않음.',
            'physical_cores':psutil.cpu_count(logical=False),'logical_cores':psutil.cpu_count(),'memory':psutil.virtual_memory()._asdict(),
            'kernels':KernelSpecManager().get_all_specs(),'xgboost':__version__,'build':build_info(),'runs':[]}
    for key,cmd in [('nvidia_smi',['nvidia-smi']),('cpu_name',['powershell','-NoProfile','-Command','Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name'])]:
        result[key]=subprocess.run(cmd,capture_output=True,text=True).stdout
    frame=load_split('split_tool',600,'train')
    # 첫 개발 장비들의 시간 순서를 유지한 동일 행렬이며 속도 측정 전용이다.
    frame=frame.iloc[:150000];x=matrix(frame,configs()['C']);y=frame[TARGET].to_numpy(dtype=np.int8)
    result['benchmark']={'rows':len(y),'features':x.shape[1],'positive':int(y.sum()),'rounds':100,'repetitions':3,'threads':THREADS,'early_stopping':False}
    params=dict(n_estimators=100,max_depth=4,max_bin=128,tree_method='hist',n_jobs=THREADS,random_state=42,learning_rate=.05)
    for device in ['cpu','cuda']:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                warm=XGBClassifier(**{**params,'n_estimators':5},device=device).fit(x[:5000],y[:5000])
                actual=json.loads(warm.get_booster().save_config())['learner']['generic_param']['device']
                if device=='cuda' and not actual.startswith('cuda'):raise RuntimeError('CUDA 요청이 CPU로 대체됨: '+actual)
                for repeat in range(3):
                    with Resources() as monitor:
                        model=XGBClassifier(**params,device=device).fit(x,y)
                    folder=OUT/'benchmark';folder.mkdir(exist_ok=True)
                    (folder/f'{device}_{repeat}_booster.json').write_text(model.get_booster().save_config(),encoding='utf-8')
                    result['runs'].append({'device':device,'repeat':repeat,'actual_device':json.loads(model.get_booster().save_config())['learner']['generic_param']['device'],**monitor.result})
                    progress(f'벤치마크 {device} {repeat+1}/3: {monitor.result["seconds"]:.3f}초')
                result[device+'_warnings']=[str(w.message) for w in caught]
        except Exception as exc:
            result[device+'_error']=repr(exc)
            result[device+'_warnings']=[str(w.message) for w in caught]
            progress(f'{device} 학습 실패: {exc}')
    med={d:float(np.median([r['seconds'] for r in result['runs'] if r['device']==d])) for d in ['cpu','cuda'] if any(r['device']==d for r in result['runs'])}
    result['median_seconds']=med;result['selected_device']=min(med,key=med.get)
    result['speedup_cpu_over_gpu']=med['cpu']/med['cuda'] if 'cuda' in med else None
    save_json(path,result);return result

if __name__=='__main__':run()
