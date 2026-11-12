"""학습 없이 로컬 CPU/RAM/GPU와 CUDA 드라이버 진입점을 확인한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ctypes
import subprocess
import psutil
import xgboost
from src.common import *

def main():
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    cuda=ctypes.WinDLL('nvcuda.dll');status=cuda.cuInit(0);count=ctypes.c_int();version=ctypes.c_int()
    if status==0:
        count_status=cuda.cuDeviceGetCount(ctypes.byref(count));version_status=cuda.cuDriverGetVersion(ctypes.byref(version))
    else:count_status=version_status=None
    r={'date':'2026-11-05','gpu_query':gpu,'ram_total_bytes':psutil.virtual_memory().total,'ram_available_bytes':psutil.virtual_memory().available,'physical_cpu_cores':psutil.cpu_count(logical=False),'logical_cpu_cores':psutil.cpu_count(),'xgboost_version':xgboost.__version__,'xgboost_build':xgboost.build_info(),'cuInit_status':status,'cuDeviceGetCount_status':count_status,'cuda_devices':count.value,'cuDriverGetVersion_status':version_status,'cuda_driver_api_version':version.value,'training_smoke_test_performed':False,'note':'장치/드라이버 탐지와 CUDA 빌드 확인. 실제 XGBoost CUDA 학습 및 VRAM 요구량은 다음 단계에서 검증.'}
    dump_json(ROOT/'outputs/validation/local_compute_profile.json',r);print(json.dumps(r,ensure_ascii=False))

if __name__=='__main__':main()
