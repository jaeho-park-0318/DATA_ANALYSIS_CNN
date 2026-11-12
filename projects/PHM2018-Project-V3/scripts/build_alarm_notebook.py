"""첫 단계 경보 평가 Notebook을 생성하고 기존 로컬 커널에서 실제 실행한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os
import time
import nbformat as nb
from nbclient import NotebookClient
from src.common import *

def main():
    os.environ['JUPYTER_PATH']=str(ROOT/'outputs/jupyter');os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'outputs/jupyter/runtime');os.environ['IPYTHONDIR']=str(ROOT/'outputs/ipython');os.environ['MPLCONFIGDIR']=str(ROOT/'outputs/.mplconfig')
    book=nb.v4.new_notebook();book.metadata['kernelspec']={'name':'phm2018-v3','display_name':'PHM2018 V3 (로컬)','language':'python'}
    blocks=[
      ('markdown','# 11 경보 평가기 — 첫 단계 실제 결과\n\n고정 무경보/항상 경보 기준으로 평가기를 점검합니다. 학습된 모델의 예측 성능이 아닙니다. validation만 사용하고 holdout은 점수화하지 않습니다.'),
      ('code',"from pathlib import Path\nimport sys, json\nROOT=Path.cwd()\nif ROOT.name=='notebooks': ROOT=ROOT.parent\nsys.path.insert(0,str(ROOT))\nimport pandas as pd\nfrom IPython.display import display, Image\nfrom src.common import TABLES, CFG\nassert Path(sys.executable).resolve()==Path(CFG['python']).resolve()\nprint('실제 실행:',sys.executable)"),
      ('markdown','## 실제 이벤트 커버리지와 전체 점수화 대상\n\n기록 이벤트/입력 있는 이벤트/검출 이벤트를 구분합니다. 미확인 라벨을 정상으로 바꾸지 않습니다.'),
      ('code',"overview=pd.read_parquet(TABLES/'alarm_stage1_overview.parquet')\ndisplay(overview)\ndisplay(Image(filename=str(ROOT/'outputs/figures/alarm_stage1/event_coverage.png')))\ndisplay(Image(filename=str(ROOT/'outputs/figures/alarm_stage1/scoring_population.png')))"),
      ('markdown','## 경보 부담과 고정 기준\n\n항상 경보의 첫 발행이 예측 창보다 이르면 새 검출로 소급하지 않습니다. 명목 관측 지지길이는 실가동시간이 아닙니다.'),
      ('code',"display(Image(filename=str(ROOT/'outputs/figures/alarm_stage1/baseline_checks.png')))\ndisplay(Image(filename=str(ROOT/'outputs/figures/alarm_stage1/alarm_categories.png')))\nepisodes=pd.read_parquet(TABLES/'alarm_stage1_episodes.parquet')\ndisplay(episodes.head(12))"),
      ('markdown','## 실제 고장 구간과 검출 근거'),
      ('code',"events=pd.read_parquet(TABLES/'alarm_stage1_events.parquet')\nselected=events[events.Tool.eq('03M02') & events.scheme.eq('temporal') & events.feature_set.eq('B') & events.baseline.eq('always_alarm')]\ndisplay(selected)\ndisplay(Image(filename=str(ROOT/'outputs/figures/alarm_stage1/actual_event.png')))"),
      ('markdown','## 학습 전용 특징과 평가 전용 라벨 분리\n\n아래 셀은 점수를 학습하지 않고 실제 첫 배치의 입력 계약만 확인합니다.'),
      ('code',"from src.pretraining import iter_scoring_batches\nX, metadata=next(iter_scoring_batches('temporal','validation','B',600,tool='03M02'))\nassert 'audit_label' not in X and 'time' not in X\nprint('입력:',X.shape,'평가 metadata:',metadata.shape)\ndisplay(X.head())\ndisplay(metadata.head())\nchecks=json.loads((ROOT/'outputs/validation/alarm_stage1_checks.json').read_text(encoding='utf-8'))\nassert all(c['passed'] for c in checks['checks'])\nprint('실제 분모/스키마/경계 검사:',len(checks['checks']))"),
      ('markdown','## CUDA 우선 학습 준비\n\n장치 인식/빌드 확인 결과입니다. 실제 GPU 학습·VRAM 사용 검증은 다음 단계에서 수행합니다.'),
      ('code',"profile=json.loads((ROOT/'outputs/validation/local_compute_profile.json').read_text(encoding='utf-8'))\ndisplay(profile)\nprint('학습 설정:',json.loads((ROOT/'config/local_compute.json').read_text(encoding='utf-8')))\nassert profile['training_smoke_test_performed'] is False")
    ]
    book.cells=[nb.v4.new_code_cell(text) if kind=='code' else nb.v4.new_markdown_cell(text) for kind,text in blocks]
    path=ROOT/'notebooks/11_alarm_evaluation.ipynb';nb.write(book,path);start=time.monotonic()
    NotebookClient(book,timeout=240,kernel_name='phm2018-v3',resources={'metadata':{'path':str(ROOT)}}).execute();nb.write(book,path)
    cells=[c for c in book.cells if c.cell_type=='code'];errors=[o for c in cells for o in c.outputs if o.output_type=='error']
    result={'notebook':'11_alarm_evaluation','code_cells':len(cells),'executed':sum(c.execution_count is not None for c in cells),'errors':len(errors),'elapsed_seconds':time.monotonic()-start,'python':CFG['python']}
    dump_json(ROOT/'outputs/validation/alarm_stage1_notebook.json',result);print(result)

if __name__=='__main__':main()
