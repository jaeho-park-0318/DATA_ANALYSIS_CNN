"""공통 코호트 6개 모델의 실제 결과와 저장 예측 재현 Notebook."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os,time
import nbformat as nb
from nbclient import NotebookClient
from src.common import *

def main():
    os.environ['JUPYTER_PATH']=str(ROOT/'outputs/jupyter');os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'outputs/jupyter/runtime');os.environ['IPYTHONDIR']=str(ROOT/'outputs/ipython');os.environ['MPLCONFIGDIR']=str(ROOT/'outputs/.mplconfig')
    book=nb.v4.new_notebook();book.metadata['kernelspec']={'name':'phm2018-v3','display_name':'PHM2018 V3 (로컬)','language':'python'}
    blocks=[
      ('markdown','# 13 공통 코호트 재학습 — A/B/C 특징 비교\n\n같은 train/validation에서 로지스틱 회귀와 CUDA XGBoost 6개를 새로 학습했습니다. 이전 모델의 점수만 잘라 비교한 실험이 아닙니다. 이 Notebook은 실제 결과 표시와 저장 모델 재검증을 수행하며 전체 재학습은 자동 실행하지 않습니다.'),
      ('code',"from pathlib import Path\nimport sys,json\nROOT=Path.cwd()\nif ROOT.name=='notebooks': ROOT=ROOT.parent\nsys.path.insert(0,str(ROOT))\nimport numpy as np,pandas as pd\nfrom IPython.display import display,Image\nfrom src.common import TABLES,CFG\nassert Path(sys.executable).resolve()==Path(CFG['python']).resolve()\nOUT=ROOT/'outputs/stage3'\ndef table(name): return pd.read_parquet(TABLES/('stage3_'+name+'.parquet'))\ndef figure(name): display(Image(filename=str(ROOT/('outputs/figures/stage3/'+name+'.png'))))\ncontract=json.loads((OUT/'source_contract.json').read_text(encoding='utf-8'))\nassert contract['cohort']=='common' and contract['holdout_scored'] is False\ndisplay(json.loads((ROOT/'config/stage3_experiment.json').read_text(encoding='utf-8')))\nprint('실제 실행:',sys.executable)"),
      ('markdown','## 1. 같은 학습 행·정답·평가 시각\n\nC의900 raw 이력까지 가용한 교집합입니다. 최대 가용 코호트보다 입력 커버리지가 줄어듭니다. 전체 기록 이벤트232개와 입력 있는53개를 구분합니다.'),
      ('code',"checks=json.loads((ROOT/'outputs/validation/stage3_common_contract.json').read_text(encoding='utf-8'))\nassert all(c['passed'] for c in checks['checks'])\ndisplay(checks)\ndisplay(table('cohort_counts'))\nfigure('common_cohort')\nruns=table('training_summary')\nassert runs.train_rows.nunique()==runs.validation_all_rows.nunique()==runs.train_positive.nunique()==1"),
      ('markdown','## 2. 실제 학습과 분류 보조 지표\n\n숫자 표준화/범주 사전은 train만 fit했습니다. 같은 설정과 조기 종료 규칙을 적용했지만 채택 tree 수는 다릅니다. AP는 알려진 POS/NEG 후보에서만 계산합니다.'),
      ('code',"display(runs)\nfigure('classification')\ndisplay(table('score_states'))\nfor f in 'ABC':\n    cfg=json.loads((OUT/f'{f}_xgboost_booster_config.json').read_text(encoding='utf-8'))\n    assert cfg['learner']['generic_param']['device']=='cuda:0'\n    print(f+' XGBoost 실제 학습 장치:',cfg['learner']['generic_param']['device'])\nfigure('training_resources')"),
      ('markdown','## 3. 같은 경보 상한에서 비교\n\n5/10/20회는 실험 전에 정한 비교용 상한이며 실제 운영 허용치가 아닙니다. 각 상한 아래에서 검출 수 최대, 동률이면 경보 부담 최소를 선택합니다. NA/EXCLUDE도 전체 점수화에 포함합니다.'),
      ('code',"curve=table('alarm_curve');points=table('budget_points')\nassert curve.available_events.nunique()==curve.observed_support_raw.nunique()==curve.available_anchors.nunique()==1\nassert points.alarm_episodes_per_10000_raw.le(points.budget_per_10000_raw).all()\ndisplay(points[['run_id','budget_per_10000_raw','threshold','detected_events','available_events','recall_available','recall_recorded','alarm_episodes_per_10000_raw','alarms_UNKNOWN_START','median_lead_raw']])\nfigure('alarm_tradeoff')\nfigure('budget_comparison')\ndisplay(table('joint_frontier')[['run_id','threshold','detected_events','alarm_episodes_per_10000_raw']])\nfigure('joint_frontier')"),
      ('markdown','## 4. 같은 이벤트의 대응 비교\n\n01~10 장비 번호 군집을 재표집하고 M01/M02를 함께 유지합니다. 2,000회·seed42, 임계값 고정 조건부95% 구간입니다. 선택 낙관·다중 비교·불완전 고장 이력은 보정하지 않습니다.'),
      ('code',"display(table('paired_deltas'))\ndisplay(table('uncertainty'))\nfigure('paired_differences')\nfigure('event_matrix')\nprint('행렬은 상한10에서 입력 있는 동일53개 고장을 모두 표시합니다.')"),
      ('markdown','## 5. 저장 모델로 실제 공통 입력을 다시 예측\n\n각 특징의 첫 배치128행을 다시 읽습니다. 새 fit 없이 저장 전처리의 transform만 사용하고 6개 모델 예측이 원래 저장 점수와 일치하는지 확인합니다.'),
      ('code',"import joblib,xgboost as xgb\nfrom src.pretraining import iter_scoring_batches\nfrom src.features import schema\nreference=None\nfor f in 'ABC':\n    X,meta=next(iter_scoring_batches('temporal','validation',f,600,'common'))\n    X=X.iloc[:128];meta=meta.iloc[:128]\n    assert list(X)==schema()[f]\n    assert not {'time','Tool','audit_label','TTF'} & set(X)\n    if reference is None: reference=meta\n    else: pd.testing.assert_frame_equal(reference,meta)\n    prep=joblib.load(OUT/f'{f}_preprocessor.joblib')\n    transformed=prep.transform(X).astype('float32')\n    for kind in ['logistic','xgboost']:\n        rid=f'{f}_{kind}';model=joblib.load(OUT/(rid+'.joblib'))\n        if kind=='xgboost': score=model.get_booster().predict(xgb.DMatrix(transformed),iteration_range=(0,model.best_iteration+1))\n        else: score=model.predict_proba(transformed)[:,1]\n        saved=pd.read_parquet(OUT/(rid+'_validation.parquet'),columns=['Tool','time','score'])\n        expected=meta[['Tool','time']].merge(saved,on=['Tool','time'],how='left',validate='one_to_one')\n        np.testing.assert_allclose(score,expected.score,rtol=1e-6,atol=1e-7)\n        print(rid,': 실제 재계산 일치',len(score),'행')"),
      ('markdown','## 해석과 다음 방향\n\n공통 코호트에서는 비교용 경보 상한5에서 B XGBoost(13/53)가 A(9/53)보다 많이 검출했고, 상한10에서는 A(17/53)가 B(13/53)보다 많이 검출했습니다. C는 정해 둔5/10/20 상한에서 앞서지 않았지만 매우 낮은 경보 부담(약0.32회/10,000 raw, 검출5/53)에서는 비지배 후보로 남습니다. 대응 차이의 불확실성이 큽니다. A/B를 주 조정 후보로 유지하고 C의 저부담 참조 후보도 보존하며, 제한적 깊이·양성 가중치 조정 및 seed 민감도를 다음 단계로 권합니다.\n\n현재 결과는 validation 탐색이며 독립 일반화 성능이 아닙니다. 최대 가용 코호트와 공통 코호트의 점수를 단순 비교하지 않습니다. holdout은 모델·경보 규칙·운영점 고정 후 평가합니다. 시간 단위·정비 종료·이력 완전성·점수의 확률 보정은 미검증입니다.')
    ]
    book.cells=[nb.v4.new_code_cell(t) if k=='code' else nb.v4.new_markdown_cell(t) for k,t in blocks]
    path=ROOT/'notebooks/13_common_cohort_training.ipynb';nb.write(book,path);started=time.monotonic();NotebookClient(book,timeout=240,kernel_name='phm2018-v3',resources={'metadata':{'path':str(ROOT)}}).execute();nb.write(book,path)
    cells=[c for c in book.cells if c.cell_type=='code'];errors=[o for c in cells for o in c.outputs if o.output_type=='error'];result={'notebook':'13_common_cohort_training','code_cells':len(cells),'executed':sum(c.execution_count is not None for c in cells),'errors':len(errors),'elapsed_seconds':time.monotonic()-started,'python':CFG['python']}
    dump_json(ROOT/'outputs/validation/stage3_notebook.json',result);print(result)

if __name__=='__main__':main()
