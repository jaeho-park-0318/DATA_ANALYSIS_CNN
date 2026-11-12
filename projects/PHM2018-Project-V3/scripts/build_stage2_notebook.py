"""저장된 실제 학습·경보 결과를 재검증하는 실행 Notebook."""
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
      ('markdown','# 12 첫 모델 학습 — A/B × 로지스틱/XGBoost\n\n실제 로컬 학습 결과를 읽고 재검증합니다. XGBoost는 CUDA, 로지스틱은 CPU를 사용했습니다. H600/L60·보수적 라벨·시간 순서 분할이며 holdout은 사용하지 않았습니다. 전체 재학습은 `scripts/train_stage2.py`로 실행합니다. 이 Notebook은 자동 재학습하지 않습니다.'),
      ('code',"from pathlib import Path\nimport sys,json\nROOT=Path.cwd()\nif ROOT.name=='notebooks': ROOT=ROOT.parent\nsys.path.insert(0,str(ROOT))\nimport numpy as np, pandas as pd\nfrom IPython.display import display,Image\nfrom src.common import TABLES,CFG\nassert Path(sys.executable).resolve()==Path(CFG['python']).resolve()\nOUT=ROOT/'outputs/stage2'\nprint('실제 실행:',sys.executable)\ncontract=json.loads((OUT/'source_contract.json').read_text(encoding='utf-8'))\nassert contract['holdout_scored'] is False\ndisplay(contract)"),
      ('markdown','## 1. 경보 로직의 독립 검증과 실제 CUDA 학습\n\n시험 점수는 경보 검증에만 썼고 모델 성능에 섞지 않았습니다. CUDA 장치 이름만 확인한 것이 아니라 실제 smoke 학습 후 booster 장치를 확인했습니다.'),
      ('code',"audit=json.loads((ROOT/'outputs/validation/alarm_independent_audit.json').read_text(encoding='utf-8'))\nassert all(c['passed'] for c in audit['checks'])\nprint('독립 검사 통과:',len(audit['checks']))\nsmoke=json.loads((OUT/'cuda_training_smoke.json').read_text(encoding='utf-8'))\nassert smoke['success'] and smoke['device']=='cuda:0'\ndisplay({k:v for k,v in smoke.items() if k!='booster_config'})\nruns=pd.read_parquet(TABLES/'stage2_training_summary.parquet')\nassert runs[runs.model.eq('xgboost')].device.eq('cuda:0').all()\ndisplay(runs)\ndisplay(Image(filename=str(ROOT/'outputs/figures/stage2/local_resources.png')))"),
      ('markdown','## 2. 분류 지표와 조기 종료\n\nAP/ROC-AUC는 알려진 POS/NEG 후보에서만 계산합니다. XGBoost aucpr과 sklearn AP는 계산이 다릅니다. B의 이른 조기 종료 원인은 후속 실험에서 분리해야 합니다.'),
      ('code',"display(Image(filename=str(ROOT/'outputs/figures/stage2/classification.png')))\ndisplay(Image(filename=str(ROOT/'outputs/figures/stage2/learning_history.png')))\ndisplay(pd.read_parquet(TABLES/'stage2_score_states.parquet'))"),
      ('markdown','## 3. 전체 가용 구간의 경보 곡선\n\nNA/EXCLUDE를 유지합니다. 오래 유지된 경보를 뒤의 예측 창에서 새 검출로 소급하지 않습니다. A/B 최대 가용 코호트가 달라 순수 특징 효과가 아닙니다.'),
      ('code',"curve=pd.read_parquet(TABLES/'stage2_alarm_curve.parquet')\npoints=pd.read_parquet(TABLES/'stage2_operating_points.parquet')\ndisplay(points[['run_id','operating_point','threshold','detected_events','available_events','recall_recorded','recall_available','alarm_episodes','alarm_episodes_per_10000_raw','median_lead_raw']])\ndisplay(Image(filename=str(ROOT/'outputs/figures/stage2/alarm_tradeoff.png')))\nassert (curve.detected_events<=curve.available_events).all()\nassert curve.recorded_events.eq(232).all()\nprint('전체 임계값 후보:',len(curve))"),
      ('markdown','## 4. 미확인 경보·장비 편차·불확실성\n\n20 Tool 군집 재표집 2,000회. 임계값 고정 조건부 구간이며 선택 낙관, 장비쌍 의존성, 이력 불완전성을 보정하지 못합니다. 미확인 경보는 확정 오경보가 아닙니다.'),
      ('code',"display(Image(filename=str(ROOT/'outputs/figures/stage2/alarm_uncertainty.png')))\ndisplay(pd.read_parquet(TABLES/'stage2_uncertainty.parquet'))\ntool_results=pd.read_parquet(TABLES/'stage2_tool_event_results.parquet')\ndisplay(tool_results[tool_results.run_id.eq('B_xgboost') & tool_results.operating_point.eq('검출최대의100%')])\nunseen=pd.read_parquet(TABLES/'stage2_unseen_process.parquet')\ndisplay(unseen.groupby('feature_set')[['rows','unknown_recipe','unknown_stage','unseen_combination']].sum())"),
      ('markdown','## 5. 실제 검출·미탐 사례\n\n원본과 저장된 전체 시계열 경보를 확대합니다. 시각·행 수는 아래 표에 있습니다.'),
      ('code',"display(pd.read_parquet(TABLES/'stage2_example_cases.parquet'))\ndisplay(Image(filename=str(ROOT/'outputs/figures/stage2/event_examples.png')))"),
      ('markdown','## 6. 저장 모델·전처리로 실제 입력 예측 재현\n\n각 특징의 실제 validation 첫 배치에서 128행을 읽고 4개 저장 모델의 예측이 저장 점수와 일치하는지 검증합니다. 전처리는 fit하지 않고 transform만 호출합니다.'),
      ('code',"import joblib,xgboost as xgb\nfrom src.pretraining import iter_scoring_batches\nfrom src.features import schema\nfor family in 'AB':\n    X,meta=next(iter_scoring_batches('temporal','validation',family,600))\n    X=X.iloc[:128];meta=meta.iloc[:128]\n    assert list(X.columns)==schema()[family]\n    assert not {'time','Tool','audit_label','TTF'} & set(X)\n    prep=joblib.load(OUT/f'{family}_preprocessor.joblib')\n    transformed=prep.transform(X).astype('float32')\n    for kind in ['logistic','xgboost']:\n        rid=f'{family}_{kind}'\n        model=joblib.load(OUT/(rid+'.joblib'))\n        if kind=='xgboost':\n            score=model.get_booster().predict(xgb.DMatrix(transformed),iteration_range=(0,model.best_iteration+1))\n        else:\n            score=model.predict_proba(transformed)[:,1]\n        saved=pd.read_parquet(OUT/(rid+'_validation.parquet'),columns=['Tool','time','score'])\n        matched=meta[['Tool','time']].merge(saved,on=['Tool','time'],how='left',validate='one_to_one')\n        np.testing.assert_allclose(score,matched.score,rtol=1e-6,atol=1e-7)\n        print(rid,': 저장 예측과 실제 재계산 일치',len(score),'행')"),
      ('markdown','## 다음 실험\n\nB XGBoost를 우선 검토하되 A/B/C 공통 코호트에서 각각 재학습해 특징 효과를 분리합니다. 이후 제한된 가중치/깊이 조정, seed 민감도, 장비 그룹 일반화, 고정 후 holdout 순으로 진행합니다. 현재 validation은 조기 종료/임계값/모델 비교에 재사용하므로 독립 성능이 아닙니다. 시간 단위·정비 종료·고장 이력 완전성은 미검증입니다.')
    ]
    book.cells=[nb.v4.new_code_cell(t) if k=='code' else nb.v4.new_markdown_cell(t) for k,t in blocks]
    path=ROOT/'notebooks/12_initial_model_training.ipynb';nb.write(book,path);started=time.monotonic();NotebookClient(book,timeout=240,kernel_name='phm2018-v3',resources={'metadata':{'path':str(ROOT)}}).execute();nb.write(book,path)
    cells=[c for c in book.cells if c.cell_type=='code'];errors=[o for c in cells for o in c.outputs if o.output_type=='error']
    result={'notebook':'12_initial_model_training','code_cells':len(cells),'executed':sum(c.execution_count is not None for c in cells),'errors':len(errors),'elapsed_seconds':time.monotonic()-started,'python':CFG['python']}
    dump_json(ROOT/'outputs/validation/stage2_notebook.json',result);print(result)

if __name__=='__main__':main()
