"""Build a portable Korean narrative notebook from completed, frozen results."""
from pathlib import Path
import sys, os, json, base64, io, hashlib, asyncio
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
os.environ['IPYTHONDIR']=str(HERE/'ipython')
os.environ['JUPYTER_PATH']=str(HERE/'jupyter')
os.environ['JUPYTER_RUNTIME_DIR']=str(HERE/'jupyter_runtime')
os.environ['MPLCONFIGDIR']=str(HERE/'mpl_cache')
if sys.platform=='win32':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import numpy as np
import pandas as pd
import nbformat
from nbclient import NotebookClient

def build():
    metrics=pd.read_csv(HERE/'model_comparison.csv')
    assert metrics.status.eq('sample_complete').all()
    for name in ['result_validation.json','preservation_check.json','notebook_execution.json']:
        assert json.loads((HERE/name).read_text())['passed']
    files=['model_comparison.csv','variable_metrics.csv','sampling_manifest.json','environment.json','pilot_fp32_validation.json','throughput_benchmark.csv','result_validation.json','preservation_check.json','progress.json']
    sources={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in files}
    # Additional descriptive summaries derived solely from already-saved predictions.
    rows=[]
    for path in sorted((HERE/'predictions').glob('*.npz')):
        fault=int(path.stem.split('_')[0][5:]);run=int(path.stem.split('_')[1][3:])
        z=np.load(path)
        for model in ['Persistence','LSTM','TimesFM']:
            e=(z[model]-z['actual']).astype(np.float64)
            rows.append(dict(fault=fault,run=run,model=model,target_rows=len(e),sse=float(np.square(e).sum()),sae=float(np.abs(e).sum())))
    snapshot={
        'metrics':metrics.to_json(orient='records'),
        'variables':pd.read_csv(HERE/'variable_metrics.csv').query("cohort == 'all'").to_json(orient='records'),
        'trajectory_errors':pd.DataFrame(rows).to_json(orient='records',double_precision=15),
        'sampling':json.loads((HERE/'sampling_manifest.json').read_text()),
        'environment':json.loads((HERE/'environment.json').read_text()),
        'checks':{name:json.loads((HERE/name).read_text()) for name in ['result_validation.json','preservation_check.json','pilot_fp32_validation.json']},
        'source_sha256':sources,
    }
    cells=[]
    def md(s):cells.append(nbformat.v4.new_markdown_cell(s.strip()))
    def code(s):cells.append(nbformat.v4.new_code_cell(s.strip()))
    def picture(name,caption):
        p=HERE/'figures'/name
        c=nbformat.v4.new_markdown_cell(f'![{caption}](attachment:{name})\n\n{caption}')
        c.attachments={name:{'image/png':base64.b64encode(p.read_bytes()).decode()}}
        cells.append(c)
    md('''# TEP 공정 변수 예측: 기존 LSTM과 Google TimesFM 비교

**과정·실험 조건·실제 결과를 함께 정리한 종합 노트북** · 기준일 2026-11-12 (Asia/Seoul)

질문은 “과거20개 관측으로 3분 뒤의52개 공정 변수값을 예측할 때, TimesFM Zero-shot이 기존 LSTM보다 정확한가?”이다.

**대표 표본에서는 TimesFM의 전체 RMSE가34.39%, MAE가18.75% 낮았다. 정상 구간은 LSTM, 고장 이후는 TimesFM이 더 정확했다.** 전체987만시점 평가는 미실행이므로 전체 Test의 확정 결론으로 일반화하지 않는다.

이 노트북은 완료된 실험의 표·검증 기록·그림을 내장한다. 원 프로젝트를 옮기지 않아도 저장된 출력과 그림을 읽을 수 있다. 아래 셀은 내장 자료로 비교를 재계산하며, 모델 설치·학습·대규모 추론을 다시 시작하지 않는다. 실제 GPU 추론이 포함된 실행용 노트북은 `timesfm_experiment/TimesFM_TEP_실험.ipynb`에 별도로 보존했다.

읽는 순서: 연구 질문 → 수행 과정 → 동일 비교 조건 → 모델 차이 → 대표 표본 → 성능·그림 → 해석·한계 → 검증·재현.''')
    md('## 1. 분석 자료와 실행 결과 불러오기\n\n다음 셀의 스냅샷은 완료된 CSV와 JSON을 내장한 것이다. 모든 오차는 낮을수록 좋고, 통합 지표의 단위는 정상 Train의 변수별 표준편차다.')
    code('''import json, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
pd.set_option('display.max_columns',12)
pd.set_option('display.max_rows',60)
SNAPSHOT = json.loads('''+repr(json.dumps(snapshot,ensure_ascii=False))+''')
metrics=pd.read_json(io.StringIO(SNAPSHOT['metrics']))
variables=pd.read_json(io.StringIO(SNAPSHOT['variables']))
trajectory_errors=pd.read_json(io.StringIO(SNAPSHOT['trajectory_errors']))
MODELS=['Persistence','LSTM','TimesFM']
COHORTS={'all':'전체 표본','healthy_context':'정상 구간','post_fault':'고장 이후','faultfree_only':'정상 전용 실행'}
display(Markdown('**완료된 표본 평가:** 105개 시계열 · 9,870개 target · 52개 변수 · 전체 Test 평가는 미실행'))''')
    md('''## 2. 실제로 수행한 과정

기존 CNN·LSTM·GRU·XGBoost와 연구 결과를 보존하면서 `timesfm_experiment`에 실험을 독립적으로 추가했다. 다음은 계획이 아니라 완료된 작업과 미실행 단계를 구분한 기록이다.''')
    code('''workflow=pd.DataFrame([
 ['1','프로젝트 조사','완료','파일1,212개 목록화; 기존 노트북17·18과 LSTM 실행 코드 확인'],
 ['2','데이터·조건 확인','완료','Harvard TEP52변수·3분 간격, Train400/Validation100/공식 Test500 family'],
 ['3','독립 환경 구성','완료','별도 .venv에 TimesFM3.0.2, CUDA12.1 PyTorch2.5.1 설치'],
 ['4','공식 가중치·GPU 확인','완료','공식 TimesFM3.0 가중치 다운로드; RTX3060 Laptop6GB 추론'],
 ['5','소규모 동작 검증','완료','scaler 일치·미래값 제외·배치 독립성 검사, FP32/BF16 처리속도 탐색'],
 ['6','대표 표본 평가','완료','5개 Test family×21종×94시점에서 세 모델 동일 조건 추론'],
 ['7','지표·그림·검증','완료','전체/구간/변수별 지표, 300dpi PNG6종, 예측 재계산·파일 보존 검사'],
 ['8','전체 Test TimesFM 평가','미실행','987만시점; 측정 속도상 수일 예상, 사용자 요청에 따라 표본 비교 우선']
],columns=['순서','단계','상태','내용'])
display(workflow)''')
    md('''## 3. 기존 프로젝트에서 확인한 근거

| 확인 자료 | 이번 비교에 사용한 내용 |
|---|---|
| `notebooks/17_CNN_기능확장_종합안내.ipynb` | CNN 분류와 별도 LSTM 회귀의 구분,20×52→3분 뒤52변수, 표본/전체 평가의 차이 |
| `notebooks/18_TEP_PHM_전체모델_종합비교.ipynb` | 기존 모델 역할·지표·분할의 종합 설명; 회귀 LSTM의 기존 평가표 |
| `experiments/next_step_forecasting/run.py` | 실제 정규화·창 생성·LSTM 구조·학습·Test 평가 구현 |
| `experiments/next_step_forecasting/models/lstm_s42.pt` | 대표 seed42의 기존 예측 LSTM 가중치 |
| `experiments/next_step_forecasting/models/scaler.joblib` | 정상 Training400개 실행으로 fit한 기존 StandardScaler |
| `data/processed/phase2/split_manifest.json` | 기존 Train/Validation/Test family 분할 |

원본 데이터는 프로젝트 루트의 `TEP_FaultFree_Training.RData`, `TEP_Faulty_Training.RData`, `TEP_FaultFree_Testing.RData`, `TEP_Faulty_Testing.RData`다. 기존 `training.npy`와 `testing.npy` 캐시를 읽었으며 원본이나 캐시를 수정하지 않았다. 별도 GitHub의41변수·1분 간격 데이터와 혼합하지 않았다.''')
    md('''## 4. 두 모델을 동일하게 비교한 조건

```text
하나의 faultNumber · simulationRun 안에서
실제 관측 [t-19, …, t] × 52변수
               ↓ 기존 정상 Train StandardScaler
          ┌────┴───────────┐
       기존 LSTM      TimesFM Zero-shot
          └────┬───────────┘
     같은 t+1 실제값 × 52변수와 비교
     Persistence 기준선: 다음값 = t의 관측값
```

Train과 Test에 같은 simulationRun 번호가 있어도 공식 파일 분할이 다르면 별도 RNG family다. 식별자는 `(공식 파일 분할, faultNumber, simulationRun, sample)`로 해석한다.''')
    code('''conditions=pd.DataFrame([
 ['공정 변수','XMEAS41개 + XMV11개, 기존 열 순서 동일'],
 ['관측 간격','3분'],['입력 / 출력','과거20시점×52변수 / 다음1시점×52변수'],
 ['Train / Validation','공식 Training의400 /100 family; 정상 시나리오 사용'],
 ['Test','공식 Testing의500 family에서 사전 고정한 대표 표본'],
 ['외부 정규화','정상 Train400×500 =200,000행으로 fit한 기존 StandardScaler 재사용'],
 ['정상 구간','정상 전용 시나리오 전체 + 고장 시나리오 target sample≤160'],
 ['고장 이후','faultNumber>0 이면서 target sample>160'],
 ['동일성','세 모델이 같은52변수·입력 창·target key를 사용'],
 ['주 비교 정밀도','float32 입력·가중치, 기존 PyTorch CUDA/cuDNN 기본 설정 유지'],
 ['전체 지표 단위','정상 Train의 변수별 표준편차 단위 (Train SD)']
],columns=['항목','조건'])
display(conditions)''')
    md('''## 5. LSTM과 TimesFM은 어떻게 다른가?

입력 관측과 예측 시점은 같지만, 구조와 사전 학습 경험이 같다는 뜻은 아니다. 이번 실험은 기존 TEP 전용 정상 학습 LSTM과 다른 데이터로 사전학습된 기반 모델의 Zero-shot 성능을 비교한다.''')
    code('''models=pd.DataFrame([
 ['모델 역할','TEP 전용 다음 시점 회귀','사전학습 기반 모델의 Zero-shot 회귀'],
 ['이번에 사용한 학습','기존 정상 Train 학습 가중치 재사용','추가 학습 없음'],
 ['구조 / 출력','2층 LSTM hidden64; 현재값+학습 변화량','TimesFM3.0 Transformer; 공식 중앙값 예측'],
 ['기존 학습 설정','dropout0.1, Huber, AdamW lr0.001, 최대12epoch/patience3','적용하지 않음'],
 ['선택 기준','정상 Validation MSE; 대표 seed42 사전 지정','공식 가중치·고정 추론 설정'],
 ['배열 형식','배치×20시점×52변수','배치 내 각 context는52변수×20시점'],
 ['다변량 처리','52변수를 같은 시계열 입력으로 처리','공식 Evaluator의32+20변수 chunk'],
 ['내부 처리','LSTM recurrent state와 residual 출력','문맥 기반 RevIN·detrending·patch mask 패딩'],
 ['이번 실행','동일 표본에서 저장 모델 재추론','공식 가중치로 동일 표본 실제 GPU 추론']
],columns=['비교 항목','기존 LSTM','Google TimesFM'])
display(models)''')
    md('''**TimesFM 다변량 입력의 중요한 조건:** 패키지는 `timesfm==3.0.2`, 모델은 `google/timesfm-3.0-pytorch`이며 revision은 `43046b85ec22d584a13f8098c2ed39c889e129c2`다. 공식 Evaluator는52개를 앞32개와 뒤20개로 나누고 마지막20개를 반복 패딩한 뒤 복원한다. 모든52변수를 예측하지만 한 번에52개 전체에 공동 attention을 적용한 설정은 아니다.

20개 실제 관측만 전달하고32길이 patch의 부족분은 모델이 mask 패딩한다. 추가 관측을 생성하지 않았다. `horizon=1`, 대칭 평균 없음, 양수 clipping 없음, 추가 z-normalization 없음, 미래 공변량·XReg·fine-tuning 없음이다. TimesFM 내부의 관측 문맥 기반 처리는 Test 전체 통계로 scaler를 새로 fit하는 것과 다르다.''')
    md('## 6. 대표 표본은 어떻게 골랐나?\n\n모델 오차를 확인하기 전에 난수 seed20261112로 Test family5개를 선택했다. 각 시나리오에서 같은 family를 사용했고, 각 시계열의 유효940개 target을 길이10의94개 구간으로 나누어 구간 안에서1개씩 무작위 선택했다. 고정 stride10의 주기적 관측 위치에만 의존하지 않도록 한 설계다.')
    code('''s=SNAPSHOT['sampling']
sample=pd.DataFrame([
 ['선택 simulationRun',', '.join(map(str,s['simulationRuns']))],
 ['시나리오','정상0 + Fault1~20 =21종'],['시계열 수',s['trajectories']],
 ['시계열당 평가 시점',s['targets_per_trajectory']],['총 target 수',s['expected_target_rows']],
 ['정상 구간 target',int(metrics.query("model=='LSTM' and cohort=='healthy_context'").target_rows.iloc[0])],
 ['고장 이후 target',int(metrics.query("model=='LSTM' and cohort=='post_fault'").target_rows.iloc[0])],
 ['전체 dense Test 대비 비율',f"{100*s['expected_target_rows']/s['full_dense_target_rows']:.1f}%"],
],columns=['표본 설계','값'])
display(sample)
assert s['trajectories']==105 and s['expected_target_rows']==9870
assert metrics.query("cohort=='all'").target_rows.eq(9870).all()''')
    md('''5개 family와 시나리오 전체를 포함했지만500개 Test family 전체를 대체하지는 못한다. 또한 rolling 입력 창이 겹치므로9,870개 시점이 모두 독립 관측인 것은 아니다. 아래는 표본에 대한 기술통계이며 통계적 유의성을 주장하지 않는다.''')
    md(r'''## 7. 평가 지표와 단위

정상 Train 평균과 표준편차로 변수별 표준화를 수행했다. 표준화된 실제값을 $z_{ij}$, 예측값을 $\hat z_{ij}$라 하면:

$$\mathrm{RMSE}=\sqrt{\frac{1}{N\cdot52}\sum_{i=1}^{N}\sum_{j=1}^{52}(\hat z_{ij}-z_{ij})^2}$$

$$\mathrm{MAE}=\frac{1}{N\cdot52}\sum_{i=1}^{N}\sum_{j=1}^{52}|\hat z_{ij}-z_{ij}|$$

전체 RMSE는52개 변수별 RMSE의 산술평균이 아니다. RMSE는 큰 오차에 더 민감하며 MAE는 일반적인 절대 편차를 보여준다. 서로 물리 단위가 다른52변수를 통합하기 때문에 Train SD 단위로 비교한다. 변수별 원 스케일 지표는 CSV에 있으며 Harvard 파일의 물리 단위가 독립 검증되지 않아 임의로 kPa·℃ 등을 붙이지 않았다.''')
    md('## 8. 전체 표본 결과: 같은9,870개 target에서 비교')
    code('''overall=metrics.query("cohort=='all'").set_index('model').loc[MODELS]
display(overall[['target_rows','normalized_rmse','normalized_mae']].rename(columns={'target_rows':'정답 시점 수','normalized_rmse':'RMSE (Train SD)','normalized_mae':'MAE (Train SD)'}))
improvement=100*(1-overall.loc['TimesFM',['normalized_rmse','normalized_mae']]/overall.loc['LSTM',['normalized_rmse','normalized_mae']])
display(Markdown(f"**TimesFM의 LSTM 대비 오차 감소:** RMSE {improvement.iloc[0]:.2f}%, MAE {improvement.iloc[1]:.2f}%."))''')
    picture('model_rmse_comparison.png','전체 대표 표본의 모델별 RMSE. 세 모델의 target과52변수가 동일하다.')
    picture('model_mae_comparison.png','전체 대표 표본의 모델별 MAE. Persistence도 함께 확인해 학습 모델의 실제 이득을 판단한다.')
    md('''## 9. 정상 구간과 고장 이후를 나누면?

**정상 구간에서는 LSTM이 더 정확하다.** TimesFM의 RMSE/MAE가 각각8.83%/6.99% 높다. **고장 이후에는 TimesFM이 더 정확하다.** RMSE/MAE가 각각37.52%/22.88% 낮다.

고장 이후8,000개가 표본의81.1%이므로 이 구간의 개선이 전체 지표에 크게 반영됐다. 따라서 “TimesFM이 정상과 고장 모두에서 우수하다”는 해석은 맞지 않는다.''')
    code('''segments=metrics.copy()
segments['평가 구간']=segments.cohort.map(COHORTS)
display(segments[['평가 구간','model','target_rows','normalized_rmse','normalized_mae']].rename(columns={'model':'모델','target_rows':'target 수','normalized_rmse':'RMSE (Train SD)','normalized_mae':'MAE (Train SD)'}))
changes=[]
for cohort,label in COHORTS.items():
    a=metrics.query('cohort==@cohort').set_index('model')
    changes.append([label,*[100*(1-a.loc['TimesFM',k]/a.loc['LSTM',k]) for k in ['normalized_rmse','normalized_mae']]])
display(pd.DataFrame(changes,columns=['평가 구간','TimesFM RMSE 감소율 (%)','TimesFM MAE 감소율 (%)']))
display(Markdown('감소율이 음수이면 TimesFM이 LSTM보다 나쁘다는 뜻이다.'))''')
    picture('normal_fault_errors.png','정상 구간에서는 LSTM, 고장 이후에서는 TimesFM이 유리하다. 전체 평균만 읽으면 이 차이가 가려진다.')
    md('''## 10. 52개 변수별 비교

RMSE와 MAE 각각52개 중42개 변수에서 TimesFM이 LSTM보다 낮은 오차를 보였다. RMSE 개선 변수 집합과 MAE 개선 변수 집합이 완전히 동일하다는 뜻은 아니다. 아래 표는 두 모델의 오차와 감소율을 실제 내장 결과로 계산한다.''')
    code('''names=[f'xmeas_{j}' for j in range(1,42)]+[f'xmv_{j}' for j in range(1,12)]
rmse=variables.pivot(index='variable',columns='model',values='normalized_rmse').loc[names]
mae=variables.pivot(index='variable',columns='model',values='normalized_mae').loc[names]
comparison=pd.DataFrame({'LSTM RMSE':rmse.LSTM,'TimesFM RMSE':rmse.TimesFM,'RMSE 감소율 (%)':100*(1-rmse.TimesFM/rmse.LSTM),'LSTM MAE':mae.LSTM,'TimesFM MAE':mae.TimesFM,'MAE 감소율 (%)':100*(1-mae.TimesFM/mae.LSTM)})
display(comparison.round(5))
display(Markdown(f"TimesFM 개선 변수 수: RMSE **{(rmse.TimesFM<rmse.LSTM).sum()}/52**, MAE **{(mae.TimesFM<mae.LSTM).sum()}/52**."))
assert (rmse.TimesFM<rmse.LSTM).sum()==42
assert (mae.TimesFM<mae.LSTM).sum()==42''')
    picture('variable_metrics_heatmap.png','52개 변수의 정규화 RMSE/MAE. 두 패널의 색 눈금은 별도이며 모든 값의 단위는 Train SD다.')
    md('''큰 RMSE 감소는 XMV10, XMEAS9, XMEAS21 등에서 관측됐다. 반대로 XMV6, XMEAS10, XMV1 등은 LSTM이 더 낮은 RMSE를 보였다. 전체 지표의 개선과 개별 변수의 개선을 구분해야 한다.''')
    md('''## 11. 고장 시나리오별 추가 확인

다음 표는 이미 저장된105개 시계열의 예측에서 고장 시나리오별 오차를 재집계한 것이다. 재학습·새 추론을 수행한 결과가 아니다. 각 시나리오의 고장 전·후 전체 표본470개 target을 포함한다. 0번은 정상 전용 시나리오다.''')
    code('''grouped=trajectory_errors.groupby(['fault','model'])[['target_rows','sse','sae']].sum().reset_index()
grouped['rmse']=np.sqrt(grouped.sse/(grouped.target_rows*52))
grouped['mae']=grouped.sae/(grouped.target_rows*52)
f_rmse=grouped.pivot(index='fault',columns='model',values='rmse')
f_mae=grouped.pivot(index='fault',columns='model',values='mae')
fault_comparison=pd.DataFrame({'LSTM RMSE':f_rmse.LSTM,'TimesFM RMSE':f_rmse.TimesFM,'RMSE 감소율 (%)':100*(1-f_rmse.TimesFM/f_rmse.LSTM),'LSTM MAE':f_mae.LSTM,'TimesFM MAE':f_mae.TimesFM,'MAE 감소율 (%)':100*(1-f_mae.TimesFM/f_mae.LSTM)})
fault_comparison.index.name='faultNumber (0=정상)'
display(fault_comparison.round(5))
# 시나리오 집계를 다시 합쳐도 기존 전체 표본 지표와 일치해야 한다.
for model in MODELS:
    a=grouped[grouped.model==model]
    np.testing.assert_allclose([np.sqrt(a.sse.sum()/(a.target_rows.sum()*52)),a.sae.sum()/(a.target_rows.sum()*52)],overall.loc[model,['normalized_rmse','normalized_mae']].to_numpy(dtype=float),rtol=2e-9,atol=2e-9)
print('시나리오별 재집계와 전체 지표 일치: PASS')''')
    md('''## 12. 실제값과 예측값, 고장 주입 시점

예시는 사전에 고정한 simulationRun50의 Fault1, 변수 XMEAS1·7·9·21이다. 오차가 좋은 사례만 사후 선택한 것이 아니다. 세 모델은 모두 한 시점 앞(3분 뒤)을 예측한다.

점선480분은 고장 주입 경계이며 고장 이후 첫 관측은 sample161(483분)이다. 시간은 sample×3분으로 표시했다. **그림은 선택된 평가 시점만 선으로 연결하며, 생략된 중간 시점을 추론했다는 뜻은 아니다.**''')
    picture('actual_predictions_fault_onset.png','실제 공정 변수값과 Persistence/LSTM/TimesFM 예측. y축은 Train SD로 표준화한 값이다.')
    picture('forecast_error_fault_onset.png','Fault1·9·15에서52변수의 시점별 RMSE. 고장 경계 전후의 오차 변화를 비교한다.')
    md('''## 13. 설치·실행 환경과 Windows 문제 해결

기존 환경의 NumPy·PyTorch 버전을 변경하지 않고 별도 `.venv`를 만들었다. 최초 pip 설치의 WinError10013은 네트워크 제한이 원인이었고 허용된 다운로드 경로로 해결했다. GPU 드라이버와 호환되는 CUDA12.1 PyTorch wheel을 사용했다.

출력 인코딩은 UTF-8로 설정했다. Jupyter의 사용자 프로필 history 읽기 전용 문제는 IPYTHONDIR·runtime 경로를 실험 폴더 안으로 지정해 해결했다. CUDA/cuDNN 배치 크기에 따른 작은 LSTM 수치 차이를 피하기 위해 저장 예측 재현 검사는 동일한94창 배치에서 수행했다. FP32가 본 비교이며 BF16은 처리속도 탐색에만 사용했다.''')
    code('''env=SNAPSHOT['environment']
display(pd.DataFrame(list(env['packages'].items()),columns=['패키지','실제 사용 버전']))
display(Markdown(f"GPU: **{env['device']}** · CUDA {env['torch_cuda']} · cuDNN TF32 허용={env['cudnn_allow_tf32']} · matmul TF32 허용={env['matmul_allow_tf32']}"))''')
    md('''## 14. 누출·지표·보존 검증

입력 창은 한 시계열 안에서 생성하며 미래 target을 포함하지 않는다. 정상 Train의 scaler 통계를 재계산해 기존 artifact와 일치함을 확인했다. 공식 Test를 scaler fit이나 추가 학습에 쓰지 않았다. TimesFM의 사전학습 corpus에 TEP가 포함됐는지까지 확인한 것은 아니므로 pretrained 데이터 중복 가능성을 배제했다는 주장은 하지 않는다.''')
    code('''checks=SNAPSHOT['checks']
assert all(checks[k]['passed'] for k in checks)
assert checks['preservation_check.json']['changed']==[]
validation=pd.DataFrame([
 ['Train 전용 scaler 재계산 일치',checks['pilot_fp32_validation.json']['scaler_train_only_verified']],
 ['Train/Validation family 분리',checks['pilot_fp32_validation.json']['train_validation_family_disjoint']],
 ['미래값 변경이 과거 입력·예측에 영향 없음',checks['pilot_fp32_validation.json']['future_perturbation_checked']],
 ['TimesFM 배치 독립성',checks['pilot_fp32_validation.json']['batch_independence_checked']],
 ['정답·Persistence가 공식 Test 캐시와 일치',checks['result_validation.json']['actual_and_persistence_match_official_Test_cache']],
 ['저장 예측으로 전체·변수별 지표 재계산',checks['result_validation.json']['summary_and_variable_metrics_recomputed_from_predictions']],
 ['기존 보호 파일409개 변경 없음',checks['preservation_check.json']['passed']],
],columns=['실제로 수행한 검증','통과'])
display(validation)
display(Markdown('본 종합 노트북은 내장 스냅샷의 지표 산술·표본 수·시나리오 재집계를 검증한다. 위의 원 데이터·모델 검사는 이전 실제 실험에서 저장된 통과 기록이다.'))''')
    md('''## 15. 결과를 어떻게 해석해야 하나?

1. **표본 전체에서는 TimesFM이 더 정확했다.** RMSE34.39%, MAE18.75% 감소했다.
2. **정상 운전은 LSTM이 유리했다.** 정상 데이터만 학습한 LSTM이 정상 구간에서 강점을 보였다.
3. **고장 이후는 TimesFM이 유리했다.** LSTM의 정상 학습 범위를 벗어난 조건에서 TimesFM의 낮은 오차가 관측됐다. 이것이 반드시 모델 구조만의 효과라는 인과 결론은 아니다.
4. **20시점·1시점 예측이라는 조건의 결과다.** 더 긴 문맥·다른 예측 길이·fine-tuning 결과로 일반화하지 않는다.
5. **표본의 우열과 전체 Test의 우열은 다르다.** 기존 전체 LSTM RMSE1.494316과 이번 표본 TimesFM RMSE0.963627을 직접 비교하면 평가 분모가 달라 잘못이다.
6. **공정 변수 회귀와 고장 유형 분류를 구분한다.** 본 RMSE/MAE는 CNN의 Accuracy 향상을 뜻하지 않는다.

학술 보고에서는 “정상·고장 시나리오 전체를 포함하는5개 Test family의 대표 표본에서 TimesFM Zero-shot의 통합 예측 오차가 낮았고, 개선은 주로 고장 이후 구간에서 관측되었다”라고 기술하는 것이 현재 증거에 맞다.''')
    md(r'''## 16. 결과 위치와 재현 방법

| 파일 | 용도 |
|---|---|
| `timesfm_experiment/experiment_summary.md` | 상세 한국어 보고서 |
| `timesfm_experiment/TimesFM_TEP_실험.ipynb` | 실제 GPU 재추론·저장 예측 검증이 포함된 실행 노트북 |
| `timesfm_experiment/model_comparison.csv` | 같은 표본의 세 모델 전체·구간별 지표 |
| `timesfm_experiment/variable_metrics.csv` | 변수별 정규화·원 스케일 지표 |
| `timesfm_experiment/predictions/` | 모든 표본의 실제값과 세 모델 예측 |
| `timesfm_experiment/figures/` | 논문·PPT용300dpi PNG6종 |
| `timesfm_experiment/requirements.txt` | 독립 환경 고정 버전 |
| `timesfm_experiment/README.md` | 실행·재개·전체 평가 안내 |

프로젝트 루트에서 PowerShell로 실행할 때:

```powershell
# 대표 표본 실험 재개/결과 생성 (기존 checkpoint 사용)
.\timesfm_experiment\run_experiment.ps1

# 별도 full_evaluation 폴더에서 전체 평가 (현재 미실행, 수일 소요 가능)
.\timesfm_experiment\run_experiment.ps1 -Full

# 이 종합 노트북만 다시 생성하고 실행
.\timesfm_experiment\.venv\Scripts\python.exe .\timesfm_experiment\build_comparison_notebook.py
```

## 17. 공식 출처와 실험 자료의 식별

- [Google Research TimesFM 공식 저장소](https://github.com/google-research/timesfm)
- [공식 모델 카드·가중치](https://huggingface.co/google/timesfm-3.0-pytorch)
- [공식 다변량 Evaluator](https://github.com/google-research/timesfm/blob/master/src/timesfm3/torch/evaluator.py)

이 노트북의 버전 설명은 이번에 설치·실행한 버전 기록이다. TimesFM3.0 가중치에는 비상업·비생산 라이선스가 적용되며 소스 Apache-2.0와 별도다. 실험 시 확인한 문서 원문과 모델 revision은 `timesfm_experiment/sources/`와 `checkpoint/`에 보존했다.''')
    code('''display(pd.DataFrame(list(SNAPSHOT['source_sha256'].items()),columns=['이번 노트북의 원 결과 파일','SHA-256']))
print('정리 완료: 대표 표본의 실제 결과 / 전체 Test TimesFM 미실행 / 기존 모델 보존')''')
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'timesfm-venv','display_name':'TimesFM isolated Python','language':'python'},'language_info':{'name':'python','version':'3.12'},'experiment_scope':'representative sample only; full TimesFM evaluation not run'})
    path=ROOT/'notebooks/19_TimesFM_LSTM_공정예측_비교.ipynb'
    NotebookClient(notebook,timeout=180,kernel_name='timesfm-venv',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.validate(notebook)
    assert not [(i,ord(ch)) for i,c in enumerate(notebook.cells) for ch in c.source if ord(ch)<32 and ch not in '\n\t']
    assert not [o for c in notebook.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
    nbformat.write(notebook,path)
    result={'passed':True,'notebook':str(path),'cells':len(cells),'executed_code_cells':sum(c.cell_type=='code' for c in cells),'embedded_figures':6,'source_sha256':sources,'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (HERE/'comparison_notebook_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':build()
