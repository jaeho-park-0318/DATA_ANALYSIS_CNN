"""Generate truthful Korean report, 300-dpi figures, and an executable notebook."""
from pathlib import Path
import os,sys,json,platform,subprocess
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'mpl_cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat

MODELS=['Persistence','LSTM','TimesFM']
COLORS=['#6B7A90','#14877F','#D46C39']
LABELS={'all':'전체 표본','healthy_context':'정상 구간','post_fault':'고장 이후','faultfree_only':'정상 전용 실행'}
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,'figure.dpi':120,'savefig.dpi':300})

def mdtable(df):
    # No optional tabulate dependency.
    def fmt(x):return f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in r)+' |' for r in df.itertuples(index=False,name=None))

def save(fig,path):
    fig.savefig(path,dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)

def figures(base=HERE):
    df=pd.read_csv(base/'model_comparison.csv');v=pd.read_csv(base/'variable_metrics.csv')
    dest=base/'figures';dest.mkdir(exist_ok=True)
    main=df[df.cohort=='all'].set_index('model').loc[MODELS]
    scope='대표 표본 · 동일 20시점 입력 / 3분 뒤 예측'
    for metric in ['rmse','mae']:
        fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
        vals=main['normalized_'+metric].to_numpy();bars=ax.bar(MODELS,vals,color=COLORS,width=.6)
        ax.bar_label(bars,labels=[f'{x:.4f}' for x in vals],padding=4)
        ax.set(title=f'모델별 {metric.upper()} 비교\n{scope}',ylabel=f'{metric.upper()} (정상 Train 표준편차 단위)',ylim=(0,max(vals)*1.18))
        ax.text(.99,.98,f'N = {int(main.target_rows.iloc[0]):,} target 시점',transform=ax.transAxes,ha='right',va='top',fontsize=9)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);save(fig,dest/f'model_{metric}_comparison.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,metric in zip(axes,['rmse','mae']):
        for j,model in enumerate(MODELS):
            values=[float(df[(df.model==model)&(df.cohort==c)]['normalized_'+metric].iloc[0]) for c in ['healthy_context','post_fault']]
            bars=ax.bar(np.arange(2)+(j-1)*.25,values,width=.25,label=model,color=COLORS[j])
            ax.bar_label(bars,fmt='%.3f',padding=2,fontsize=8)
        ax.set(xticks=[0,1],xticklabels=['정상 구간','고장 이후'],ylabel=f'{metric.upper()} (Train SD)',title=f'구간별 {metric.upper()}')
        ax.legend(loc='upper left');ax.set_ylim(0,ax.get_ylim()[1]*1.3);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f'정상·고장 구간 예측 오차 — {scope}');save(fig,dest/'normal_fault_errors.png')
    names=[f'xmeas_{i}' for i in range(1,42)]+[f'xmv_{i}' for i in range(1,12)]
    fig,axes=plt.subplots(1,2,figsize=(10,15),layout='constrained')
    for ax,metric in zip(axes,['rmse','mae']):
        mat=v[v.cohort=='all'].pivot(index='variable',columns='model',values='normalized_'+metric).loc[names,MODELS].to_numpy()
        im=ax.imshow(mat,aspect='auto',cmap='YlOrRd');ax.set(xticks=range(3),xticklabels=MODELS,yticks=range(52),yticklabels=names,title=f'변수별 {metric.upper()} (Train SD)')
        for i in range(52):
            for j in range(3):ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if mat[i,j]>mat.max()*.55 else '#202020')
        fig.colorbar(im,ax=ax,shrink=.65,label=f'{metric.upper()} (Train SD)')
    fig.suptitle('52개 공정 변수 예측 오차 — 대표 표본');save(fig,dest/'variable_metrics_heatmap.png')
    manifest=json.loads((base/'sampling_manifest.json').read_text());run=manifest['simulationRuns'][0]
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,j in zip(axes.flat,[0,6,8,20]):
        z=np.load(base/f'example_fault1_run{run}.npz');t=z['target_sample']*3
        ax.plot(t,z['actual'][:,j],color='#182331',label='실제값',lw=1.6)
        for model,color in zip(MODELS,COLORS):ax.plot(t,z[model][:,j],color=color,label=model,lw=1,alpha=.85)
        ax.axvline(480,color='#AD294D',ls='--',label='고장 주입 (480분)')
        ax.set(title=f'Fault 1 · simulationRun {run} · {names[j]}',xlabel='시작 이후 시간 (분; sample × 3)',ylabel='값 (정상 Train 표준편차 단위)')
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('실제 공정 변수와 3분 뒤 예측 — 대표 표본의 평가 시점만 연결')
    save(fig,dest/'actual_predictions_fault_onset.png')
    fig,axes=plt.subplots(3,1,figsize=(12,9),layout='constrained')
    for ax,fault in zip(axes,[1,9,15]):
        z=np.load(base/f'example_fault{fault}_run{run}.npz');t=z['target_sample']*3
        for model,color in zip(MODELS,COLORS):
            e=(z[model]-z['actual']).astype(np.float64)
            ax.plot(t,np.sqrt(np.mean(e**2,axis=1)),label=model,color=color,lw=1.2)
        ax.axvline(480,color='#AD294D',ls='--',label='고장 주입 (480분)')
        ax.set(title=f'Fault {fault} · simulationRun {run}',xlabel='시작 이후 시간 (분; sample × 3)',ylabel='52변수 RMSE (Train SD)');ax.grid(alpha=.2)
    axes[0].legend(ncol=4,fontsize=9);fig.suptitle('고장 발생 전후의 시점별 예측 오차 — 대표 표본')
    save(fig,dest/'forecast_error_fault_onset.png')

def validate_saved(base=HERE):
    """Recompute reported values from every saved paired prediction."""
    df=pd.read_csv(base/'model_comparison.csv');v=pd.read_csv(base/'variable_metrics.csv')
    all_acc={};keys=set();normal=0;post_count=0
    manifest=json.loads((base/'sampling_manifest.json').read_text())
    from experiment import data_setup,windows
    scaler,raw=data_setup()
    mean=scaler.mean_.astype(np.float32);scale=scaler.scale_.astype(np.float32)
    faults=set();runs=set();trajectories=0
    for path in sorted((base/'predictions').glob('*.npz')):
        fault=int(path.stem.split('_')[0][5:]);run=int(path.stem.split('_')[1][3:])
        z=np.load(path);t=z['target_sample'];n=len(t)
        faults.add(fault);runs.add(run);trajectories+=1
        assert z['actual'].shape==(n,52) and np.all(np.diff(t)>0)
        x,y=windows(raw[fault,run-1],t-2,mean,scale)
        np.testing.assert_array_equal(y,z['actual'])
        np.testing.assert_array_equal(x[:,-1],z['Persistence'])
        if manifest['mode']=='representative':
            assert n==94 and np.array_equal((t-21)//10,np.arange(94))
        for s in t:
            assert (fault,run,int(s)) not in keys
            keys.add((fault,run,int(s)))
        post=(fault>0)&(t>160)
        for model in MODELS:
            assert np.isfinite(z[model]).all()
            e=(z[model]-z['actual']).astype(np.float64)
            for cohort,mask in [('all',np.ones(n,bool)),('healthy_context',~post),('post_fault',post),('faultfree_only',np.full(n,fault==0))]:
                if not mask.any():continue
                k=(model,cohort)
                if k not in all_acc:all_acc[k]=[0,np.zeros(52),np.zeros(52)]
                a=all_acc[k];a[0]+=int(mask.sum());a[1]+=np.square(e[mask]).sum(0);a[2]+=np.abs(e[mask]).sum(0)
    for (model,cohort),(n,sse,sae) in all_acc.items():
        row=df[(df.model==model)&(df.cohort==cohort)].iloc[0]
        assert int(row.target_rows)==n
        np.testing.assert_allclose([np.sqrt(sse.sum()/(n*52)),sae.sum()/(n*52)],[row.normalized_rmse,row.normalized_mae],rtol=1e-12,atol=1e-12)
        vv=v[(v.model==model)&(v.cohort==cohort)].set_index('variable')
        names=[f'xmeas_{i}' for i in range(1,42)]+[f'xmv_{i}' for i in range(1,12)]
        np.testing.assert_allclose(vv.loc[names,'normalized_rmse'],np.sqrt(sse/n),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(vv.loc[names,'normalized_mae'],sae/n,rtol=1e-12,atol=1e-12)
    assert len(keys)==manifest['expected_target_rows']
    assert faults==set(range(21)) and runs==set(manifest['simulationRuns']) and trajectories==manifest['trajectories']
    result={'passed':True,'unique_target_keys':len(keys),'paired_predictions':True,'all_52_variables':True,'summary_and_variable_metrics_recomputed_from_predictions':True,'actual_and_persistence_match_official_Test_cache':True,'stratified_time_bins_verified':True,'all_faults':len(faults),'trajectories':trajectories,'sample_only':manifest['mode']=='representative'}
    (base/'result_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result

def summary(base=HERE):
    df=pd.read_csv(base/'model_comparison.csv');v=pd.read_csv(base/'variable_metrics.csv');m=json.loads((base/'sampling_manifest.json').read_text());pilot=json.loads((HERE/'pilot_fp32_validation.json').read_text())
    main=df[df.cohort=='all'].set_index('model');l=main.loc['LSTM'];tf=main.loc['TimesFM'];p=main.loc['Persistence']
    rmse_pct=(tf.normalized_rmse/l.normalized_rmse-1)*100;mae_pct=(tf.normalized_mae/l.normalized_mae-1)*100
    wins={}
    for metric in ['rmse','mae']:
        t=v[v.cohort=='all'].pivot(index='variable',columns='model',values='normalized_'+metric)
        wins[metric]=int((t.TimesFM<t.LSTM).sum())
    healthy=df[df.cohort=='healthy_context'].set_index('model')
    post=df[df.cohort=='post_fault'].set_index('model')
    healthy_rmse_pct=(healthy.loc['TimesFM','normalized_rmse']/healthy.loc['LSTM','normalized_rmse']-1)*100
    healthy_mae_pct=(healthy.loc['TimesFM','normalized_mae']/healthy.loc['LSTM','normalized_mae']-1)*100
    post_rmse_pct=(post.loc['TimesFM','normalized_rmse']/post.loc['LSTM','normalized_rmse']-1)*100
    post_mae_pct=(post.loc['TimesFM','normalized_mae']/post.loc['LSTM','normalized_mae']-1)*100
    show=df[['model','cohort','target_rows','normalized_rmse','normalized_mae']].copy();show['cohort']=show.cohort.map(LABELS)
    var=v[v.cohort=='all'].pivot(index='variable',columns='model',values='normalized_rmse');delta=(var.TimesFM-var.LSTM).sort_values()
    text=f'''# TimesFM–TEP Zero-shot 예측 실험 (대표 표본 평가 완료)

기준일: 2026-11-12 (Asia/Seoul). **전체 Test 평가가 아니라 대표성 확보를 위해 사전 정의한 소규모 표본 비교다.** 사용자의 후속 요청에 따라 소규모 비교를 우선 완료했다. 전체 9,870,000시점 TimesFM 평가는 실행하지 않았으며 완료로 표시하지 않는다.

## 핵심 결과

같은 {int(l.target_rows):,}개 정답 시점과 52개 변수를 비교했다. TimesFM의 전체 표본 정규화 RMSE는 **{tf.normalized_rmse:.6f}**, LSTM은 **{l.normalized_rmse:.6f}**로, TimesFM이 LSTM 대비 **{abs(rmse_pct):.2f}% {'높아 악화' if rmse_pct>0 else '낮아 개선'}**됐다. MAE는 TimesFM **{tf.normalized_mae:.6f}**, LSTM **{l.normalized_mae:.6f}**로 **{abs(mae_pct):.2f}% {'높아 악화' if mae_pct>0 else '낮아 개선'}**됐다. Persistence는 RMSE {p.normalized_rmse:.6f}, MAE {p.normalized_mae:.6f}다.

52변수 중 TimesFM이 LSTM보다 낮은 오차를 보인 변수는 RMSE {wins['rmse']}개, MAE {wins['mae']}개다. 이 수치는 표본에 대한 기술통계이며 유의성 검정이나 전체 Test 우열 확정이 아니다.

{mdtable(show)}

## 1. 기존 프로젝트 분석

- 프로젝트 전체 파일 목록: `project_inventory.json` (의존성·Git·새 실험 디렉터리 제외); 기존 보호 파일 SHA-256: `protected_before.json`.
- 원본 Harvard TEP: 루트의 `TEP_FaultFree_Training.RData`, `TEP_Faulty_Training.RData`, `TEP_FaultFree_Testing.RData`, `TEP_Faulty_Testing.RData`. 별도의 GitHub 기반 41변수·1분 데이터와 혼합하지 않았다.
- 실제 사용한 기존 캐시: `data/processed/phase2/training.npy` (21,500,500,52), `testing.npy` (21,500,960,52), dtype float32. 원본 변환은 기존 `src/data/phase2.py`에서 faultNumber·simulationRun으로 그룹화하고 sample 연속성을 검증했다. 캐시를 새로 생성하거나 수정하지 않았다.
- `17_CNN_기능확장_종합안내.ipynb`: 52변수 회귀 LSTM은 CNN 고장 분류와 독립된 경로. 20×52 → 다음 3분의 52변수, 정상 Train 전용 학습과 정규화, 대표 seed42. 표본(common)과 전체(dense) 평가는 별도이며 MAE 결론이 달라질 수 있다고 명시돼 있다.
- `18_TEP_PHM_전체모델_종합비교.ipynb`: LSTM 회귀와 LSTM 분류를 구분하고 기존 예측 평가표를 내장한다. 새 학습 코드가 아니라 종합 열람용이다. 셀별 관련 원문은 `notebook_analysis.json`에 저장했다.
- 실제 재현 근거: `experiments/next_step_forecasting/run.py`, `config.json`, `models/scaler.joblib`, `models/lstm_s42.pt`, 기존 `reports/next_step_forecasting/tables/test_summary.csv`.
- 동일 run family의 시나리오를 묶어 공식 Training의 500 family를 Train400/Validation100으로 분할. 공식 Testing의 500 family는 별도 RNG family다. Training과 Testing에서 simulationRun 번호 1~500이 재사용되지만 같은 trajectory를 뜻하지 않는다. 키는 (공식 파일 분할, faultNumber, simulationRun, sample)이다.
- scaler는 정상 Training400실행×500관측 = 200,000행으로 fit한 StandardScaler를 그대로 사용했다. mean/std를 원 Training 데이터로 다시 계산해 저장 scaler와 일치함을 검증했다. Test 정규화 통계 추정은 하지 않았다.
- 기존 LSTM: 2층, hidden64, dropout0.1, 현재값+학습 변화량의 residual 출력. 정상 Train48,000창(stride4), Validation48,000창(stride1). AdamW lr0.001, weight_decay1e-4, Huber loss, 최대12epoch/patience3. Validation normalized MSE로 epoch 선택, 대표 seed42는 Test 이전 지정. 저장된 가중치를 로드해 표본 전체를 실제 재추론했다. 재학습은 하지 않았다.
- 기존 전체(dense) 평가: target sample21~960의 940시점×500run×21scenario = 9,870,000행. common은 stride10의 987,000행. 본 대표 표본은 dense Test의 부분집합이며 세 모델에서 동일하다.

## 2. 대표 표본 설계와 평가 공정성

- 모든 시나리오 faultNumber0~20 (정상 전용 1종 + 고장20종)을 포함한다.
- seed20261112로 오류·값을 보지 않고 공식 Test simulationRun **{', '.join(map(str,m['simulationRuns']))}**를 선택했다. 모든 시나리오에서 같은 5개 family를 사용해 총105개 trajectory다.
- 각 trajectory의 940개 유효 입력 끝점을 길이10의 94개 구간으로 나누고, 각 구간에서 난수로 1개씩 선택했다. seed는 (20261112, faultNumber, zero-based run index)로 결정해 재현 가능하다. 고정 stride10에서 관측된 주기적 표본 편향을 피하고자 시간 bin 내부 위치를 무작위화했다.
- 총9,870개 target = 전체 dense Test의 0.1%. 정상 시나리오는 전 구간 정상, 고장 시나리오는 target sample≤160이 정상·sample>160이 고장 이후다. 모든 변수·세 모델의 target key가 완전히 같다.
- 겹치는 입력 창은 정상적인 rolling one-step 평가다. 각 창은 하나의 (fault,run) 안에서만 생성하며 target은 입력 마지막보다 정확히 한 sample 뒤다. 미래 정답을 읽어 모델 입력·정규화·설정 선택에 사용하지 않았다.
- 전체 RMSE = sqrt(모든 표본·52변수의 제곱오차 합 / (행수×52)); MAE = 절대오차 합/(행수×52). 변수별 RMSE를 산술평균한 값과 다르다. 서로 물리 단위가 다른 변수는 정상 Train 표준편차 단위로 통합한다.
- 변수별 원 스케일 RMSE/MAE도 CSV에 제공한다. 해당 Harvard 파일의 물리 단위는 독립 검증되지 않아 원 단위를 임의로 kPa·℃ 등으로 붙이지 않았다.

## 3. TimesFM 버전·입력·Zero-shot 구성

- 공식 PyPI 배포 `timesfm==3.0.2`, 공식 가중치 `google/timesfm-3.0-pytorch`, revision `43046b85ec22d584a13f8098c2ed39c889e129c2`. 조회 원본은 `sources/`에 보존했다. 패키지 버전과 모델 세대(3.0)는 서로 다른 식별자다.
- 독립 `.venv`, Python {platform.python_version()}, PyTorch 2.5.1+cu121. 기존 Anaconda/프로젝트 `.deps` 패키지 버전은 변경하지 않았다. 실제 사용 환경은 `requirements.txt`, `environment.json`에 기록한다.
- GPU: {pilot['gpu']}, 6GB VRAM. Windows 드라이버546.92의 CUDA12.1 호환 PyTorch wheel을 별도 환경에 설치했다. 최초 pip 실패 WinError10013은 네트워크 제한이 원인이었고 허용된 다운로드 경로에서 재설치해 해결했다. Windows 출력 인코딩은 UTF-8로 설정했다.
- TimesFM3는 네이티브 다변량 입력을 지원한다. 공식 `TimesFM3Evaluator.predict_batch`에 trajectory별 **(52,20)** float32 context를 넘기고 `horizon=1`로 호출한다. 출력은 (52,1), 모델이 제공하는 중앙값(0.5 quantile)을 point forecast로 사용한다.
- 공식 Evaluator는 32변수 제한에 따라 52개를 **앞32개/뒤20개**로 처리하고 뒤20개 그룹을 32개가 되도록 반복 패딩한 후 원20개를 복원한다. 52개 출력 모두 평가하지만 한 번에 52개 전체가 공동 attention하는 설정은 아니다. 변수 순서는 기존 scaler의 XMEAS1~41/XMV1~11 순서 그대로다. 이를 자체적으로 바꾸거나 최적화하지 않았다.
- 외부 전처리는 기존 LSTM과 같은 Train StandardScaler다. TimesFM 내부의 context 기반 RevIN·detrending과 patch 패딩은 공식 모델 구조이며 Train/Test 전체를 사용한 추가 정규화가 아니다. 20개 관측만 전달하며 32길이 patch의 부족분은 모델이 mask 패딩한다. 추가 실제 관측이나 생성 데이터는 넣지 않았다.
- 추가 학습·가중치 업데이트·XReg 회귀·공변량 미래값 사용 없음. `return_quantiles=False`, `use_symmetric_averaging=False`, `make_positive=False`, `sort_quantiles=True`, `use_znorm=False`, `padding_mode='none'`. FP32를 주 비교에 사용하고 BF16은 속도 탐색에만 사용했다. Test 오차로 inference 설정을 선택하지 않았다.
- 3.0 가중치는 별도 비상업·비생산 라이선스이며 공식 저장소 소스의 Apache-2.0와 구분된다. 라이선스 원문은 `checkpoint/LICENSE`에 있다. 이번 결과는 로컬 연구 비교다.

## 4. 결과 해석과 한계

**정상 구간:** LSTM의 RMSE/MAE는 {healthy.loc['LSTM','normalized_rmse']:.6f}/{healthy.loc['LSTM','normalized_mae']:.6f}, TimesFM은 {healthy.loc['TimesFM','normalized_rmse']:.6f}/{healthy.loc['TimesFM','normalized_mae']:.6f}다. TimesFM 오차가 각각 {healthy_rmse_pct:.2f}%/{healthy_mae_pct:.2f}% 높아 정상 운전에서는 LSTM이 더 정확했다.

**고장 이후:** LSTM의 RMSE/MAE는 {post.loc['LSTM','normalized_rmse']:.6f}/{post.loc['LSTM','normalized_mae']:.6f}, TimesFM은 {post.loc['TimesFM','normalized_rmse']:.6f}/{post.loc['TimesFM','normalized_mae']:.6f}다. TimesFM 오차가 각각 {abs(post_rmse_pct):.2f}%/{abs(post_mae_pct):.2f}% 낮았다. 고장 이후 시점이 8,000/9,870개(약81.1%)여서 이 개선이 전체 표본 지표에 크게 반영됐다. 따라서 정상 운전과 고장 이후 모두에서 TimesFM이 우수하다는 결론은 맞지 않는다.

정상 구간은 정상 운전만 학습한 LSTM의 학습 분포에 가깝다. 고장 이후 구간은 LSTM에 외삽 조건이며 TimesFM도 TEP 전용으로 학습하지 않았다. 따라서 결과는 ‘52변수·20관측·1관측 뒤’라는 짧은 문맥 조건에서의 비교다. 긴 문맥·긴 horizon·fine-tuning 결과로 일반화할 수 없다. TimesFM은 32길이 patch를 사용하므로 입력20개가 사전학습 모델에 충분한 문맥인지 별도 연구가 필요하다.

RMSE는 큰 오차에 더 민감하고 MAE는 일반적인 절대 편차를 보여준다. 두 지표가 서로 다른 방향이면 일괄적인 우열을 주장하지 않는다. 변수별 오차와 정상/고장 이후를 함께 확인해야 한다. 표본 5개 family는 모든500개 Test family의 완전한 대체가 아니며, 시점들은 겹치는 입력 창으로 독립 표본도 아니다. 이 보고서는 통계적 유의성을 주장하지 않는다.

TimesFM–LSTM 변수별 RMSE 차이가 작은 순서(음수면 TimesFM 개선):

{mdtable(delta.head(6).rename('TimesFM_minus_LSTM_RMSE').reset_index())}

차이가 큰 순서:

{mdtable(delta.tail(6).sort_values(ascending=False).rename('TimesFM_minus_LSTM_RMSE').reset_index())}

CNN의 고장 유형 Accuracy는 분류 평가이고 여기의 RMSE/MAE는 공정 변수 회귀 오차다. 이 실험으로 기존 CNN·GRU·XGBoost의 분류 성능이 향상되었다고 해석하지 않는다. 사전학습 corpus에 TEP가 포함됐는지까지 확인한 것은 아니므로 pretrained 데이터 중복 가능성을 엄밀히 배제했다는 주장도 하지 않는다.

## 5. 실행·검증 상태

- 완료: 기존 파일 분석, 별도 환경 설치, 공식 가중치 다운로드, GPU Zero-shot 소규모 실행, 정상 Train scaler 재계산 일치, causal 입력과 batch 독립성 검증, 대표 표본 세 모델 실제 추론, 지표·그림·실행 노트북 생성.
- 완료: 저장된 모든 paired 예측에서 지표를 독립 재계산하고 CSV와 일치 확인 (`result_validation.json`). 기존 파일 보존 검사는 `preservation_check.json` 참조.
- Windows 노트북의 사용자 프로필 history 읽기 전용 문제는 IPYTHONDIR와 Jupyter runtime 경로를 새 실험 폴더 안으로 지정해 해결했다. CUDA/cuDNN은 배치 크기에 따라 LSTM 출력에 작은 수치 차이가 생겨, 재현 검사는 본 평가와 동일한94창 배치에서 수행했다. 기존 PyTorch 기본 TF32 허용 설정을 유지하고 `environment.json`에 기록했다.
- 미실행: 전체 dense Test의 TimesFM 추론. 초기 FP32 pilot 속도는 {pilot['windows_per_second']:.2f}창/초이며 987만창 기준 단순 추정 {pilot['estimated_dense_hours']:.1f}시간이었다. 배치8/16/32/64/128 BF16 탐색 결과는 `throughput_benchmark.csv`다. 표본 우선이라는 후속 요청에 따라 전체 평가를 완료한 것처럼 표시하거나 백그라운드로 계속 실행하지 않았다.
- 기존 전체 Test 숫자는 참고용이다: LSTM RMSE1.4943157993, MAE0.7378341926; Persistence RMSE1.6301365023, MAE0.6905302629. **이 숫자와 본 표본 TimesFM 숫자를 직접 비교해 순위를 만들면 안 된다.**

## 6. 그림 읽기

모든 PNG는300dpi다. `model_rmse_comparison.png`, `model_mae_comparison.png`, `normal_fault_errors.png`, `variable_metrics_heatmap.png`, `actual_predictions_fault_onset.png`, `forecast_error_fault_onset.png`를 제공한다. 예시 변수는 사전에 XMEAS1·7·9·21로 고정했으며 오차가 큰 사례를 골라 결론을 바꾸지 않았다. 시계열 그림은 표본 평가 시점만 선으로 연결한 것으로, 중간 시점의 실제 예측을 수행했다는 뜻이 아니다. x축은 sample×3분, 고장 주입 경계는480분, 고장 이후 첫 관측은 sample161(483분)이다. 정상 run에는 고장 주입이 없다.

## 공식 출처

- [공식 저장소와 현재 사용 예제](https://github.com/google-research/timesfm)
- [공식 패키지 정의](https://github.com/google-research/timesfm/blob/master/pyproject.toml)
- [공식 모델 카드](https://huggingface.co/google/timesfm-3.0-pytorch)
- [공식 다변량 Evaluator](https://github.com/google-research/timesfm/blob/master/src/timesfm3/torch/evaluator.py)
- [공식 Forecaster](https://github.com/google-research/timesfm/blob/master/src/timesfm3/torch/timesfm3_forecaster.py)
'''
    (base/'experiment_summary.md').write_text(text,encoding='utf-8')

def notebook(base=HERE):
    n=nbformat.v4.new_notebook();cells=[]
    cells.append(nbformat.v4.new_markdown_cell('# TimesFM–TEP 실험: 대표 표본의 실제 실행 결과\n\n정상·고장20종 / 52변수 / 과거20시점 → 3분 뒤1시점. 기존 모델은 보존했다. **전체 Test 평가는 미실행**이며 아래는 동일 표본에서 실제 재추론한 비교다. 학습 없이 TimesFM 사전학습 모델을 사용했다.'))
    cells.append(nbformat.v4.new_code_cell("from pathlib import Path\nimport json, numpy as np, pandas as pd\nfrom IPython.display import display, Image, Markdown\nHERE=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'model_comparison.csv').exists() and (p/'experiment.py').exists())\ndisplay(pd.read_csv(HERE/'model_comparison.csv'))\ndisplay(json.loads((HERE/'sampling_manifest.json').read_text()))"))
    cells.append(nbformat.v4.new_markdown_cell('## 기존 실험 조건과 공식 TimesFM 입력\n\n정상 Train400실행으로 fit한 StandardScaler와 seed42 LSTM을 그대로 로드했다. Validation100실행, 공식 Test500 family를 유지한다. TimesFM 공식 Evaluator 입력은 (52,20)이며 32+20 변수 chunk를 사용한다. 예측 target은 마지막 입력보다1시점 뒤, fault>0이고 target sample>160이면 고장 이후다.'))
    cells.append(nbformat.v4.new_code_cell("import sys\nsys.path.insert(0,str(HERE))\nfrom experiment import data_setup, windows, lstm_model, timesfm_model, tf_predict\nimport torch\ntorch.set_num_threads(4)\nscaler,raw=data_setup()\nmanifest=json.loads((HERE/'sampling_manifest.json').read_text())\nr=manifest['simulationRuns'][0]-1\nz=np.load(HERE/'predictions'/f'fault01_run{r+1:03d}.npz')\nendpoints=z['target_sample']-2\nx,y=windows(raw[1,r],endpoints,scaler.mean_.astype(np.float32),scaler.scale_.astype(np.float32))\ndevice='cuda' if torch.cuda.is_available() else 'cpu'\nlstm=lstm_model(device)\nwith torch.inference_mode(): lp=lstm(torch.from_numpy(x).to(device)).cpu().numpy()\ntfm=timesfm_model(device,128)\ntp=tf_predict(tfm,x,'fp32')\nnp.testing.assert_allclose(lp,z['LSTM'],rtol=2e-5,atol=2e-5)\nnp.testing.assert_allclose(tp,z['TimesFM'],rtol=2e-5,atol=2e-5)\nprint('실제 GPU 재추론 94창, 동일 배치 저장 예측 일치:',device,tp.shape)\ndel tfm,lstm\nif device=='cuda': torch.cuda.empty_cache()"))
    cells.append(nbformat.v4.new_markdown_cell('## 저장 예측 전체로부터 지표 재검증'))
    cells.append(nbformat.v4.new_code_cell("from report import validate_saved\ndisplay(validate_saved(HERE))\nv=pd.read_csv(HERE/'variable_metrics.csv')\ndisplay(v[v.cohort=='all'])"))
    for name in ['model_rmse_comparison','model_mae_comparison','normal_fault_errors','variable_metrics_heatmap','actual_predictions_fault_onset','forecast_error_fault_onset']:
        cells.append(nbformat.v4.new_code_cell(f"display(Image(filename=str(HERE/'figures/{name}.png')))"))
    cells.append(nbformat.v4.new_code_cell("display(Markdown((HERE/'experiment_summary.md').read_text(encoding='utf-8')))"))
    cells.append(nbformat.v4.new_markdown_cell('## 재현\n\n`run_experiment.ps1`은 대표 표본을 실행하고 보고서·노트북을 재생성한다. 재실행 시 완료된 동일설정 trajectory는 checkpoint에서 복원한다. `run_experiment.ps1 -Full`은 별도 `full_evaluation` 폴더에서 전체987만시점을 평가한다. 현재 전체평가는 실행하지 않았다. 긴 전체평가에는 수일이 걸릴 수 있다.'))
    n.cells=cells;n.metadata={'kernelspec':{'display_name':'Python (TimesFM isolated)','language':'python','name':'python3'},'language_info':{'name':'python','version':platform.python_version()}}
    nbformat.write(n,base/'TimesFM_TEP_실험.ipynb')

def main():
    validate_saved();figures();summary();notebook()
    print('Report, figures, notebook created.',flush=True)

if __name__=='__main__':main()
