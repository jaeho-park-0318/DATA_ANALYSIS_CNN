"""Standalone next-step regression report, including the persistence benchmark's tradeoff."""
from pathlib import Path
import os,sys,json,textwrap
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'reports/next_step_forecasting'
os.environ['MPLCONFIGDIR']=str(HERE/'cache/.matplotlib')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nbf

def md(df):
    def f(v):return f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v).replace('|','/')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(f(v) for v in r)+' |' for r in df.itertuples(index=False,name=None))

def run():
    result=pd.read_csv(OUT/'tables/test_summary.csv');history=pd.read_csv(OUT/'tables/training_history.csv')
    variables=pd.read_csv(OUT/'tables/per_variable_metrics.csv');validation=pd.read_csv(OUT/'tables/validation_summary.csv')
    dense=result[(result.seed==42)&(result['mode']=='dense')]
    mean=result[result['mode']=='common'].groupby(['model','cohort']).agg(rmse_mean=('normalized_rmse','mean'),rmse_sd=('normalized_rmse','std'),mae_mean=('normalized_mae','mean'),mae_sd=('normalized_mae','std')).reset_index()
    mean.to_csv(OUT/'tables/common_seed_summary.csv',index=False)
    lstm=dense[(dense.model=='LSTM')&(dense.cohort=='all')].iloc[0]
    naive=dense[(dense.model=='Persistence')&(dense.cohort=='all')].iloc[0]
    rmse_change=(1-lstm.normalized_rmse/naive.normalized_rmse)*100
    mae_change=(lstm.normalized_mae/naive.normalized_mae-1)*100
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,'figure.dpi':140})
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for seed,g in history.groupby('seed'):ax.plot(g.epoch,g.validation_normalized_mse,label=f'LSTM seed{seed}')
    ax.axhline(history.persistence_validation_mse.iloc[0],ls='--',color='#888888',label='Persistence')
    ax.set(xlabel='Epoch',ylabel='정상 Validation 정규화 MSE',title='정상 Train400개 family만 학습 / 정상 Validation100개 family로 epoch 선택');ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'figures/validation_learning.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');cohorts=['all','faultfree_only','post_fault']
    for ax,metric,title in zip(axes,['normalized_rmse','normalized_mae'],['정규화 RMSE (낮을수록 좋음)','정규화 MAE (낮을수록 좋음)']):
        for j,model in enumerate(['LSTM','Persistence']):
            values=[dense[(dense.model==model)&(dense.cohort==c)][metric].iloc[0] for c in cohorts]
            ax.bar(np.arange(3)+(j-.5)*.34,values,.34,label=model,color=['#257d91','#c68a46'][j])
        ax.set(xticks=range(3),xticklabels=['전체 Test','정상 전용 실행','주입 후 고장 구간'],ylabel=metric,title=title);ax.legend()
    fig.suptitle('주 seed42 / 전체9,870,000개 다음시점 target / 정상Train 채널표준편차 단위\n출처: test_summary.csv · 공통간격 비교표와 평가시점이 다름',fontsize=10)
    fig.savefig(OUT/'figures/lstm_vs_persistence.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(14,4.8),layout='constrained')
    for model,style in [('LSTM','-'),('Persistence','--')]:
        q=variables[(variables.model==model)&(variables.cohort=='all')]
        ax.plot(np.arange(52),q.normalized_rmse,style,label=model)
    sensors=variables[(variables.model=='LSTM')&(variables.cohort=='all')].variable.tolist()
    ax.set(xticks=range(52),xticklabels=sensors,xlabel='공정 변수',ylabel='정규화 RMSE',title='전체 Test의 변수별 예측 오차 / 주 seed42 / 정상 Train 표준편차 단위');ax.tick_params(axis='x',rotation=90,labelsize=7);ax.legend()
    fig.savefig(OUT/'figures/per_variable_forecast_error.png');plt.close(fig)
    cases=pd.read_csv(OUT/'tables/forecast_examples.csv');fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    for ax,fault,variable in [(axes[0],0,'xmeas_7'),(axes[1],1,'xmeas_1')]:
        q=cases[(cases.fault==fault)&(cases.variable==variable)]
        for col,label,style in [('actual','실제값','-'),('lstm','LSTM 예측','-'),('persistence','Persistence','--')]:ax.plot(q.target_sample*3/60,q[col],style,label=label,lw=.8)
        if fault>0:ax.axvline(8,color='#555555',ls=':',label='명목 주입8h')
        ax.set(xlabel='예측 target 명목 시간(h)',ylabel='정상Train 표준편차 단위',title=f'Test Fault{fault} / run1 / {variable}');ax.legend(fontsize=8)
    fig.suptitle('다음3분 예측 사례 / 현재·과거20개 관측만 입력 / 출처: forecast_examples.csv',fontsize=10)
    fig.savefig(OUT/'figures/forecast_examples.png');plt.close(fig)
    report=f'''# 별도 LSTM 다음 시점 공정 변수 예측

## 구성과 기존 CNN의 분리

사용자 요청에 따라 기존 CNN 유형 분류 검증을 먼저 수행한 뒤, **과거20×52→다음3분의52개 변수**를 예측하는 별도 LSTM을 추가했다. 기존 CNN은 재학습하지 않았고 LSTM 예측값을 CNN 입력에 넣지 않았다. 따라서 이 예측기가 CNN 분류를 개선했다고 주장하지 않는다.

LSTM은 hidden64·2layers·dropout0.1이고, 현재값에 학습한 변화량을 더하는 residual 출력이다. zero-initialized 출력층은 Persistence를 출발점으로 삼는다. Huber loss와 AdamW lr0.001, 최대12epoch·patience3을 사용했다. 원래 실행 family 분할400/100/500을 유지했으며 scaler와 학습은 **정상 Training400개 실행**에서만 구성했다. 각실행20관측창의다음target을쓰되실행경계를넘지않고미래target을입력에넣지않았다.

학습48,000창, 정상Validation48,000창, seed42/7/2026이다. 이3개 seed는LSTM가중치학습변동이며가용성실험의누락seed와다르다. epoch는정상Validation MSE로선택했고대표seed42는사전지정했다. Test를보고재선택하지않았다. 정상기준예측이라는아이디어는[참고GitHub](https://github.com/min3237/tep-fault-diagnosis-dl)를참고했지만10관측/하이브리드분류구조의동일재현은아니다.

{md(validation)}

![학습](figures/validation_learning.png)

## 실측 성능: Persistence와 비교

Persistence는다음값을현재값그대로예측하는기준선이다. 정규화오차는정상Train채널별표준편차단위이며, 서로단위가다른52개를이단위로평균했다. 원래열단위RMSE/MAE도변수별표에기록했으나B의독립단위검증이없어단위표시는UNKNOWN이다.

전체시점평가는9,870,000개의target이며마지막관측에서한칸뒤정답이없는창을제외했다. 원래분류평가9,880,500시점과분모가다르다. 대표seed42의전체평가결과는다음과같다.

{md(dense[['model','cohort','target_rows','normalized_rmse','normalized_mae']])}

**전체RMSE는{naive.normalized_rmse:.4f}→{lstm.normalized_rmse:.4f}로{rmse_change:.2f}%감소했지만, MAE는{naive.normalized_mae:.4f}→{lstm.normalized_mae:.4f}로{mae_change:.2f}%증가했다.** 큰오차의제곱평균관점과평균절대오차관점의판단이다르므로전반적으로항상우수한예측기라고결론내리지않는다. 정상전용실행에서는두오차가개선됐으나고장후MAE는더나빴다. 정상만학습했으므로고장구간은범위밖외삽성격이있다.

![예측기준비교](figures/lstm_vs_persistence.png)
![변수별오차](figures/per_variable_forecast_error.png)

## 동일 공통 시점의3개 seed 비교

{md(mean)}

공통비교는987,000개target, 전체평가는9,870,000개target이다.10샘플간격의고정위상표본과전체시점에서오차분포가달라MAE개선방향도달라졌다. 표본과지표를선택해우수성을주장하지않고전체시점을주결과로제시한다. 서로다른공통/전체점수를같은평가집합으로섞지않는다.

![예측사례](figures/forecast_examples.png)

## 한계와 실행

이모델은다음3분변수회귀예측기이며고장유형분류나RUL예측기가아니다. 잔차로새경보임계값을학습하거나기존CNN에연결하지않았다. 센서결측이있으면추론API는입력을거부하며미래값보간을하지않는다. 여러장비/운전모드·실제설비일반화는미검증이다. 기존Test의후속연구평가이지완전히미노출인새Test가아니다.

모델·scaler는`experiments/next_step_forecasting/models/`에저장했다. 입력은한실행의52개정규센서열과연속관측이다.

```powershell
python experiments/next_step_forecasting/predict.py --input one_run.csv --output next_step_predictions.csv
```

출력`target_sample`은입력종료의다음시점이며`predicted_xmeas_*`/`predicted_xmv_*`는원래열단위예측이다. API는마지막관측의다음값도반환하되그정답이없는경우실측오차를만들지않는다. 검증노트북은`16_next_step_forecasting.ipynb`이다.
'''
    (OUT/'final_report.md').write_text(report,encoding='utf-8')
    create_notebook()
    print('Forecast report and notebook authored',flush=True)

def create_notebook():
    parts=[('markdown','''# 16. 별도 LSTM 다음3분 공정 변수 예측
    기존CNN유형진단과분리한회귀과제다. 과거20×52→다음52개변수를예측한다.
    정상Train400개family만학습하고동일Validation100/Test500분할을유지했다. 이노트북은저장결과만읽고재학습하지않는다.'''),
    ('code','''from pathlib import Path
import json,pandas as pd,numpy as np
from IPython.display import display,Image
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'configs/phase2.json').exists())
OUT=ROOT/'reports/next_step_forecasting'
def table(name):return pd.read_csv(OUT/'tables'/name)
def show(frame):display(frame.style.hide(axis='index').format(precision=5,na_rep='해당없음'))
def image(name):display(Image(filename=str(OUT/'figures'/name)))
v=json.loads((OUT/'validation.json').read_text())
assert not v['cnn_modified'] and not v['forecast_values_fed_to_CNN'] and not v['test_selection_or_tuning']'''),
    ('markdown','''## 1. 정상 패턴 학습과 Validation 선택
    LSTM64·2layers·residual현재값출력, Huber loss, AdamW. 예측값을기존CNN에넣지않았다.
    정상Train의평균/표준편차만fit했다. 모든입력은target보다앞선20개관측이다.'''),
    ('code',"show(table('validation_summary.csv'))\nimage('validation_learning.png')"),
    ('markdown','''## 2. 전체 Test: 단순 Persistence와 비교
    다음값=현재값 기준선과같은target에서비교한다. 대표seed42,전체9,870,000개target이다.
    RMSE가좋아도MAE가나빠질수있으며실제로전체/고장후MAE는LSTM이더나빴다. 특정지표만으로우수성을단정하지않는다.'''),
    ('code',"results=table('test_summary.csv')\nshow(results[(results.seed==42)&(results['mode']=='dense')][['model','cohort','target_rows','normalized_rmse','normalized_mae']])\nimage('lstm_vs_persistence.png')\nshow(table('per_variable_metrics.csv').query('cohort == \"all\"'))\nimage('per_variable_forecast_error.png')"),
    ('markdown','''## 3.3개 seed와 예측 사례
    공통987,000시점의비교는전체9,870,000시점과표본분포가다르다.두표를섞지않는다.
    변수그래프는정상Train표준편차단위다. B원단위대응은독립검증하지않아UNKNOWN으로기록했다.'''),
    ('code',"show(table('common_seed_summary.csv'))\nimage('forecast_examples.png')"),
    ('markdown','''## 4. 기능과 해석 범위
    예측기는다음3분변수값을반환한다. 고장ID·고장원인·RUL을예측하는기능이아니다.
    원래CNN의21출력·입력·threshold는그대로다. 예측잔차의새경보임계값도고르지않았다.
    `python experiments/next_step_forecasting/predict.py --input one_run.csv --output next_step_predictions.csv`로사용한다.
    52개필수열·한실행·연속관측·20개이력조건을지켜야하며결측은대체하지않는다.''')]
    cells=[]
    for kind,s in parts:
        s=textwrap.dedent(s).strip()
        if kind=='code':compile(s,'forecast_notebook','exec');cells.append(nbf.v4.new_code_cell(s))
        else:cells.append(nbf.v4.new_markdown_cell(s))
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'TEP next-step forecast','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    nbf.write(nb,HERE/'16_next_step_forecasting.ipynb')

if __name__=='__main__':run()
