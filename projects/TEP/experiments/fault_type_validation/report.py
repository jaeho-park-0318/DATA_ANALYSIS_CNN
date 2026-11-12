"""Independent diagnosis report and executable notebook, preserving the frozen CNN."""
import os,textwrap
from evaluate import *
os.environ['MPLCONFIGDIR']=str(HERE/'cache/.matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nbf

def table_md(frame):
    def f(v):
        if isinstance(v,(float,np.floating)):return f'{v:.4f}' if np.isfinite(v) else '미검출/해당 없음'
        return str(v).replace('|','/')
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(f(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))

def run():
    validation=json.loads((OUT/'validation.json').read_text());classes=pd.read_csv(OUT/'tables/per_class_metrics.csv')
    runs=pd.read_csv(OUT/'tables/run_type_results.csv');timing=pd.read_csv(OUT/'tables/per_fault_diagnosis_time.csv')
    av=pd.read_csv(OUT/'tables/availability_detection_vs_diagnosis.csv');fault_av=pd.read_csv(OUT/'tables/availability_per_fault.csv')
    summary=pd.read_csv(OUT/'tables/classification_summary.csv')
    early=[]
    for h in [30,60,120,240]:
        for fault,g in [('ALL',runs),*list(runs.groupby('fault'))]:
            early.append(dict(horizon_minutes=h,fault=fault,runs=len(g),detection_rate_by_horizon=g.detection_delay_minutes.le(h).mean(),correct_diagnosis_rate_by_horizon=g.correct_diagnosis_delay_minutes.le(h).mean()))
    early=pd.DataFrame(early);early.to_csv(OUT/'tables/early_diagnosis_horizons.csv',index=False)
    means=av.groupby('rate').mean(numeric_only=True).reset_index();means.to_csv(OUT/'tables/availability_mean.csv',index=False)
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,'figure.dpi':140})
    fig,axes=plt.subplots(1,3,figsize=(18,6),layout='constrained')
    for ax,mode,label in zip(axes,['argmax21','gated21','conditional20'],['출력21개 단순 argmax','고정 탐지 임계값 적용21상태','실제 고장 구간 조건부20종']):
        cm=np.load(OUT/'tables'/f'{mode}_confusion.npy');a=cm/np.maximum(cm.sum(1,keepdims=True),1)
        im=ax.imshow(a,vmin=0,vmax=1,cmap='Blues');start=1 if mode=='conditional20' else 0
        ax.set(xticks=range(len(cm)),yticks=range(len(cm)),xticklabels=range(start,len(cm)+start),yticklabels=range(start,len(cm)+start),xlabel='예측 클래스',ylabel='실제 클래스',title=label)
        ax.tick_params(labelsize=7)
    fig.colorbar(im,ax=axes,label='실제 클래스 내 비율')
    fig.suptitle('동일 고정 CNN / 공식 Test / 전체유효9,880,500시점(조건부는고장8,000,000시점)\n출처: 저장된21클래스 확률; 재학습·threshold 변경 없음',fontsize=10)
    fig.savefig(OUT/'figures/confusions_three_definitions.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,8),layout='constrained')
    for ax,mode,label in zip(axes,['conditional20','gated21'],['고장 조건부 분류','실제 탐지 포함 진단']):
        c=classes[(classes['mode']==mode)&(classes['class']>0)].sort_values('class')
        im=ax.imshow(c[['precision','recall','f1']],vmin=0,vmax=1,aspect='auto',cmap='YlGnBu')
        ax.set(xticks=range(3),xticklabels=['Precision','Recall','F1'],yticks=range(20),yticklabels=range(1,21),ylabel='Fault ID',title=label)
        for i,row in enumerate(c[['precision','recall','f1']].to_numpy()):
            for j,value in enumerate(row):ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=8,color='white' if value>.6 else 'black')
    fig.colorbar(im,ax=axes,label='비율');fig.suptitle('Fault별 성능 / 공식 Test / 각Fault400,000개 post시점\n출처: per_class_metrics.csv',fontsize=10)
    fig.savefig(OUT/'figures/per_fault_classification.png');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
    for field,label in [('run_detection_rate','고장 검출 성공'),('run_correct_diagnosis_rate','올바른 유형 진단 성공'),('first_type_accuracy','최초 확정 유형 정확도')]:axes[0].plot(timing.fault,timing[field],'o-',label=label)
    axes[0].set(ylim=(-.03,1.05),ylabel='실행 비율',title='실행 단위 검출과 올바른 진단은 다름');axes[0].legend(fontsize=8)
    axes[1].bar(timing.fault-.18,timing.detection_delay_median,.36,label='검출 지연',color='#257d91')
    axes[1].bar(timing.fault+.18,timing.correct_diagnosis_delay_median,.36,label='올바른 유형 진단 지연',color='#ce893b')
    axes[1].set(xlabel='Fault ID',ylabel='성공 실행의 지연 중앙값(분)',title='명목 주입8h 기준 / 검출·진단 각각 성공한 실행만 집계');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xticks(range(1,21));ax.grid(axis='y',alpha=.15)
    fig.suptitle('공식Test Fault별500실행 / 3연속 고정규칙 / 출처: per_fault_diagnosis_time.csv',fontsize=10)
    fig.savefig(OUT/'figures/detection_vs_diagnosis_time.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for field,label,color,style in [('overall_detection_rate','전체 검출률','#257d91','-'),('conditional_detection_rate','가용 실행 검출률','#257d91','--'),('overall_correct_diagnosis_rate','전체 올바른 진단률','#c67b35','-'),('conditional_correct_diagnosis_rate','가용 실행 올바른 진단률','#c67b35','--')]:
        mean=av.groupby('rate')[field].mean();sd=av.groupby('rate')[field].std()
        ax.errorbar(mean.index*100,mean*100,yerr=sd*100,fmt='o'+style,label=label,color=color,capsize=3)
    ax.set(xlabel='평가불가 고장 실행 비율(%)',ylabel='실행 성공률(%)',ylim=(0,105),xticks=means.rate*100,title='입력 상실: 이상 검출과 올바른 유형 진단의 영향 비교')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.suptitle('고정 CNN / 고장10,000실행 / 누락seed42·43·44 평균±SD\n출처: availability_detection_vs_diagnosis.csv',fontsize=10)
    fig.savefig(OUT/'figures/availability_detection_vs_diagnosis.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,7),layout='constrained')
    a=fault_av.groupby(['fault','rate']).overall_diagnosis_rate.mean().unstack()
    im=ax.imshow(a,vmin=0,vmax=1,aspect='auto',cmap='YlGnBu');ax.set(xticks=range(5),xticklabels=[f'{r:.1%}' for r in a.columns],yticks=range(20),yticklabels=range(1,21),xlabel='입력 불가 실행 비율',ylabel='Fault ID',title='전체 실행 기준 올바른 유형 진단률 / 고정 CNN·3개 누락seed 평균')
    fig.colorbar(im,ax=ax,label='실행 진단률');fig.savefig(OUT/'figures/availability_per_fault_diagnosis.png');plt.close(fig)
    earlyall=early[early.fault=='ALL']
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    ax.plot(earlyall.horizon_minutes,earlyall.detection_rate_by_horizon,'o-',label='해당 시간 내 검출')
    ax.plot(earlyall.horizon_minutes,earlyall.correct_diagnosis_rate_by_horizon,'s-',label='해당 시간 내 올바른 유형 진단')
    ax.set(xlabel='명목 고장 이후 시간 한도(분)',ylabel='전체10,000실행 내 비율',ylim=(0,1.05),xticks=[30,60,120,240],title='사전 고정30·60·120·240분 내 검출/진단, Test 설정 재선택 없음')
    ax.legend();ax.grid(alpha=.2);fig.savefig(OUT/'figures/early_correct_diagnosis.png');plt.close(fig)
    first=runs[runs.first_confirmed_type.notna()]
    run_overview=pd.DataFrame([
        ['고장 실행',len(runs)],['한번이라도 고장 검출',int(runs.detected.sum())],['올바른 유형3연속 진단',int(runs.correctly_diagnosed.sum())],
        ['최초 유형 확정 실행',len(first)],['최초 확정 유형이 정답',int(first.first_confirmed_type_correct.sum())],
        ['최초 유형 정확도(유형 확정 실행 분모)',validation['first_confirmed_type_accuracy']],
        ['올바른 진단 성공 실행 지연 중앙값(분)',validation['diagnosis_delay_median']],
        ['검출~올바른 진단 추가 지연의 중앙값(분)',validation['additional_diagnosis_delay_median']]
    ],columns=['항목','결과']);run_overview.to_csv(OUT/'tables/run_summary.csv',index=False)
    comparison=pd.DataFrame([
        ['구조','LSTM 다음시점 예측+실제9행/예측1행 분류','기존20×52 CNN의21출력을 그대로 평가'],
        ['참고한 점','클래스별 Precision/Recall/F1, 결과 표시','유형별 지표·혼동·시간·가용성 영향으로 확장'],
        ['성능 직접 비교','자료 구성·시점·평가규칙이 다름','GitHub 수치와 모델 우열로 비교하지 않음'],
        ['이번 고정 CNN 검증','새 forecaster를 CNN에 연결하지 않음','재학습·입력·가중치·threshold 변경 없음']
    ],columns=['항목','참고 GitHub README/설명','본 검증'])
    comparison.to_csv(OUT/'tables/reference_comparison.csv',index=False)
    compact=means[['rate','available_runs','detected_runs','correct_diagnosis_runs','overall_detection_rate','conditional_detection_rate','overall_correct_diagnosis_rate','conditional_correct_diagnosis_rate']].copy()
    main=means[means.rate==.531].iloc[0]
    report=f'''# 기존 CNN 고장 유형 분류 추가 검증

## 결론

**기존CNN21출력에 유형 분류가 이미 있으며, 이번 작업은 재학습 없는 독립 검증이다.** 동일 Test의 조건부20종 Macro F1={validation['classification_metrics']['conditional20']['macro_f1']:.4f}, 고정 탐지threshold 적용21상태 Macro F1={validation['classification_metrics']['gated21']['macro_f1']:.4f}로 기존 결과와 일치했다.

실행단위 고장 검출률98.51%와 달리 **올바른 유형3연속 진단률은90.15%**다. 최초로 확정한 유형의 정확도는 **{validation['first_confirmed_type_accuracy']:.2%}**로 낮았다. 충분히 기다린 최종진단 성공이 초기진단 정확성을 보장하지 않는다.

## 1. 비교 정의와 분모

{table_md(summary)}

- argmax21: 현재21개 출력의최대클래스. threshold를 적용하지 않는다.
- gated21: 기존threshold={validation['threshold']}에서`1-P(정상)`이 낮으면0, 높으면Fault1~20 후보를출력한다. 정상구간오탐과미탐을포함한다.
- conditional20: 실제고장8,000,000시점에서정상출력을제외하고20종을선택한다. 고장상태를안다는가정이있다.

argmax/gated는warm-up19개를제외한Test9,880,500시점이고 고장별400,000개post샘플을포함한다. 세조건은동일한가중치·같은입력의결정규칙차이다.

![혼동행렬](figures/confusions_three_definitions.png)
![유형별성능](figures/per_fault_classification.png)

전체 Precision/Recall/F1/support는`per_class_metrics.csv`, 오분류쌍은`confusion_pairs.csv`에 있다. 유형후보의조건부확률이높아도고장확률이낮으면정상판정일수있다. 확률을보정된신뢰도로주장하지않는다.

## 2. 고장 검출 시간과 올바른 유형 진단 시간

기존고장3연속경보규칙을유지했다. 진단확정은동일한양의예측ID가3회연속일때로사전정의했다. 실제Fault ID는정답대조에만사용하고예측에넣지않았다. 고장주입후카운터를초기화한다. 첫유형확정이틀린후나중에맞힐수있다.

{table_md(run_overview)}
{table_md(timing)}

![검출과진단](figures/detection_vs_diagnosis_time.png)
![조기진단](figures/early_correct_diagnosis.png)

올바른진단지연중앙값69분은진단성공9,015개실행만의값이다. 일반검출48분과의차이21분은동일집합의짝지은추가지연이아니다. 두사건이모두발생한실행에서실제추가지연중앙값은12분이다. 미검출·미진단실행지연을0으로채우지않았다.40h안에한번성공했다는결과를조기진단률로해석하지않도록고정30/60/120/240분한도도제시했다.

Fault9는고장경보500/500에도올바른유형진단15/500(3%)뿐이며, Fault15는고장경보351/500에도올바른유형진단0개였다. 강한고장에서도처음에는다른유형으로확정될수있어최초유형정확도를별도로보고했다.

## 3. 입력 가용성 저하: 검출과 진단의 차이

기존센서가용성연구의동일마스크·선정run·seed42/43/44를재사용했다. 선택된실행은post-fault유효창0개이며미선택실행의기존출력을유지한다. 정상예측값으로누락시점을채우지않는다. 고정CNNseed는42이고3개seed는누락선정의변동이다.

{table_md(compact)}

![가용성비교](figures/availability_detection_vs_diagnosis.png)
![고장별가용성](figures/availability_per_fault_diagnosis.png)

53.1%입력불가에서전체검출은{main.overall_detection_rate:.2%}, 전체올바른진단은{main.overall_correct_diagnosis_rate:.2%}다. 가용실행내검출은{main.conditional_detection_rate:.2%}, 가용실행내올바른진단은{main.conditional_correct_diagnosis_rate:.2%}로분리된다. 가용시점의Macro F1과전체실행의진단범위를섞지않는다. 부분가용대조군은기존유효성마스크로진단시간을별도재평가했으며`partial_selected_diagnosis.csv`에기록했다.

## 4. GitHub 참고와 변경하지 않은 항목

{table_md(comparison)}

참고자료: [원저장소](https://github.com/min3237/tep-fault-diagnosis-dl), [노트북로직설명](https://github.com/min3237/tep-fault-diagnosis-dl/blob/main/NOTEBOOK_LOGIC_EXPLAINED.md). 공개문서는`experiments/fault_type_validation/references/`에읽기전용참고로보존했다. 코드실행·저장소모델복제·동등한수치재현을주장하지않는다. LSTM다음시점예측은별도과제로구성하며기존CNN을예측값입력방식으로바꾸지않는다.

## 5. 검증·한계·산출물

기존gated21혼동행렬·조건부Macro F1·기존이벤트검출/지연·가용성실험의전체검출률이일치했다. 모델·기존노트북등보호파일{validation['unchanged_protected_files']}개의SHA256이변경되지않았다. 물리적원인이공개되지않은Fault16~20은UNKNOWN으로유지한다. 분류ID를맞힌것은실제원인변수를발견한것과다르다. 기존Test를추가로분석했으므로새로운독립Test실험이라고부르지않는다. 새로운구조/threshold선택을하지않았다.

상세표는`tables/`, 그림은`figures/`, 재현코드는`experiments/fault_type_validation/`에있다. `15_fault_type_validation.ipynb`은저장된표·그림을재사용하는실행가능노트북이다.
'''
    (OUT/'final_report.md').write_text(report,encoding='utf-8')
    notebook()
    print('Diagnosis report and notebook authored',flush=True)

def notebook():
    parts=[('markdown','''# 15. 기존 CNN 고장 유형 분류 검증
    **재학습 없이21개 클래스 출력을 독립적으로 평가한다.** 유형별 성능 → 올바른 진단 시각 → 가용성의 검출/진단 영향 순으로 읽는다.
    모델·threshold·라벨·분할은 기존값이다. Run All은 저장 결과만 읽으며 학습·모델선택을 하지 않는다.'''),
    ('code','''from pathlib import Path
import json,pandas as pd,numpy as np
from IPython.display import display,Image,Markdown
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'configs/phase2.json').exists())
OUT=ROOT/'reports/fault_type_validation'
def table(name):return pd.read_csv(OUT/'tables'/name)
def show(df):display(df.style.hide(axis='index').format(precision=5,na_rep='미진단/해당없음'))
def image(name):display(Image(filename=str(OUT/'figures'/name)))
validation=json.loads((OUT/'validation.json').read_text())
assert validation['all_reference_results_match'] and validation['no_retraining'] and not validation['changed_files']'''),
    ('markdown','''## 1. 정의가 다른 세 가지 분류 지표
    argmax21은Softmax최대, gated21은기존경보임계값적용, conditional20은실제고장구간을안다고가정한다.
    Accuracy와Macro F1은분모·결정규칙을함께읽는다. 출처: 기존동결CNN의전체유효Test확률.'''),
    ('code',"show(table('classification_summary.csv'))\nimage('confusions_three_definitions.png')\nshow(table('per_class_metrics.csv'))\nimage('per_fault_classification.png')"),
    ('markdown','''## 2. 어떤 고장으로 오분류하는가?
    정상0으로빠지는미탐과다른Fault로잘못판정하는오진을분리한다. 조건부후보가확률상높아도고장확률이낮으면경보가아니다.'''),
    ('code',"pairs=table('confusion_pairs.csv')\nshow(pairs.groupby('mode',sort=False).head(8))\nshow(table('prediction_examples.csv'))"),
    ('markdown','''## 3. 검출과 올바른 유형 진단의 시간 차이
    탐지는3연속경보, 진단확정은같은양의예측ID3연속이다. 실제Fault는정답대조에만쓴다.
    첫유형판정정확도와40h내한번이라도올바르게진단한비율을구분한다. 지연은각사건성공실행에서만계산한다.'''),
    ('code',"show(table('run_summary.csv'))\nshow(table('per_fault_diagnosis_time.csv'))\nimage('detection_vs_diagnosis_time.png')\nshow(table('early_diagnosis_horizons.csv').query('fault == \"ALL\"'))\nimage('early_correct_diagnosis.png')"),
    ('markdown','''## 4. 입력 가용성 저하가 검출·진단에 미치는 영향
    기존가용성실험과동일한5개비율×3개누락seed를재사용했다. CNN은같은seed42체크포인트다.
    전체실행과가용실행의분모를각각유지하며, 불가시점에정상/고장확률을만들지않는다.'''),
    ('code',"show(table('availability_mean.csv')[['rate','available_runs','overall_detection_rate','conditional_detection_rate','overall_correct_diagnosis_rate','conditional_correct_diagnosis_rate','conditional20_macro_f1','gated21_macro_f1']])\nimage('availability_detection_vs_diagnosis.png')\nimage('availability_per_fault_diagnosis.png')"),
    ('markdown','''## 5. 참고 저장소와 본 검증의 경계
    참고GitHub는LSTM다음시점예측과분류의2단계구조이며, 본검증은기존CNN을그대로사용한다. 클래스별지표와결과표시방식을참고했다.
    서로다른자료구성·시점·정상라벨·경보규칙이므로README점수로직접우열을정하지않는다.
    [참고저장소](https://github.com/min3237/tep-fault-diagnosis-dl). 물리원인적중률은미평가이며Fault16~20은UNKNOWN이다.'''),
    ('code',"show(table('reference_comparison.csv'))\ndisplay(Markdown(f\"고정모델: `{validation['model']}` · threshold={validation['threshold']:.6f} · 재학습없음 · 보호파일 {validation['unchanged_protected_files']}개 변경없음\"))")]
    cells=[]
    for kind,s in parts:
        s=textwrap.dedent(s).strip()
        if kind=='code':compile(s,'diagnosis_notebook','exec');cells.append(nbf.v4.new_code_cell(s))
        else:cells.append(nbf.v4.new_markdown_cell(s))
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'TEP diagnosis validation','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    nbf.write(nb,HERE/'15_fault_type_validation.ipynb')

if __name__=='__main__':run()
