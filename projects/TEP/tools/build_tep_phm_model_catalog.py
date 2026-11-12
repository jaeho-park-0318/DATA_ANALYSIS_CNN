"""Read-only cross-project model catalog; all new outputs stay in the TEP workspace."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,base64,zlib,asyncio
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/all_models'; OUT.mkdir(parents=True,exist_ok=True)
SNAP=OUT/'tables'; SNAP.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(OUT/'runtime/matplotlib')
import numpy as np,pandas as pd,nbformat as nbf
from nbclient import NotebookClient
V1=ROOT.parent/'PHM2018-Project'; V2=ROOT.parent/'PHM2018-Project-V2'; V3=ROOT.parent/'PHM2018-Project-V3'
sources=[]; tables={}; extra={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p):
    p=p.resolve()
    if not any(r['path']==str(p) for r in sources):sources.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
def csv(key,p):record(p); tables[key]=pd.read_csv(p); return tables[key]
def js(key,p):record(p);extra[key]=json.loads(p.read_text(encoding='utf-8-sig'));return extra[key]

for directory in [V1,V2,V3]:
    for name in ['README.md','AGENTS.md']:record(directory/name)
for key,name in [('tep_test','test_comparison'),('tep_means','model_performance_summary'),('tep_imbalance','imbalance_comparison')]:
    csv(key,ROOT/f'reports/tables/phase2/{name}.csv')
csv('tep_accuracy',ROOT/'reports/model_extension_overview/accuracy_model_summary.csv')
csv('diagnosis_classes',ROOT/'reports/fault_type_validation/tables/classification_summary.csv')
csv('diagnosis_faults',ROOT/'reports/fault_type_validation/tables/per_fault_diagnosis_time.csv')
csv('availability',ROOT/'reports/fault_type_validation/tables/availability_mean.csv')
csv('forecast_test',ROOT/'reports/next_step_forecasting/tables/test_summary.csv')
csv('forecast_train',ROOT/'reports/next_step_forecasting/tables/validation_summary.csv')
js('diagnosis',ROOT/'reports/fault_type_validation/validation.json')
js('tep_dense',ROOT/'reports/sources/phase2/dense_metrics.json')
js('tep_config',ROOT/'configs/phase2.json')
js('v1_summary',V1/'outputs/stage4/summary.json')
js('v1_cv_summary',V1/'outputs/stage4_5/summary.json')
js('phm_classification',V3/'outputs/stage6/holdout_classification.json')
for stage in [2,3,4,5,7]:csv(f'phm_stage{stage}_train',V3/f'outputs/tables/stage{stage}_training_summary.csv')
for key,file in [('phm_stage2_points','stage2_operating_points'),('phm_stage3_points','stage3_budget_points'),
                 ('phm_stage4_ranges','stage4_seed_ranges'),('phm_stage5_points','stage5_budget_points'),
                 ('phm_holdout','stage6_holdout_summary'),('phm_specialist','stage7_comparison_display')]:
    csv(key,V3/f'outputs/tables/{file}.csv')

# Read only small evaluation Parquets and two columns of saved Holdout scores using PHM's existing environment.
parquets={'v1_metrics':V1/'outputs/stage4/metrics.parquet',
          'v1_cv':V1/'outputs/stage4_5/cv_metrics.parquet',
          'v1_cv_means':V1/'outputs/stage4_5/cv_summary.parquet'}
for p in parquets.values():record(p)
pred=V3/'outputs/stage6/holdout_predictions.parquet';record(pred)
payload={'parquets':{k:str(p.resolve()) for k,p in parquets.items()},'prediction':str(pred.resolve()),
         'thresholds':tables['phm_holdout'][['operating_point','threshold']].to_dict('records')}
reader=r'''
import sys,json,pandas as pd,numpy as np
x=json.load(sys.stdin); out={'tables':{}}
for k,p in x['parquets'].items():out['tables'][k]=pd.read_parquet(p).to_csv(index=False)
d=pd.read_parquet(x['prediction'],columns=['audit_label','score']); known=d[d.audit_label.isin([0,1])]
rows=[]
for r in x['thresholds']:
 y=known.audit_label.to_numpy()==1; p=known.score.to_numpy()>=r['threshold']
 tp=int((y&p).sum());tn=int((~y&~p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum())
 rows.append(dict(**r,all_input_rows=len(d),known_rows=len(known),positive=int(y.sum()),negative=int((~y).sum()),
                  tp=tp,tn=tn,fp=fp,fn=fn,accuracy=(tp+tn)/len(y),recall=tp/(tp+fn),
                  precision=tp/(tp+fp) if tp+fp else None))
out['tables']['phm_holdout_accuracy']=pd.DataFrame(rows).to_csv(index=False)
print(json.dumps(out,ensure_ascii=False))
'''
env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONIOENCODING']='utf-8'
res=subprocess.run([str(Path('C:/Users/fl27/anaconda3/envs/phm2018/python.exe')),'-c',reader],input=json.dumps(payload),
                   text=True,encoding='utf-8',capture_output=True,env=env,check=True)
from io import StringIO
for key,value in json.loads(res.stdout)['tables'].items():tables[key]=pd.read_csv(StringIO(value))
assert len(tables['v1_metrics'].query("split == 'evaluation'"))==58
assert len(tables['v1_cv'])==32
assert (tables['phm_holdout_accuracy'].known_rows==extra['phm_classification']['known_anchors']).all()
assert (tables['phm_holdout_accuracy'].positive==extra['phm_classification']['positive']).all()
for r in tables['phm_holdout_accuracy'].itertuples():assert r.tp+r.tn+r.fp+r.fn==r.known_rows

# Derive TEP Accuracy from saved gated confusion matrices; binary-only rows use saved recall/FAR and fixed supports.
test=tables['tep_test'].copy(); aa=[]
for r in test.itertuples():
    cmfile=ROOT/f'reports/tables/phase2/{r.model_id}_confusion.npy'
    if cmfile.exists():
        record(cmfile);cm=np.load(cmfile);assert cm.sum()==r.samples
        a2=(cm[0,0]+cm[1:,1:].sum())/cm.sum();a21=np.trace(cm)/cm.sum();method='saved confusion'
    else:
        positives=800000;negatives=r.samples-positives
        a2=(r.detection_recall*positives+(1-r.sample_false_alarm_rate)*negatives)/r.samples
        a21=np.nan;method='saved binary recall/FAR + supports'
    aa.append([a2,a21,method])
test[['binary_accuracy','class21_accuracy','accuracy_source']]=aa
test['binary_accuracy']=pd.to_numeric(test.binary_accuracy);test['class21_accuracy']=pd.to_numeric(test.class21_accuracy)
tables['tep_all_results']=test
cmfile=ROOT/'reports/tables/phase2/primary_dense_confusion.npy';record(cmfile);cm=np.load(cmfile)
extra['tep_dense_accuracy']={'binary':float((cm[0,0]+cm[1:,1:].sum())/cm.sum()),'class21':float(np.trace(cm)/cm.sum())}

families=pd.DataFrame([
 ['TEP','Logistic Regression','21종 분류','선형 기준선','현재 52 / 통계 312','Validation 구성·임계값 선택'],
 ['TEP','Random Forest','21종 분류','비선형 트리 앙상블','현재 52 / 통계 312','가중치 구성 비교'],
 ['TEP','XGBoost','21종 분류','부스팅 트리','현재 52 / 통계 312','현재값과 과거 통계 비교'],
 ['TEP','LSTM 분류기','21종 분류','순환 시계열 모델','20×52','후속 LSTM 회귀기와 별개'],
 ['TEP','GRU 분류기','21종 분류','게이트 순환 모델','20×52','CNN과 동일 분할로 비교'],
 ['TEP','1D-CNN','21종 분류·이상 검출','선택된 주 모델','20×52','seed42 고정; 유형·지연 추가 평가'],
 ['TEP','Binary XGBoost','정상/고장 2종','별도 이진 기준선','통계 312','natural/balanced/under/oversample'],
 ['TEP','Unknown Binary XGBoost','미지 고장 검출 연구','Fault 3·9·15를 학습에서 제외','통계 312','미지/기지 평가 분리'],
 ['TEP','LSTM 회귀기','다음 3분 52변수 예측','별도 새 예측기','20×52 → 52','정상만 학습; CNN 연결 없음'],
 ['TEP','Persistence','다음값 예측 기준선','다음값=현재값','현재 52','학습 없는 회귀 비교 기준'],
 ['PHM V1','Dummy / 압력 임계값','고장 전 이진 경보','학습 없는/단순 기준선','당시 공통 후보','Accuracy 착시 확인'],
 ['PHM V1','Random Forest','고장 전 이진 예측','A/B/C 특징 비교','A8 / B80 / C83','horizon300/600/1800, 장비/시간 분할'],
 ['PHM V1','XGBoost','고장 전 이진 예측','주 비교 모델·진단 변형','A8 / B80 / C83','ID 보조진단; 4-Fold 후속 연구'],
 ['PHM V2','학습 모델 없음','원본·라벨·입력 감사','학습 전 진단 단계','A8 / B32 / C44','데이터 불확실성으로 학습 차단'],
 ['PHM V3','Logistic Regression','고장 전 이진 예측','A/B 및 공통 A/B/C 기준선','범주형 One-Hot 후 입력','시간 Validation 지표'],
 ['PHM V3','XGBoost 공통 모델','고장 전 이진 예측','A/B/C·튜닝·seed·그룹 비교','A8 / B32 / C44','stage2 B를 동결하여 최종 평가'],
 ['PHM V3','XGBoost 장비 전용','03M02 고장 전 예측','장비 한 대 파일럿','B32','기존 후반 데이터 재사용 연구']
],columns=['프로젝트','모델/기준선','목표','역할','입력','특징/상태'])
tables['model_catalog']=families
for key,df in tables.items():df.to_csv(SNAP/f'{key}.csv',index=False,encoding='utf-8-sig')

picture_paths={
 'diag':ROOT/'reports/fault_type_validation/figures/detection_vs_diagnosis_time.png',
 'availability':ROOT/'reports/fault_type_validation/figures/availability_detection_vs_diagnosis.png',
 'forecast':ROOT/'reports/next_step_forecasting/figures/lstm_vs_persistence.png',
 'phm_group':V3/'outputs/figures/final_evaluation/group_results.png',
 'phm_specialist':V3/'outputs/figures/stage7/comparison.png'}
images={}
for key,p in picture_paths.items():record(p);images[key]=base64.b64encode(p.read_bytes()).decode()
packed=base64.b64encode(zlib.compress(json.dumps({'tables':{k:d.to_csv(index=False) for k,d in tables.items()},
          'extra':extra,'images':images},ensure_ascii=False).encode('utf-8'),9)).decode()
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
md('''# 지금까지 만든 모델 전체 정리 — TEP와 PHM 2018

**모델이 어떤 문제를 풀고, 무엇을 비교했으며, 어느 결과까지 믿을 수 있는지**를 한 파일에 정리했습니다. 확인 범위는 로컬 TEP, PHM2018-Project(V1), V2, V3 및 V1 재분석 폴더입니다. 다른 프로젝트(SRAM·WM811K 등)는 이번 범위에 포함하지 않았습니다.

TEP는 **고장 주입 후 검출·유형 진단**, PHM은 **기록된 고장 정지 이전 조기 예측**, 새 TEP LSTM은 **다음 시점 센서값 회귀**입니다. 과제가 다르므로 Accuracy나 검출률로 서로 순위를 매기지 않습니다.

기준일: 2026-11-12. 원본 데이터·기존 모델·노트북을 변경하지 않았습니다. 기존 평가표와 저장 예측을 읽었으며 **새 학습·새 모델 추론·임계값 선택은 하지 않았습니다.** PHM은 읽기 전용으로 확인했습니다.

아래 표·그림은 **내장 스냅샷**으로 표시됩니다. 다른 폴더로 옮겨도 기존 출력으로 열람할 수 있고, Python·pandas·NumPy·matplotlib·IPython 환경에서 원 프로젝트 없이 재실행할 수 있습니다. 내장 자료는 원 센서 데이터나 모델 바이너리를 포함하지 않습니다.''')
code('''import base64, zlib, json, io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown
pd.set_option('display.max_rows', 90)
pd.set_option('display.max_columns', 20)
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'figure.dpi':120})
'''+f"PAYLOAD = '{packed}'\n"+'''
snapshot=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)).decode('utf-8'))
T={k:pd.read_csv(io.StringIO(s)) for k,s in snapshot['tables'].items()}
E=snapshot['extra']
def show(key,columns=None):
    frame=T[key] if columns is None else T[key][columns]
    display(frame.style.hide(axis='index'))
def picture(key):display(Image(data=base64.b64decode(snapshot['images'][key])))
display(Markdown('**내장 평가 결과 준비 완료 — 모델을 학습하거나 추론하지 않습니다.**'))''')
md('''## 1. 먼저 보는 전체 모델 지도

‘모델 종류’, ‘학습 실행 수’, ‘평가 행 수’는 다릅니다. 같은 XGBoost를 특징·seed·분할별로 학습하면 여러 실행이 됩니다. 임계값만 달리 평가한 것은 새 학습 모델로 세지 않습니다. 아래는 기준선을 포함한 역할별 목록입니다.''')
code("show('model_catalog')")
md('''## 2. 프로젝트별 연구 진행과 실험 규모

PHM V1 README의 초기 EDA 상태보다 후속 4·4.5단계 보고서와 실제 결과표를 우선했습니다. V1 `reanalysis_v2`는 계획·폴더만 있고 학습 산출물은 확인되지 않았습니다. PHM V2도 학습 모델이 없습니다. **누락된 성능을 0으로 채우지 않았습니다.**''')
code('''coverage=pd.DataFrame([
 ['TEP Phase2',len(T['tep_test']),'공통 Test 평가 model_id 수; seed·구성 변형 포함'],
 ['TEP 별도 LSTM 예측',len(T['forecast_train']),'학습 seed 수; Persistence는 학습하지 않음'],
 ['PHM V1 stage4',len(T['v1_metrics'].query("split == 'evaluation'")),'48개 본/확장 + 10개 ID 보조진단; 단순 기준선 포함'],
 ['PHM V1 stage4.5',E['v1_cv_summary']['experiments'],'4-Fold × 4설정; 정책 2개 평가이므로 성능표 32행'],
 ['PHM V2',0,'감사·대표 2대 후보 생성; 학습 없음'],
 ['PHM V3 stage2',len(T['phm_stage2_train']),'A/B × Logistic/XGBoost 최대가용 코호트'],
 ['PHM V3 stage3',len(T['phm_stage3_train']),'A/B/C × Logistic/XGBoost 공통 코호트'],
 ['PHM V3 stage4',len(T['phm_stage4_train']),'21개 보존 실행 중 신규 18개·stage3 재사용 3개'],
 ['PHM V3 stage5',len(T['phm_stage5_train']),'보호된 장비 그룹 검증용 XGBoost'],
 ['PHM V3 stage6',0,'기존 stage2 B를 동결; 새 학습 없이 최종 Holdout 1회'],
 ['PHM V3 stage7',len(T['phm_stage7_train']),'03M02 장비 전용 신규 학습 1개']
],columns=['단계','실행/모델 수','분모 및 의미'])
display(coverage.style.hide(axis='index'))''')
md('''## 3. 지표를 읽는 공통 기준

| 지표 | 질문 | 주의점 |
|---|---|---|
| Accuracy | 시점별 정답을 얼마나 맞혔나? | 정상 비중이 높으면 고장을 전부 놓쳐도 높음 |
| Precision / Recall / F1 | 경보의 정확성과 고장 포착이 어떤가? | 임계값과 평가 라벨 집합을 확인 |
| Macro-F1 | 각 클래스의 F1을 동등하게 평균하면? | 고장 유형 불균형을 보완 |
| ROC-AUC / AP | 점수의 순위 구분 능력은? | 임계값 적용 Accuracy와 다름; AP는 양성 비율 영향 |
| 전체 이벤트 검출률 | 기록된 전체 고장을 얼마나 검출했나? | 입력 없는 고장도 분모에 포함 |
| 조건부 이벤트 검출률 | 입력이 있었던 고장 중 얼마나 검출했나? | 입력 부족으로 평가 불가한 사건을 숨길 수 있음 |
| 지연 / 선행시간 | 고장 후 얼마나 늦게 / 고장 전 얼마나 일찍? | 성공한 사건만의 통계인지 확인 |
| RMSE / MAE | 연속값 예측 오차는? | LSTM 센서 예측에 사용; 분류 Accuracy를 적용하지 않음 |

이번에 추가 계산한 Accuracy는 저장 혼동행렬 또는 저장 예측·고정 임계값으로 계산했습니다. 이미 보고된 원 지표와 구분해 설명합니다. PHM V1은 당시 보고서가 초 해석을 지지했지만 V3는 raw 단위를 유지합니다. 통합 비교에서는 PHM 시간을 임의로 분/시간으로 변환하지 않습니다.''')
md('''## 4. TEP — 모델 구성과 비교 조건

Harvard TEP, 52변수, 3분 간격, 실행 family 기준 Train400 / Validation100 / 공식 Test500 분할입니다. 현재 특징은52개, 통계 특징은312개, 시계열 특징은20×52입니다. 같은 run family의 정상·고장 시나리오를 같은 분할에 두었습니다.

Logistic/RF/XGBoost의 현재·통계 특징과 class weight를 비교했고, LSTM·GRU·CNN은 같은 시계열 입력으로 비교했습니다. Binary XGBoost는 정상/고장 전용이며 Unknown Binary는 Fault 3·9·15를 학습에서 제외한 별도 실험입니다. 주 CNN은 Validation으로 선택한 **seed42 단일 모델**이며 ensemble이 아닙니다.''')
code('''a=T['tep_accuracy'].copy()
for c in ['binary_accuracy_mean','binary_accuracy_sd','class21_accuracy_mean','class21_accuracy_sd']:a[c]*=100
display(a.rename(columns={'binary_accuracy_mean':'2종 Accuracy 평균(%)','binary_accuracy_sd':'2종 SD(%p)',
 'class21_accuracy_mean':'21종 Accuracy 평균(%)','class21_accuracy_sd':'21종 SD(%p)'}).style.format(precision=2).hide(axis='index'))
q=T['tep_means']
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
names=q.model+' / '+q.features
axes[0].barh(names,q.detection_f1_mean,color='#2b8291');axes[0].set(xlim=(0,1),xlabel='정상/고장 F1',title='TEP 공통 시점 · 3개 seed 평균')
axes[1].barh(names,q.diagnosis_macro_f1_mean,color='#c48643');axes[1].set(xlim=(0,1),xlabel='고장 조건부 Macro-F1',title='유형 구분 능력 · 검출 실패와 분리')
for ax in axes:ax.invert_yaxis()
plt.show()
show('tep_means',['model','features','weight','seeds','detection_f1_mean','diagnosis_macro_f1_mean','end_to_end_macro_f1_mean'])''')
md('''### TEP의 별도 이진 탐지와 미지 고장

정상/고장 전용 XGBoost는 natural/balanced/undersample/oversample을 비교했습니다. Unknown Binary의 미학습 고장 성능은 같은 고장 유형을 학습한 CNN의 결과와 같은 조건이 아닙니다. 분류 유형을 출력하지 않는 이진 모델의 21종 Accuracy는 공란입니다.''')
code('''z=T['tep_all_results']; binary=z[z.model_id.str.contains('BinaryXGBoost')]
display(binary[['model_id','binary_accuracy','detection_f1','detection_roc_auc',
                'unseen_faults_detection_f1','seen_faults_detection_f1']].style.format(precision=4).hide(axis='index'))''')
md('''### 최종 CNN의 전체 Test 결과와 추가된 기능

공통 비교는997,500시점이고 최종 CNN 전체 평가는9,880,500시점입니다. 평가 시점 차이 때문에 두 표의 점수가 다릅니다. 유형별 진단 평가는 이미 존재한21개 출력을 재사용했으며 새 분류 head를 학습하지 않았습니다. 고정 임계값에서 고장3연속이면 검출, 동일 유형3연속이면 유형 확정입니다.''')
code('''d=E['tep_dense']['joint_primary'];v=E['diagnosis']
display(pd.DataFrame([
 ['정상/고장 Accuracy',E['tep_dense_accuracy']['binary']],['임계값 적용 21종 Accuracy',E['tep_dense_accuracy']['class21']],
 ['고장 조건부 20종 Accuracy',v['classification_metrics']['conditional20']['accuracy']],
 ['정상/고장 F1',d['detection_f1']],['정상/고장 ROC-AUC',d['detection_roc_auc']],
 ['임계값 적용 21종 Macro-F1',d['end_to_end_macro_f1']],
 ['실행 검출률',v['detection_run_rate']],['실행 정답 유형 진단률',v['correct_diagnosis_run_rate']],
 ['최초 확정 유형 정확도',v['first_confirmed_type_accuracy']]
],columns=['지표','측정값']).style.format({'측정값':'{:.2%}'}).hide(axis='index'))
picture('diag')''')
md('''검출률98.51%와 정답 유형 진단률90.15%는 실행 단위 결과입니다. 최초 확정 유형 정확도44.85%는 최초 유형이 확정된 실행이 분모입니다. Fault9는500실행 중15개만 정답 진단, Fault15는0개였습니다. 검출 지연 중앙48분, 정답 진단 지연 중앙69분은 각각 성공한 실행만의 통계입니다.''')
md('''## 5. TEP — 입력 가용성 저하와 PHM 취약점 모사

전체 고장 실행 중0/10/25/40/53.1%를 입력 부족으로 평가 불가능하게 만들었습니다. 샘플 삭제율이 아닙니다. 고장별 균형 선정, seed42·43·44 반복, 모델·임계값·라벨은 고정입니다. 유효20시점 창만 남겨 평가하고 임의 예측이나 미래 보간은 하지 않았습니다. 짧은 누락 뒤 유효 입력이 남으면 부분 가용 실행으로 집계했습니다.''')
code('''show('availability',['rate','available_runs','unavailable_runs','overall_detection_rate',
 'conditional_detection_rate','overall_correct_diagnosis_rate','conditional_correct_diagnosis_rate'])
picture('availability')''')
md('''53.1% 조건의 전체 검출률46.17%, 전체 정답 진단률42.29%와 달리, 가용 실행의 조건부 성능은 약98.45%,90.16%입니다. **모델의 분류 능력 저하를 직접 보인 결과가 아니라, 입력 상실로 적용 범위가 줄어드는 효과**입니다.''')
md('''## 6. TEP — 별도 LSTM 공정 변수 예측

이 LSTM은 앞의 21종 LSTM 분류기와 다릅니다. 정상 Train400실행으로20×52→다음3분52변수를 학습했습니다. 2층hidden64, 현재값+예측 변화량 구조이며, seed42/7/2026을 비교했습니다. Validation으로 epoch를 선택하고 대표seed42의 전체9,870,000개 정답 시점을 평가했습니다. CNN에 연결하거나 결측 복원에 쓰지 않았습니다.

정상 Train 표준편차로 정규화한 오차입니다. 전체 RMSE는 개선됐지만 MAE는 악화됐으므로 전반적으로 항상 더 좋은 예측기라고 말할 수 없습니다.''')
code('''show('forecast_train')
f=T['forecast_test'];display(f[(f.seed==42)&(f['mode']=='dense')][['model','cohort','target_rows','normalized_rmse','normalized_mae']].style.hide(axis='index'))
picture('forecast')''')
md('''## 7. PHM V1 — 최초 모델 비교

Dummy·압력 임계값·Random Forest·XGBoost를 비교했습니다. A8=현재 센서6개+recipe/stage, B80=현재+과거60/300통계, C83=공정 전환 등 추가입니다. 이는 **V3의 A8/B32/C44와 다른 정의**입니다.

horizon300/600/1800, 장비 분할·시간 분할, A/B/C를 비교한48실험과 Tool/time/Lot/사용량을 넣어 의존성을 보는10진단 실험을 보존했습니다. ID 진단 모델은 주 모델과 분리합니다. 알려진 라벨·충분한 이력이 있는 관측만의 결과이며 미확인 라벨을 정상으로 바꾸지 않았습니다.''')
code('''m=T['v1_metrics'];main=m[(m.split=='evaluation')&(m.horizon==600)&(~m.identity_diagnostic)]
display(main[['scheme','name','config','n','positive','accuracy','f1','pr_auc_ap','roc_auc','event_recall']].style.format(precision=4).hide(axis='index'))
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for ax,scheme in zip(axes,['split_tool','split_time']):
 g=main[main.scheme==scheme];x=np.arange(len(g))
 ax.bar(x-.18,g.accuracy,.36,label='Accuracy');ax.bar(x+.18,g.f1,.36,label='고장 F1')
 ax.set(xticks=x,xticklabels=g.name,ylim=(0,1.05),title=f'PHM V1 H600 / {scheme}')
 ax.tick_params(axis='x',rotation=45);ax.legend()
plt.show()''')
md('''**Accuracy가 높아도 고장을 잘 찾는 것은 아닙니다.** 양성이 적은 PHM에서 항상 정상으로 판단하는 Dummy도 높은 Accuracy가 나올 수 있습니다. F1·AP·이벤트 검출과 함께 읽어야 합니다. 아래는 당시 Validation AP로 선정한 대표 모델의 실제 평가 결과이며, 현재 표를 보고 새로 선택한 모델이 아닙니다.''')
code('''selected=pd.DataFrame(E['v1_summary']['selected_per_family'])
selected['accuracy']=(selected.tp+selected.tn)/(selected.tp+selected.tn+selected.fp+selected.fn)
display(selected[['scheme','selected_by_validation_AP','accuracy','f1','pr_auc_ap','roc_auc','tp','fn']].style.format(precision=4).hide(axis='index'))
display(m[m.split=='evaluation'][['scheme','horizon','name','config','identity_diagnostic','accuracy','f1','pr_auc_ap','event_recall']].style.format(precision=4).hide(axis='index'))''')
md('''## 8. PHM V1 — 4.5단계 장비 일반화 개선 실험

개발 장비의4-Fold × A/B/C/C_regularized로 XGBoost16개를 학습했습니다. 각 모델에 F1정책과 이벤트/FAR정책 두 경보 방식을 평가하여 성능표는32행입니다. 정책 수를 학습 수로 세지 않습니다. 학습·조기 종료·임계값 보정·외부 Fold의 장비를 분리했고, 기존 평가09/10계열은 새 학습·선택에서 제외했습니다.

결론은 **일반화 개선이 확실히 입증되지 않음**입니다. 기존 데이터의 후향적 후속 분석이며 새로운 미관측 외부 검증 결과가 아닙니다.''')
code('''cv=T['v1_cv'];agg=cv.groupby(['model','policy']).agg(
 accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),f1_mean=('f1','mean'),
 AP_mean=('pr_auc_ap','mean'),event_recall_mean=('event_recall','mean')).reset_index()
display(agg.style.format(precision=4).hide(axis='index'))
show('v1_cv_means',['model','policy','f1_mean','event_recall_mean','false_alarms_per_active_hour_mean'])''')
md('''## 9. PHM V2 — 모델 성능을 제시하지 않는 이유

V2는 전체20대 원본 감사와 대표2대 후보 특징·라벨·분할 진단 단계입니다. 시간 단위·고장 이력 완전성·정비 종료가 확정되지 않아 정식 학습 데이터 배포와 모델 학습을 차단했습니다. A8/B32/C44 후보가 있다는 것은 모델이 학습됐다는 뜻이 아닙니다. Accuracy·F1은 **미산출**입니다. V1 재분석 폴더도 모델 재학습 결과가 확인되지 않아 별도 모델로 세지 않았습니다.''')
md('''## 10. PHM V3 — 실제로 학습한 모델과 단계

센서6개+recipe/stage의 A8, 과거300raw mean/std/min/max를 더한 B32, 추가 공정 이력을 더한 C44를 비교했습니다. 수치 표준화·범주형 One-Hot은 Train에서만 fit했습니다. 범주형 인코딩 후 열 수는 학습 집합마다 달라집니다. 시간/장비ID/TTF/미래 라벨은 입력에서 제외했습니다.

stage2는 각 특징의 최대가용 입력, stage3는 A/B/C가 모두 가능한 동일 공통 입력입니다. 분모가 다르므로 stage2와3의 점수 차이를 특징 개선으로 단정하지 않습니다. 알려진 라벨 AP/ROC-AUC는 보조 분류 지표이고, NA/EXCLUDE까지 점수화한 이벤트 검출과 경보 부담을 함께 봅니다.''')
code('''cols=['run_id','feature_set','model','train_rows','validation_known_rows','encoded_features','iterations','AP_known','ROC_AUC_known','train_seconds']
display(Markdown('**stage2: 최대가용 코호트 4개**'));show('phm_stage2_train',cols)
display(Markdown('**stage3: 공통 코호트 6개**'));show('phm_stage3_train',cols)
points=T['phm_stage3_points'];p=points[points.budget_per_10000_raw==10]
display(p[['run_id','recorded_events','available_events','detected_events','recall_recorded','recall_available','alarm_episodes_per_10000_raw']].style.format(precision=4).hide(axis='index'))
fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
ax.bar(p.run_id,p.recall_available*100,color='#427c9a')
ax.set(ylabel='가용 고장 검출률(%)',title='PHM V3 stage3 · Validation 경보 비교상한10 · 동일 공통 코호트')
ax.tick_params(axis='x',rotation=30);plt.show()''')
md('''### stage4: 제한 튜닝·seed 민감도

A/B의 depth·양성 가중치를 비교하고 설정을 동결한 뒤 seed43/44를 확인했습니다. 18개 신규 학습과3개 기존 실행 재사용으로21행입니다. seed42에서의 개선이 다른seed에서 일관되지 않아 기존 기준을 유지했습니다. 아래 AP/ROC-AUC는 동일 Validation 기반이며 최종 Holdout 성능이 아닙니다. 임계값 고정 이전과 seed별 재선택도 분리해 보존했습니다.''')
code('''show('phm_stage4_train',['run_id','feature_set','depth','weight_mode','seed','reused_stage3','phase','iterations','AP_known','ROC_AUC_known'])
show('phm_stage4_ranges')''')
md('''### stage5: 장비 그룹 검증

01–06학습 /07–08검증, 기존 시간 Holdout을 제외한6개 XGBoost를 학습했습니다. 최대가용 A/B와 공통A/B/조정A/C입니다. 시간 Validation·그룹 Validation을 함께 고려해 사전 규칙으로 최종 B를 선택했습니다. 이 그룹 모델을 최종 모델로 교체한 것이 아니라 **기존 stage2 B를 최종 모델로 동결**했습니다.''')
code('''show('phm_stage5_train',['run_id','cohort','train_rows','validation_known_rows','iterations','AP_known','ROC_AUC_known'])
picture('phm_group')''')
md('''## 11. PHM V3 — 최종 동결 모델과 Holdout

최종 artifact는 stage2의 B XGBoost(seed42, depth3, 선택tree2)와 Train 전용 전처리입니다. train+validation 재학습 없이 모델과 두 임계값을 동결했습니다. 각 장비 마지막20%를 한 번 추론한 뒤 저장 예측만 재사용했습니다.

고장 전 `[t_fault-600, t_fault-60]` raw 구간에 시작한 유효 경보를 평가합니다. 기록194개 중91개만 입력이 있었고103개는 평가 불가능했습니다. 아래 두 운영점은 모두 사전에 동결됐으며 최종 결과를 보고 재선택하지 않았습니다.''')
code('''show('phm_holdout',['operating_point','threshold','recorded_events','available_events','detected_events',
 'recall_recorded','recall_available','alarm_episodes','alarm_episodes_per_10000_raw','median_lead_raw'])
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
q=T['phm_holdout'];x=np.arange(len(q))
ax.bar(x-.18,q.recall_recorded*100,.36,label='전체194개 기준')
ax.bar(x+.18,q.recall_available*100,.36,label='입력 가용91개 기준')
ax.set(xticks=x,xticklabels=q.operating_point,ylabel='고장 검출률(%)',ylim=(0,100),title='PHM V3 최종 Holdout · 동일 모델의 사전 고정 운영점')
ax.legend();plt.show()''')
md('''### PHM 최종 모델의 Accuracy도 계산할 수 있나?

저장된 Holdout 예측의 `audit_label`이0/1인 **정답 확인32,645개 시점**에서 고정 임계값별 Accuracy를 추가 계산했습니다. 나머지 미확인/제외 입력을 임의 정상으로 바꾸지 않았습니다. 이는 입력736,646개 전체의 Accuracy나 고장194개 전체의 검출률이 아닙니다. 아래의 ‘항상 정상 Accuracy’는 라벨 불균형을 보여주는 **학습 없는 계산 기준**입니다.

고정 모델의 알려진 라벨 AP=0.0654, ROC-AUC=0.5012였습니다. 높은 Accuracy가 나오더라도 실제 고장 조기 예측이 충분하다는 뜻은 아닙니다.''')
code('''acc=T['phm_holdout_accuracy'].copy();acc['always_normal_accuracy']=acc.negative/acc.known_rows
display(acc.style.format({'accuracy':'{:.2%}','always_normal_accuracy':'{:.2%}','recall':'{:.2%}','precision':'{:.2%}'}).hide(axis='index'))
display(pd.DataFrame([E['phm_classification']]).drop(columns=['label_counts']).style.hide(axis='index'))''')
md('''## 12. PHM V3 — 장비 전용 모델 후속 파일럿

최대3대 후보 중 최소 학습/검증 조건을 통과한03M02 한 대에만 B XGBoost를 학습했습니다. 전용 모델과 공통 모델의 장비별 임계값 조정, 공통 모델 기존 임계값을 비교했습니다.

후반 구간은 이미 본 데이터를 다시 사용한 연구 비교로 독립 최종 Holdout이 아닙니다. 27개 고장 중24개는 입력이 없고3개만 가용했습니다. 전용 모델은0개를 검출했습니다. 공통 모델 기존 높은 경보 설정은2개를 검출했지만 경보 부담이 더 크므로 동일 부담에서의 우위를 주장하지 않습니다.''')
code('''show('phm_stage7_train')
show('phm_specialist')
picture('phm_specialist')''')
md('''## 13. TEP와 PHM에서 배운 차이

| 항목 | TEP | PHM2018 V3 |
|---|---|---|
| 목적 | 주입 후 검출·유형 진단 | 정지 이벤트 이전 조기 예측 |
| 데이터 | 규칙적3분 간격 시뮬레이션52변수 | 실제 장비 센서·공정, 공백·라벨 미확인 존재 |
| 최종 주 모델 | 1D-CNN 21클래스 | B XGBoost 이진 위험점수 |
| 사건 단위 | 고장 실행10,000개 | 기록 고장194개 |
| 원본 평가 입력 가용성 | 모든 실행에 고장 이후 유효 입력 | 91개 가용 /103개 불가 |
| 입력53.1% 손실 비교 | 전체 검출46.17%, 조건부98.45% | 전체17.01%, 조건부36.26% (고정 상한10 운영점) |
| 해석 | 적용 범위 손실과 유형 혼동 분리 | 입력·라벨·공정 분포·장비 일반화 제약 |

표의 검출률은 같은 과제의 우열 비교가 아닙니다. PHM의 실제 pre-fault 입력조건과 TEP의20시점 창 조건도 다릅니다. TEP에서는 ‘기회가0이면 평가 불가’라는 의미를 모사했습니다.

실행이 입력 불가하면 모델 교체만으로 해결되지 않습니다. 입력이 있어도 검출/정답 진단에 실패할 수 있으므로 전체와 조건부 성공률을 함께 봐야 합니다. 학습하지 않은 기준선·튜닝 변형·평가 후속 분석을 새로운 모델 성능 향상으로 표현하지 않았습니다.''')
md('''## 14. TEP 모든 평가 모델 — 전체 기록

아래는 seed·가중치 후보와 Binary/Unknown 실험을 포함한44개 model_id입니다. Accuracy는 저장된 혼동행렬로 계산했고, 별도 이진 모델은 저장 recall/FAR와 공통 평가의 양성800,000·음성197,500으로 계산했습니다. Unknown 모델의 아래 전체 성능에는 기지·미지 고장이 섞이므로 미지 성능은4절을 확인합니다.''')
code('''show('tep_all_results',['model_id','samples','binary_accuracy','class21_accuracy','detection_f1',
 'diagnosis_macro_f1','end_to_end_macro_f1','detection_roc_auc'])''')
md('''## 15. 자료 위치와 재확인 방법

| 자료 | 위치 |
|---|---|
| TEP 기능 확장 안내 | `TEP/notebooks/17_CNN_기능확장_종합안내.ipynb` |
| TEP 모든 model_id 원 평가 | `TEP/reports/tables/phase2/test_comparison.csv` |
| PHM V1 최초 모델 비교 | `PHM2018-Project/reports/04_model_training_summary.md` |
| PHM V1 4-Fold 개선 | `PHM2018-Project/reports/04_5_model_improvement_summary.md` |
| PHM V2 학습 전 진단 | `PHM2018-Project-V2/README.md` |
| PHM 전체 기존 수행 결과 | `PHM2018-Project-V3/notebooks/00_PHM2018_종합_수행결과.ipynb` |
| PHM 최종 모델·평가 | `PHM2018-Project-V3/reports/19_group_and_holdout.md` |
| PHM 장비 전용 후속 | `PHM2018-Project-V3/reports/20_equipment_specialists.md` |
| 이번 내장 자료 CSV | `TEP/reports/all_models/tables/` |
| 이번 자료 해시·출처 | `TEP/reports/all_models/source_manifest.json` |

노트북의 `T` 사전에서 원 평가표를 다시 확인할 수 있습니다. 예를 들어 `T['tep_all_results']`는 TEP 전체 모델, `T['v1_metrics']`는 V1 전체 분할 결과, `T['phm_stage4_train']`은 V3 튜닝 실행입니다. 원본 결과를 바꾸거나 새 학습을 시작할 필요가 없습니다.

공란은 미산출/해당 없음입니다. PHM 미확인 시작 경보는 확정 오경보가 아니며, raw당 전체 경보 부담을 TEP 정상 시간당 오경보와 직접 비교하지 않습니다. 이 노트북은 로컬에서 확인된 실험을 정리한 것이며 별도의 외부 검증이나 배포 검증을 추가하지 않았습니다.''')
code('''assert len(T['tep_all_results'])==44
assert len(T['v1_metrics'].query("split == 'evaluation'"))==58
assert len(T['phm_stage4_train'])==21
assert (~T['phm_stage4_train'].reused_stage3).sum()==18
assert (T['phm_holdout_accuracy'].known_rows==32645).all()
assert (T['phm_holdout'].recorded_events-T['phm_holdout'].available_events==103).all()
display(Markdown('**내장 표의 실험 수·재사용 수·PHM 라벨 분모·입력 불가 수 검증 통과.**'))''')

target=ROOT/'notebooks/18_TEP_PHM_전체모델_종합비교.ipynb'
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},
 'language_info':{'name':'python'},'catalog':{'snapshot_embedded':True,'no_training':True,'no_inference':True}})
nbf.validate(nb)
runtime=OUT/'runtime';kernel=runtime/'jupyter/kernels/tep-catalog';kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
 'display_name':'TEP PHM catalog','language':'python','env':{'PYTHONPATH':str(ROOT/'.deps'),
 'PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(runtime/'matplotlib'),'IPYTHONDIR':str(runtime/'ipython')}}),encoding='utf-8')
os.environ['JUPYTER_PATH']=str(runtime/'jupyter');os.environ['JUPYTER_RUNTIME_DIR']=str(runtime/'sessions');(runtime/'sessions').mkdir(exist_ok=True)
if sys.platform=='win32':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
NotebookClient(nb,timeout=180,kernel_name='tep-catalog',resources={'metadata':{'path':str(runtime)}}).execute()
outputs=[o for c in nb.cells for o in c.get('outputs',[])]
assert not any(o.output_type=='error' for o in outputs)
changed=[r['path'] for r in sources if sha(Path(r['path']))!=r['sha256']];assert not changed
nbf.validate(nb);nbf.write(nb,target)
(OUT/'source_manifest.json').write_text(json.dumps({'sources':sources,'changed':changed,'raw_sensor_files_read':False,
 'phm_saved_predictions_read_only':True},ensure_ascii=False,indent=2),encoding='utf-8')
validation={'notebook':str(target.relative_to(ROOT)),'cells':len(cells),'code_cells':sum(c.cell_type=='code' for c in cells),
 'images':sum('image/png' in o.get('data',{}) for o in outputs),'errors':0,'source_files_verified':len(sources),
 'changed_source_files':changed,'bytes':target.stat().st_size,'tables':len(tables),'executed_without_source_paths':True}
(OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(validation,ensure_ascii=False,indent=2))
