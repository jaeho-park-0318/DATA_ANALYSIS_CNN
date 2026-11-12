"""V1/V2/V3 수행 기록을 자료 내장형 단일 Notebook으로 종합한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import base64,gzip,hashlib,io,json,os,re,time
import nbformat as nb
import pandas as pd
from nbclient import NotebookClient
from src.common import ROOT,CFG,dump_json

DEST=ROOT/'notebooks/00_PHM2018_종합_수행결과.ipynb'
BUILD=ROOT/'outputs/consolidated'
V1=ROOT.parent/'PHM2018-Project'
V2=ROOT.parent/'PHM2018-Project-V2'


def sha(data):return hashlib.sha256(data).hexdigest()
def readjson(path):return json.loads(path.read_text(encoding='utf-8'))
def origin(path):return str(path.relative_to(ROOT.parent)).replace('\\','/')


def main():
    BUILD.mkdir(parents=True,exist_ok=True)
    # 과거 모델/예측/결과의 보호와 생성 전후 해시 대조.
    from src.specialist_experiment import verify_previous
    verify_previous()
    protected={ROOT/p:h for p,h in readjson(ROOT/'outputs/stage7/prior_artifacts.json').items()}
    protected.update({ROOT/p:h for p,h in readjson(ROOT/'outputs/stage7/frozen_manifest.json')['sha256'].items()})
    payload={'tables':{},'json':{},'sources':{},'created_date':'2026-11-12','scope':'V1/V2/V3의 저장된 수행 결과; 학습/모델 추론 없음'}
    catalog=[];table_sources=[]
    def add_table(path,key):
        data=path.read_bytes();frame=pd.read_parquet(io.BytesIO(data))
        payload['tables'][key]=dict(data=base64.b64encode(data).decode(),sha256=sha(data),source=origin(path),rows=len(frame),columns=frame.columns.tolist())
        catalog.append(dict(표=key,행=len(frame),열=len(frame.columns),출처=origin(path),SHA256=sha(data)))
        table_sources.append(path)
    for path in sorted((ROOT/'outputs/tables').glob('*.parquet')):add_table(path,'V3/'+path.stem)
    for stage in ['stage4','stage4_5']:
        for path in sorted((V1/'outputs'/stage).glob('*.parquet')):add_table(path,'V1/'+stage+'/'+path.stem)
    add_table(ROOT/'outputs/stage6/holdout_predictions.parquet','V3/최종_저장예측')
    for path in sorted((ROOT/'outputs/stage7').glob('*_later_predictions.parquet')):add_table(path,'V3/전용_'+path.stem)
    json_paths=[*sorted((ROOT/'outputs/validation').glob('*.json')),
                ROOT/'outputs/stage6/frozen_manifest.json',ROOT/'outputs/stage6/holdout_access.json',ROOT/'outputs/stage6/holdout_classification.json',ROOT/'outputs/stage6/postprocessing_recovery.json',
                ROOT/'outputs/stage7/selection.json',ROOT/'outputs/stage7/frozen_manifest.json',ROOT/'outputs/stage7/evaluation_access.json',ROOT/'outputs/stage7/checks.json',ROOT/'outputs/stage7/completion.json',
                *sorted((ROOT/'config').glob('*.json')),
                V1/'outputs/stage4/summary.json',V1/'outputs/stage4_5/summary.json',V1/'outputs/stage4_5/hardware.json',V1/'outputs/stage4_5/hardware_supplement.json',
                V2/'outputs/release_status.json',V2/'outputs/pilot_summary.json',V2/'outputs/validation.json',V2/'outputs/reproducibility.json',V2/'outputs/notebook_validation.json']
    for path in json_paths:
        key=origin(path);data=path.read_bytes();payload['json'][key]=json.loads(data);payload['sources'][key]=sha(data)
    report_paths=[*sorted((V1/'reports').glob('*.md')),*sorted((V2/'reports').glob('*.md')),*sorted((ROOT/'reports').glob('*.md')),V1/'reanalysis_v2/README.md']
    reports={origin(p):p.read_text(encoding='utf-8') for p in report_paths}
    for p in report_paths:payload['sources'][origin(p)]=sha(p.read_bytes())

    figure_paths=[];figure_info={}
    manifests=[ROOT/'outputs/figure_manifest.json',ROOT/'outputs/alarm_stage1_figure_manifest.json',ROOT/'outputs/stage2_figure_manifest.json',ROOT/'outputs/stage3_figure_manifest.json',ROOT/'outputs/stage4_figure_manifest.json',ROOT/'outputs/final_evaluation_figure_manifest.json',ROOT/'outputs/stage7/figure_manifest.json']
    for path in manifests:
        for item in readjson(path):
            p=ROOT/item['file'];figure_paths.append(p);figure_info[p.resolve()]=item
    historical_dirs=[V1/'reports/local/eda/figures',V1/'outputs/stage4/figures',V1/'outputs/stage4_5/figures',V2/'reports/figures']
    for directory in historical_dirs:figure_paths.extend(sorted(directory.glob('*.png')))
    for path in report_paths:
        for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',reports[origin(path)]):
            p=(path.parent/target).resolve()
            if p.suffix.lower()=='.png' and p.exists():figure_paths.append(p)
    figure_paths=list(dict.fromkeys(p.resolve() for p in figure_paths))
    figure_ids={p:'fig-'+str(i+1).zfill(3) for i,p in enumerate(figure_paths)}
    figure_bytes={p:p.read_bytes() for p in figure_paths}
    for p,data in figure_bytes.items():payload['sources'][origin(p)]=sha(data)

    old_notebooks=[]
    for version,folder in [('V1',V1),('V2',V2),('V3',ROOT)]:
        for path in sorted((folder/'notebooks').glob('*.ipynb')):
            if path==DEST:continue
            old=nb.read(path,as_version=4);codes=[c for c in old.cells if c.cell_type=='code']
            old_notebooks.append(dict(버전=version,노트북=path.name,코드셀=len(codes),실행셀=sum(c.execution_count is not None for c in codes),저장오류=sum(o.output_type=='error' for c in codes for o in c.get('outputs',[])),출처=origin(path)))
    payload['notebooks']=old_notebooks
    book=nb.v4.new_notebook();book.metadata['kernelspec']={'name':'phm2018-v3','display_name':'PHM2018 V3 (로컬)','language':'python'}
    book.metadata['phm_consolidated']={'created_date':'2026-11-12','standalone_results':True,'model_training':False,'model_inference':False,'original_versions':['V1','V2','V3']}
    cells=[];shown=set()
    def markdown(text):cells.append(nb.v4.new_markdown_cell(text))
    def code(text):cells.append(nb.v4.new_code_cell(text))
    def figures(paths):
        for path in paths:
            path=path.resolve()
            if path in shown:continue
            shown.add(path);info=figure_info.get(path,{})
            title=info.get('title',path.stem.replace('_',' '));context=info.get('context','이전 버전 작성 당시의 저장 그림. 현재 V3의 분모·라벨·시간 규약과 직접 합산하지 않음.')
            text=f'<a id="{figure_ids[path]}"></a>\n\n**{title}**\n\n![{title}](attachment:figure.png)\n\n출처: `{origin(path)}`\n\n{context}'
            c=nb.v4.new_markdown_cell(text);c['attachments']={'figure.png':{'image/png':base64.b64encode(figure_bytes[path]).decode()}};cells.append(c)
    def gallery(prefix):figures([p for p in figure_paths if origin(p).startswith(prefix)])
    def section(number,title,body):markdown(f'<a id="section-{number}"></a>\n\n## {number}. {title}\n\n{body}')

    markdown('''# PHM2018 종합 수행 결과 — V1 · V2 · V3

**정리 기준일: 2026-11-12 · 실제 데이터의 저장된 수행 결과 · 시간 표기는 raw**

원본 점검/EDA → V1 모델58개·그룹 교차검증16개 → V2 원본 재감사·대표2대 진단 → V3 전체20대 전처리/시각화 → 공통 모델 비교 → seed 민감도 → 보호된 장비 그룹 검증 → 모델/임계값 동결·최종 시간 평가 →03M02 전용 모델 파일럿을 종합했습니다.

표와 그림, 설정·검사 기록을 파일 안에 내장했습니다. 저장된 출력으로 바로 읽을 수 있고, 다른 폴더에서 전체 셀을 재실행할 수 있습니다. 실행에 pandas/PyArrow/IPython이 필요합니다. 원본 CSV나 모델 파일 없이 저장 결과를 조회하며 새 학습·모델 추론·임계값 탐색은 수행하지 않습니다. 원본82백만 행과 모든 모델 바이너리를 포함한 학습 재현 패키지는 아닙니다.

**현재 핵심 결과:** V3 최종 B 공통 모델은 두 고정 설정에서7/194개와33/194개를 검출했습니다. 입력 가용91개 기준은7.7%/36.3%입니다. 장비 전용 파일럿03M02는 후반 입력 가용3개 고장을 검출하지 못했습니다. 입력 공백·미확인 정답·경보 부담 때문에 현재는 연구 기준선으로 보존합니다.

**읽는 순서:** [버전 차이](#section-1) · [V1 연구](#section-2) · [V2 재검증](#section-3) · [V3 원본/전처리](#section-4) · [라벨](#section-5) · [특징/이벤트](#section-6) · [분할](#section-7) · [경보 평가](#section-8) · [최초 학습](#section-9) · [공통 코호트](#section-10) · [튜닝/seed](#section-11) · [장비 그룹](#section-12) · [동결/최종 평가](#section-13) · [장비 전용](#section-14) · [자원/검증](#section-15) · [종합 판단](#section-16) · [전체 표 목록](#section-17) · [원문 보고서](#section-18)

이전 보고서와 그림의 “미학습/holdout 미사용/다음 단계” 문구는 **당시 시점의 기록**입니다. V1은 A/B/C=8/80/83, V2/V3는8/32/44로 정의가 다릅니다. V1에 초 단위 해석을 지지한 기록이 있어도 이 종합본에서 새 단위 검증을 수행한 것은 아니므로 V3 원칙대로 raw를 유지합니다. V1/V3 성능을 같은 평가로 합산하거나 개선 폭으로 계산하지 않습니다.''')
    raw_payload=json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode();packed=base64.b64encode(gzip.compress(raw_payload,compresslevel=6,mtime=0)).decode()
    loader="""# 내장 자료 읽기: 원본·모델 파일을 열지 않습니다.
import base64,gzip,hashlib,io,json
import numpy as np,pandas as pd
from IPython.display import display,Markdown
"""+f"_PACKED={packed!r}\n"+"""DATA=json.loads(gzip.decompress(base64.b64decode(_PACKED)))
del _PACKED
def T(name):
    key=name if name.startswith(('V1/','V3/')) else 'V3/'+name
    entry=DATA['tables'][key]
    raw=base64.b64decode(entry['data'])
    assert hashlib.sha256(raw).hexdigest()==entry['sha256'],key
    return pd.read_parquet(io.BytesIO(raw))
def J(path):
    return DATA['json'][path]
def show(name,columns=None,rows=30):
    d=T(name)
    if columns is not None:d=d[columns]
    display(Markdown(f'**{name}** · 전체 {len(d):,}행 × {len(d.columns)}열'+(f' · 앞 {rows}행 표시, 전체 값은 `T({name!r})`로 조회' if len(d)>rows else '')))
    with pd.option_context('display.max_rows',rows,'display.max_columns',None,'display.max_colwidth',90,'display.width',180):display(d.head(rows))
def show_frame(d):
    with pd.option_context('display.max_columns',None,'display.max_rows',100,'display.max_colwidth',100):display(d)
print('내장 표:',len(DATA['tables']),'개 | 설정·감사 JSON:',len(DATA['json']),'개 | 이전 Notebook:',len(DATA['notebooks']),'개')
"""
    code(loader);cells[-1].metadata={'jupyter':{'source_hidden':True},'tags':['내장자료','hide-input']}
    section(1,'버전과 평가 조건의 차이','V1은 초기 표본/모델 연구, V2는 정답 근거 재감사, V3는 명시적 가용성·경보 부담 평가를 강화한 연구입니다. V2가 학습을 중단한 사실과 이후 V3가 연구용 라벨로 실험을 진행한 사실을 함께 기록합니다. 정비 종료·이력 완전성 검증이 끝났다는 뜻은 아닙니다.')
    code("show_frame(pd.DataFrame([['V1','8/80/83','원본/EDA·58개 모델·16개 그룹CV','당시 정답·관측 조건의 평가; V3와 직접 비교 불가'],['V2','8/32/44','20대 감사·2대 진단; 모델0회','확정 학습 데이터0행; 배포 차단'],['V3 공통','8/32/44','단계2~6 학습·동결·시간 평가','최종 B; 시간 뒤20% 점수는 선택 전 보류'],['V3 전용','B32','03M02 전용 모델1개','이미 본 뒤20%의 후속 연구 비교'] ],columns=['버전','A/B/C 특징 수','수행 내용','평가 해석']))\ntool=T('tool_summary');evt=T('events')\nassert len(tool)==20 and int(tool.raw_rows.sum())==82189440 and len(evt)==1211\nshow_frame(pd.DataFrame({'항목':['장비','원본 train 행','고유 고장','원본 고장 행','중복 제거 행'],'값':[len(tool),int(tool.raw_rows.sum()),len(evt),int(tool.raw_fault_rows.sum()),int(tool.raw_fault_rows.sum())-len(evt)]}))")
    section(2,'V1 원본 점검·EDA·최초 모델과 그룹 교차검증','초기 파일65개, train82,189,440행, 공식 test7,198,948행을 점검했습니다. test 파일 점검이 공식 test 성능 평가를 뜻하지 않습니다. Dummy/압력 기준/RF/XGBoost와 식별 보조 변수 진단까지58개 실험, 이후4-Fold×4구성의16개 CUDA 그룹 실험을 수행했습니다. 당시 조건부 이벤트 분모와 V3 전체 기록 분모는 다릅니다. 낮은 조건부 오경보율은 전체 현장 FAR가 아닙니다. 아래 수치는 저장된 과거 집계의 재표시입니다.')
    code("v1=J('PHM2018-Project/outputs/stage4/summary.json')\nassert v1['checks']['experiments']==58\nshow_frame(pd.DataFrame(v1['selected_per_family']))\nshow_frame(pd.DataFrame(v1['event_metrics'])[['scheme','model','events','detected','event_recall','median_lead_raw']])\nshow_frame(pd.DataFrame(v1['prospective']))\nshow('V1/stage4/metrics',rows=12)\nshow('V1/stage4_5/cv_summary',rows=20)\ndisplay(Markdown('V1의 시간당 지표는 당시 초 해석에 따른 역사적 값입니다. 현재 V3의 raw 경보 부담과 수치 비교하지 않습니다. 그룹 CV에서 정규화 강화 C의 개선도 Fold 간 변동이 커 안정적 일반화를 입증하지 못했습니다.'))")
    for directory in historical_dirs[:3]:figures(sorted(directory.glob('*.png')))
    section(3,'V2 원본 재감사·대표2대 전처리 진단','20대 원본을 감사하고01M01/03M02 두 장비만 진단 Parquet로 만들었습니다. 승인된 학습 자료는0행, 모델 학습0회이며 단위·이력 유효범위·정비 종료 미확정으로 전체 확정 데이터 생성을 차단했습니다. 블록 처리로 메모리 문제를 개선하고 진단 자료 재생성의 동일 SHA256을 확인한 기록이 있습니다.')
    code("display(J('PHM2018-Project-V2/outputs/release_status.json'))\ndisplay(J('PHM2018-Project-V2/outputs/reproducibility.json'))\ndisplay(J('PHM2018-Project-V2/outputs/notebook_validation.json'))\nshow_frame(pd.DataFrame([['01M01',3124704,662570,19,13,2],['03M02',5060498,679083,356,111,52]],columns=['장비','원본 행','격자 Anchor','고유 이벤트','센서 가용 격자 이벤트','900이력 양성 격자 이벤트']))")
    figures(sorted(historical_dirs[3].glob('*.png')))
    section(4,'V3 전체20대 원본·관측 타임라인과 품질','같은 시각 센서를 평균 집계해 정규화하고 원본 행 출처를 보존했습니다. dt>12 raw 또는 공정 충돌에서 세그먼트를 끊고 공백을 보간하지 않았습니다. 공정 충돌은 recipe/stage/recipe_step/Lot/runnum의 동일 시각 불일치입니다. 전체 관측 범위와 고정100,000 raw 구간/연속 세그먼트를 함께 진단했습니다.')
    code("show('tool_summary',rows=20)\nshow('gap_summary',rows=20)\nshow('preprocessing_preservation',rows=12)\nshow('time_segment_summary',rows=8)")
    gallery('PHM2018-Project-V3/outputs/figures/tool_overview/');gallery('PHM2018-Project-V3/outputs/figures/timeline/')
    section(5,'라벨 정책·TTF Null·예측 창','H=300/600/1800 raw, 최소 선행 L=60입니다. d=60은 양성,0≤d<60은 임박 EXCLUDE를 최우선 적용합니다. 보수적/제한 연구용/완전 이력 가정의 세 정책을 비교했으며 기본 학습은 보수적 통합 라벨입니다. TTF Null을 정상으로 바꾸지 않습니다. [NoWaferID] 포함/제외로 다음 고장이 바뀌는 시점은 NA로 격리합니다. 고장 정지 기록은 이상 시작이나 정비 종료를 뜻하지 않습니다.')
    code("labels=T('label_status_summary')\nz=labels[labels.sample_level.eq('anchor')&labels.fault_type.eq('any')].groupby(['policy','H'])[['positive','negative','NA','excluded','denominator']].sum()\nshow_frame(z.reset_index())\nshow('ttf_label_relationship',rows=20)")
    gallery('PHM2018-Project-V3/outputs/figures/label_analysis/')
    section(6,'A/B/C 특징 가용성·고장 보존·대표 사례','A8=현재 센서6+recipe/stage, B32=A+과거300 raw 평균/표준편차/최소/최대, C44=B+과거900 raw 기울기/변화량입니다. 특징 창은(t−w,t]이며 같은 세그먼트의90% 이상 관측을 요구합니다. 격자60 raw, 최근 현재값 지연은12 raw 이내입니다. 과거 이력을 늘리면 가용 입력과 고장이 줄어듭니다. 정답 존재와 입력 가용성을 구분하고 입력 없는 고장도 기록 분모에 남겼습니다. 아래에는12개 대표 사례와 상대시간 분석의 기존 그림을 모두 포함했습니다.')
    code("f=T('feature_availability_summary');z=f[f.policy.eq('conservative')&f.H.eq(600)&f.fault_type.eq('any')].groupby(['cohort','feature_set'])[['available_anchors','positive_anchors','negative_anchors','NA_anchors','positive_events']].sum().reset_index();show_frame(z)\nshow('feature_unavailability_reasons')\nshow('case_catalog',rows=20)\nshow('case_sensor_changes',rows=12)\nshow('event_exclusion_reasons',rows=12)")
    for prefix in ['feature_availability','event_preservation','fault_events']:gallery('PHM2018-Project-V3/outputs/figures/'+prefix+'/')
    section(7,'학습 후보·누수 방지·시간/장비 분할','시간 범위60/20/20%로 분할합니다(행 수 비율 아님). 모든 A/B/C·H에서 t−900≥분할 시작, t+1800<분할 끝을 요구합니다. 다음 동일 유형 고장이 분할 밖이면 음성 근거를 NA로 바꿔 라벨을 재계산했습니다. 입력에는 Tool/시각/TTF/라벨/미래 이력을 넣지 않습니다. 전처리 fit은 train만 사용했습니다. 초기 그룹 후보01–06/07–08/09–10 정의와 실제 단계5 보호 그룹 실험을 구분해야 합니다. 실제 단계5는 앞80% 안에서01–06 학습/07–08 검증만 수행했고09–10 그룹 최종 시험은 하지 않았습니다.')
    code("show('pretraining_readiness',rows=24)\nshow('pretraining_split_boundaries',rows=12)\nshow('pretraining_unseen_conditions')")
    section(8,'학습 전 경보 평가기 검증','60 raw 간격·동일 세그먼트의 연속 발화를 한 에피소드로 묶습니다. 경보 시작이 고장 전[−600,−60] 안에 있어야 검출하며, 더 일찍 시작해 유지된 경보를 새 검출로 소급하지 않습니다. NA 시작은 미확인 부담으로 남깁니다. 관측 지지길이는 센서 명목4 raw 지지와 점수 유지 구간의 교집합이며 실제 가동시간은 아닙니다. 합성/시험 점수는 평가기 검사에만 사용했고 실제 모델 성능에 섞지 않았습니다.')
    code("show('alarm_stage1_overview',rows=20)\nshow('alarm_stage1_label_states')")
    gallery('PHM2018-Project-V3/outputs/figures/alarm_stage1/')
    section(9,'첫 모델 학습: A/B × 로지스틱/CUDA XGBoost','최대가용 코호트의4개 모델을 학습했습니다. 숫자 표준화/범주 인코딩은 train만 fit하고, XGBoost는 CUDA hist로 조기 종료했습니다. A/B 입력 가용성이 달라 이 단계의 차이를 순수 특징 효과로 해석하지 않습니다. AP/AUC는 알려진 라벨만의 보조 지표이며 NA/EXCLUDE까지 포함한 전체 가용 구간에서 경보를 평가했습니다.')
    code("show('stage2_training_summary')\nshow('stage2_operating_points',columns=['run_id','operating_point','threshold','detected_events','available_events','recorded_events','alarm_episodes_per_10000_raw'],rows=20)\nshow('stage2_uncertainty',rows=12)\nshow('stage2_tool_event_results',rows=20)")
    gallery('PHM2018-Project-V3/outputs/figures/stage2/')
    section(10,'공통 코호트 A/B/C 재학습','C 이력까지 확보된 동일 학습309,273행과 validation309,203행에서6개 모델을 새로 학습했습니다. 입력 가용 고장은53/기록232개입니다. 상한5에서 B XGBoost13개, A9개였고 상한10에서는 A17개/B13개였습니다. C는 매우 낮은 부담의 참조 후보였습니다. 장비쌍10개 군집 부트스트랩은 고정 임계값에 조건부이며 선택 낙관이나 다중 비교를 보정하지 않습니다.')
    code("show('stage3_cohort_counts')\nshow('stage3_training_summary')\nshow('stage3_budget_points',columns=['run_id','budget_per_10000_raw','threshold','detected_events','available_events','recorded_events','alarm_episodes_per_10000_raw'],rows=30)\nshow('stage3_paired_deltas',rows=30)\nshow('stage3_joint_frontier',rows=12)")
    gallery('PHM2018-Project-V3/outputs/figures/stage3/')
    section(11,'깊이/양성 가중치와 seed 민감도','seed42에서 A/B 깊이2/3/4×가중1/√(train NEG/POS)의12개 설정을 탐색하고 후보 설정을 고정한 뒤43/44를 확인했습니다. 새 CUDA 학습18개와 기존 기준3개를 재사용한21개 실행 결과입니다. 상한10에서 기존 A 검출17/19/19개, 조정 A21/17/15개로 개선이 seed43/44에 유지되지 않았습니다. seed별 임계값 재선택과 seed42 숫자 임계값 이전을 구분했습니다. B seed43은 일부 고정 임계값에서 경보와 검출이0이었습니다. 좋은 seed로 사후 교체하지 않았습니다.')
    code("show('stage4_training_summary',rows=24)\nshow('stage4_seed_ranges',rows=40)\nshow('stage4_retuned_points',columns=['config_id','seed','budget_per_10000_raw','detected_events','alarm_episodes_per_10000_raw'],rows=24)\nfixed=T('stage4_fixed_points');show_frame(fixed[fixed.exceeds_budget][['config_id','seed','budget_per_10000_raw','detected_events','alarm_episodes_per_10000_raw','exceeds_budget']])")
    gallery('PHM2018-Project-V3/outputs/figures/stage4/')
    section(12,'최종 시간 구간을 보호한 장비 그룹 검증','앞80%에서01–06의12대만 학습,07–08의4대를 validation으로 사용했습니다. 최대가용 A/B와 공통 A/B/조정A/C의6개 모델을 CUDA 학습했습니다. 최종 시간 뒤20%는 이 실험에서 제외했습니다. 그룹 validation은 조기 종료·임계값 선택에 사용하고 앞선 연구에서 해당 장비를 이미 봤으므로 완전 미사용 외부 장비 시험은 아닙니다. B 최대가용은5/56개(기록121개)를 검출해 일반화가 약했습니다.')
    code("show('stage5_training_summary')\ng=T('stage5_budget_points');show_frame(g[g.budget_per_10000_raw.isin([5,10])][['run_id','budget_per_10000_raw','detected_events','available_events','recorded_events','alarm_episodes_per_10000_raw']])")
    figures([ROOT/'outputs/figures/final_evaluation/protected_split.png',ROOT/'outputs/figures/final_evaluation/group_results.png'])
    section(13,'모델/임계값 동결과 최종 시간 평가','사전 규칙에 따라 상한5/10에서 시간·그룹 validation 중 낮은 기록 검출률을 평균해 A/B 중 B를 선정했습니다. 최종 모델은20대의 시간 train을 합쳐 학습한 기존 stage2 B 공통 모델(seed42,깊이3,가중1,선택2트리)입니다. train+validation 재학습 없이 모델/전처리와 두 숫자 임계값을 동결한 뒤 뒤20%를 한 번 추론했습니다. 기록194개 중 가용91개, 검출7개/33개입니다. 전체 원본 품질 진단은 이전에 확인했으므로 완전 맹검 외부 시험은 아닙니다. 단계7에서 이미 확인한 후반을 연구에 재사용한 사실도 별도로 표시합니다.')
    code("show('stage6_selection_ranking')\nshow('stage6_selection_evidence')\nF=J('PHM2018-Project-V3/outputs/stage6/frozen_manifest.json');A=J('PHM2018-Project-V3/outputs/stage6/holdout_access.json')\nfrom datetime import datetime\nassert datetime.fromisoformat(F['frozen_at_utc'])<datetime.fromisoformat(A['opened_at_utc'])\nassert A['full_prediction_passes']==1 and not A['threshold_search'] and not A['refit']\nshow_frame(pd.DataFrame(F['operating_points']))\nh=T('stage6_holdout_summary');show_frame(h[['operating_point','threshold','detected_events','recorded_events','available_events','recall_recorded','recall_available','alarm_episodes','alarm_episodes_per_10000_raw','median_lead_raw']])\nassert h.detected_events.tolist()==[7.0,33.0] and h.recorded_events.eq(194).all()\nfor p in F['operating_points']:assert h[h.operating_point.eq(p['operating_point'])].threshold.iloc[0]==p['threshold']\nshow('stage6_holdout_uncertainty')\nshow('stage6_holdout_by_tool',rows=40)\ndisplay(J('PHM2018-Project-V3/outputs/stage6/holdout_classification.json'))")
    markdown('後半予測736,646행 저장 후 bool 자료형 집계 오류가 발생했으며, 저장 점수의 bool 자료형만 변환해 집계를 복구했습니다. 추가 전체 추론0회, 재학습0회, 임계값 탐색0회입니다. 예측 파일과 동결 파일 해시를 보존했습니다. 아래 접근/복구 기록은 내장된 원문입니다.'.replace('後半予測','후반 예측'))
    code("display(J('PHM2018-Project-V3/outputs/stage6/holdout_access.json'))\ndisplay(J('PHM2018-Project-V3/outputs/stage6/postprocessing_recovery.json'))")
    gallery('PHM2018-Project-V3/outputs/figures/final_evaluation/')
    section(14,'장비 전용 모델 파일럿: 03M02','최대3대/약20분 범위에서 최소 데이터 기준을 충족한03M02 한 대만 선정했습니다. 학습/검증 고장군집8/3개, 양성100/20행, 음성1,000/1,000행이 기준이었고 결과를 보고 완화하지 않았습니다.03M02 학습123,796행(양성500), validation6,115개 알려진 정답으로 CUDA 전용 모델을 학습했습니다. 기존 공통 모델/장비별 임계값 조정/장비 전용을 같은 입력으로 비교했습니다. **이미 평가한 뒤20%의 후속 연구**이므로 독립 Holdout 성능으로 부르지 않습니다.')
    code("show('stage7_eligibility',rows=20)\nshow('stage7_training_summary')\nshow('stage7_comparison_display')\nshow('stage7_paired_events')\nshow('stage7_later_classification')\ns=T('stage7_later_summary');assert s[s.method.eq('specialist')].detected_events.eq(0).all()\ndisplay(J('PHM2018-Project-V3/outputs/stage7/evaluation_access.json'))")
    markdown('전용 모델은0/27개(입력 가용3개 중0개)를 검출했고 경보228/289개를 냈습니다. 공통·장비별 임계값도0개/141경보였습니다. 기존 공통 높은 설정은2개를 검출했지만5,411경보·20.253/10,000 raw로 장비별 비교 상한10을 넘었습니다. 따라서 동일 부담에서의 우월성 비교가 아닙니다. 후반 알려진 정답21행은 모두 양성이므로 AP/ROC-AUC는 미산출입니다. 전용 모델을 공통 모델의 대체로 채택할 근거는 확인되지 않았습니다.')
    gallery('PHM2018-Project-V3/outputs/figures/stage7/')
    section(15,'로컬 CUDA 자원·검사·재현성','RTX3060 Laptop,CPU Ryzen7 6800H,물리8/논리16코어,RAM 약32GB 환경을 사용했습니다. V3 XGBoost는 실제 cuda:0 학습을 검증했고 모델을 순차 처리했습니다. CPU12스레드/경보 평가3프로세스와 메모리 감시를 활용했습니다. GPU 장치 전체 메모리는 다른 프로세스를 포함할 수 있고2초 표본은 순간 피크를 보장하지 않습니다. 원본 보호는 주로 파일 크기/mtime 검사이며 전체 원본 내용 해시와는 다릅니다. 종합본 생성 시에는 기존 동결 모델·점수·결과의 내용 해시도 대조합니다.')
    code("audits=[]\nfor path,record in DATA['json'].items():\n    if not isinstance(record,dict):continue\n    cs=record.get('checks')\n    if isinstance(cs,list) and cs and all(isinstance(c,dict) and 'passed' in c for c in cs):\n        audits.append({'검사 기록':path,'검사 수':len(cs),'통과':sum(bool(c['passed']) for c in cs)})\nshow_frame(pd.DataFrame(audits))\nshow_frame(pd.DataFrame(DATA['notebooks']))\nfor stage in [2,3,5,7]:\n    resource=T(f'stage{stage}_resource_telemetry')\n    show_frame(pd.DataFrame([{'단계':stage,'RAM 표본 최대 GiB':resource.rss_bytes.max()/2**30,'장치 전체 GPU 표본 최대 MiB':resource.gpu_used_mib.max(),'자원 표본 수':len(resource)}]))")
    section(16,'종합 판단과 남은 과제','현재까지의 결과로 현장 자동 적용을 권할 근거는 부족합니다. 분류 지표가 높거나 확인 음성에서 경보가 적더라도 미확인 구간 부담과 입력 없는 고장을 포함한 시스템 성능은 별도입니다. 우선순위는①고장 전 입력 공백 원인②TTF Null/후속 고장 이력의 완전성③정비 종료·상태·시간 단위④공정 변화에 따른 장비 일반화입니다. 이 근거를 확인한 뒤 새로운 외부/미래 자료로 개선 모델을 평가해야 합니다. 이미 확인한 평가 구간을 반복 튜닝 후 독립 Holdout이라고 부르지 않습니다. 이 종합본은 새로운 연구 결과를 추가한 것이 아니라 기존 수행 자료를 통합한 문서입니다.')
    section(17,'내장 표 전체 목록과 조회 방법','아래 목록의 집계표는 모든 행을 내장했습니다. 본문에서 길게 표시하지 않은 표도 `T("V3/stage6_holdout_events")`처럼 조회할 수 있습니다. 예측 점수는 `T("V3/최종_저장예측")`에 있습니다. CSV 원본이나 모델 바이너리는 내장 대상이 아닙니다. 각 표의 Parquet SHA256을 재계산해 무결성을 검사합니다.')
    code("catalog=pd.DataFrame([{'표':k,'행':v['rows'],'열':len(v['columns']),'출처':v['source']} for k,v in DATA['tables'].items()])\nshow_frame(catalog)\nfor key,entry in DATA['tables'].items():\n    raw=base64.b64decode(entry['data']);assert hashlib.sha256(raw).hexdigest()==entry['sha256'],key\nprint('내장 표 전체 무결성 검사 통과:',len(catalog),'개')\nprint('재학습0회 · 모델 재추론0회 · 원본 재처리0회')")
    # 아직 본문에 배치하지 않은, 원문에서만 연결된 그림도 빠뜨리지 않는다.
    remaining=[p for p in figure_paths if p not in shown]
    if remaining:
        markdown('### 추가 보관 그림\n\n원문 보고서에 연결된 추가 그림입니다. 버전과 작성 당시 평가 범위는 경로를 확인하세요.');figures(remaining)
    section(18,'원문 보고서 전체 보관','아래에는 V1/V2/V3의 기존 보고서 원문과 재분석 계획 상태를 모두 내장했습니다. 펼침 항목은 작성 당시 문서입니다. 오래된 미학습/미사용/초 단위 해석·당시 다음 단계 문구는 현재 상태를 덮어쓰지 않습니다. 원문 수치나 과거 판단을 현재 결과에 맞춰 수정하지 않았습니다. 그림은 본문 내장 그림으로 연결하고, 외부/모델/데이터 링크는 원래 보관 위치를 가리킵니다.')
    for path in report_paths:
        text=reports[origin(path)]
        def replace_image(match):
            p=(path.parent/match.group(2)).resolve()
            if p in figure_ids:return f'[그림: {match.group(1)}](#{figure_ids[p]})'
            return f'그림 보관 경로: `{origin(p) if p.is_relative_to(ROOT.parent) else str(p)}`'
        text=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',replace_image,text)
        # 본문 파일을 복제하지 않으면서 과거 상대 링크의 출처를 보존한다.
        def replace_link(match):
            label,target=match.groups()
            if target.startswith(('http:','https:','#','attachment:')):return match.group(0)
            dest=(path.parent/target).resolve()
            return f'[{label}]({dest.as_posix()})'
        text=re.sub(r'(?<!!)\[([^\]]*)\]\(([^)]+)\)',replace_link,text)
        markdown(f'<details>\n<summary>{origin(path)} — 작성 당시 기록</summary>\n\n> 원문 내용 해시: `{sha(path.read_bytes())}`. 현재 상태는 본문 단계별 설명을 기준으로 읽으세요.\n\n{text}\n\n</details>')
    book.cells=cells
    nb.validate(book);nb.write(book,DEST)
    # 프로젝트 데이터 경로가 없는 빈 작업 폴더에서 실행하여 단일 파일 자립성을 검증.
    for key,rel in [('JUPYTER_PATH','outputs/jupyter'),('JUPYTER_RUNTIME_DIR','outputs/jupyter/runtime'),('IPYTHONDIR','outputs/ipython'),('MPLCONFIGDIR','outputs/.mplconfig')]:os.environ[key]=str(ROOT/rel)
    empty=BUILD/'empty_execution_directory';empty.mkdir(exist_ok=True);assert not list(empty.iterdir())
    started=time.monotonic();NotebookClient(book,timeout=300,kernel_name='phm2018-v3',resources={'metadata':{'path':str(empty)}}).execute();nb.write(book,DEST);nb.validate(book)
    codes=[c for c in book.cells if c.cell_type=='code'];errors=[o for c in codes for o in c.outputs if o.output_type=='error']
    assert not errors and all(c.execution_count is not None for c in codes)
    assert len(shown)==len(figure_paths) and not list(empty.iterdir())
    for path,h in protected.items():assert sha(path.read_bytes())==h,str(path)
    verify_previous()
    dump_json(BUILD/'table_catalog.json',catalog)
    dump_json(BUILD/'source_manifest.json',dict(sources=payload['sources'],table_sources={k:v['sha256'] for k,v in payload['tables'].items()},reports=[origin(p) for p in report_paths],figures=[dict(file=origin(p),sha256=sha(figure_bytes[p]),anchor=figure_ids[p]) for p in figure_paths]))
    result=dict(notebook=str(DEST.relative_to(ROOT)).replace('\\','/'),created_date='2026-11-12',code_cells=len(codes),executed=len(codes),errors=len(errors),embedded_tables=len(catalog),embedded_figures=len(figure_paths),archived_reports=len(report_paths),previous_notebooks=len(old_notebooks),execution_seconds=time.monotonic()-started,notebook_bytes=DEST.stat().st_size,notebook_sha256=sha(DEST.read_bytes()),standalone_execution_in_empty_directory=True,new_training=0,new_model_inference=0,protected_previous_artifacts_unchanged=True)
    dump_json(BUILD/'execution_validation.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
