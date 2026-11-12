"""Create and execute only the three new notebooks; no training/inference on Run All."""
from common import *
import textwrap,os
import nbformat as nbf
from nbclient import NotebookClient

SETUP='''
from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display,Markdown,Image
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'experiments/sensor_availability/config.json').exists())
OUT=ROOT/'reports/sensor_availability'
def js(name): return json.loads((OUT/name).read_text(encoding='utf-8'))
def table(name): return pd.read_csv(OUT/'tables'/name)
def show(frame,title):
    display(frame.style.hide(axis='index').set_caption(title).format(precision=5,na_rep='해당 없음')
            .set_properties(**{'padding':'6px 9px','text-align':'left'})
            .set_table_styles([{'selector':'caption','props':[('text-align','left'),('font-weight','bold')]},
                              {'selector':'th','props':[('background-color','#eef3f6'),('text-align','left')]}]))
def figure(name): display(Image(filename=str(OUT/'figures'/name)))
baseline=js('baseline.json')
phm=js('sources/phm_definition_audit.json')
full=table('full_summary.csv')
summary=table('summary_mean_sd.csv')
'''

def write(name,parts):
    cells=[]
    for kind,text in parts:
        text=textwrap.dedent(text).strip()
        if kind=='code':compile(text,name,'exec');cells.append(nbf.v4.new_code_cell(text))
        else:cells.append(nbf.v4.new_markdown_cell(text))
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'TEP sensor availability','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    nbf.validate(nb);nbf.write(nb,HERE/name)

def create():
    write('11_availability_baseline.ipynb',[
    ('markdown','''# 11. 입력 가용성 실험 — Baseline과 판정 규칙 검증

    정리일2026-11-12. 기존 TEP/PHM 파일은 읽기 전용으로 재사용했다. **이 노트북은 이미 수행한 실측 결과를 읽어 검증·표시하며 재학습이나 재추론을 실행하지 않는다.**

    실험의 순서는①기존 CNN Baseline 전체 재현②PHM V3 입력 불가 규칙 대조③실제 NaN 입력/윈도우 마스크 동등성④파일럿⑤전체 조건이다. 모델·분할·라벨·입력20×52·threshold·3연속 경보 규칙을 고정했다.'''),
    ('code',SETUP),
    ('markdown','''## 1. 전체 Test에서 기존 Baseline을 새로 재현한 결과

    중단 전 기존 최종 CNN을 로드하여 원본 Test의 유효9,880,500시점을 새로 추론했다. GPU 추론 실측67.17초이며 원본 보호 체크섬·후처리 시간은 제외한다. 저장된21개 확률과 차이0, 아래 지표와10,000개 이벤트별 검출·최초시각·지연이 일치했다. 하나라도 불일치하면 실험을 중단하도록 gate를 구현했다.'''),
    ('code',"""
    show(pd.DataFrame([
        ['모델',baseline['model']],['임계값',baseline['threshold']],
        ['ROC-AUC',baseline['detection_metrics']['detection_roc_auc']],['Detection F1',baseline['detection_metrics']['detection_f1']],
        ['전체 고장 실행',baseline['total_fault_runs']],['검출 실행',baseline['detected_fault_runs']],
        ['Run-level 검출률',baseline['run_detection_rate']],['검출 지연 중앙값(분)',baseline['median_delay_minutes']],
        ['정상 alarm episodes',baseline['normal_alarm_episodes']],['정상 관측지원시간(h)',baseline['normal_exposure_hours']],
        ['정상 episodes/h',baseline['normal_alarm_episodes_per_hour']],['새 추론 실측시간(s)',baseline['fresh_full_test_inference_seconds']]
    ],columns=['지표','실측 결과']),'원본 Test Baseline 재현 / 출처: baseline.json')
    checks=table('baseline_reproduction.csv')
    assert checks.passed.all()
    show(checks,'기존 값과 새 재현 값 / 허용오차1e-10, 이벤트별 정확 일치')
    """),
    ('markdown','''## 2. PHM V3의 평가 불가능 조건 직접 대조

    PHM `evaluate()`는 이벤트별닫힌 사전 예측창`[tf-600,tf-60] raw`의 유효score anchor가1개 이상이면available=True,0개면NO_INPUT이다. B특징은현재센서+과거300raw 통계로서 lag≤12raw·같은세그먼트·coverage≥90% 등을 요구한다.

    TEP는고장 이후sample161~960의800시점에서 유효20×52입력이1개 이상이면평가 가능이다. 같은 것은**평가 구간에 예측 기회가0개인지**라는 의미이며, 전후 방향·시간 단위·입력 규칙은 다르다. PHM raw를 초/분으로 환산하지 않았다.'''),
    ('code',"""
    show(pd.DataFrame([
        ['전체 기록',phm['total_events']],['가용',phm['available_events']],['NO_INPUT',phm['unavailable_events']],
        ['검출',phm['detected_events']],['전체 검출률',phm['overall_detection_rate']],['가용 검출률',phm['conditional_detection_rate']]
    ],columns=['PHM 고정 운영점10','원표 검증 결과']),'PHM V3 stage6 실제 이벤트·요약표에서 검증')
    ev=table('phm_verified_events.csv')
    assert len(ev)==194 and ev.available.sum()==91 and ev.detected.sum()==33
    assert np.array_equal(ev.available,ev.eligible_anchors>0)
    show(ev.groupby('status').size().rename('이벤트 수').reset_index(),'PHM 상태별 실제 이벤트 수')
    """),
    ('markdown','''## 3. 짧은 공백과 평가 불가능 실행을 구분

    주 실험은선택 실행의sample141~960을가려 고장 후 입력창이0개임을 보장한다. 이는TEP 고장 후40h를덮는 의도적인 검열 가정이며 PHM 실제 공백 길이 분포를 복원한 것이 아니다.

    짧은 공백(sample141~180)에서는sample200부터20개 연속 입력이복구돼761/800개 post-fault시점이남는다. 따라서이실행은95.125% **부분 가용**이며평가가능분모에포함한다. 유효창이딱1개·2개인경우도평가가능으로검증했다.'''),
    ('code',"""
    figure('availability_masks.png')
    mask=js('sources/mask_validation.json')
    assert mask['passed']
    show(pd.DataFrame(mask['checks']),'가용성·경보 연속성·미래참조·선정 재현성 검사')
    show(table('mask_equivalence_checks.csv'),'실제 NaN 입력의 유효창과 마스크 필터링 비교')
    """),
    ('markdown','''남은입력창은원본과정확히같고,같은배치의예측도정확히같았다. 다른GPU배치포장에서는작은반올림차이가있을수있으며, 주실험은동일한기존예측을유효마스크로선택했다. 불가시점에정상/고장확률을만들지않았다. 경보계산중결측은연속성을끊는표식이며정상예측이아니다.

    ## 4. 진행 gate와 기존 파일 보존'''),
    ('code',"""
    pilot=js('sources/pilot_gate.json');preserve=js('sources/preservation_check.json')
    assert pilot['passed'] and preserve['all_unchanged']
    show(pd.DataFrame([
        ['Baseline gate',js('sources/baseline_gate.json')['passed']],['마스크 검증',mask['passed']],
        ['0/25/53.1% × seed42/43/44 파일럿',pilot['passed']],
        ['sklearn 파일럿 ROC-AUC 차이',pilot['sklearn_auc_difference']],['sklearn 파일럿 F1 차이',pilot['sklearn_f1_difference']],
        ['보존한 기존 TEP 파일',preserve['protected_TEP_files']],['보존한 PHM 참조파일',preserve['protected_PHM_source_files']],
        ['기존 보호파일 변경 없음',preserve['all_unchanged']]
    ],columns=['검사','결과']),'파일럿 통과 후에만 전체 조건으로 확장')
    """)])

    write('12_availability_degradation.ipynb',[
    ('markdown','''# 12. 입력 가용성 저하 — 전체 검출 범위와 조건부 성능

    고장20종×500실행=10,000개를유지하고 고장별균형·seed42/43/44로센서입력만가렸다. 선택비율은**실행비율**이며전체샘플삭제율이아니다. 모델재학습·임계값재선택·라벨변경은없다. 이노트북은저장된실측표·그림만읽는다.'''),
    ('code',SETUP),
    ('markdown','''## 1. 필수 결과: 전체/평가가능 분모를 함께 표시

    전체검출률=검출/전체10,000, 조건부검출률=검출/가용실행이다. 아래는3seed실측평균이며 분모정수는같고 검출수평균은소수가될수있다. 조건별부분가용은주실험에서0개이고,짧은공백대조는별도로집계한다.'''),
    ('code',"""
    show(table('compact_results.csv'),'0/10/25/40/53.1% 결과 / 검출률은%, AUC·F1은0~1, 실측3seed평균')
    figure('coverage_and_detection.png')
    detail=full[['rate','seed','evaluable_fault_runs','unavailable_fault_runs','detected_fault_runs','conditional_detection_rate','overall_detection_rate']]
    show(detail,'각seed의 실제 정수 검출수 및 두 분모')
    assert np.allclose(full.overall_detection_rate,(full.evaluable_fault_runs/10000)*full.conditional_detection_rate)
    """),
    ('markdown','''## 2. A: 입력 없음 / B: 입력은 있으나 미검출

    A+B+검출=10,000이다. A에는원래모델도검출하지못했던실행이포함되므로 A수와신규검출손실을구분한다. 53.1%에서5,310개의기회를없앴고기존검출실행의손실은약5,234개였다. 유효시점만의ROC-AUC/F1이유사해도전체검출범위는절반수준으로줄었다.'''),
    ('code',"""
    show(full[full.rate==.531][['seed','cause_A_no_input','cause_B_input_but_no_detection','detected_fault_runs','baseline_detections_lost','lost_previously_detected_due_unavailability']],
         '53.1% 조건의 배타적 결과와 실제 Baseline 검출 손실')
    show(summary[['rate','detection_roc_auc','detection_f1','predicted_samples']] if 'predicted_samples' in summary else summary[['rate','detection_roc_auc','detection_f1']],
         '실제 유효 시점에서만 계산한 ROC-AUC/F1 / 평가집합 변경에 주의')
    figure('fault_detection_heatmap.png')
    """),
    ('markdown','''## 3. Detection Delay: 사례구성 변화와 실제 지연 증가 구분

    주실험의입력이남은동일실행은원예측·시간축·임계값이같아짝지은지연변화가0이다. 전체요약중앙값도48분이지만평가불가실행은지연통계에서빠졌으므로입력손실의영향이없다는뜻이아니다.'''),
    ('code',"""
    show(full[['rate','seed','detected_fault_runs','median_delay_minutes','mean_delay_minutes','p90_delay_minutes','paired_median_delay_change_minutes']],
         '주실험: 검출된실행의지연, 단위분; 미검출/입력없음은제외')
    figure('delay_comparison.png')
    """),
    ('markdown','''## 4. 짧은 공백 대조군: 부분 가용은 평가 가능에 포함

    25%·seed42의동일2,500개실행에sample141~180만가렸다. 이후761/800입력창이남아평가불가는0,부분가용은2,500이다. 전체검출9,849개,전체지연중앙값72분이다. 공백선택·전후공동검출2,471개실행만짝지어보면지연차이중앙값은+81분이다. 두집단median의차이78분과paired차이median81분은다른통계다.'''),
    ('code',"""
    control=js('tables/summary_partial_control_r0.250_s42.json')
    show(pd.DataFrame([{k:control[k] for k in ['total_fault_runs','evaluable_fault_runs','unavailable_fault_runs','partial_fault_runs','mean_post_input_availability_partial','detected_fault_runs','overall_detection_rate','median_delay_minutes','detection_f1']}]),
         '부분가용 대조군 / 입력가용비율95.125%, 전체평가범위유지')
    show(table('partial_paired_delays.csv'),'동일실행의전후짝지은탐지지연, 단위분')
    show(full[['rate','seed','normal_runs','normal_alarm_episodes','normal_exposure_hours','normal_alarm_episodes_per_hour']],
         '정상실행은가리지않아모든조건에서705 episodes/23,525h로동일')
    """),
    ('markdown','''**해석 한계:** 이실험은가용시점의기존예측을선택한평가범위실험이다. 분류함수자체의성능저하를입증한것이아니다. 부분대조의F1이상승해도초기어려운시점이빠진영향일수있어모델개선이라고해석하지않는다. 입력1~2창만남은경우가용미검출에는3연속경보부족도포함될수있다.''')])

    write('13_phm_comparison.ipynb',[
    ('markdown','''# 13. PHM 2018 V3와 TEP의 평가 가능성 비교

    **고장전조기예측(PHM)과고장후이상탐지(TEP)의모델성능순위를비교하지않는다.** 비교하는것은입력기회가없어전체이벤트를포괄하지못하는구조다. PHM값은V3코드·원표에서확인한정수이고,TEP값은가용성조작후3개seed의실측평균이다.'''),
    ('code',SETUP),
    ('markdown','''## 1. 같은 형식으로 분모를 표시

    PHM운영점10:전체194개,가용91개,불가103개,검출33개. 전체검출률17.01%,조건부검출률36.26%다. PHM103/194=53.0928%를TEP목표53.1%(5,310/10,000)로근사했으며서로정확히같은비율은아니다.'''),
    ('code',"""
    comparison=table('phm_tep_comparison.csv')
    show(comparison[['dataset','condition','total','available','unavailable','partial','detected','conditional_rate','overall_rate','task']],
         'PHM V3 원표 vs TEP 가용성실험 / 조건부·전체분모 병기')
    figure('phm_tep_availability.png')
    """),
    ('markdown','''## 2. 가용성 계수와 모델 검출률의 곱

    `전체검출률 = 가용실행비율 × 조건부검출률`은두분모의정의에따른산술관계다. TEP원본의조건부성능을유지한다는가정으로Baseline×(1-r)를계산하면실측과가깝다. 이계산은가정에기반한예상값이며,실제seed별균형선택결과를대체하지않는다.'''),
    ('code',"""
    identity=comparison[['dataset','condition','available','total','conditional_rate','overall_rate']].copy()
    identity['coverage_factor']=identity.available/identity.total
    identity['factor_times_conditional']=identity.coverage_factor*identity.conditional_rate
    identity['difference']=identity.overall_rate-identity.factor_times_conditional
    assert np.max(np.abs(identity.difference))<1e-12
    show(identity,'실측분모관계: 평가가능비율×조건부검출률')
    show(summary[['rate','overall_detection_rate','analytical_MCAR_detection_rate']],
         '실제측정평균 vs Baseline×(1-r) 가정계산')
    """),
    ('markdown','''## 3. 맞춰 재현한 것과 재현하지 않은 것

    | 항목 | 범위 |
    |---|---|
    | 재현한 의미 | 해당이벤트평가구간에유효입력기회가0개인경우NO_INPUT |
    | PHM 평가구간 | `[tf-600,tf-60] raw`의사전예측anchor; B특징·coverage규칙 |
    | TEP 평가구간 | 고장주입후sample161~960(40h),20×52유효창 |
    | 가정 | TEP고장별균형랜덤실행선정,선택실행은종료까지공백 |
    | 재현하지않음 | PHM실제공백길이분포,장비·공정·고장징후와누락의연관성 |
    | 모델변경 | 없음: 재학습·threshold조정·구조변경없음 |
    | 오경보비교 | PHM전체경보/raw와TEP정상경보/h는정의·단위가달라직접비교불가 |

    입력손실의학습영향은미평가다. 가용미검출을전부분류능력저하로단정하지않는다. 세seed의SD는누락선택변동이며모델일반화의신뢰구간이아니다.'''),
    ('code',"""
    phm_sources=pd.DataFrame([{'참조파일':k,'SHA256':v['sha256']} for k,v in phm['files'].items()])
    show(phm_sources,'PHM V3 규칙·결과를읽기전용으로검증한출처')
    preservation=js('sources/preservation_check.json')
    assert preservation['all_unchanged']
    show(pd.DataFrame([{'기존TEP보호파일':preservation['protected_TEP_files'],
                       'PHM참조파일':preservation['protected_PHM_source_files'],
                       '변경없음':preservation['all_unchanged']}]),'기존원본·모델·분할·00~10번노트북보존')
    """),
    ('markdown','''## 4. 재현과 보고서

    [최종 보고서](../../reports/sensor_availability/final_report.md)를참조한다. 실험코드·새노트북은이폴더에,새수치·그림은`reports/sensor_availability/`에만기록했다.

    프로젝트루트에서`experiments/sensor_availability/run.ps1`을실행하면Baseline→PHM감사→마스크검증→파일럿→전체조건→보고서→새노트북순으로진행한다. `-ResultsOnly`는저장된결과의검증·보고서·새노트북만작성한다. 기존분석파일은덮어쓰지않는다.

    출처: PHM2018-Project-V3의stage6실제원표·src평가코드,TEP고정모델의Test예측. [PHM공식과제설명](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/)에서사전예측과제임을확인했다.''')])

def execute():
    import asyncio
    if sys.platform=='win32':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    kernel=HERE/'.jupyter/kernels/sensor-availability';kernel.mkdir(parents=True,exist_ok=True)
    write_json(kernel/'kernel.json',{'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'TEP sensor availability','language':'python',
        'env':{'PYTHONPATH':str(ROOT/'.deps'),'PYTHONDONTWRITEBYTECODE':'1','MPLCONFIGDIR':str(CACHE/'.matplotlib'),'IPYTHONDIR':str(CACHE/'.ipython')}})
    os.environ['JUPYTER_PATH']=str(HERE/'.jupyter')
    os.environ['JUPYTER_RUNTIME_DIR']=str(CACHE/'.jupyter_runtime')
    os.environ['JUPYTER_CONFIG_DIR']=str(CACHE/'.jupyter_config')
    Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(exist_ok=True)
    results=[]
    for name in ['11_availability_baseline.ipynb','12_availability_degradation.ipynb','13_phm_comparison.ipynb']:
        path=HERE/name;nb=nbf.read(path,as_version=4)
        NotebookClient(nb,timeout=180,kernel_name='sensor-availability',resources={'metadata':{'path':str(ROOT)}}).execute()
        nbf.validate(nb);nbf.write(nb,path)
        codes=[c for c in nb.cells if c.cell_type=='code'];outputs=[o for c in codes for o in c.get('outputs',[])]
        errors=sum(o.output_type=='error' for o in outputs)
        assert errors==0 and all(c.execution_count is not None for c in codes)
        results.append(dict(notebook=name,executed_cells=len(codes),error_outputs=errors,embedded_images=sum('image/png' in o.get('data',{}) for o in outputs)))
        print('Executed',name,flush=True)
    write_json(OUT/'sources/notebook_validation.json',results)

if __name__=='__main__':create();execute()
