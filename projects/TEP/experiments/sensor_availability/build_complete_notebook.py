"""Compose the entire TEP study into a new notebook without editing source notebooks."""
from pathlib import Path
import sys,os,json,hashlib,copy,re,textwrap,asyncio
sys.dont_write_bytecode=True
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'reports/sensor_availability'
CACHE=HERE/'cache/complete_notebook'
CACHE.mkdir(parents=True,exist_ok=True)
TARGET=ROOT/'notebooks/14_TEP_종합_연구_노트북.ipynb'
SOURCES=[ROOT/'notebooks/00_TEP_전체_연구_과정과_결과.ipynb']+[HERE/f'{n}_{name}.ipynb' for n,name in [
    (11,'availability_baseline'),(12,'availability_degradation'),(13,'phm_comparison')]]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def markdown(text):return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())
def code(text):return nbf.v4.new_code_cell(textwrap.dedent(text).strip())

def build():
    protected=[p for p in (ROOT/'notebooks').glob('*.ipynb') if p!=TARGET]+list(HERE.glob('1[123]_*.ipynb'))
    protected.extend(p for p in (ROOT/'models').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    original=nbf.read(SOURCES[0],as_version=4)
    cells=copy.deepcopy(original.cells)
    cells[0]=markdown('''
    # TEP 종합 연구 노트북
    ## 데이터 감사·모델 개발·입력 가용성 후속 실험의 전체 과정과 결과

    **정리 기준일: 2026-11-12 · 모든 설명과 결과는 실제 프로젝트 산출물에 근거한다.**

    이 파일은 현재까지 수행한 TEP 연구를 한 흐름으로 정리한다. 기존 Phase 1·2 종합 노트북과
    센서 입력 가용성 후속 실험 11~13번 노트북을 재사용하고, 전체 연구의 연결 관계와 결론을 보완했다.
    기존 노트북·모델은 수정하지 않았다.

    | 부분 | 절 | 포함 내용 |
    |---|---|---|
    | Ⅰ. 데이터와 공정 이해 | 1~5 | 핵심 결과, 수행 흐름, 공정·변수·고장, 원본 확보, 품질·호환성 |
    | Ⅱ. 고정 모델 개발·평가 | 6~20 | 실행 단위 분할, 특징·학습·선택, 모델 비교, 전체 Test, 고장별 지연, SHAP, 미지 고장, 비용·한계 |
    | Ⅲ. 입력 가용성 후속 연구 | 21~23 | Baseline 재현, PHM V3 판정 근거, 결측 마스크 검증, 5개 비율×3개 seed, 부분 가용, PHM 비교 |
    | Ⅳ. 전체 결론과 재현 | 24 | 단계별 답변, 평가 범위·진단의 구분, 보존·실행 검증, 원 산출물 |

    **읽는 방법:** 표와 그림은 실행된 출력으로 내장되어 있어 파일을 열면 바로 읽을 수 있다.
    `Run All`은 기존 CSV·JSON·그림에서 결과를 재구성하며 **모델 재학습·임계값 재선택·전체 Test 재추론을 하지 않는다.**
    재실행에는 이 프로젝트의 산출물과 분석 패키지가 필요하다. 14절의 실행별 확률 그림은 저장된 확률 캐시를 사용한다.

    > 결과를 구분해서 읽는다: **고장 조건부 분류 / 실제 경보를 포함한 end-to-end 진단 / 입력이 없는 실행까지 포함한 전체 검출 범위 / 물리적 원인 식별**은 서로 다른 개념이다.
    ''')
    snapshot=code('''
    # 전체 연구의 서로 다른 평가 과제를 한 표에서 구분한다.
    availability_mean=pd.read_csv(ROOT/'reports/sensor_availability/tables/summary_mean_sd.csv')
    availability_531=availability_mean.loc[np.isclose(availability_mean.rate,.531)].iloc[0]
    unknown_study=pd.read_csv(ROOT/'reports/tables/phase2/test_comparison.csv')
    unknown_study=unknown_study[unknown_study.model_id.str.startswith('Unknown')]
    paired_availability=pd.read_csv(ROOT/'reports/sensor_availability/tables/partial_paired_delays.csv')
    selected_pair=paired_availability[paired_availability.group=='공백 선택 실행'].iloc[0]
    overview=pd.DataFrame([
        ['원본 Test 이상 탐지',f"F1 {primary['detection_f1']:.4f}",'유효9,880,500시점, 고정 CNN 단일 seed42'],
        ['원본 Test 고장 조건부 분류',f"Macro F1 {primary['diagnosis_macro_f1']:.4f}",'실제 고장 구간을 안다고 가정한20종 분류'],
        ['탐지 실패 포함 진단',f"Macro F1 {primary['end_to_end_macro_f1']:.4f}",'정상+20종 고장, 임계값 적용21상태'],
        ['미지 고장 일반화',f"Recall {unknown_study.unseen_faults_detection_recall.mean():.2%}",'별도 이진 모델에서 Fault3·9·15를 양성 학습에서 제외'],
        ['입력53.1% 불가: 전체 검출',f"{availability_531.overall_detection_rate:.2%}",'전체 고장10,000개 실행을 분모로 유지, 3개 누락 seed 평균'],
        ['입력53.1% 불가: 조건부 검출',f"{availability_531.conditional_detection_rate:.2%}",'입력이 남은4,690개 실행만 분모로 사용'],
        ['짧은 공백의 지연 영향',f"+{selected_pair.paired_delay_change_median:.0f}분",'공백 선택·전후 공동검출2,471개 실행의 짝지은 지연 차이 중앙값']
    ],columns=['질문 / 실험','실측 결과','분모와 해석'])
    show_table(overview,'전체 연구 결과 한눈에 보기',
               'phase2/dense_metrics.json; test_comparison.csv; sensor_availability/summary_mean_sd.csv; partial_paired_delays.csv')
    note('핵심 결론',
         '관측 가능한 데이터에서의 분류 성능이 높아도 입력 자체가 없으면 전체 고장을 포괄할 수 없다. '
         '원본 모델의 Fault9·15 진단과 미지 고장 탐지에도 약점이 남는다. '
         'PHM의 사전 예측과 TEP의 사후 탐지는 같은 과제의 모델 순위로 비교하지 않는다.')
    ''')
    # The existing setup is retained, then the new executive overview is inserted.
    cells.insert(2,snapshot)
    cells.insert(3,markdown('''
    ### 연구가 확장된 이유

    1. **Phase 1:** 이름이 같은 TEP 자료라도 변수·주기·고장 정의·독립성이 같은지 확인했다.
    2. **Phase 2:** Harvard 자료만으로 실행 family를 분리하고, Validation에서 고정한 모델을 공식 Test에 평가했다.
    3. **후속 연구:** PHM V3에서 유효 입력이 없어 103/194개 이벤트를 평가할 수 없었던 문제를 조사한 뒤,
       동일한 의미의 ‘평가 구간 내 예측 기회 0개’를 TEP에 모사했다.

    Phase 2의 모델 개발과 후속 실험의 입력 마스크 조작을 구분한다. 후속 실험에서 기존 학습 모델이나 라벨을 다시 맞추지 않았다.
    '''))
    cells.append(markdown('''
    # Ⅲ. 센서 입력 가용성 후속 연구

    앞의 1~20절은 규칙적으로 관측된 TEP 데이터에서 모델을 개발·평가한 기록이다.
    이어지는 21~23절은 그 모델을 그대로 고정하고 **입력 가용성만 바꾼 후속 연구**다.

    PHM2018-Project-V3의 실제 코드·Holdout 원표를 읽기 전용으로 대조했다.
    PHM은 `[tf-600, tf-60] raw`의 고장 이전 예측 기회, TEP는 주입 이후40시간의 유효20×52창을 판정한다.
    같게 모사한 것은 ‘평가 구간에서 유효한 입력 기회가0개’라는 의미이며, 실제 공백의 길이 분포나 예측 과제가 같은 것은 아니다.
    '''))
    for major,source in zip([21,22,23],SOURCES[1:]):
        part=nbf.read(source,as_version=4)
        for index,cell in enumerate(part.cells):
            new=copy.deepcopy(cell)
            if new.cell_type=='code' and 'def js(name)' in new.source and major!=21:
                continue # Share one availability setup instead of repeating imports/globals.
            if new.cell_type=='markdown':
                if index==0:
                    headings={21:'입력 가용성 연구: Baseline 재현과 판정 규칙',22:'입력 가용성 저하: 전체 검출 범위와 조건부 성능',23:'PHM 2018 V3와 TEP의 평가 가능성 비교'}
                    new.source=re.sub(r'^# [^\n]+',f'## {major}. {headings[major]}',new.source,count=1)
                new.source=re.sub(r'^## (\d+)\.',lambda m:f'### {major}.{m.group(1)}.',new.source,flags=re.M)
                # This new notebook lives in the root notebooks/ directory.
                new.source=new.source.replace('../../reports/sensor_availability/final_report.md','../reports/sensor_availability/final_report.md')
                new.source=new.source.replace('실험코드·새노트북은이폴더에','실험코드·개별후속노트북은`experiments/sensor_availability/`에')
            cells.append(new)
    cells.append(markdown('''
    # Ⅳ. 종합 결론
    ## 24. 전체 연구에서 확인한 범위와 남은 문제

    - **데이터 감사:** A의 중복과 헤더 문제, Fault21 정의 충돌, A/B 저장 주기·변수 차이를 확인했다. A와 B를 직접 병합하지 않았다.
    - **모델 개발:** 실행 family 분할과 미래 참조 방지, Train 전용 전처리, Validation 선택·임계값 고정 후 Test 평가를 수행했다.
    - **성능 해석:** 높은 조건부 고장 분류 점수와 실제 탐지·정확한 ID 판정은 다르다. Fault9·15 및 미지 고장 탐지에는 한계가 있다.
    - **입력 가용성:** 모델이 동일해도 입력 부족으로 예측 기회가 사라지면 전체 검출 범위가 줄었다. 유효 시점만 계산한 F1은 이 손실을 숨길 수 있다.
    - **부분 가용:** 짧은 공백 뒤 입력이 복구되면 해당 실행은 평가 가능이다. 공백으로 인한 검출 지연은 동일 실행의 전후 결과를 짝지어 확인했다.
    - **물리적 원인:** SHAP·관측 센서 반응·실제 고장 원인을 구분했으며, 대응이 미확인인 원인 적중률은 계산하지 않았다.
    - **일반화:** 실제 설비·다른 운전모드·학습 중 센서 누락 영향은 검증하지 않았다. 이 결과는 화학공정 시뮬레이션과 지정한 가용성 조작에 한정된다.

    ### 재현과 원본 보존

    이 노트북은 기존 산출물을 읽어 표·그림을 재구성하는 문서다. **학습을 재현하는 명령**과 **저장 결과를 다시 표시하는 작업**을 구분한다.

    | 목적 | 프로젝트 루트에서의 명령 |
    |---|---|
    | 원래 Phase 1 재계산 | `.\\run_phase1.ps1` |
    | Phase 2 전체 재현 | `.\\run_phase2.ps1` |
    | 가용성 실험 전체 재현 | `.\\experiments\\sensor_availability\\run.ps1` |
    | 가용성 저장 결과만 재구성 | `.\\experiments\\sensor_availability\\run.ps1 -ResultsOnly` |
    | 이 종합 노트북만 생성·실행 검증 | `python experiments/sensor_availability/build_complete_notebook.py` |

    프로젝트의 Python 환경과 `.deps` 패키지를 사용한다. 노트북에 내장된 표·그림은 실행 없이 읽을 수 있고,
    전체 셀을 다시 실행하려면 프로젝트의 원 결과 파일이 필요하다. 14절의 확률 캐시는 없을 경우 명시적으로 안내하고 대체 예측을 만들지 않는다.

    원 연구: `reports/markdown/01~13`, `notebooks/00~10`.
    후속 연구: `reports/sensor_availability/final_report.md`, `experiments/sensor_availability/11~13`.
    모델 번들: `models/phase2/TEP_final_model.zip`.
    '''))
    cells.append(code('''
    # 전체 통합 문서의 수치·기록 일관성을 확인한다. 재학습/재추론 없음.
    model_result=read_json(ROOT/'reports/sources/phase2/dense_metrics.json')['joint_primary']
    follow_result=pd.read_csv(ROOT/'reports/sensor_availability/tables/full_summary.csv')
    baseline_result=read_json(ROOT/'reports/sensor_availability/baseline.json')
    final_check=read_json(ROOT/'reports/sensor_availability/sources/final_validation.json')
    prior_preservation=read_json(ROOT/'reports/sensor_availability/sources/preservation_check.json')
    assert baseline_result['all_checks_passed'] and final_check['passed']
    assert abs(model_result['detection_f1']-baseline_result['detection_metrics']['detection_f1'])<1e-12
    assert len(follow_result)==15 and set(follow_result.seed)=={42,43,44}
    assert np.allclose(follow_result.overall_detection_rate,
                       follow_result.conditional_detection_rate*follow_result.evaluable_fault_runs/10000)
    assert prior_preservation['all_unchanged']
    final_status=pd.DataFrame([
        ['Phase 1·2','원본·실행 분할·학습·고정 Test·고장별 평가·SHAP·미지 고장 포함'],
        ['후속 실험','Baseline 일치 + 마스크 동등성 + 파일럿 통과 + 전체15조건 검증'],
        ['보존 기록',f"TEP {prior_preservation['protected_TEP_files']}개, PHM {prior_preservation['protected_PHM_source_files']}개 보호 파일 변경 없음"],
        ['이 문서의 실행','기존 결과 표시·검증만 수행; 재학습·임계값 선택·전체 Test 추론 없음']
    ],columns=['범위','확인 내용'])
    show_table(final_status,'전체 연구의 기록 확인','Phase2 및 sensor_availability 검증 기록')
    '''))
    for index,cell in enumerate(cells):
        cell.id=f'complete-{index:03d}'
        if cell.cell_type=='code':
            cell.outputs=[];cell.execution_count=None
            compile(cell.source,f'complete_cell_{index}','exec')
    nb=nbf.v4.new_notebook(cells=cells,metadata={
        'title':'TEP 종합 연구: Phase1·Phase2·입력 가용성',
        'kernelspec':{'display_name':'TEP complete study','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12'},
        'source_notebooks':[str(p.relative_to(ROOT)) for p in SOURCES],
        'scope':'read-only synthesis and verification of completed studies; no retraining'})
    nbf.validate(nb);nbf.write(nb,TARGET)
    return before

def execute(before):
    if sys.platform=='win32':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    kernel=CACHE/'.jupyter/kernels/tep-complete-study';kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'TEP complete study','language':'python','env':{
            'PYTHONPATH':str(ROOT/'.deps'),'PYTHONDONTWRITEBYTECODE':'1',
            'MPLCONFIGDIR':str(CACHE/'.matplotlib'),'IPYTHONDIR':str(CACHE/'.ipython')}},indent=2),encoding='utf-8')
    os.environ['JUPYTER_PATH']=str(CACHE/'.jupyter')
    os.environ['JUPYTER_RUNTIME_DIR']=str(CACHE/'.runtime')
    os.environ['JUPYTER_CONFIG_DIR']=str(CACHE/'.config')
    Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(exist_ok=True)
    nb=nbf.read(TARGET,as_version=4)
    NotebookClient(nb,timeout=240,kernel_name='tep-complete-study',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.validate(nb);nbf.write(nb,TARGET)
    changed=[name for name,h in before.items() if sha(ROOT/name)!=h]
    assert not changed,changed
    outputs=[out for c in nb.cells for out in c.get('outputs',[])]
    codes=[c for c in nb.cells if c.cell_type=='code']
    errors=[o for o in outputs if o.output_type=='error']
    assert not errors and all(c.execution_count is not None for c in codes)
    tables=sum('<table' in o.get('data',{}).get('text/html','') for o in outputs)
    images=sum('image/png' in o.get('data',{}) for o in outputs)
    assert tables>=40 and images>=20,(tables,images)
    checks={'notebook':str(TARGET.relative_to(ROOT)),'cells':len(nb.cells),'executed_code_cells':len(codes),
        'rendered_tables':tables,'embedded_images':images,'error_outputs':len(errors),
        'protected_original_notebooks_and_model_files':len(before),'changed_original_files':changed,
        'source_notebooks':{str(p.relative_to(ROOT)):sha(p) for p in SOURCES},
        'bytes':TARGET.stat().st_size,'no_training_or_threshold_selection':True}
    (OUT/'sources/complete_notebook_validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':execute(build())
