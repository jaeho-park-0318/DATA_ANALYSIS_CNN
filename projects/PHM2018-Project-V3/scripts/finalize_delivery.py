"""완성 파일과 문서 링크를 점검하고 최종 재현 정보를 기록한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ast
import re
from src.common import *

def main():
    required=['tool_summary','time_segment_summary','fault_event_summary','label_status_summary','feature_availability_summary','event_exclusion_reasons']
    for name in required:
        assert (TABLES/(name+'.csv')).is_file()
        assert (TABLES/(name+'.parquet')).is_file()
    documents=[ROOT/'README.md',ROOT/'dashboard/README.md',*(ROOT/'reports').glob('*.md'),*(ROOT/'experiments').rglob('*.md')]
    links=0
    for doc in documents:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#')):continue
            assert (doc.parent/target).resolve().exists(),f'{doc}: {target}'
            links+=1
    python_files=[p for folder in ['src','scripts','dashboard','tests'] for p in (ROOT/folder).rglob('*.py')]
    for path in python_files:ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    figures=json.loads((ROOT/'outputs/figure_manifest.json').read_text(encoding='utf-8'))
    cases=json.loads((ROOT/'outputs/case_catalog.json').read_text(encoding='utf-8'))
    for case in cases:
        for name in ['anchor_evidence.csv','process_conditions.csv','event_exclusion.csv','sensor_changes.csv','segment_status.csv','metadata.json']:
            assert (ROOT/'outputs/cases'/case['case_id']/name).is_file()
    actual=json.loads((ROOT/'outputs/validation/actual_checks.json').read_text(encoding='utf-8'))
    visual=json.loads((ROOT/'outputs/validation/visual_checks.json').read_text(encoding='utf-8'))
    notebooks=json.loads((ROOT/'outputs/validation/notebook_execution.json').read_text(encoding='utf-8'))
    assert all(c['passed'] for c in actual['checks']+visual['checks'])
    assert len(notebooks)==5 and all(n['errors']==0 for n in notebooks)
    stage_notebook=ROOT/'outputs/validation/alarm_stage1_notebook.json'
    if stage_notebook.exists():
        n=json.loads(stage_notebook.read_text(encoding='utf-8'));assert n['errors']==0 and n['executed']==n['code_cells'];notebooks.append(n)
    stage_figures=ROOT/'outputs/alarm_stage1_figure_manifest.json'
    if stage_figures.exists():figures+=json.loads(stage_figures.read_text(encoding='utf-8'))
    stage2_notebook=ROOT/'outputs/validation/stage2_notebook.json'
    training_performed=stage2_notebook.exists()
    if training_performed:
        n=json.loads(stage2_notebook.read_text(encoding='utf-8'));assert n['errors']==0 and n['executed']==n['code_cells'];notebooks.append(n)
        figures+=json.loads((ROOT/'outputs/stage2_figure_manifest.json').read_text(encoding='utf-8'))
        trained=json.loads((ROOT/'outputs/validation/stage2_checks.json').read_text(encoding='utf-8'));assert all(c['passed'] for c in trained['checks']) and trained['holdout_scored'] is False
    stage3_notebook=ROOT/'outputs/validation/stage3_notebook.json'
    if stage3_notebook.exists():
        n=json.loads(stage3_notebook.read_text(encoding='utf-8'));assert n['errors']==0 and n['executed']==n['code_cells'];notebooks.append(n)
        figures+=json.loads((ROOT/'outputs/stage3_figure_manifest.json').read_text(encoding='utf-8'))
        trained=json.loads((ROOT/'outputs/validation/stage3_checks.json').read_text(encoding='utf-8'));assert all(c['passed'] for c in trained['checks']) and trained['holdout_scored'] is False
    stage4_notebook=ROOT/'outputs/validation/stage4_notebook.json'
    if stage4_notebook.exists():
        n=json.loads(stage4_notebook.read_text(encoding='utf-8'));assert n['errors']==0 and n['executed']==n['code_cells'];notebooks.append(n)
        figures+=json.loads((ROOT/'outputs/stage4_figure_manifest.json').read_text(encoding='utf-8'))
        trained=json.loads((ROOT/'outputs/validation/stage4_checks.json').read_text(encoding='utf-8'));assert all(c['passed'] for c in trained['checks']) and trained['holdout_scored'] is False
    final_notebook=ROOT/'outputs/validation/final_notebook.json'
    holdout_scored=False
    final_counts={}
    if final_notebook.exists():
        n=json.loads(final_notebook.read_text(encoding='utf-8'));assert n['errors']==0 and n['executed']==n['code_cells'] and n['additional_holdout_inference']==0;notebooks.append(n)
        final_figures=json.loads((ROOT/'outputs/final_evaluation_figure_manifest.json').read_text(encoding='utf-8'));figures+=final_figures
        for stage in ['stage5','stage6']:
            audit=json.loads((ROOT/f'outputs/validation/{stage}_checks.json').read_text(encoding='utf-8'));assert all(c['passed'] for c in audit['checks']);final_counts[stage+'_checks']=len(audit['checks'])
        qa=json.loads((ROOT/'outputs/validation/final_visual_checks.json').read_text(encoding='utf-8'));assert all(c['passed'] for c in qa['checks']) and len(qa['checks'])==len(final_figures)
        state=json.loads((ROOT/'outputs/stage6/holdout_access.json').read_text(encoding='utf-8'));assert state['status']=='complete' and state['full_prediction_passes']==1 and not state['threshold_search'] and not state['refit'];holdout_scored=True
        frozen=json.loads((ROOT/'outputs/stage6/frozen_manifest.json').read_text(encoding='utf-8'))
        for name,h in frozen['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    specialist_notebook=ROOT/'outputs/stage7/notebook_execution.json'
    if specialist_notebook.exists():
        from src.specialist_experiment import verify_previous
        verify_previous()
        n=json.loads(specialist_notebook.read_text(encoding='utf-8'));assert n['errors']==0 and n['executed']==n['code_cells'] and n['additional_later_inference']==0;notebooks.append(n)
        sf=json.loads((ROOT/'outputs/stage7/figure_manifest.json').read_text(encoding='utf-8'));figures+=sf
        sc=json.loads((ROOT/'outputs/stage7/checks.json').read_text(encoding='utf-8'));assert all(c['passed'] for c in sc['checks'])
        sq=json.loads((ROOT/'outputs/stage7/visual_checks.json').read_text(encoding='utf-8'));assert all(c['passed'] for c in sq['checks']) and len(sq['checks'])==len(sf)
        access7=json.loads((ROOT/'outputs/stage7/evaluation_access.json').read_text(encoding='utf-8'));assert access7['status']=='complete' and access7['independent_final_test'] is False
        final_counts.update(stage7_checks=len(sc['checks']),stage7_figures=len(sf),stage7_specialist_tools=len(access7['prediction_sha256']),stage7_independent_final_test=False,stage7_prediction_passes_per_tool=access7['specialist_prediction_passes_per_tool'],holdout_prediction_scope='stage6 원래 동결 공통 모델의 횟수. stage7 전용 모델 후속 연구는 별도 기록')
    consolidated=ROOT/'outputs/consolidated/execution_validation.json'
    if consolidated.exists():
        n=json.loads(consolidated.read_text(encoding='utf-8'));assert n['errors']==0 and n['executed']==n['code_cells'] and n['standalone_execution_in_empty_directory'] and n['new_training']==n['new_model_inference']==0
        assert hashlib.sha256((ROOT/n['notebook']).read_bytes()).hexdigest()==n['notebook_sha256'];notebooks.append(n)
        final_counts.update(consolidated_embedded_tables=n['embedded_tables'],consolidated_embedded_figures=n['embedded_figures'],consolidated_archived_reports=n['archived_reports'],consolidated_standalone_execution=True)
    assert snapshot()==json.loads((ROOT/'outputs/source_snapshot_before.json').read_text(encoding='utf-8'))
    provenance={'files':{str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src','scripts','dashboard','tests','config'] for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.js','.html','.css','.json']},'source_metadata_only':True,'raw_content_hashes_not_computed':True}
    dump_json(ROOT/'outputs/code_provenance.json',provenance)
    packages=environment()['packages']
    used=['pandas','numpy','pyarrow','matplotlib','nbformat','nbclient','ipykernel']
    if training_performed:
        for p in ['scipy','scikit-learn','xgboost','joblib','threadpoolctl','psutil']:packages[p]=importlib.metadata.version(p);used.append(p)
    (ROOT/'requirements-local.txt').write_text('# 현재 phm2018 환경의 재현용 버전 기록. 이 작업은 환경을 변경하지 않았습니다.\n'+'\n'.join(f'{p}=={packages[p]}' for p in used)+'\nPillow=='+importlib.metadata.version('Pillow')+'\n',encoding='utf-8')
    summary={'tools':actual['tools'],'raw_rows':int(pd.read_parquet(TABLES/'tool_summary.parquet').raw_rows.sum()),'unique_events':len(pd.read_parquet(TABLES/'events.parquet')),'figures':len(figures),'cases':len(cases),'executed_notebooks':len(notebooks),'notebook_code_cells':sum(n['executed'] for n in notebooks),'actual_checks':len(actual['checks']),'visual_checks':len(visual['checks']),'verified_document_links':links,'python_syntax_files':len(python_files),'training_performed':training_performed,'holdout_scored':False,'raw_metadata_unchanged':True}
    summary.update(holdout_scored=holdout_scored,**final_counts)
    if holdout_scored:summary.update(holdout_prediction_passes=1,holdout_additional_prediction_passes=0,frozen_files_unchanged=True,final_visual_checks=len(qa['checks']))
    dump_json(ROOT/'outputs/validation/completion.json',summary)
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
