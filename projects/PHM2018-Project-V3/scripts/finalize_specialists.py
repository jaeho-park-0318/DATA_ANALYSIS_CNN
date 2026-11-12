"""장비 전용 추가 프로젝트의 완료와 재현 해시 기록."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ast,re
from src.specialist_experiment import *


def main():
    verify_previous();frozen=readjson(OUT/'frozen_manifest.json')
    for path,h in frozen['sha256'].items():assert digest(ROOT/path)==h,path
    checks=readjson(OUT/'checks.json');qa=readjson(OUT/'visual_checks.json');notebook=readjson(OUT/'notebook_execution.json');figures=readjson(OUT/'figure_manifest.json');state=readjson(OUT/'evaluation_access.json')
    assert all(c['passed'] for c in checks['checks']+qa['checks']) and notebook['executed']==notebook['code_cells'] and notebook['errors']==0 and state['status']=='complete'
    assert len(figures)==len(qa['checks'])==6
    docs=[ROOT/'reports/20_equipment_specialists.md',ROOT/'experiments/equipment_specialists/README.md']
    links=0
    for doc in docs:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            assert (doc.parent/target).resolve().exists(),target
            links+=1
    source=[ROOT/'src/specialist_experiment.py',*sorted((ROOT/'scripts').glob('*specialist*.py')),PROTOCOL_PATH]
    for p in source:
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
    artifacts=[*source,*docs,ROOT/'notebooks/16_equipment_specialists.ipynb',*sorted(TABLES.glob('stage7_*')),*[ROOT/f['file'] for f in figures],*[p for p in OUT.iterdir() if p.is_file() and p.name not in ['artifacts.json','completion.json']]]
    dump_json(OUT/'artifacts.json',dict(sha256={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in artifacts},packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','xgboost','joblib','nbformat','nbclient']},python=sys.executable))
    result=dict(status='complete',completed_at_utc=utc(),selected_tools=frozen['tools'],device='cuda:0',checks=len(checks['checks']),figures=len(figures),executed_notebook_cells=notebook['executed'],notebook_errors=0,verified_document_links=links,independent_final_test=False,previous_final_artifacts_unchanged=True,selected_tools_limit=3,additional_later_inference_in_notebook=0)
    dump_json(OUT/'completion.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
