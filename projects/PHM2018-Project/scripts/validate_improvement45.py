"""기존 파일 보존, 실제 모델 예측, 입력 및 Notebook 실행을 최종 검증한다."""
from pathlib import Path
import sys,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import duckdb,numpy as np,pandas as pd,nbformat
from xgboost import XGBClassifier
from src.improvement_common import *
from src.group_validation45 import load_tools
from src.evaluate_models import sample_metrics

def run():
    old=read_json(OUT/'preservation.json');changed=[]
    for item in old['files']:
        p=Path(item['path'])
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:changed.append(str(p))
    assert not changed,changed
    assert old['raw']==snapshot()
    con=duckdb.connect();con.execute('SET threads=4');con.execute("SET memory_limit='2GB'")
    con.execute(f"CREATE VIEW cached AS SELECT * FROM read_parquet('{(OUT/'cache/*.parquet').as_posix()}')")
    con.execute(f"CREATE VIEW original AS SELECT * FROM read_parquet('{(DEST/'training/*/*.parquet').as_posix()}')")
    duplicate=con.execute('SELECT count(*) FROM (SELECT Tool,row_id,count(*) n FROM cached GROUP BY Tool,row_id HAVING n>1)').fetchone()[0]
    future=con.execute('SELECT count(*) FROM cached WHERE source_max_time>time OR history_min_time>time').fetchone()[0]
    changed_labels=con.execute(f'SELECT count(*) FROM cached c JOIN original o USING(Tool,row_id) WHERE c.{TARGET} IS DISTINCT FROM o.{TARGET}').fetchone()[0]
    lost_known=con.execute(f'SELECT count(*) FROM original o LEFT JOIN cached c USING(Tool,row_id) WHERE o.{TARGET} IS NOT NULL AND c.row_id IS NULL').fetchone()[0]
    counts=con.execute(f'SELECT count(*) rows,count({TARGET}) known FROM cached').fetchone()
    assert not any([duplicate,future,changed_labels,lost_known])
    checked=[]
    for folder in sorted((OUT/'cv').glob('fold_*/*')):
        r=read_json(folder/'result.json');assert r['completed'] and r['actual_device']=='cuda:0'
        pred=pd.read_parquet(folder/'evaluation_predictions.parquet');known=pred[pred[TARGET].notna()]
        for policy,threshold in r['thresholds'].items():
            recalculated=sample_metrics(known[TARGET].astype(int),known.score,threshold)
            for key,value in recalculated.items():
                oldvalue=r['evaluation'][policy][key]
                assert (value is None and oldvalue is None) or np.isclose(value,oldvalue,rtol=1e-6,atol=1e-9),(folder,key)
        # 장비별 시간순 첫·마지막 256개 점수로 저장 모델의 재로딩을 확인한다.
        frame=load_tools(r['plan']['evaluation']);sample=pd.concat([pd.concat([g.head(256),g.tail(256)]) for _,g in frame.groupby('Tool')]).drop_duplicates(['Tool','row_id'])
        expected=sample[['Tool','row_id']].merge(pred[['Tool','row_id','score']],on=['Tool','row_id'],validate='one_to_one').score.to_numpy()
        model=XGBClassifier();model.load_model(folder/'model.ubj');model.set_params(device='cpu',n_jobs=4)
        score=model.predict_proba(matrix(sample,r['columns']))[:,1]
        assert np.allclose(expected,score,rtol=1e-6,atol=1e-7)
        checked.append({'fold':r['fold'],'model':r['name'],'prediction_rows_checked':len(sample),'metric_rows_checked':len(known)})
    test=subprocess.run([sys.executable,'-X','utf8','-m','unittest','tests.test_improvement45','-v'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    (OUT/'tests.txt').write_text(test.stdout+test.stderr,encoding='utf-8');assert test.returncode==0
    notebooks=[]
    for name in ['09_dataset_audit.ipynb','10_group_cross_validation.ipynb','11_threshold_analysis.ipynb']:
        nb=nbformat.read(ROOT/'notebooks'/name,as_version=4);nbformat.validate(nb)
        cells=[c for c in nb.cells if c.cell_type=='code'];assert all(c.execution_count is not None for c in cells)
        errors=[o for c in cells for o in c.outputs if o.output_type=='error'];assert not errors
        notebooks.append({'name':name,'executed_cells':len(cells),'errors':len(errors)})
    save_json(OUT/'delivery_checks.json',{'passed':True,'preexisting_files_preserved':len(old['files']),'raw_files_preserved':len(old['raw']),
        'cached_rows':counts[0],'known_rows':counts[1],'duplicate_keys':duplicate,'future_feature_rows':future,'changed_labels':changed_labels,'lost_known_rows':lost_known,
        'models':checked,'unit_tests':4,'notebooks':notebooks,'python':sys.executable})
    print('최종 검증 통과: 원본·기존 결과 보존, 16개 모델·라벨·누수·Notebook 확인',flush=True)

if __name__=='__main__':run()
