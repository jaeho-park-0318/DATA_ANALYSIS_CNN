"""실제 모델 산출물 수집과 최종 재현성 검사."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from src.model_common import *
from src.evaluate_models import sample_metrics


def collect():
    records=[];objects=[]
    for path in sorted((OUT/"experiments").glob("*/*/result.json")):
        result=read_json(path);objects.append(result)
        for split,metrics in result['metrics'].items():
            scalar={k:v for k,v in metrics.items() if not isinstance(v,(dict,list))}
            records.append({"scheme":result['scheme'],"horizon":result['horizon'],"name":result['name'],"config":result['config'],
                            "split":split,"threshold":result['threshold'],"identity_diagnostic":result['details'].get('identity_diagnostic_only',False),**scalar})
    frame=pd.DataFrame(records);frame.to_parquet(OUT/'metrics.parquet',index=False,compression='zstd')
    return objects,frame


def selected_per_family(objects,horizon=600):
    selected=[]
    for scheme in SCHEMES:
        for family in ('RF','XGB'):
            choices=[r for r in objects if r['scheme']==scheme and r['horizon']==horizon and r['name'] in [family+'_'+c for c in ('A','B','C')]]
            best=max(choices,key=lambda r:r['metrics']['validation']['pr_auc_ap'])
            selected.append(best)
    return selected


def final_checks():
    objects,frame=collect()
    assert len(objects)==58,("실험 수",len(objects))
    assert len([r for r in objects if r['details'].get('identity_diagnostic_only')])==10
    for result in objects:
        assert result['completed']
        assert not result['details'].get('columns') or result['details'].get('identity_diagnostic_only') or set(result['details']['columns']).issubset(set(feature_columns()))
        folder=OUT/'experiments'/f"{result['scheme']}_h{result['horizon']}"/result['name']
        for split in ('validation','evaluation'):
            pred=pd.read_parquet(folder/f'{split}_predictions.parquet')
            assert np.isfinite(pred.score).all() and pred.score.between(0,1).all()
            y=pred[f"label_any_h{result['horizon']}raw"]
            assert y.notna().all()
            fresh=sample_metrics(y,pred.score,result['threshold'])
            for key in ('tn','fp','fn','tp','n'):assert fresh[key]==result['metrics'][split][key]
            for key in ('f1','precision','recall','pr_auc_ap','roc_auc'):
                expected=result['metrics'][split][key]
                assert fresh[key] is None if expected is None else np.isclose(fresh[key],expected)
    previous=read_json(OUT/'preexisting_files.json')
    unchanged=all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in previous)
    assert unchanged,'기존 코드·Notebook·보고서 변경'
    assert snapshot()==read_json(OUT/'preflight.json')['source_snapshot'],'원본 메타데이터 변경'
    result={'experiments':len(objects),'core_experiments':48,'identity_experiments':10,
            'all_prediction_metrics_checked':True,'prior_files_unchanged':unchanged,
            'prior_files_checked':len(previous),'original_metadata_unchanged':True,
            'features':{c:len(v) for c,v in configs().items()},'passed':True}
    save_json(OUT/'final_checks.json',result)
    return result
