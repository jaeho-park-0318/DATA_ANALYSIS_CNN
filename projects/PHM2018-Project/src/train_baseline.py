"""Dummy·학습 기반 압력 임계값·Random Forest. 평가 데이터로 조정하지 않는다."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from src.model_common import *
from src.evaluate_models import choose_threshold,evaluate_and_save


def train(scheme="split_tool",horizon=600,include_simple=True,config_names=("A","B","C")):
    frames={s:load_split(scheme,horizon,s) for s in ("train","validation","evaluation")}
    y={s:f[f"label_any_h{horizon}raw"].to_numpy(dtype=np.int8) for s,f in frames.items()}
    results=[]
    if include_simple:
        folder=OUT/"experiments"/f"{scheme}_h{horizon}"/"Dummy"
        if (folder/"result.json").exists():results.append(read_json(folder/"result.json"))
        else:
            model=DummyClassifier(strategy="prior",random_state=SEED).fit(np.zeros((len(y['train']),1)),y['train'])
            scores={s:model.predict_proba(np.zeros((len(v),1)))[:,1] for s,v in y.items()}
            folder.mkdir(parents=True,exist_ok=True);joblib.dump(model,folder/"model.joblib")
            results.append(evaluate_and_save("Dummy",scheme,horizon,"none",frames,scores,.5,{"strategy":"prior","threshold_policy":"고정 0.5"}))
        folder=OUT/"experiments"/f"{scheme}_h{horizon}"/"Pressure"
        if (folder/"result.json").exists():results.append(read_json(folder/"result.json"))
        else:
            pressure={s:f.FLOWCOOLPRESSURE__current.to_numpy() for s,f in frames.items()}
            direction=1 if average_precision_score(y['train'],pressure['train'])>=average_precision_score(y['train'],-pressure['train']) else -1
            reference=np.sort(direction*pressure['train'])
            scores={s:np.searchsorted(reference,direction*x,side="right")/len(reference) for s,x in pressure.items()}
            threshold=choose_threshold(y['validation'],scores['validation'])
            folder.mkdir(parents=True,exist_ok=True);joblib.dump({"direction":direction,"train_reference":reference,"threshold":threshold},folder/"model.joblib")
            results.append(evaluate_and_save("Pressure",scheme,horizon,"pressure",frames,scores,threshold,
                                            {"direction":direction,"feature":"FLOWCOOLPRESSURE__current","score":"학습 경험적 CDF 순위; 보정 확률 아님","threshold_policy":"검증 F1 최대화"}))
    for config in config_names:
        name="RF_"+config;folder=OUT/"experiments"/f"{scheme}_h{horizon}"/name
        if (folder/"result.json").exists():results.append(read_json(folder/"result.json"));continue
        cols=configs()[config];xs={s:matrix(f,cols) for s,f in frames.items()}
        parameters={"n_estimators":64,"max_depth":10,"min_samples_leaf":20,"max_features":"sqrt",
                    "class_weight":"balanced_subsample","max_samples":.5,"n_jobs":4,"random_state":SEED}
        start=time.monotonic();model=RandomForestClassifier(**parameters).fit(xs['train'],y['train'])
        scores={s:model.predict_proba(x)[:,1] for s,x in xs.items()}
        threshold=choose_threshold(y['validation'],scores['validation'])
        folder.mkdir(parents=True,exist_ok=True);joblib.dump({"model":model,"columns":cols,"threshold":threshold},folder/"model.joblib",compress=3)
        save_json(folder/"importance.json",dict(zip(cols,model.feature_importances_)))
        results.append(evaluate_and_save(name,scheme,horizon,config,frames,scores,threshold,
                                        {"columns":cols,"parameters":parameters,"fit_seconds":time.monotonic()-start,"threshold_policy":"검증 F1 최대화","preprocessing":"대치/스케일링 없음"}))
    return results


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--scheme",default="split_tool",choices=SCHEMES);p.add_argument("--horizon",type=int,default=600)
    a=p.parse_args();train(a.scheme,a.horizon)
