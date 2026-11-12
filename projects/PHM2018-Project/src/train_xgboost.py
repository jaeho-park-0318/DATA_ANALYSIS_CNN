"""불균형 가중치·검증 Early Stopping·검증 임계값의 XGBoost 실험."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from xgboost import XGBClassifier
from src.model_common import *
from src.evaluate_models import choose_threshold,evaluate_and_save


def parameters(y,rounds=500):
    return {"n_estimators":rounds,"max_depth":4,"learning_rate":.05,"min_child_weight":10,
            "subsample":.8,"colsample_bytree":.8,"reg_lambda":5,"reg_alpha":.1,
            "scale_pos_weight":float((y==0).sum()/(y==1).sum()),"tree_method":"hist","max_bin":128,
            "n_jobs":4,"random_state":SEED,"eval_metric":"aucpr","early_stopping_rounds":30}


def train(scheme="split_tool",horizon=600,config_names=("A","B","C")):
    frames={s:load_split(scheme,horizon,s) for s in ("train","validation","evaluation")}
    y={s:f[f"label_any_h{horizon}raw"].to_numpy(dtype=np.int8) for s,f in frames.items()}
    results=[]
    for config in config_names:
        name="XGB_"+config;folder=OUT/"experiments"/f"{scheme}_h{horizon}"/name
        if (folder/"result.json").exists():results.append(read_json(folder/"result.json"));continue
        cols=configs()[config];xs={s:matrix(f,cols) for s,f in frames.items()}
        params=parameters(y['train']);start=time.monotonic()
        model=XGBClassifier(**params).fit(xs['train'],y['train'],eval_set=[(xs['validation'],y['validation'])],verbose=False)
        scores={s:model.predict_proba(x)[:,1] for s,x in xs.items()}
        threshold=choose_threshold(y['validation'],scores['validation'])
        folder.mkdir(parents=True,exist_ok=True);model.save_model(folder/"model.ubj")
        save_json(folder/"importance.json",dict(zip(cols,model.feature_importances_)))
        save_json(folder/"learning_curve.json",model.evals_result())
        results.append(evaluate_and_save(name,scheme,horizon,config,frames,scores,threshold,
                                        {"columns":cols,"parameters":params,"best_iteration":model.best_iteration,"fit_seconds":time.monotonic()-start,
                                         "threshold_policy":"검증 F1 최대화","early_stopping":"검증 aucpr만 사용","preprocessing":"대치/스케일링 없음"}))
    return results


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--scheme",default="split_tool",choices=SCHEMES);p.add_argument("--horizon",type=int,default=600)
    a=p.parse_args();train(a.scheme,a.horizon)
