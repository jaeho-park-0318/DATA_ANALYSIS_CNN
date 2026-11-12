"""대표성·식별자 영향·정답 미확인 구간의 전향적 추론을 분리해 점검한다."""
from __future__ import annotations
import gc
import numpy as np
import pandas as pd
import joblib
import pyarrow.parquet as pq
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from src.model_common import *
from src.dataset_common import read_batches,tool_split
from src.train_xgboost import parameters
from src.evaluate_models import choose_threshold,evaluate_and_save,operational_metrics,sample_metrics

AUX=["Lot","runnum","ETCHSOURCEUSAGE","ETCHAUXSOURCETIMER","ETCHAUX2SOURCETIMER"]


def build_identity_aux():
    summaries=[]
    for file in sorted((DEST/"training").rglob("dataset.parquet")):
        tool=file.parent.name;path=OUT/"identity_aux"/f"{tool}.parquet"
        if path.exists():continue
        candidates=pd.read_parquet(file,columns=["time","label_any_h600raw"])
        times=candidates.loc[candidates.label_any_h600raw.notna(),"time"].to_numpy()
        if not len(times):continue
        raw=DATA/"train"/(tool[:2]+"_"+tool[2:]+"_DC_train.csv")
        retained=[]
        for chunk in pd.read_csv(raw,usecols=["time"]+AUX,dtype="float64",chunksize=200_000):
            selected=chunk[chunk.time.isin(times)]
            if len(selected):retained.append(selected)
        small=pd.concat(retained,ignore_index=True)
        group=small.groupby("time",sort=True)
        result=group[AUX].mean()
        # 동시각 wafer 식별자가 모순되면 임의의 식별자를 부여하지 않는다.
        result.loc[group.Lot.nunique(dropna=False)>1,"Lot"]=np.nan
        result=result.reset_index();result["Tool"]=tool
        assert len(result)==len(times)
        path.parent.mkdir(parents=True,exist_ok=True);result.to_parquet(path,index=False,compression="zstd")
        summaries.append({"Tool":tool,"rows":len(result)})
        print("식별자 진단용 현재 정보 수집:",tool,len(result),flush=True)
    return summaries


def identity_experiments():
    build_identity_aux()
    aux=pd.concat([pd.read_parquet(p) for p in sorted((OUT/"identity_aux").glob("*.parquet"))],ignore_index=True)
    groups={"Tool":["Tool"],"Time":["time"],"Lot":["Lot"],"Usage":AUX[1:],"All":["Tool","time"]+AUX}
    for scheme in SCHEMES:
        frames={s:load_split(scheme,600,s).merge(aux,on=["Tool","time"],how="left",validate="one_to_one") for s in ("train","validation","evaluation")}
        y={s:f.label_any_h600raw.to_numpy(dtype=np.int8) for s,f in frames.items()}
        for group,extras in groups.items():
            name="XGB_C_ID_"+group;folder=OUT/"experiments"/f"{scheme}_h600"/name
            if (folder/"result.json").exists():continue
            base=configs()['C'];xs={s:matrix(f,base) for s,f in frames.items()};encoders={};unknown={}
            for c in extras:
                if c in ("Tool","Lot"):
                    encoder=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)
                    train=frames['train'][[c]].astype("str").fillna("missing")
                    encoder.fit(train);encoders[c]=encoder
                    for split,frame in frames.items():
                        z=encoder.transform(frame[[c]].astype("str").fillna("missing"))
                        xs[split]=np.column_stack([xs[split],z.astype(np.float32)])
                        unknown[f"{split}_{c}"]=float((z<0).mean())
                else:
                    for split,frame in frames.items():xs[split]=np.column_stack([xs[split],frame[c].to_numpy(dtype=np.float32)])
            model=XGBClassifier(**parameters(y['train'])).fit(xs['train'],y['train'],eval_set=[(xs['validation'],y['validation'])],verbose=False)
            scores={s:model.predict_proba(x)[:,1] for s,x in xs.items()};threshold=choose_threshold(y['validation'],scores['validation'])
            folder.mkdir(parents=True,exist_ok=True);model.save_model(folder/"model.ubj");joblib.dump(encoders,folder/"encoders.joblib")
            evaluate_and_save(name,scheme,600,"C+"+group,frames,scores,threshold,
                              {"columns":base+extras,"identity_diagnostic_only":True,"unknown_category_fraction":unknown,
                               "best_iteration":model.best_iteration,"preprocessing":"식별자 인코더는 학습에만 적합; 수치 결측은 XGBoost 기본 처리"})
        del frames;gc.collect()


def load_prospective(scheme,split,columns,horizon=600):
    """정답 때문에 탈락한 구간도 예측한다. feature_ready 조건만 유지한다."""
    outputs=[];target=f"label_any_h{horizon}raw"
    names=list(dict.fromkeys(columns+META))
    for folder in sorted((DEST/"features").iterdir()):
        tool=folder.name
        if scheme=="split_tool" and tool_split(tool)!=split:continue
        pieces=[f[f.feature_ready] for f in read_batches(folder,list(dict.fromkeys(names+["feature_ready"]))) ]
        f=pd.concat(pieces,ignore_index=True)
        if not len(f):continue
        ids=f.row_id.to_numpy();pieces=[]
        labelnames=["row_id",scheme,target]+[f"label_{c}_h{horizon}raw" for c in ("f1","f2","f3")]+EVENT_COLS
        for label in read_batches(DEST/"labels"/tool,labelnames):
            subset=label[(label[scheme]==split)&label.row_id.isin(ids)]
            if len(subset):pieces.append(subset)
        if not pieces:continue
        joined=f.merge(pd.concat(pieces,ignore_index=True),on="row_id",how="inner",validate="one_to_one")
        if scheme=="split_time":
            bounds=pd.read_parquet(DEST/"segments"/f"{tool}.parquet")
            low=float(bounds.start.min());high=float(bounds.end.max())
            cut=low+(.6 if split=="validation" else .8)*(high-low)
            joined=joined[joined.history_min_time>=cut]
        outputs.append(joined)
    return pd.concat(outputs,ignore_index=True).sort_values(["Tool","time"],kind="stable").reset_index(drop=True)


def prospective_evaluation():
    selections=[]
    for scheme in SCHEMES:
        candidates=[read_json(p) for p in (OUT/"experiments"/f"{scheme}_h600").glob("*/result.json") if p.parent.name in ("RF_A","RF_B","RF_C","XGB_A","XGB_B","XGB_C")]
        best=max(candidates,key=lambda r:(r["metrics"]["validation"]["pr_auc_ap"],r["metrics"]["validation"]["f1"]))
        cols=best['details']['columns'];frame=load_prospective(scheme,"evaluation",cols)
        folder=OUT/"experiments"/f"{scheme}_h600"/best['name']
        if best['name'].startswith('RF'):
            model=joblib.load(folder/"model.joblib")["model"]
        else:model=XGBClassifier();model.load_model(folder/"model.ubj")
        score=model.predict_proba(matrix(frame,cols))[:,1]
        keep=META+["label_any_h600raw"]+[f"label_{c}_h600raw" for c in ("f1","f2","f3")]+EVENT_COLS
        predictions=frame[list(dict.fromkeys(keep))].copy();predictions['score']=score
        destination=OUT/"prospective"/scheme;destination.mkdir(parents=True,exist_ok=True)
        predictions.to_parquet(destination/"predictions.parquet",index=False,compression="zstd")
        known=predictions.label_any_h600raw.notna()
        metrics=operational_metrics(predictions,best['threshold'],600,scheme,"evaluation")
        metrics.update({"all_feature_ready_rows":len(frame),"unknown_label_rows":int((~known).sum()),
                        "unknown_row_alarm_fraction":float((score[~known]>=best['threshold']).mean()) if (~known).any() else None})
        result={"scheme":scheme,"selected_name":best['name'],"selection_rule":"검증 AP 최대; 평가 성능 미사용",
                "threshold":best['threshold'],"metrics":metrics}
        save_json(destination/"result.json",result);selections.append(result)
        print("전향적 평가 완료:",scheme,best['name'],len(frame),flush=True)
    return selections
