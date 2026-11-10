"""Train/validation selection only; evaluate locked original Test once."""
import sys, os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.packages'))
os.environ['OMP_NUM_THREADS']='8'
import argparse,json,time,hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from wafer_features import extract_features,CLASSES

OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
cache=OUT/'features.npz'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset',type=Path,default=ROOT.parent/'data'/'WM811K.pkl',help='WM-811K pandas pickle (default: data/WM811K.pkl)')
parser.add_argument('--rebuild-cache',action='store_true',help='Recompute features when changing datasets or feature code')
args=parser.parse_args()

def scalar_label(value):
    # The original WM-811K release stores labels in nested singleton arrays.
    while isinstance(value,(list,tuple,np.ndarray)):
        flat=np.asarray(value,dtype=object).reshape(-1)
        if len(flat)==0:
            return ''
        if len(flat)!=1:
            raise ValueError('Expected a single label per wafer')
        value=flat[0]
    return str(value)

if args.rebuild_cache or not cache.exists():
    print('Loading source dataset',flush=True)
    if not args.dataset.is_file():
        raise FileNotFoundError(f'Dataset not found: {args.dataset}. Pass --dataset PATH.')
    df=pd.read_pickle(args.dataset)
    required={'waferMap','failureType','trainTestLabel','lotName'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing dataset columns: {required-set(df.columns)}')
    for column in ['failureType','trainTestLabel']:
        df[column]=df[column].map(scalar_label)
    df=df[df.failureType.isin(CLASSES)].copy()
    print('Labels/splits:',pd.crosstab(df.failureType,df.trainTestLabel).to_string(),flush=True)
    rows=[]; hashes=[]
    for i,a in enumerate(df.waferMap):
        rows.append(extract_features(a))
        hashes.append(hashlib.sha256(np.asarray(a.shape,dtype=np.int32).tobytes()+a.astype(np.uint8).tobytes()).hexdigest())
        if i%2000==0: print('Features',i,'/',len(df),flush=True)
    np.savez_compressed(cache,X=np.stack(rows),y=np.array([CLASSES.index(v) for v in df.failureType]),groups=df.lotName.astype(str).to_numpy(dtype=str),split=df.trainTestLabel.astype(str).to_numpy(dtype=str),indices=df.index.to_numpy(),hashes=np.array(hashes))
    del df,rows
d=np.load(cache); X,y,g=d['X'],d['y'],d['groups']; train=np.where(d['split']=='Training')[0]; test=np.where(d['split']=='Test')[0]
assert len(train)==17625 and len(test)==7894,(len(train),len(test))
tr,va=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(X[train],y[train],g[train]))
tr,va=train[tr],train[va]
assert not set(g[tr])&set(g[va])
# Remove exact map copies from fit if present in validation.
tr=tr[~np.isin(d['hashes'][tr],d['hashes'][va])]
models={
 'extra_trees':ExtraTreesClassifier(n_estimators=500,min_samples_leaf=1,max_features=.8,class_weight='balanced',n_jobs=8,random_state=42),
 'random_forest':RandomForestClassifier(n_estimators=400,min_samples_leaf=1,max_features=.7,class_weight='balanced_subsample',n_jobs=8,random_state=42),
 'hist_gradient':HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=23,l2_regularization=3,learning_rate=.07,early_stopping=False,random_state=42),
}
scores=[]
for name,m in models.items():
    print('Fitting',name,flush=True); m.fit(X[tr],y[tr]); p=m.predict(X[va])
    score={'model':name,'accuracy':accuracy_score(y[va],p),'macro_f1':f1_score(y[va],p,average='macro')}; scores.append(score)
    print(score,flush=True)
    (OUT/'validation.json').write_text(json.dumps(scores,indent=2))
best=max(scores,key=lambda s:(s['accuracy'],s['macro_f1']))['model']
model=models[best]
print('Selected on validation:',best,'; refitting all original Training',flush=True)
model.fit(X[train],y[train]); p=model.predict(X[test])
joblib.dump({'model':model,'classes':CLASSES,'feature_version':1},OUT/'spatial_model.joblib')
report={'selected_model':best,'selection_metric':'validation accuracy; macro F1 tie break','train_count':len(train),'test_count':len(test),'validation_fit_count':len(tr),'validation_count':len(va),'train_test_lot_overlap':len(set(g[train])&set(g[test])),'train_test_exact_map_overlap_test_count':int(np.isin(d['hashes'][test],d['hashes'][train]).sum()),'accuracy':accuracy_score(y[test],p),'macro_f1':f1_score(y[test],p,average='macro'),'classification_report':classification_report(y[test],p,target_names=CLASSES,output_dict=True),'confusion_matrix':confusion_matrix(y[test],p).tolist(),'validation_candidates':scores,'pdf_baseline_accuracy':.7009,'pdf_transfer_accuracy':.639}
clean=~np.isin(d['hashes'][test],d['hashes'][train])
report['test_without_training_duplicates']={'count':int(clean.sum()),'accuracy':accuracy_score(y[test][clean],p[clean]),'macro_f1':f1_score(y[test][clean],p[clean],average='macro')}
(OUT/'evaluation.json').write_text(json.dumps(report,indent=2))
pd.DataFrame({'source_index':d['indices'][test],'true':[CLASSES[i] for i in y[test]],'predicted':[CLASSES[i] for i in p],'lot':g[test]}).to_csv(OUT/'test_predictions.csv',index=False)
np.savez_compressed(OUT/'split_indices.npz',fit=tr,validation=va,training=train,test=test)
print(json.dumps(report,indent=2),flush=True)
