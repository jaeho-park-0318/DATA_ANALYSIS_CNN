"""Validation-only threshold selection and explicit detection/diagnosis metrics."""
import numpy as np
from sklearn.metrics import (precision_recall_curve,precision_recall_fscore_support,
    balanced_accuracy_score,roc_auc_score,average_precision_score,confusion_matrix,f1_score)

def threshold(y,score,max_far=.01):
    binary=np.asarray(y)>0
    p,r,t=precision_recall_curve(binary,score)
    tp=r[:-1]*binary.sum();fp=tp*(1/np.maximum(p[:-1],1e-15)-1)
    far=fp/(~binary).sum();f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-15)
    allowed=far<=max_far+1e-12
    if not np.any(allowed):return float(np.nextafter(np.max(score),np.inf))
    return float(t[np.argmax(np.where(allowed,f,-1))])

def binary_metrics(y,score,t):
    y=np.asarray(y)>0;p=score>=t
    prec,rec,f,_=precision_recall_fscore_support(y,p,average='binary',zero_division=0)
    return dict(detection_precision=prec,detection_recall=rec,detection_f1=f,
        detection_macro_f1=f1_score(y,p,average='macro'),detection_balanced_accuracy=balanced_accuracy_score(y,p),
        detection_roc_auc=roc_auc_score(y,score),detection_pr_auc=average_precision_score(y,score),
        sample_false_alarm_rate=float(p[~y].mean()),threshold=float(t),samples=len(y))

def joint_metrics(y,proba,t,auc=True):
    score=1-proba[:,0];d=(score>=t)
    diag=np.argmax(proba[:,1:],axis=1)+1
    pred=np.where(d,diag,0)
    positive=y>0
    result=binary_metrics(y,score,t)
    pp,rr,ff,_=precision_recall_fscore_support(y[positive],diag[positive],labels=np.arange(1,21),average='macro',zero_division=0)
    result.update(diagnosis_macro_precision=pp,diagnosis_macro_recall=rr,diagnosis_macro_f1=ff,
        diagnosis_balanced_accuracy=balanced_accuracy_score(y[positive],diag[positive]),
        end_to_end_macro_f1=f1_score(y,pred,labels=np.arange(21),average='macro',zero_division=0))
    if auc:
        q=proba[positive,1:].astype(np.float64);q/=np.maximum(q.sum(axis=1,keepdims=True),1e-15)
        result['diagnosis_roc_auc_macro_ovr']=roc_auc_score(y[positive],q,labels=np.arange(1,21),multi_class='ovr',average='macro')
        result['diagnosis_pr_auc_macro']=float(np.mean([average_precision_score(y[positive]==i,q[:,i-1]) for i in range(1,21)]))
    return result,confusion_matrix(y,pred,labels=np.arange(21))

def episodes(alarm,k=3):
    """A confirmation needs k consecutive positives; contiguous positives form one episode."""
    alarm=np.asarray(alarm,dtype=bool)
    starts=np.flatnonzero(alarm & ~np.r_[False,alarm[:-1]])
    ends=np.flatnonzero(alarm & ~np.r_[alarm[1:],False])+1
    return [(int(s+k-1),int(e)) for s,e in zip(starts,ends) if e-s>=k]
