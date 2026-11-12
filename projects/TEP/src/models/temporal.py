"""Past-window LSTM/GRU/1D-CNN candidates with validation early stopping."""
from pathlib import Path
import sys,json,time,random,copy
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.data.phase2 import CFG,trajectories
from src.features.phase2 import load
from src.evaluation.phase2 import threshold,joint_metrics
M=ROOT/'models/phase2';T=ROOT/'reports/tables/phase2'

class TemporalNet(nn.Module):
    def __init__(self,kind,hidden=64):
        super().__init__();self.kind=kind
        if kind=='LSTM':self.encoder=nn.LSTM(52,hidden,batch_first=True)
        elif kind=='GRU':self.encoder=nn.GRU(52,hidden,batch_first=True)
        else:self.encoder=nn.Sequential(nn.Conv1d(52,hidden,5,padding=2),nn.ReLU(),nn.Conv1d(hidden,hidden,3,padding=1),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(hidden+52,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,21))
    def forward(self,x):
        if self.kind in ['LSTM','GRU']:h=self.encoder(x)[0][:,-1]
        else:h=self.encoder(x.transpose(1,2)).mean(dim=2)
        return self.head(torch.cat([h,x[:,-1]],dim=1))

def gather(meta,split,scaler):
    raw=trajectories(split)
    w=raw[meta[:,0,None],meta[:,1,None],meta[:,2,None]-20+np.arange(20)[None,:]]
    return ((w-scaler.mean_.astype(np.float32))/scaler.scale_.astype(np.float32)).astype(np.float32)

def predict(model,w,device,batch=1024):
    model.eval();parts=[]
    with torch.inference_mode():
        for i in range(0,len(w),batch):
            logits=model(torch.from_numpy(w[i:i+batch]).to(device));parts.append(logits.softmax(1).cpu().numpy())
    return np.concatenate(parts)

def run():
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('Temporal device',device,flush=True)
    X,y,meta=load('train');V,vy,vmeta=load('validation')
    scaler=StandardScaler().fit(X[:,:52]);joblib.dump(scaler,M/'temporal_scaler.joblib')
    train=gather(meta,'train',scaler);valid=gather(vmeta,'validation',scaler)
    del X,V
    rows=[];history=[]
    weights=compute_class_weight('balanced',classes=np.arange(21),y=y).astype(np.float32)
    for kind in ['LSTM','GRU','CNN1D']:
        for seed in CFG['seeds']:
            tag=f'{kind}_Temporal_balanced_s{seed}';summary=M/(tag+'.json')
            if summary.exists():rows.append(json.loads(summary.read_text()));continue
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            if device=='cuda':torch.cuda.manual_seed_all(seed)
            model=TemporalNet(kind).to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
            lossfn=nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
            loader=DataLoader(TensorDataset(torch.from_numpy(train),torch.from_numpy(y)),batch_size=CFG['deep_batch_size'],shuffle=True,num_workers=0)
            best=-1;state=None;patience=0;start=time.perf_counter();bestrow=None
            for epoch in range(1,CFG['deep_epochs']+1):
                model.train();total=0
                for xx,yy in loader:
                    xx=xx.to(device);yy=yy.to(device);optimizer.zero_grad(set_to_none=True)
                    loss=lossfn(model(xx),yy);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();total+=float(loss.detach())*len(yy)
                p=predict(model,valid,device);th=threshold(vy,1-p[:,0]);result,_=joint_metrics(vy,p,th,auc=False)
                score=(result['diagnosis_macro_f1']+result['detection_f1'])/2
                history.append(dict(model_id=tag,epoch=epoch,loss=total/len(y),selection_score=score,**result))
                print('EPOCH',tag,epoch,'diag',round(result['diagnosis_macro_f1'],4),'detect',round(result['detection_f1'],4),flush=True)
                pd.DataFrame(history).to_csv(T/'temporal_history.csv',index=False)
                if score>best+1e-5:
                    best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};patience=0
                    bestrow=dict(model_id=tag,family=kind,feature='Temporal',seed=seed,weight='balanced',epoch=epoch,selection_score=score,device=device,train_rows=len(y),validation_rows=len(vy),**result)
                else:patience+=1
                if patience>=CFG['deep_patience']:break
            bestrow['training_seconds']=time.perf_counter()-start
            torch.save({'state_dict':state,'kind':kind,'hidden':64,'threshold':bestrow['threshold'],'metadata':bestrow},M/(tag+'.pt'))
            bestrow['model_bytes']=(M/(tag+'.pt')).stat().st_size
            summary.write_text(json.dumps(bestrow,indent=2));rows.append(bestrow)
            pd.DataFrame(rows).to_csv(T/'validation_temporal.csv',index=False)
            del model;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(T/'validation_temporal.csv',index=False)

if __name__=='__main__':run()
