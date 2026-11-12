"""Portable inference API for one simulation trajectory at a time."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn

class TemporalNet(nn.Module):
    def __init__(self,kind,hidden=64):
        super().__init__();self.kind=kind
        if kind=='LSTM':self.encoder=nn.LSTM(52,hidden,batch_first=True)
        elif kind=='GRU':self.encoder=nn.GRU(52,hidden,batch_first=True)
        else:self.encoder=nn.Sequential(nn.Conv1d(52,hidden,5,padding=2),nn.ReLU(),nn.Conv1d(hidden,hidden,3,padding=1),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(hidden+52,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,21))
    def forward(self,x):
        h=self.encoder(x)[0][:,-1] if self.kind in ['LSTM','GRU'] else self.encoder(x.transpose(1,2)).mean(2)
        return self.head(torch.cat([h,x[:,-1]],1))

class TEPModel:
    def __init__(self,bundle,device='cpu'):
        self.bundle=Path(bundle);self.device=device
        self.names=json.loads((self.bundle/'sensor_names.json').read_text())
        checkpoint=torch.load(self.bundle/'model.pt',map_location='cpu',weights_only=False)
        self.model=TemporalNet(checkpoint['kind'],checkpoint['hidden']).to(device)
        self.model.load_state_dict(checkpoint['state_dict']);self.model.eval()
        self.threshold=float(checkpoint['threshold']);self.scaler=joblib.load(self.bundle/'scaler.joblib')
        detector=joblib.load(self.bundle/'binary_detector.joblib')
        self.detector=detector['model'];self.binary_threshold=float(detector['threshold'])
    def predict(self,frame):
        if frame.columns.duplicated().any():raise ValueError('Duplicate columns')
        missing=set(self.names)-set(frame.columns)
        if missing:raise ValueError(f'Missing process variables: {sorted(missing)}')
        x=frame[self.names].to_numpy(dtype=np.float32)
        if not np.isfinite(x).all():raise ValueError('Non-finite sensor value')
        if 'sample' in frame.columns and not np.all(np.diff(frame['sample'].to_numpy())==1):raise ValueError('Input samples must be consecutive')
        if 'simulationRun' in frame.columns and frame.simulationRun.nunique()!=1:raise ValueError('Pass exactly one simulationRun at a time')
        if 'faultNumber' in frame.columns and frame.faultNumber.nunique()!=1:raise ValueError('Pass exactly one fault scenario at a time')
        result=pd.DataFrame({'sample':frame['sample'].to_numpy() if 'sample' in frame.columns else np.arange(1,len(x)+1),'eligible':np.arange(len(x))>=19})
        for c in ['fault_probability','conditional_fault_id','gated_fault_id','alarm_episode_confirmed','dedicated_binary_probability','dedicated_binary_alarm']:result[c]=np.nan
        if len(x)<20:return result
        output=[];binary=[]
        with torch.inference_mode():
            for first in range(19,len(x),1024):
                ix=np.arange(first,min(first+1024,len(x)))
                w=x[ix[:,None]+np.arange(-19,1)[None,:]]
                standardized=((w-self.scaler.mean_.astype(np.float32))/self.scaler.scale_.astype(np.float32)).astype(np.float32)
                output.append(self.model(torch.from_numpy(standardized).to(self.device)).softmax(1).cpu().numpy())
                f=np.concatenate([w[:,-1],w.mean(1),w.std(1),w.min(1),w.max(1),(w[:,-1]-w[:,0])/57.],1)
                binary.append(self.detector.predict_proba(f)[:,1])
        proba=np.concatenate(output);p=1-proba[:,0];bp=np.concatenate(binary);diag=proba[:,1:].argmax(1)+1
        def confirmation(flag):
            n=0;out=[]
            for v in flag:n=n+1 if v else 0;out.append(int(n>=3))
            return out
        result.loc[19:,'fault_probability']=p
        result.loc[19:,'conditional_fault_id']=diag
        result.loc[19:,'gated_fault_id']=np.where(p>=self.threshold,diag,0)
        result.loc[19:,'alarm_episode_confirmed']=confirmation(p>=self.threshold)
        result.loc[19:,'dedicated_binary_probability']=bp
        result.loc[19:,'dedicated_binary_alarm']=confirmation(bp>=self.binary_threshold)
        return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',default=str(Path(__file__).resolve().parent));parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();model=TEPModel(args.bundle,args.device)
    model.predict(pd.read_csv(args.input)).to_csv(args.output,index=False)
