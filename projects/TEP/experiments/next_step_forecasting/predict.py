"""Predict raw next-step52 process values for one complete3-minute trajectory."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import argparse,json
import numpy as np
import pandas as pd
import joblib,torch
from run import NextStepLSTM,predict,HERE,ROOT

class Forecaster:
    def __init__(self,device='cpu',seed=42):
        self.device=device
        self.names=json.loads((ROOT/'data/processed/phase2/sensor_names.json').read_text())
        self.scaler=joblib.load(HERE/'models/scaler.joblib')
        ck=torch.load(HERE/'models'/f'lstm_s{seed}.pt',map_location='cpu',weights_only=False)
        self.model=NextStepLSTM().to(device);self.model.load_state_dict(ck['state_dict']);self.model.eval()
    def predict_next(self,frame):
        if len(frame)<20:raise ValueError('At least20 past observations required')
        if frame.columns.duplicated().any():raise ValueError('Duplicate columns')
        missing=set(self.names)-set(frame.columns)
        if missing:raise ValueError(f'Missing process columns: {sorted(missing)}')
        for name in ['simulationRun','faultNumber']:
            if name in frame and frame[name].nunique()!=1:raise ValueError('Only one trajectory per input')
        if 'sample' in frame and not np.all(np.diff(frame['sample'].to_numpy())==1):raise ValueError('Consecutive3-minute observations required')
        x=frame[self.names].to_numpy(dtype=np.float32)
        if not np.isfinite(x).all():raise ValueError('Missing inputs are not imputed')
        endpoints=np.arange(19,len(x))
        w=x[endpoints[:,None]+np.arange(-19,1)[None,:]]
        scaled=((w-self.scaler.mean_.astype(np.float32))/self.scaler.scale_.astype(np.float32)).astype(np.float32)
        output=predict(self.model,scaled,self.device)*self.scaler.scale_+self.scaler.mean_
        out=pd.DataFrame(output,columns=[f'predicted_{n}' for n in self.names])
        out.insert(0,'target_sample',frame['sample'].to_numpy()[endpoints]+1 if 'sample' in frame else endpoints+2)
        return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    args=p.parse_args();Forecaster(args.device).predict_next(pd.read_csv(args.input)).to_csv(args.output,index=False)
