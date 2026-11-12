"""Separate LSTM next-step forecaster. The frozen diagnostic CNN is never altered."""
from pathlib import Path
import sys,os,json,time,copy,random
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
OUT=ROOT/'reports/next_step_forecasting'
for p in [HERE/'models',HERE/'cache',OUT,OUT/'tables',OUT/'figures']:p.mkdir(parents=True,exist_ok=True)
CFG={'window':20,'horizon_samples':1,'horizon_minutes':3,'train_stride':4,'comparison_stride':10,'seeds':[42,7,2026],
    'hidden':64,'layers':2,'dropout':.1,'epochs':12,'patience':3,'batch_size':256,'learning_rate':.001,
    'fit_cohort':'faultfree official Training runs assigned to Train only','selection':'lowest normal Validation normalized MSE; early stopping; primary seed42 predetermined',
    'coupling':'standalone forecasting only; predictions are NOT fed into the existing CNN'}
CACHE=ROOT/'data/processed/phase2'
SPLIT=json.loads((CACHE/'split_manifest.json').read_text())

class NextStepLSTM(nn.Module):
    def __init__(self):
        super().__init__();self.lstm=nn.LSTM(52,64,num_layers=2,dropout=.1,batch_first=True);self.delta=nn.Linear(64,52)
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
    def forward(self,x):return x[:,-1]+self.delta(self.lstm(x)[0][:,-1])

def gather(raw,runs,scaler,stride):
    endpoint=np.arange(19,raw.shape[2]-1,stride)
    xx=np.concatenate([raw[0,r][endpoint[:,None]+np.arange(-19,1)[None,:]] for r in runs])
    yy=np.concatenate([raw[0,r,endpoint+1] for r in runs])
    xx=((xx-scaler.mean_.astype(np.float32))/scaler.scale_.astype(np.float32)).astype(np.float32)
    yy=((yy-scaler.mean_.astype(np.float32))/scaler.scale_.astype(np.float32)).astype(np.float32)
    assert endpoint.max()+1<raw.shape[2]
    return xx,yy

def predict(model,x,device):
    model.eval();parts=[]
    with torch.inference_mode():
        for i in range(0,len(x),1024):parts.append(model(torch.from_numpy(x[i:i+1024]).to(device)).cpu().numpy())
    return np.concatenate(parts)

def train():
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    raw=np.load(CACHE/'training.npy',mmap_mode='r')
    normal=np.asarray(raw[0,SPLIT['train']]).reshape(-1,52)
    scaler=StandardScaler().fit(normal);joblib.dump(scaler,HERE/'models/scaler.joblib')
    x,y=gather(raw,SPLIT['train'],scaler,4);v,vy=gather(raw,SPLIT['validation'],scaler,1)
    persist_mse=float(np.square(v[:,-1]-vy,dtype=np.float64).mean())
    rows=[];history=[]
    for seed in CFG['seeds']:
        dest=HERE/'models'/f'lstm_s{seed}.pt'
        if dest.exists():rows.append(torch.load(dest,map_location='cpu',weights_only=False)['metadata']);continue
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        if device=='cuda':torch.cuda.manual_seed_all(seed)
        model=NextStepLSTM().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        lossfn=nn.HuberLoss(delta=1.)
        loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=256,shuffle=True,num_workers=0)
        best=np.inf;state=None;patience=0;start=time.perf_counter();best_epoch=0
        for epoch in range(1,13):
            model.train();loss_sum=0
            for xx,yy in loader:
                xx=xx.to(device);yy=yy.to(device);opt.zero_grad(set_to_none=True)
                loss=lossfn(model(xx),yy);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();loss_sum+=float(loss.detach())*len(yy)
            p=predict(model,v,device);mse=float(np.square(p-vy,dtype=np.float64).mean())
            history.append(dict(seed=seed,epoch=epoch,train_huber=loss_sum/len(x),validation_normalized_mse=mse,persistence_validation_mse=persist_mse))
            pd.DataFrame(history).to_csv(OUT/'tables/training_history.csv',index=False)
            print('Forecast epoch',seed,epoch,'MSE',round(mse,6),'persistence',round(persist_mse,6),flush=True)
            if mse<best-1e-6:
                best=mse;state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()};patience=0;best_epoch=epoch
            else:patience+=1
            if patience>=3:break
        row=dict(seed=seed,epoch=best_epoch,validation_normalized_mse=best,persistence_validation_mse=persist_mse,train_windows=len(x),validation_windows=len(vy),training_seconds=time.perf_counter()-start,device=device)
        torch.save({'state_dict':state,'metadata':row,'config':CFG},dest);rows.append(row)
        del model;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(OUT/'tables/validation_summary.csv',index=False)
    return scaler

def evaluate(scaler):
    raw=np.load(CACHE/'testing.npy',mmap_mode='r');device='cuda' if torch.cuda.is_available() else 'cpu'
    sensors=json.loads((CACHE/'sensor_names.json').read_text())
    all_summary=[];variables=[];cases=[]
    for seed in CFG['seeds']:
        model=NextStepLSTM().to(device);ck=torch.load(HERE/'models'/f'lstm_s{seed}.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict']);model.eval()
        modes=['common','dense'] if seed==42 else ['common']
        for mode in modes:
            endpoint=np.arange(19,959,10 if mode=='common' else 1);target_sample=endpoint+2
            total_count=0;start=time.perf_counter()
            acc={}
            for fault in range(21):
                for first in range(0,500,10):
                    runs=range(first,first+10)
                    w=np.concatenate([raw[fault,r][endpoint[:,None]+np.arange(-19,1)[None,:]] for r in runs])
                    target=np.concatenate([raw[fault,r,endpoint+1] for r in runs])
                    xx=((w-scaler.mean_.astype(np.float32))/scaler.scale_.astype(np.float32)).astype(np.float32)
                    actual=((target-scaler.mean_.astype(np.float32))/scaler.scale_.astype(np.float32)).astype(np.float32)
                    pred=predict(model,xx,device);naive=xx[:,-1];total_count+=len(pred)
                    sample=np.tile(target_sample,len(runs));post=(sample>160)&(fault>0)
                    for name,output in [('LSTM',pred),('Persistence',naive)]:
                        error=(output-actual).astype(np.float64)
                        for cohort,mask in [('all',np.ones(len(pred),bool)),('healthy_context',~post),('post_fault',post),('faultfree_only',np.full(len(pred),fault==0))]:
                            if not mask.any():continue
                            key=(name,cohort,fault)
                            if key not in acc:acc[key]=[0,np.zeros(52),np.zeros(52)]
                            acc[key][0]+=int(mask.sum());acc[key][1]+=np.square(error[mask]).sum(0);acc[key][2]+=np.abs(error[mask]).sum(0)
                    if seed==42 and mode=='dense' and first==0 and fault in [0,1,9,15]:
                        for j in [0,6,8,17,20,21]:
                            for k,t in enumerate(target_sample):
                                cases.append(dict(fault=fault,run=1,target_sample=int(t),variable=sensors[j],actual=float(actual[k,j]),lstm=float(pred[k,j]),persistence=float(naive[k,j]),unit='healthy Train std units'))
                print('Forecast Test',seed,mode,'Fault',fault,flush=True)
            for name in ['LSTM','Persistence']:
                for cohort in ['all','healthy_context','post_fault','faultfree_only']:
                    entries=[v for (n,c,f),v in acc.items() if n==name and c==cohort]
                    if not entries:continue
                    n=sum(v[0] for v in entries);sse=sum(v[1] for v in entries);sae=sum(v[2] for v in entries)
                    all_summary.append(dict(seed=seed,mode=mode,model=name,cohort=cohort,target_rows=n,normalized_rmse=float(np.sqrt(sse.sum()/(n*52))),normalized_mae=float(sae.sum()/(n*52)),evaluation_seconds=time.perf_counter()-start))
                    if seed==42 and mode=='dense':
                        for j,sensor in enumerate(sensors):variables.append(dict(model=name,cohort=cohort,variable=sensor,target_rows=n,normalized_rmse=float(np.sqrt(sse[j]/n)),normalized_mae=float(sae[j]/n),raw_rmse=float(np.sqrt(sse[j]/n)*scaler.scale_[j]),raw_mae=float(sae[j]/n*scaler.scale_[j]),raw_unit='UNKNOWN; B column units not independently verified'))
            pd.DataFrame(all_summary).to_csv(OUT/'tables/test_summary.csv',index=False)
            pd.DataFrame(variables).to_csv(OUT/'tables/per_variable_metrics.csv',index=False)
            pd.DataFrame(cases).to_csv(OUT/'tables/forecast_examples.csv',index=False)
        del model;torch.cuda.empty_cache()
    (OUT/'validation.json').write_text(json.dumps({'fit_run_families':400,'validation_run_families':100,'test_run_families':500,'same_original_split':True,'future_target_not_input':True,'input_window':20,'horizon_samples':1,'scaler_fit_normal_training_rows':int(scaler.n_samples_seen_),'cnn_modified':False,'forecast_values_fed_to_CNN':False,'test_selection_or_tuning':False},indent=2),encoding='utf-8')

if __name__=='__main__':
    (HERE/'config.json').write_text(json.dumps(CFG,indent=2),encoding='utf-8')
    evaluate(train())
