"""개발 장비만 사용하는 외부 그룹 검증과 별도 내부 조기종료·경보 보정."""
import argparse, gc, json, time, warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from src.improvement_common import *
from src.evaluate_models import episodes, choose_threshold, sample_metrics

KEEP=['Tool','time','row_id','segment_id','stage',TARGET]+[f'label_{c}_h600raw' for c in ['f1','f2','f3']]+[f'next_event_{c}' for c in ['f1','f2','f3']]

def fold_plan():
    tools=[f'{i:02}M{j:02}' for i in range(1,7) for j in [1,2]]
    plans=[]
    for fold in range(4):
        test=tools[fold::4];rest=[t for t in tools if t not in test]
        # 장비명만으로 고정하고 정답·성능을 이용해 재분할하지 않는다.
        shift=(fold*2)%len(rest);rest=rest[shift:]+rest[:shift]
        plan={'fold':fold,'evaluation':test,'early_stop':rest[:2],'calibration':rest[2:4],'train':rest[4:]}
        sets=[set(plan[k]) for k in ['train','early_stop','calibration','evaluation']]
        assert sum(map(len,sets))==len(set.union(*sets))==12
        plans.append(plan)
    return plans

def load_tools(tools):
    frame=pd.concat([pd.read_parquet(OUT/'cache'/f'{t}.parquet') for t in tools],ignore_index=True)
    assert (frame.source_max_time<=frame.time).all()
    assert np.isfinite(frame[configs()['C']].to_numpy()).all()
    return frame.sort_values(['Tool','time']).reset_index(drop=True)

def operational(frame,threshold):
    """NA를 포함한 예측 시간선을 유지하고 고유 이벤트와 조건부 FAR를 계산한다."""
    data=episodes(frame,threshold,600);detected={};eligible=set();proposals={};type_counts={}
    events=event_table();events=events[(~events.tagged)&events.Tool.isin(data.Tool.unique())]
    groups={t:g for t,g in data.groupby('Tool',sort=False)}
    for event in events.itertuples():
        group=groups[event.Tool]
        own=group[(group.time>=event.time-600)&(group.time<event.time)&(group[f'label_{event.fault_code}_h600raw']==1)&(group['next_event_'+event.fault_code]==event.event_key)]
        if len(own):eligible.add(event.event_key)
        for ep,g in own[own.alarm].groupby('episode_id'):
            proposals.setdefault(int(ep),[]).append((event.time,event.event_key,float(event.time-g.time.min())))
    matched=set()
    for ep,values in proposals.items():
        first=min(v[0] for v in values)
        for ft,key,lead in values:
            if ft==first:detected[key]=max(detected.get(key,0),lead);matched.add(ep)
    alarm=data[data.alarm]
    agg=alarm.groupby('episode_id').agg(rows=(TARGET,'size'),known=(TARGET,'count'),positive=(TARGET,'sum'),active=('stage','min'))
    agg['active']=agg.active>0
    unmatched=agg.loc[~agg.index.isin(matched)]
    false=unmatched[(unmatched.known==unmatched.rows)&(unmatched.positive==0)]
    unknown=unmatched[~unmatched.index.isin(false.index)]
    prev=data.shift();dt=data.time-prev.time
    exposure=(data.Tool==prev.Tool)&(data.segment_id==prev.segment_id)&(dt>0)&(dt<=72)&(data[TARGET]==0)&(prev[TARGET]==0)
    active=exposure&(data.stage>0)&(prev.stage>0)
    hours=float(dt[active].sum()/3600)
    for code in ['f1','f2','f3']:
        keys=[k for k in eligible if k.endswith(':'+code)]
        hits=[k for k in keys if k in detected]
        type_counts[code]={'eligible':len(keys),'detected':len(hits),'recall':len(hits)/len(keys) if keys else None}
    return {'eligible_events':len(eligible),'detected_events':len(detected),'event_recall':len(detected)/len(eligible) if eligible else None,
            'false_episodes':len(false),'active_false_episodes':int(false.active.sum()),'active_negative_hours':hours,
            'false_alarms_per_active_hour':float(false.active.sum()/hours) if hours else None,
            'unknown_episodes':len(unknown),'unknown_rows':int(data[TARGET].isna().sum()),
            'unknown_alarm_rows':int((data[TARGET].isna()&data.alarm).sum()),
            'lead_time_median_raw':float(np.median(list(detected.values()))) if detected else None,
            'per_type':type_counts,'lead_times':detected}

def metric(frame,threshold):
    known=frame[frame[TARGET].notna()]
    return {**sample_metrics(known[TARGET].astype(int),known.score,threshold),**operational(frame,threshold)}

def threshold_grid(frame):
    known=frame[frame[TARGET].notna()]
    best=choose_threshold(known[TARGET].astype(int),known.score)
    thresholds=sorted(set([0.,.1,.25,.5,.75,.9,.99,1.000001,best]+list(np.quantile(frame.score,[.5,.75,.9,.95,.98,.99,.995,.999]))))
    results=[{'threshold':float(t),**metric(frame,t)} for t in thresholds]
    feasible=[r for r in results if r['false_alarms_per_active_hour'] is not None and r['false_alarms_per_active_hour']<=.1 and r['event_recall'] is not None]
    # 사전 고정한 조건부 가동 FAR 0.1/h 제약 아래 Event Recall을 최대화한다.
    event=max(feasible,key=lambda r:(r['event_recall'],-r['unknown_episodes'],r['precision'] or 0,r['threshold'])) if feasible else None
    return best,event['threshold'] if event else 1.000001,results

def run(pilot=False):
    initialize();plans=fold_plan();save_json(OUT/'fold_plan.json',plans)
    hardware=read_json(OUT/'hardware.json');device=hardware['selected_device']
    for plan in plans[:1] if pilot else plans:
        fold=plan['fold'];frames={k:load_tools(plan[k]) for k in ['train','early_stop','calibration','evaluation']}
        counts=[]
        for role,f in frames.items():
            for tool,g in f.groupby('Tool'):
                counts.append({'role':role,'Tool':tool,'rows':len(g),'positive':int((g[TARGET]==1).sum()),'negative':int((g[TARGET]==0).sum()),'unknown':int(g[TARGET].isna().sum()),
                               **{c:int(g.loc[g[f'label_{c}_h600raw']==1,'next_event_'+c].nunique()) for c in ['f1','f2','f3']}})
        save_json(OUT/f'fold_{fold}_counts.json',counts)
        for name in (['A'] if pilot else ['A','B','C','C_regularized']):
            folder=OUT/('pilot' if pilot else 'cv')/f'fold_{fold}'/name;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'result.json').exists():continue
            cols=configs()[name[0]];known={k:frames[k][frames[k][TARGET].notna()] for k in ['train','early_stop']}
            y=known['train'][TARGET].to_numpy(dtype=np.int8)
            assert y.sum()>0 and known['early_stop'][TARGET].sum()>0
            params=dict(n_estimators=30 if pilot else 500,max_depth=3 if name=='C_regularized' else 4,
                        min_child_weight=30 if name=='C_regularized' else 10,reg_lambda=20 if name=='C_regularized' else 5,
                        reg_alpha=.1,learning_rate=.05,subsample=.8,colsample_bytree=.8,tree_method='hist',max_bin=128,
                        device=device,n_jobs=THREADS,random_state=42,scale_pos_weight=float((y==0).sum()/y.sum()),eval_metric='aucpr',early_stopping_rounds=30)
            x=matrix(known['train'],cols);xe=matrix(known['early_stop'],cols)
            progress(f'학습 시작 fold={fold}, {name}, {device}, {len(y):,}행')
            fallback=None
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                with Resources() as monitor:
                    try:model=XGBClassifier(**params).fit(x,y,eval_set=[(xe,known['early_stop'][TARGET].astype(int))],verbose=False)
                    except Exception as exc:
                        if device!='cuda':raise
                        fallback=repr(exc);params['device']='cpu'
                        model=XGBClassifier(**params).fit(x,y,eval_set=[(xe,known['early_stop'][TARGET].astype(int))],verbose=False)
                booster=json.loads(model.get_booster().save_config())
                model.save_model(folder/'model.ubj');save_json(folder/'booster_config.json',booster)
                # CPU 호스트 행렬 추론은 CPU로 명시해 디바이스 불일치 경고와 전송 비용을 피한다.
                model.set_params(device='cpu')
                preds={}
                for role in ['calibration','evaluation']:
                    f=frames[role];p=f[KEEP].copy();p['score']=model.predict_proba(matrix(f,cols))[:,1];preds[role]=p
                    p.to_parquet(folder/f'{role}_predictions.parquet',index=False,compression='zstd')
                train_score=model.predict_proba(x)[:,1]
            f1_threshold,event_threshold,grid=threshold_grid(preds['calibration'])
            evaluations={policy:metric(preds['evaluation'],t) for policy,t in [('f1',f1_threshold),('event_far',event_threshold)]}
            result={'fold':fold,'name':name,'columns':cols,'plan':plan,'params':params,'actual_device':booster['learner']['generic_param']['device'],
                    'best_iteration':model.best_iteration,'resources':monitor.result,'fallback':fallback,'warnings':[str(w.message) for w in caught],
                    'thresholds':{'f1':f1_threshold,'event_far':event_threshold},'calibration_grid':grid,'evaluation':evaluations,
                    'training':sample_metrics(y,train_score,f1_threshold),'importance':dict(zip(cols,model.feature_importances_)),'completed':True}
            save_json(folder/'result.json',result)
            progress(f'완료 fold={fold}, {name}: AP={evaluations["f1"]["pr_auc_ap"]:.4f}, F1={evaluations["f1"]["f1"]:.4f}, 학습 {monitor.result["seconds"]:.2f}초')
            del x,xe,model,preds;gc.collect()
        del frames;gc.collect()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();run(a.pilot)
