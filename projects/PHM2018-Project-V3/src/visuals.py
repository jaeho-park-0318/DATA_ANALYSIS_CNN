"""실제 집계와 원본 구간의 정적 진단 그림. 합성 데이터 사용 없음."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from .query import manifest,read_window,broken_line
from .detail import get_detail

font=Path('C:/Windows/Fonts/malgun.ttf')
if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=['Malgun Gothic','DejaVu Sans']
plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold','savefig.dpi':140})
FIGURES=ROOT/'outputs/figures'
REGISTRY=[]

def save(fig,rel,title,source,context,values=None):
    path=FIGURES/(rel+'.png');path.parent.mkdir(parents=True,exist_ok=True)
    fig.suptitle(title,fontsize=15,y=.985)
    fig.text(.02,.012,'출처: '+source+'\n'+context,fontsize=8,color='#475e6a',va='bottom')
    fig.tight_layout(rect=(.01,.08,.99,.95));fig.savefig(path);plt.close(fig)
    REGISTRY.append({'file':str(path.relative_to(ROOT)).replace('\\','/'),'title':title,'source':source,'context':context,'plot_values':values})
    return path

def heatmap(ax,data,xlabels,ylabels,title,vmin=0,vmax=1):
    im=ax.imshow(np.asarray(data,float),aspect='auto',vmin=vmin,vmax=vmax,cmap='YlGnBu')
    ax.set_xticks(range(len(xlabels)),xlabels,rotation=45,ha='right');ax.set_yticks(range(len(ylabels)),ylabels);ax.set_title(title)
    plt.colorbar(im,ax=ax,shrink=.8,label='비율')

def overview_figures():
    t=pd.read_parquet(TABLES/'tool_summary.parquet');ev=pd.read_parquet(TABLES/'events.parquet');lab=pd.read_parquet(TABLES/'label_status_summary.parquet');feat=pd.read_parquet(TABLES/'feature_availability_summary.parquet');seg=pd.read_parquet(TABLES/'time_segment_summary.parquet')
    tools=t.Tool.tolist();x=np.arange(len(t));context=f'전체 {len(t)}대 · 원본 {t.raw_rows.sum():,}행 · 고유 이벤트 {len(ev):,}개 · 샘플링 없음 · 시간 단위 raw'
    fig,ax=plt.subplots(figsize=(13,5));ax.bar(x,t.raw_rows,color='#087e80');ax.set_xticks(x,tools,rotation=45);ax.set_ylabel('원본 Sensor 행 수');ax.ticklabel_format(axis='y',style='plain')
    save(fig,'tool_overview/raw_rows','장비별 원본 데이터 규모','tool_summary.csv',context,{'Tool':tools,'raw_rows':t.raw_rows.tolist()})
    fig,ax=plt.subplots(figsize=(13,5));bottom=np.zeros(len(t))
    for k,c in zip(FAULTS,['#d55e00','#0072b2','#009e73']):ax.bar(x,t['fault_'+k],bottom=bottom,label=FAULT_KO[k],color=c);bottom+=t['fault_'+k]
    ax.set_xticks(x,tools,rotation=45);ax.set_ylabel('고유 고장 이벤트 수');ax.legend()
    save(fig,'tool_overview/fault_types','장비별 고장 유형 분포','events.csv / tool_summary.csv',context,{'fault_events':bottom.astype(int).tolist(),'Tool':tools})
    fig,ax=plt.subplots(figsize=(13,4));heatmap(ax,[t['ttf_null_ratio_'+k] for k in FAULTS],tools,[FAULT_KO[k] for k in FAULTS],'원본 행을 분모로 계산한 TTF Null 비율')
    save(fig,'tool_overview/ttf_null','장비·고장 유형별 TTF Null','tool_summary.csv',context)
    fig,ax=plt.subplots(figsize=(14,7));observed=[];gap_lines=[]
    for i,tool in enumerate(tools):
        q=seg[seg.Tool.eq(tool)&seg.kind.eq('continuous')]
        observed.extend([[(r.start,i),(r.end,i)] for r in q.itertuples()])
        g=pd.read_parquet(CACHE/'source'/tool/'gaps.parquet');g=g[g.long_gap]
        gap_lines.extend([[(r.start,i),(r.end,i)] for r in g.itertuples()])
    ax.add_collection(LineCollection(gap_lines,colors='#e1e6ea',linewidths=7,label='긴 공백'))
    ax.add_collection(LineCollection(observed,colors='#087e80',linewidths=5,label='관측 세그먼트'))
    for k,c in zip(FAULTS,['#d55e00','#0072b2','#b564ba']):
        q=ev[ev.fault_type.eq(k)];ax.scatter(q.fault_time,q.Tool.map({v:i for i,v in enumerate(tools)}),s=10,marker='|',color=c,label=FAULT_KO[k],zorder=4)
    ax.set_xlim(t.time_start.min(),t.time_end.max());ax.set_ylim(-1,len(tools));ax.set_yticks(x,tools);ax.set_xlabel('원시시간 (raw)');ax.ticklabel_format(axis='x',style='plain');ax.legend(ncol=5,loc='lower center',bbox_to_anchor=(.5,1.02))
    save(fig,'timeline/all_tools','전체 장비 관측·공백·고장 타임라인','time_segment_summary.csv / events.csv / gaps.parquet',context)
    fixed=seg[seg.kind.eq('fixed')].copy();fixed['bin']=(fixed.start//CFG['bin_width']).astype(int)
    density=fixed.pivot(index='Tool',columns='bin',values='density').reindex(tools)
    fig,ax=plt.subplots(figsize=(14,7));im=ax.imshow(density,aspect='auto',vmin=0,vmax=1,cmap='YlGnBu',extent=[density.columns.min()*CFG['bin_width'],(density.columns.max()+1)*CFG['bin_width'],len(tools)-.5,-.5]);ax.set_yticks(x,tools);ax.set_xlabel('원시시간 (raw)');fig.colorbar(im,ax=ax,label='고유 관측 수 / 기대 관측 수')
    save(fig,'timeline/density','고정 구간별 관측 밀도','time_segment_summary.csv',context+f' · 구간 폭 {CFG["bin_width"]:,} raw · 미관측=0, 범위 밖=빈칸')
    gapstats=[];fig,(ax,bx)=plt.subplots(1,2,figsize=(14,5))
    for tool in tools:
        m=manifest(tool);a=np.array(sorted((int(k),v) for k,v in m['dt_hist'].items() if int(k)>4),dtype=float)
        if not len(a):continue
        bins=np.geomspace(4,max(5,a[:,0].max()),40);hist,edges=np.histogram(a[:,0],bins=bins,weights=a[:,1]);ax.stairs(hist,edges,label=tool,alpha=.5)
        bx.plot(a[:,0],np.cumsum(a[:,1])/a[:,1].sum(),alpha=.65,label=tool)
        cumulative=np.cumsum(a[:,1]);q=np.searchsorted(cumulative,np.array([.5,.9,.99])*cumulative[-1]);gapstats.append({'Tool':tool,'gaps_above_4':int(cumulative[-1]),'p50':a[q[0],0],'p90':a[q[1],0],'p99':a[q[2],0],'max':a[-1,0]})
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('관측 간격 (raw, 로그)');ax.set_ylabel('간격 수 (로그)');bx.set_xscale('log');bx.set_xlabel('관측 간격 (raw, 로그)');bx.set_ylabel('누적 비율');bx.legend(ncol=4,fontsize=7)
    save(fig,'timeline/gap_distribution','장비별 공백 길이 Histogram과 누적분포','원본 시간 간격 전체 집계',context+' · 분모: 4 raw 초과 간격');save_table(pd.DataFrame(gapstats),'gap_summary')
    fig,ax=plt.subplots(figsize=(14,6))
    for i,tool in enumerate(tools):
        d=pd.read_parquet(CACHE/'source'/tool/'duplicates.parquet');ax.scatter(d.time,np.full(len(d),i),s=2,color='#97a6ad');c=d[d.state_conflict];ax.scatter(c.time,np.full(len(c),i),s=16,color='#d55e00',marker='x')
    ax.set_yticks(x,tools);ax.set_xlabel('원시시간 (raw)');ax.set_title('회색: 동시각 중복 · 주황: 공정 상태 충돌')
    save(fig,'timeline/duplicates_conflicts','중복과 공정 충돌 발생 위치','장비별 duplicates.parquet',context+' · recipe/stage/recipe_step/Lot/runnum 검사')
    for p in CFG['policies']:
        q=lab[lab.policy.eq(p)&lab.H.eq(600)&lab.fault_type.eq('any')&lab.sample_level.eq('raw_rows')].set_index('Tool').reindex(tools)
        fig,ax=plt.subplots(figsize=(14,5));bottom=np.zeros(len(q))
        for col,code in [('positive','POS'),('negative','NEG'),('NA','NA'),('excluded','EXCLUDE')]:
            v=q[col]/q.denominator;ax.bar(x,v,bottom=bottom,label={'POS':'양성','NEG':'정책상 음성','NA':'미확인','EXCLUDE':'제외'}[code],color=COLORS[code]);bottom+=v
        ax.set_xticks(x,tools,rotation=45);ax.set_ylabel('원본 행 대비 비율');ax.legend(ncol=4)
        save(fig,'label_analysis/'+p,'H=600 · '+POLICY_KO[p]+' 통합 라벨','label_status_summary.csv',context+' · L=60 · 가정 음성은 정상 확정이 아님',{'policy':p,'status_totals':{c:int(q[c].sum()) for c in ['positive','negative','NA','excluded']}})
    q=lab[lab.fault_type.eq('any')&lab.sample_level.eq('anchor')].groupby(['policy','H'])[['positive','negative','NA','excluded']].sum()
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,p in zip(axes,CFG['policies']):
        z=q.loc[p];z.div(z.sum(axis=1),axis=0).plot.bar(stacked=True,ax=ax,color=[COLORS[k] for k in ['POS','NEG','NA','EXCLUDE']],legend=False);ax.set_title(POLICY_KO[p]);ax.set_xlabel('H (raw)');ax.set_ylabel('Anchor 비율')
    axes[-1].legend(['양성','정책상 음성','NA','제외'],loc='lower center',fontsize=8)
    save(fig,'label_analysis/horizon_policies','H와 정책에 따른 통합 라벨 변화','label_status_summary.csv',context+' · 분모: 고정 격자 Anchor')
    q=lab[lab.H.eq(600)&lab.sample_level.eq('anchor')&lab.policy.eq('conservative')&~lab.fault_type.eq('any')].groupby('fault_type')[['positive','negative','NA','excluded']].sum()
    fig,ax=plt.subplots(figsize=(9,5));q.div(q.sum(axis=1),axis=0).rename(index=FAULT_KO).plot.bar(stacked=True,ax=ax,color=[COLORS[k] for k in ['POS','NEG','NA','EXCLUDE']]);ax.set_ylabel('Anchor 비율');ax.set_xlabel('고장 유형');ax.legend(['양성','음성','NA','제외'])
    save(fig,'label_analysis/fault_types','고장 유형별 보수적 라벨 상태','label_status_summary.csv',context+' · H=600, L=60 · Anchor 기준')
    nr=pd.read_parquet(TABLES/'ttf_label_relationship.parquet');fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,k in zip(axes,FAULTS):
        z=nr[nr.fault_type.eq(k)].groupby(['TTF_null','status']).anchors.sum().unstack(fill_value=0).reindex(index=[False,True],columns=[0,1,2,3],fill_value=0)
        im=ax.imshow(z.div(z.sum(axis=1).replace(0,np.nan),axis=0),vmin=0,vmax=1,cmap='YlGnBu',aspect='auto');ax.set_xticks(range(4),['음성','양성','NA','제외']);ax.set_yticks([0,1],['TTF 유한','TTF Null']);ax.set_title(FAULT_KO[k])
        for i in range(2):
            for j in range(4):ax.text(j,i,f'{int(z.iloc[i,j]):,}',ha='center',va='center',fontsize=7,color='#192d38',bbox={'facecolor':'white','alpha':.65,'edgecolor':'none'})
    save(fig,'label_analysis/ttf_null_relation','TTF Null과 라벨 NA는 같은 개념이 아니다','ttf_label_relationship.csv',context+' · Anchor TTF는 검증된 이벤트에서 재구성 · 보수적 H600')
    fig,ax=plt.subplots(figsize=(13,4));heatmap(ax,[t['available_'+f]/t.anchors for f in 'ABC'],tools,list('ABC'),'가용 Anchor / 각 장비 전체 Anchor')
    save(fig,'feature_availability/tool_heatmap','장비별 A/B/C 특징 가용성','tool_summary.csv',context+' · A8 / B32 / C44 · 독립 분기')
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5));q=feat[feat.policy.eq('conservative')&feat.H.eq(600)&feat.fault_type.eq('any')].groupby(['cohort','feature_set'])[['available_anchors','positive_events']].sum()
    for off,cohort,col in [(-.18,'maximum','#087e80'),(.18,'common','#9cc9c6')]:
        z=q.loc[cohort].reindex(list('ABC'));ax.bar(np.arange(3)+off,z.available_anchors,width=.35,label=cohort,color=col);bx.bar(np.arange(3)+off,z.positive_events,width=.35,label=cohort,color=col)
    for a in [ax,bx]:a.set_xticks(range(3),list('ABC'));a.legend(['최대 가용','공통'])
    ax.set_ylabel('가용 Anchor 수');bx.set_ylabel('양성 Anchor를 가진 고유 이벤트 수')
    save(fig,'feature_availability/cohorts','동일 코호트와 최대 가용 코호트 비교','feature_availability_summary.csv',context+' · 보수적 H600 · 동일 이벤트는 장비 내 1회만 계산')
    ex=pd.read_parquet(TABLES/'event_exclusion_reasons.parquet');fig,axes=plt.subplots(3,1,figsize=(13,9))
    for ax,p in zip(axes,CFG['policies']):
        z=ex[ex.policy.eq(p)&ex.H.eq(600)].groupby(['feature_set','Tool']).preserved.sum().unstack().reindex(index=list('ABC'),columns=tools)
        heatmap(ax,z/t.set_index('Tool').fault_events,tools,list('ABC'),POLICY_KO[p]+' · H600')
    save(fig,'event_preservation/tool_policy_heatmap','정책·특징별 고유 이벤트 보존률','event_exclusion_reasons.csv / tool_summary.csv',context+' · 분모: 태그 포함 원본 고유 이벤트')
    fig,ax=plt.subplots(figsize=(12,5));z=ex[ex.policy.eq('conservative')&ex.H.eq(600)&ex.feature_set.eq('C')].reason.value_counts();ax.barh(z.index,z.values,color='#748994');ax.set_xlabel('고유 이벤트 수')
    save(fig,'event_preservation/exclusion_reasons','C 특징에서 이벤트가 제외되는 이유','event_exclusion_reasons.csv',context+' · 보수적 H600 · 이벤트별 배타적 요약 이유')
    proc=pd.read_parquet(TABLES/'process_feature_summary.parquet');proc['condition']=proc.recipe.astype(str)+' / '+proc.stage.astype(str);proc=proc.groupby('condition')[['anchors','A_available','B_available','C_available']].sum().nlargest(20,'anchors')
    fig,ax=plt.subplots(figsize=(14,5));heatmap(ax,[1-proc[f+'_available']/proc.anchors for f in 'ABC'],proc.index.tolist(),list('ABC'),'Anchor 수 상위 20개 Recipe / Stage의 특징 불가 비율')
    save(fig,'feature_availability/process_conditions','공정 조건별 특징 가용성 차이','process_feature_summary.csv',context+' · 표시: Anchor 수 상위 20개 공정 조건; 전체 조건은 표 보존')
    pr=pd.read_parquet(TABLES/'preprocessing_preservation.parquet');fig,axes=plt.subplots(1,3,figsize=(16,6))
    for ax,f in zip(axes,'ABC'):
        z=pr[pr.feature_set.eq(f)&pr.policy.eq('conservative')&pr.H.eq(600)].groupby(['stage_order','stage','unit']).output_rows.sum().reset_index().sort_values('stage_order')
        ax.barh(z.stage,z.output_rows,color=['#93a4ad' if u!='Anchor' else '#087e80' for u in z.unit]);ax.set_xscale('log');ax.invert_yaxis();ax.set_title(f+' 독립 분기');ax.set_xlabel('단계별 단위의 수 (로그)')
        ax.set_xlim(100000,150000000);ax.set_xticks([1e5,1e6,1e7,1e8],['10만','100만','1천만','1억']);ax.xaxis.set_minor_formatter(NullFormatter())
        for i,v in enumerate(z.output_rows):ax.text(v*1.05,i,f'{v:,}',va='center',fontsize=8)
    save(fig,'event_preservation/stage_rows','전처리 단계별 행·시각·Anchor 수','preprocessing_preservation.csv',context+' · 분모가 바뀌는 단계는 제외율 계산 안 함')
    fig,axes=plt.subplots(1,3,figsize=(16,6))
    for ax,f in zip(axes,'ABC'):
        z=pr[pr.feature_set.eq(f)&pr.policy.eq('conservative')&pr.H.eq(600)].groupby(['stage_order','stage'])[['events_with_eligible_history','source_events']].sum().reset_index().sort_values('stage_order')
        ax.barh(z.stage,z.events_with_eligible_history,color='#087e80');ax.axvline(len(ev),color='#93a4ad',ls='--',label='원본 고유 이벤트');ax.invert_yaxis();ax.set_xlim(0,len(ev)*1.12);ax.set_title(f+' 독립 분기');ax.set_xlabel('고유 이벤트 수');ax.legend(fontsize=8,loc='lower right')
        for i,v in enumerate(z.events_with_eligible_history):ax.text(v+8,i,f'{v:,}',va='center',fontsize=8)
    save(fig,'event_preservation/stage_events','전처리 단계별 고유 이벤트 보존','preprocessing_preservation.csv',context+' · 보수적 H600 · 앞 단계: 경보 구간 센서 존재; 특징 단계: 같은 Anchor에서 조건 충족; 마지막: 사용 가능 양성 Anchor 존재')
    fig,ax=plt.subplots(figsize=(10,5));z=ex[ex.policy.eq('conservative')&ex.H.eq(600)].groupby('feature_set').preserved.sum().reindex(list('ABC'));ax.bar(['원본 고유 이벤트','A 현재값','B 300 이력','C 900 이력'],[len(ev),*z],color=['#93a4ad','#087e80','#36a09a','#83c1bd']);ax.set_ylabel('고유 이벤트 수')
    save(fig,'event_preservation/history_length','요구 과거 이력에 따른 이벤트 보존','events.csv / event_exclusion_reasons.csv',context+' · 보수적 H600 · A/B/C를 독립 비교')

def select_cases():
    events=pd.read_parquet(TABLES/'fault_event_summary.parquet');ex=pd.read_parquet(TABLES/'event_exclusion_reasons.parquet');cases=[]
    for tool in CFG['pilot_tools']:
        m=manifest(tool);own=events[events.Tool.eq(tool)]
        def add(category,t,event_id='',description='',start=None,end=None):
            cases.append({'case_id':f'{tool}-{len(cases)+1:02d}','Tool':tool,'category':category,'start':float(t-1800 if start is None else start),'end':float(t+120 if end is None else end),'event_id':event_id,'description':description})
        gaps=pd.read_parquet(CACHE/'source'/tool/'gaps.parquet');g=gaps.loc[gaps.gap_raw.idxmax()]
        add('긴 센서 공백',g.end,description=f'실제 공백 {int(g.start)}~{int(g.end)} raw, 길이 {int(g.gap_raw)} raw. 그림은 공백 끝 주변을 확대.',end=g.end+900)
        dup=pd.read_parquet(CACHE/'source'/tool/'duplicates.parquet');conf=dup[dup.state_conflict]
        if len(conf):add('동일 시각 공정 충돌',conf.time.iloc[0],description='recipe/stage/recipe_step/Lot/runnum 중 하나 이상이 같은 시각에 다름.',start=conf.time.iloc[0]-300,end=conf.time.iloc[0]+300)
        good=own[(own.last_sensor_lag<=12)&(own.coverage_pre300>=.9)&(~own.tagged)].sort_values('fault_time')
        if len(good):r=good.iloc[0];add('고장 직전 센서 존재',r.fault_time,r.event_id,'고장 전 300 raw 관측률이 90% 이상인 실제 이벤트.')
        a=ex[ex.Tool.eq(tool)&ex.policy.eq('conservative')&ex.H.eq(600)&ex.feature_set.eq('A')&ex.preserved]
        c=ex[ex.Tool.eq(tool)&ex.policy.eq('conservative')&ex.H.eq(600)&ex.feature_set.eq('C')&~ex.preserved]
        candidates=a[a.event_id.isin(c.event_id)].sort_values('fault_time')
        if len(candidates):r=candidates.iloc[0];add('고장은 있으나 장기 이력 부족',r.fault_time,r.event_id,'A 양성 후보는 있으나 C 양성 후보가 없는 실제 이벤트.')
        null_added=False;transition_added=False
        for item in m['normalized_index']:
            if null_added and transition_added:break
            d=pd.read_parquet(CACHE/'source'/tool/item['file'],columns=['time','process_change','ttf_low','ttf_high','ttf_leak'])
            if not null_added:
                q=d[d[['ttf_low','ttf_high','ttf_leak']].isna().sum(axis=1)>=2]
                if len(q):t=q.time.iloc[len(q)//2];add('TTF Null 집중',t,description='동일 시각에 2개 이상 유형의 제공 TTF가 Null. 정상이라는 의미는 아님.',start=t-900,end=t+900);null_added=True
            if not transition_added:
                q=d[d.process_change]
                if len(q):t=q.time.iloc[min(5,len(q)-1)];add('공정 전환과 센서 변화',t,description='실제로 관측된 공정 조건 전환. 값의 변화가 고장 전조인지 별도로 구분.',start=t-300,end=t+600);transition_added=True
        if not null_added:
            for item in reversed(m['normalized_index']):
                d=pd.read_parquet(CACHE/'source'/tool/item['file'],columns=['time','ttf_low','ttf_high','ttf_leak']);q=d[d[['ttf_low','ttf_high','ttf_leak']].isna().any(axis=1)]
                if len(q):t=q.time.iloc[len(q)//2];add('TTF Null 집중',t,description='최소 1개 유형의 제공 TTF Null 구간.',start=t-900,end=t+900);break
    dump_json(ROOT/'outputs/case_catalog.json',cases);save_table(pd.DataFrame(cases),'case_catalog');return cases

def detail_figure(d):
    fig,axes=plt.subplots(9,1,figsize=(13,17),sharex=True)
    for j,s in enumerate(SENSORS):
        z=d['series'][s];ax=axes[j];ax.plot(z['raw_x'],z['raw'],'.',ms=2,color='#8b979e',label='원본 센서')
        ax.plot(z['x'],z['normalized'],lw=.8,color='#087e80',label='동시각 평균');ax.plot(z['mean60']['x'],z['mean60']['y'],lw=1,color='#c17a32',label='과거60 평균');ax.set_ylabel(SENSOR_KO[j]+'\n익명화값');ax.legend(loc='upper left',fontsize=7,ncol=3)
        ax.ticklabel_format(axis='y',useOffset=False)
        if not d['raw_rows']:ax.text(.5,.5,'이 구간에 원본 센서 없음',transform=ax.transAxes,ha='center')
    for k,j in zip(FAULTS,range(3)):
        z=d['ttf'][k];axes[6].scatter(z['x'],np.full(len(z['x']),j),c=['#087e80' if v else '#929aa4' for v in z['valid']],s=3,marker='s')
    axes[6].set_yticks([0,1,2],[FAULT_KO[k]+' TTF' for k in FAULTS]);axes[6].set_ylabel('유한: 청록\nNull: 회색')
    for i,family in enumerate('ABC'):
        valid=[r[family+'_available'] for r in d['anchor_table']]
        axes[7].scatter(d['anchor_times'],np.full(len(valid),i),c=['#087e80' if v else '#929aa4' for v in valid],s=10,marker='s')
    axes[7].set_yticks(range(3),list('ABC'));axes[7].set_ylabel('특징 가용\n청록=가능')
    for i,k in enumerate(list(FAULTS)+['any']):
        v=d['label_strips'].get(d['policy'],{}).get(k,[]);axes[8].scatter(d['anchor_times'],np.full(len(v),i),c=[COLORS[x] for x in v],s=10,marker='s')
    axes[8].set_yticks(range(4),[FAULT_KO[k] for k in list(FAULTS)+['any']]);axes[8].set_ylabel('라벨 상태');axes[8].set_xlabel('원시시간 (raw)')
    for ax in axes:
        for g in d['gaps']:ax.axvspan(max(d['start'],g['start']),min(d['end'],g['end']),color='#ccd5dd',alpha=.45)
        if d['event']:
            tf=d['event']['fault_time'];ax.axvspan(tf-d['H'],tf-d['L'],color='#efc2aa',alpha=.35);ax.axvline(tf,color='#243b46',lw=1.3);ax.axvline(tf-d['L'],color='#d55e00',ls='--',lw=.8)
        for t in d['process_changes']:ax.axvline(t,color='#36a09a',alpha=.35,lw=.6,ls=':')
        ax.set_xlim(d['start'],d['end']);ax.ticklabel_format(axis='x',style='plain',useOffset=False);ax.grid(alpha=.15)
    return fig

def case_figures(cases):
    changes=[]
    for c in cases:
        d=get_detail(c['Tool'],c['start'],c['end'],c['event_id'] or None)
        fig=detail_figure(d);eid=c['event_id'] or '관련 고장 이벤트 없음'
        fault=FAULT_KO[d['event']['fault_type']] if d['event'] else '선택 고장 없음'
        context=f'{eid} · {fault} · {c["start"]:.0f}~{c["end"]:.0f} raw · 원본 {d["raw_rows"]:,}행 / 관측 {d["observations"]:,}개 / 범위 내 이벤트 {d["unique_events_in_range"]}개\n원본 해상도 100% · 회색=공백, 주황 음영=[tf−600,tf−60], 점선=선행시간/공정 전환 · 라벨: 양성 주황 / 음성 파랑 / NA 회색 / 제외 보라'
        save(fig,'fault_events/'+c['case_id'],c['Tool']+' · '+c['category'],d['source'],context)
        casepath=ROOT/'outputs/cases'/c['case_id'];casepath.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(d['anchor_table']).to_csv(casepath/'anchor_evidence.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(d['conditions']).to_csv(casepath/'process_conditions.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(d['sensor_summary']).to_csv(casepath/'sensor_changes.csv',index=False,encoding='utf-8-sig')
        changes.extend([dict(case_id=c['case_id'],Tool=c['Tool'],event_id=c['event_id'],**r) for r in d['sensor_summary']])
        pd.DataFrame(d['exclusions']).to_csv(casepath/'event_exclusion.csv',index=False,encoding='utf-8-sig')
        dump_json(casepath/'metadata.json',{k:v for k,v in d.items() if k not in ['series','anchor_table','label_strips','ttf']})
        # 실제 구간에서 모든 센서의 변화율/표준편차/단기-장기 차이를 확인한다.
        if c['category'] in ['고장 직전 센서 존재','공정 전환과 센서 변화']:
            fig,axes=plt.subplots(6,3,figsize=(15,16),sharex=True)
            for i,s in enumerate(SENSORS):
                for j,key in enumerate(['std60','rate','short_long']):
                    z=d['series'][s][key];axes[i,j].plot(z['x'],z['y'],lw=.8,color=['#087e80','#0072b2','#d55e00'][j]);axes[i,j].set_title(SENSOR_KO[i]+' · '+{'std60':'과거60 표준편차','rate':'변화율 / raw','short_long':'평균60−평균300'}[key],fontsize=9)
                    if d['event']:axes[i,j].axvline(d['event']['fault_time'],color='#333',lw=.6)
                    for t in d['process_changes']:axes[i,j].axvline(t,color='#36a09a',lw=.4,alpha=.3)
            for ax in axes[-1]:ax.set_xlabel('원시시간 (raw)')
            save(fig,'fault_events/'+c['case_id']+'_changes',c['Tool']+' · 센서 변화 진단',d['source'],context)
            n,_=read_window(c['Tool'],c['start'],c['end']);n['condition']=n.recipe.astype(str)+'/'+n.stage.astype(str)
            conditions=n.condition.value_counts().head(6).index.tolist();fig,axes=plt.subplots(2,3,figsize=(14,7))
            for ax,s in zip(axes.flat,SENSORS):
                vals=[n.loc[n.condition.eq(k),s].dropna() for k in conditions]
                if vals:ax.boxplot(vals,tick_labels=conditions,showfliers=False);ax.set_title(s,fontsize=9);ax.tick_params(axis='x',rotation=30)
            save(fig,'fault_events/'+c['case_id']+'_conditions',c['Tool']+' · 구간 내 공정 상태별 센서 분포',d['source'],context+' · 상위 6개 Recipe/Stage · 상자그림 이상점 표시는 생략, 값 삭제 없음')
    save_table(pd.DataFrame(changes),'case_sensor_changes')

def relative_profiles():
    events=pd.read_parquet(TABLES/'fault_event_summary.parquet');rows=[]
    for r in events[events.Tool.isin(CFG['pilot_tools'])&~events.tagged].itertuples():
        d,_=read_window(r.Tool,r.fault_time-1800,r.fault_time-60)
        if d.empty:continue
        ref=d.iloc[-1]
        if r.fault_time-60-ref.time>12 or ref.state_conflict or not np.isfinite([ref.recipe,ref.stage,ref.recipe_step]).all():continue
        q=d[d.recipe.eq(ref.recipe)&d.stage.eq(ref.stage)&d.recipe_step.eq(ref.recipe_step)&~d.state_conflict].copy()
        q['relative_bin']=np.floor((q.time-r.fault_time)/60).astype(int)*60
        z=q.groupby('relative_bin')[SENSORS].median().reset_index();z['Tool']=r.Tool;z['fault_type']=r.fault_type;z['recipe']=ref.recipe;z['stage']=ref.stage;z['recipe_step']=ref.recipe_step;z['event_id']=r.event_id
        rows.append(z)
    data=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    if data.empty:return
    save_table(data,'relative_event_profiles');groups=data.groupby(['Tool','fault_type','recipe','stage','recipe_step']).event_id.nunique().sort_values(ascending=False)
    selected=groups[groups>=3].head(6);summary=[]
    for key,count in selected.items():
        tool,k,recipe,stage,step=key;q=data[data.Tool.eq(tool)&data.fault_type.eq(k)&data.recipe.eq(recipe)&data.stage.eq(stage)&data.recipe_step.eq(step)]
        fig,axes=plt.subplots(2,3,figsize=(14,7))
        for ax,s in zip(axes.flat,SENSORS):
            g=q.groupby('relative_bin')[s];median=g.median();lo=g.quantile(.1);hi=g.quantile(.9);n=g.count()
            grid=np.arange(-1800,1,60);median=median.reindex(grid);lo=lo.reindex(grid);hi=hi.reindex(grid)
            ax.plot(grid,median,color='#087e80');ax.fill_between(grid,lo,hi,color='#98c9c4',alpha=.5);ax.axvline(0,color='#333',lw=.8);ax.set_title(s,fontsize=9);ax.set_xlabel('기록된 고장 기준 상대시간 (raw)');ax.set_ylabel('익명화 센서값')
            for t in n.index:summary.append({'Tool':tool,'fault_type':k,'recipe':recipe,'stage':stage,'recipe_step':step,'sensor':s,'relative_time':t,'contributing_events':int(n[t]),'median':g.median()[t],'q10':g.quantile(.1)[t],'q90':g.quantile(.9)[t]})
        save(fig,f'fault_events/relative_{tool}_{k}_{int(recipe)}_{int(stage)}_{int(step)}',f'{tool} · {FAULT_KO[k]} · Recipe={recipe:g}, Stage={stage:g}, Step={step:g} · 이벤트 정렬','relative_event_profiles.csv',f'고유 이벤트 {count}개 · 60 raw 구간 내 이벤트별 중앙값을 먼저 계산 후 이벤트 간 중앙값/10~90% · 보간 없음\n같은 Tool/고장/Recipe/Stage/Step만 비교 · 상대시간 0=기록된 정지 시각 · 고장 -60 raw 이후는 입력 비교 제외 · 전조/원인 입증 아님')
    if summary:save_table(pd.DataFrame(summary),'relative_event_quantiles')

def main():
    overview_figures();cases=select_cases();case_figures(cases);relative_profiles()
    dump_json(ROOT/'outputs/figure_manifest.json',REGISTRY)
    print(f'실제 데이터 그림 {len(REGISTRY)}개, 상세 사례 {len(cases)}개 생성')

if __name__=='__main__':main()
