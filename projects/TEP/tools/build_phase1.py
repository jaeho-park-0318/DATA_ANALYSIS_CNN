"""Build Korean Phase 1 reports, descriptive figures, and executable notebooks."""
from pathlib import Path
import sys, json, re, platform, importlib.metadata
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data.audit import series_hash

T=ROOT/'reports/tables'; R=ROOT/'reports/markdown'; F=ROOT/'reports/figures'
for p in [R,F,ROOT/'notebooks',ROOT/'src/features',ROOT/'src/models',ROOT/'src/evaluation',ROOT/'src/visualization',ROOT/'data/processed']: p.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':9,'figure.dpi':130})
SOURCES='''
## 출처와 검증 범위

- Dataset A: https://github.com/Blound04/tennessee-eastman-dataset (MIT). 로컬 README, simulator/auto_run.m, Mode_1_Init.m, MultiLoop_mode1.mdl, TEplot.m, temexd_mod.c를 대조했다. 로컬 checkout의 commit은 미확인이다. 다운로드한 Excel은 원래 LFS 포인터의 SHA256과 일치한다.
- Dataset B: https://doi.org/10.7910/DVN/6C3JR1 . 공식 API의 V1.0 메타데이터를 reports/sources/harvard_metadata.json에 보존했다. 배포 설명은 public-domain dedication과 면책 조건을 포함한다. 로컬 파일의 공식 MD5 일치 여부는 감사 표에 기록했다.
- 수치 결과는 로컬 원본의 재현 가능한 기술통계다. Phase 1에서 모델을 학습하거나 Test로 모델을 선택하지 않았다. 자료는 화학공정 시뮬레이션이며 실제 반도체 공정 검증 결과가 아니다.
'''

def write(name,text): (R/name).write_text(text+'\n'+SOURCES,encoding='utf-8')
def table(df):
    cols=list(df.columns)
    def fmt(v):
        if isinstance(v,float): return f'{v:.5g}'
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,cols))+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def variables():
    text=(ROOT/'tennessee-eastman-dataset-main/simulator/TEplot.m').read_text()
    names={int(i):s for i,s in re.findall(r"TEdata.title\{(\d+)\}='([^']+)';",text)}
    units={int(i):s for i,s in re.findall(r"TEdata.ylabel\{(\d+)\}='([^']+)';",text)}
    for i in range(23,29): names[i]=f'Component {"ABCDEF"[i-23]} to Reactor'
    for i in range(29,37): names[i]=f'Component {"ABCDEFGH"[i-29]} in Purge'
    for i in range(37,42): names[i]=f'Component {"DEFGH"[i-37]} in Product'
    rows=[]
    for i in range(1,42):
        device='Feed' if i<=6 else 'Reactor' if i<=9 else 'Purge' if i==10 else 'Separator' if i<=14 else 'Stripper' if i<=19 else 'Compressor' if i==20 else 'Reactor cooling' if i==21 else 'Condenser cooling' if i==22 else 'Reactor feed analyzer' if i<=28 else 'Purge analyzer' if i<=36 else 'Product analyzer'
        rows.append(dict(variable=f'xmeas_{i}',A_header=f'xmv-{i}',meaning=names.get(i,'UNKNOWN'),unit=units.get(i,'mol %' if i>=23 else 'UNKNOWN'),device=device,expected_response='관련 유량·온도·압력·조성 교란 및 제어 보상에 반응 가능(가설)',evidence='TEplot.m; temexd_mod.c Output 1'))
    mvnames={int(i):s for i,s in re.findall(r"TEmvs.title\{(\d+)\}='([^']+)';",text)}
    for i in range(1,13): rows.append(dict(variable=f'xmv_{i}',A_header='미포함',meaning=mvnames[i],unit='%',device='제어 밸브' if i<12 else 'Reactor agitator',expected_response='제어기 보상으로 변화 가능(가설)',evidence='TEplot.m; B는 xmv_1~11만 제공'))
    df=pd.DataFrame(rows); df.to_csv(T/'variable_dictionary.csv',index=False); return df

def faults():
    names=['A/C 원료 비율(B 일정)','B 조성(A/C 일정)','D 원료 온도','Reactor 냉각수 입구 온도','Condenser/Separator 냉각수 입구 온도','A 원료 공급 손실','C header 압력 손실','A/B/C 원료 조성','D 원료 온도','C 원료 온도','Reactor 냉각수 입구 온도','Condenser 냉각수 입구 온도','반응 동역학','Reactor 냉각수 밸브 고착','Condenser 냉각수 밸브 고착']+['UNKNOWN']*5+['정의 충돌: README 밸브 고정 / 소스 A 원료 온도']
    kinds=['Step']*7+['Random']*5+['Drift','Stiction','Stiction']+['UNKNOWN']*5+['CONFLICT']
    related=['Feed','Feed','Feed D','Reactor cooling','Condenser cooling','Feed A','Feed C','Feed','Feed D','Feed C','Reactor cooling','Condenser cooling','Reactor','Reactor cooling','Condenser cooling']+['UNKNOWN']*5+['UNKNOWN']
    possible=['xmeas_23~28','xmeas_23~28','xmeas_9','xmeas_9,21','xmeas_11,22','xmeas_1','xmeas_4','xmeas_23~28','xmeas_9','xmeas_9','xmeas_9,21','xmeas_11,22','xmeas_9,23~41','xmeas_9,21','xmeas_11,22']+['UNKNOWN']*6
    rows=[]
    for i in range(21):
        rows.append(dict(fault=i+1,cause=names[i],form=kinds[i],device=related[i],direct_measured_variable='xmeas_1' if i==5 else 'UNKNOWN',indirect_candidates=possible[i],candidate_status='공정 가설이며 원인 적중 정답으로 사용 불가',definition_status='문서/소스 일치' if i<15 else '원인 미공개; 소스 구현 단서는 별도' if i<20 else '충돌; 격리 필요'))
    df=pd.DataFrame(rows);df.to_csv(T/'fault_catalog.csv',index=False);return df

def figures(a,stats,vd):
    base=np.load(ROOT/'data/interim/github/mode1_normal_50.npz')
    # Warm-up exclusion is an analysis choice, not a documented injection boundary.
    normal=base['values'][base['time']>=5]
    mu=normal.mean(axis=0); sd=normal.std(axis=0,ddof=1); sd[sd==0]=np.nan
    pd.DataFrame({'variable':vd.variable[:41],'reference_mean':mu,'reference_std':sd}).to_csv(T/'a_mode1_reference.csv',index=False)
    responses=[]; onsets=[]
    for fault in range(1,22):
        files=a[(a.fault==fault)&(a['mode']==1)]
        run_effect=[]
        unique=[]; seen=set()
        for _,r in files.iterrows():
            cache=np.load(ROOT/'data/interim/github'/ (Path(r.path).stem+'.npz'))
            x=cache['values'];t=cache['time']; h=series_hash(x)
            if h in seen: continue
            seen.add(h);unique.append(Path(r.path).stem)
            effect=(x[t>=10].mean(axis=0)-mu)/sd
            run_effect.append(effect)
        effect=np.nanmean(run_effect,axis=0)
        top=np.argsort(np.abs(effect))[-5:][::-1]
        for j in range(41): responses.append(dict(fault=fault,variable=f'xmeas_{j+1}',standardized_mean_difference=effect[j],unique_series=len(unique),reference='mode1_normal_50, t>=5h',window='fault file t>=10h (model nominal injection)',causal=False))
        r=files.sort_values('run').iloc[0]
        cache=np.load(ROOT/'data/interim/github'/(Path(r.path).stem+'.npz'))
        t=cache['time'];x=cache['values']; z=(x-mu)/sd
        smooth=pd.DataFrame(z).rolling(30,min_periods=30).mean().to_numpy()
        starts=[]
        for j in range(41):
            exceed=np.abs(smooth[:,j])>3
            sustained=np.convolve(exceed.astype(int),np.ones(10,dtype=int),'valid')==10
            ids=np.flatnonzero(sustained & (t[:len(sustained)]>=10))
            onset=float(t[ids[0]+9]) if len(ids) else np.nan
            starts.append(onset)
            onsets.append(dict(fault=fault,run=1,variable=f'xmeas_{j+1}',first_sustained_deviation_hour=onset,definition='30 sample trailing mean |z|>3 for 10 consecutive samples; time at confirmation; search t>=10h',nominal_injection_hour=10,nominal_delay_minutes=(onset-10)*60,verified_injection_hour='UNKNOWN',detection_delay='NOT_VALIDATED'))
        fig,axes=plt.subplots(3,2,figsize=(14,12),layout='constrained')
        for j in top[:3]:
            axes[0,0].plot(t,x[:,j],label=f'xmeas_{j+1} ({vd.iloc[j].unit})',linewidth=.8)
            axes[0,1].plot(t,z[:,j],label=f'xmeas_{j+1}',linewidth=.8)
        axes[0,0].set(title='주요 센서 원 단위 시계열 (혼합 단위)',xlabel='시뮬레이션 시간 (h)')
        axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
        for ax in axes[0]: ax.axvline(10,color='#555555',linestyle='--',linewidth=1)
        axes[0,1].set(title='정상 대비 표준화 편차',xlabel='시뮬레이션 시간 (h)',ylabel='z (무차원)')
        corr=pd.DataFrame(x[t>=5]).corr().to_numpy()-pd.DataFrame(normal).corr().to_numpy()
        im=axes[1,0].imshow(corr,vmin=-2,vmax=2,cmap='RdBu_r');fig.colorbar(im,ax=axes[1,0],label='상관계수 차이')
        axes[1,0].set(title='정상 대비 상관관계 변화 (41개 측정값)',xlabel='xmeas 번호 - 1',ylabel='xmeas 번호 - 1')
        first=x[(t>=5)&(t<10)].mean(axis=0);last=x[t>=10].mean(axis=0)
        axes[1,1].bar(np.arange(1,42),(last-first)/sd,color='#3979a5')
        axes[1,1].set(title='명목 10h 주입 전(5~10h)/후(10h~) 변화',xlabel='xmeas 번호',ylabel='평균 차이 / 정상 표준편차')
        valid=np.isfinite(starts)
        axes[2,0].scatter(np.arange(1,42)[valid],np.array(starts)[valid],s=14)
        axes[2,0].set(title='지속 편차 최초 확인 시각 (미관찰 변수 제외)',xlabel='xmeas 번호',ylabel='시간 (h), 실제 주입 기준 지연 미검증')
        axes[2,1].barh([f'xmeas_{j+1}' for j in top][::-1],effect[top][::-1],color='#266c86')
        axes[2,1].set(title=f'고유 시계열 {len(unique)}개 평균의 주요 반응',xlabel='정상 대비 표준화 평균 차이')
        fig.suptitle(f'Dataset A | Mode 1 | Fault {fault} | 대표 run 1 | 공식 Train/Test 없음\n출처: GitHub Excel, 정상 참조 t≥5h. 점선 10h는 배포 모델 명목 주입, 파일별 기록 미확인',fontsize=12)
        fig.savefig(F/f'fault_{fault:02d}_analysis.png');plt.close(fig)
    response=pd.DataFrame(responses);response.to_csv(T/'a_fault_response.csv',index=False)
    pd.DataFrame(onsets).to_csv(T/'a_deviation_onsets.csv',index=False)
    matrix=response.pivot(index='fault',columns='variable',values='standardized_mean_difference').reindex(columns=[f'xmeas_{i}' for i in range(1,42)])
    fig,ax=plt.subplots(figsize=(15,7),layout='constrained')
    im=ax.imshow(np.clip(matrix,-10,10),aspect='auto',cmap='RdBu_r',vmin=-10,vmax=10)
    ax.set_xticks(range(41),range(1,42),fontsize=7);ax.set_yticks(range(21),range(1,22))
    ax.set(xlabel='xmeas 번호',ylabel='Fault ID (21 정의 충돌)',title='Dataset A / Mode 1 / 고유 시계열 평균 반응 / 공식 Train-Test 없음\n정상 참조 mode1_normal_50 t≥5h, 고장 파일 t≥10h(명목), 출처: GitHub Excel')
    fig.colorbar(im,ax=ax,label='표준화 평균 차이 (표시 범위 ±10, 무차원)')
    fig.savefig(F/'fault_response_overview.png');plt.close(fig)
    return response

def build():
    a=pd.read_csv(T/'a_inventory.csv'); stats=pd.read_csv(T/'a_variable_statistics.csv'); ah=pd.read_csv(T/'a_run_hashes.csv')
    vd=variables(); fc=faults(); response=figures(a,stats,vd)
    group=a.groupby(['mode','fault']).agg(files=('path','count'),samples=('rows','sum')).reset_index()
    unique=ah.groupby(['mode','fault']).full.nunique().reset_index(name='unique_series')
    counts=group.merge(unique);counts.to_csv(T/'a_class_counts.csv',index=False)
    duplicates=ah[ah.duplicated('full',keep=False)].sort_values('full');duplicates.to_csv(T/'a_duplicate_series.csv',index=False)
    overlaps=[]
    from itertools import combinations
    for _,g in ah.groupby('prefix20'):
        for left,right in combinations(g.itertuples(),2):
            if left.rows==right.rows: continue
            lx=np.load(ROOT/'data/interim/github'/f'{left.file}.npz')['values']
            rx=np.load(ROOT/'data/interim/github'/f'{right.file}.npz')['values']
            n=min(len(lx),len(rx))
            if np.array_equal(lx[:n],rx[:n]): overlaps.append(dict(left=left.file,right=right.file,shared_prefix_rows=n))
    pd.DataFrame(overlaps,columns=['left','right','shared_prefix_rows']).to_csv(T/'a_partial_overlaps.csv',index=False)
    # Preserve a full checkout inventory, including pointers and non-data resources.
    from src.data.audit import digest
    checkout=[]
    for p in sorted((ROOT/'tennessee-eastman-dataset-main').rglob('*')):
        if p.is_file():
            first=p.open('rb').read(200)
            checkout.append(dict(path=str(p.relative_to(ROOT)),format=p.suffix,bytes=p.stat().st_size,sha256=digest(p),lfs_pointer=first.startswith(b'version https://git-lfs.github.com/spec/v1')))
    pd.DataFrame(checkout).to_csv(T/'supplied_checkout_inventory.csv',index=False)
    audits=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(T.glob('TEP_*_audit.json'))]
    b=pd.concat([pd.read_csv(p) for p in sorted(T.glob('TEP_*_runs.csv'))],ignore_index=True)
    b.to_csv(T/'b_all_runs.csv',index=False)
    bdup=b[b.duplicated('full',keep=False)];bdup.to_csv(T/'b_duplicate_series.csv',index=False)
    prefix_groups=b.groupby('prefix20').agg(trajectories=('run','count'),distinct_run_ids=('run','nunique'),files=('file',lambda s:','.join(sorted(set(s)))),faults=('fault',lambda s:','.join(map(str,sorted(set(s)))))).reset_index()
    prefix_groups.to_csv(T/'b_prefix_groups.csv',index=False)
    train_prefix=set(b[b.file.str.contains('Training')].prefix20)
    test_prefix=set(b[b.file.str.contains('Testing')].prefix20)
    shared_prefix_count=len(train_prefix & test_prefix)
    # Compare full measured trajectories for equal lengths, plus first 20 samples under candidate column correspondence.
    cross=ah.merge(b,left_on=['rows','full'],right_on=['rows','measured_full'],suffixes=('_a','_b'))
    cross.to_csv(T/'cross_dataset_exact_matches.csv',index=False)
    # Sampling differs; compare both possible regular 3-minute phases and documented B run lengths.
    candidates=[]
    bhashes=set(b.measured_full); bprefix=set(b.measured_prefix20)
    for p in sorted((ROOT/'data/interim/github').glob('*.npz')):
        x=np.load(p)['values']
        for offset in range(3):
            down=x[offset::3]
            for n in [500,960]:
                if len(down)>=n:
                    candidates.append(dict(file=p.stem,offset_minutes=offset,rows=n,full_match=series_hash(down[:n]) in bhashes,prefix20_match=series_hash(down[:20]) in bprefix))
    cand=pd.DataFrame(candidates);cand.to_csv(T/'cross_dataset_downsample_checks.csv',index=False)
    bs=pd.DataFrame([dict(file=v['file'],rows=v['rows'],columns=len(v['columns']),run_groups=v['run_groups'],md5_verified=v['checksum_match'],missing=v['missing']) for v in audits])
    bs.to_csv(T/'b_inventory.csv',index=False)
    bc=b.groupby(['file','fault']).agg(run_groups=('run','count'),samples=('rows','sum'),unique_series=('full','nunique')).reset_index();bc.to_csv(T/'b_class_counts.csv',index=False)
    write('01_dataset_audit.md',f'''# Phase 1 데이터 감사

검사 기준일: 2026-11-10 (Asia/Seoul). 폴더는 Dataset A, 프로젝트 루트 RData는 Dataset B다. 원본을 이동하거나 수정하지 않았다.

## Dataset A 확보와 구조

사용자 폴더의 Excel 214개는 모두 Git LFS 포인터였다. 공식 media.githubusercontent.com에서 실제 데이터를 data/raw/github에 복원했고 모든 포인터 SHA256을 대조했다. reports/sources/github_download.json에 결과를 남겼다. 재다운로드: `python tools/fetch_dataset_a.py`. 수동 경로는 공식 GitHub 각 파일의 Raw 다운로드 또는 Git LFS가 설치된 환경에서 `git lfs pull`이다.

전체 {a.rows.sum():,}행, 42열(Time (h) + 41 측정값), 시각 0부터 시작, 실제 저장 주기 1분이다. 원본 헤더 xmv-1~41은 auto_run.m의 잘못된 명명이며 simout 측정 출력이다. 별도 조작변수 xmv와 교란 플래그 idv는 Excel에 없다. 파일 경로에서 mode/fault/run을 추출했다. 공식 Train/Test 분할은 없다.

{table(counts)}

README의 고장별 5개/105개 설명과 달리 실제 고장 파일은 210개(21×10)다. Fault 6은 약 17.05~17.15h에 끝나며 종료 이유는 파일만으로 미확인이다. 정상 500h 파일은 30,001행이며 나머지 완전한 50h 파일은 3,001행이다.

## 품질과 독립성

결측 {a.missing.sum()}개, 비유한값 {a.nonfinite.sum()}개, 시간 역전/중복 {a.nonmonotonic.sum()}개, 파일 내부 전체 측정값 동일 중복행 {a.duplicate_sensor_rows.sum()}개다. 상수 변수와 음수값은 a_inventory.csv, a_variable_statistics.csv에 기록했다. 물리적 범위 위반은 확정하지 않았으며 음수만으로 제거하지 않았다. 예를 들어 steam flow의 음수 측정은 잡음 또는 측정 정의 확인이 필요하다.

전체 측정 시계열 SHA256 고유 수는 {ah.full.nunique()}개로, 파일 수 214와 다르다. 중복 추가 파일은 {ah.full.duplicated().sum()}개다. 이번 정확 일치 중복은 모두 같은 Fault 내부에서 발견됐으며 서로 다른 Fault 간 전체 시계열 일치는 없었다. 길이가 다른 파일의 짧은 쪽 전체가 긴 쪽 선두와 일치하는 쌍은 {len(overlaps)}개다(a_partial_overlaps.csv). 고유 시계열 수도 난수 독립성을 보장하는 실험 횟수는 아니다. 원래 seed 로그가 없어 진정한 독립 실행 수는 UNKNOWN이다. 모델 분할 전 중복/공유 구간 그룹을 통째로 묶어야 한다.

## Dataset B 원본 검증

{table(bs)}

공식 배포는 faultNumber, simulationRun, sample과 52개 공정 변수(41 xmeas + 11 xmv)다. 공식 설명에 따르면 각 고장/분할에 500개 RNG 상태가 있고 Train/Test seed는 겹치지 않는다. simulationRun 숫자 자체는 분할별로 재사용된다. 또한 서로 다른 고장 사이의 같은 번호가 독립 seed라는 증거는 없다. 파일/고장/실행별 샘플 수와 해시는 b_all_runs.csv에 있다.

전체 B 실행 그룹 {len(b):,}개, 전체 {bs.rows.sum():,}행. 실행별 비유한값 합 {b.nonfinite.sum()}, sample 연속성 오류 {b.sample_errors.sum()}, 중복 sample 키 {b.duplicate_keys.sum()}다. 전체 52변수 동일 시계열의 추가 중복 수 {b.full.duplicated().sum()}개. 공식 seed 설명과 실측 중복 검사는 독립성의 서로 다른 증거다.

B의 선두 20개 샘플 전체 52변수 해시는 {b.prefix20.nunique()}개뿐이며 추가 중복은 {b.prefix20.duplicated().sum():,}개다. Train/Test 간 공유 prefix 해시는 {shared_prefix_count}개다. 동일 분할의 같은 simulationRun은 정상 및 여러 Fault에서 동일한 주입 전 궤적을 공유하므로 21,000개 궤적을 모두 독립 실험으로 취급하면 안 된다. 실행 분할은 같은 분할의 simulationRun 전체를 함께 묶어야 한다. 이 해시 결과는 난수 seed 원값을 복원한 것이 아니다.
''')
    normal_stats=stats[stats.fault==0][['file','mode','variable','mean','std','minimum','maximum']]
    normal_stats.to_csv(T/'a_normal_mode_statistics.csv',index=False)
    bnormal=pd.read_csv(T/'TEP_FaultFree_Training_run1_statistics.csv')
    av=stats[stats.file=='mode1_normal_50'][['variable','mean','std']]
    baseline=av.merge(bnormal[['variable','mean','std']],on='variable',suffixes=('_A','_B'))
    baseline['difference_over_B_run1_sd']=(baseline.mean_A-baseline.mean_B)/baseline.std_B
    baseline.to_csv(T/'candidate_normal_distribution_comparison.csv',index=False)
    write('02_process_variables.md',f'''# 공정과 변수 사전

원료 A/C/D/E가 Reactor로 들어가고 반응 혼합물은 Condenser에서 냉각된다. Separator가 기상과 액상을 나누며, 기상은 purge 및 Compressor recycle 경로로 이어진다. 액상은 Stripper에서 분리되어 제품으로 나간다. 냉각수와 steam, 각 유량 제어 밸브가 온도·압력·수위·조성을 조절한다. 아래 도식은 배포 시뮬레이터의 장치/stream 정의를 요약한 개념도이며 상세 배관도는 아니다.

```mermaid
flowchart LR
 Feed[원료 공급] --> Reactor
 Reactor --> Condenser --> Separator
 Separator --> Compressor --> Reactor
 Separator --> Purge
 Separator --> Stripper --> Product
 Cooling[냉각수] -.-> Reactor
 Cooling -.-> Condenser
 Steam -.-> Stripper
```

{table(vd)}

kscmh는 원문 표기를 보존했다. xmv_12는 로컬 시뮬레이터에 있지만 Harvard에는 없다. 변수 사전은 Dataset A 배포 소스의 정의이며 Harvard의 동일 단위·보정·운전 조건을 자동 확정하지 않는다. 예상 반응은 관측 결과와 별개의 물리적 가설이다.
''')
    write('03_fault_catalog.md',f'''# 고장 시나리오

{table(fc)}

Fault 16~20은 공개 카탈로그 기준 원인 UNKNOWN으로 유지한다. 소스 주석에는 16~19 구현 단서가 있지만 이를 공통 벤치마크의 공개 원인 정답으로 채택하지 않았다. Fault 21의 정의 충돌은 특히 심각하다. README는 Stream 4 valve fixed이고 auto_run.m은 dist(21)=1을 설정하지만 temexd_mod.c의 IDV(21)은 A feed temperature random disturbance다. 배포 MEX와 생성 시점 코드의 일치도 확인되지 않아 Fault 21의 실제 물리 원인을 확정할 수 없다. Dataset B에는 21이 없다.

직접 고장 원인인 냉각수 입구 온도는 xmeas_21/22(출구 온도)와 다르다. 밸브 고착도 출력 온도와 같은 개념이 아니다. 위 간접 후보는 진단 가설이며 정량 원인 적중률의 정답표로 사용할 수 없다.
''')
    top=response.assign(abs_effect=response.standardized_mean_difference.abs()).sort_values(['fault','abs_effect'],ascending=[True,False]).groupby('fault').head(3)
    write('04_fault_sensor_response.md',f'''# 고장별 센서 반응

## 방법과 해석 범위

Mode 1 정상 50h 중 t≥5h를 기술통계 참조로 사용했다. 5h 제외는 초기 과도 영향을 줄이기 위한 분석 선택이다. 모델 전처리 학습이나 경보 임계값 최적화가 아니다. 고장별 t≥10h(모델 명목 주입 이후) 평균을 정상 평균/표준편차로 표준화했고 같은 Fault 내 완전히 동일한 시계열을 하나로 세어 실행별 평균에 동일 가중치를 주었다. 다른 고장 간 중복은 별도 검사했다. 기준 정상 실행이 하나여서 신뢰구간과 인과 추론을 제공하지 않는다.

상관관계 차이(t≥5h 전체)와 대표 run 1의 시계열, 정상 대비 편차, 명목 주입 전 5~10h/후 10h~ 비교 및 지속 편차 최초 확인 시각을 각 Fault에 대해 생성했다. 서로 다른 단위 원시 시계열은 크기를 직접 비교할 수 없다. 단위 비교는 z 패널을 사용한다.

고장 주입 시각은 Excel에 없다. 배포 모델 배선을 확인하면 Constant dist(SID 3)가 VariableTransportDelay(SID 250)를 지나 TE Plant 입력 13으로 들어가고, Constant 10(SID 249)이 지연 입력에 연결된다. 따라서 **배포 모델의 명목 주입은 10h**다. auto_run.m에서 dist를 미리 설정하는 것과 실제 Plant 적용 시각은 다르다. 시계열에서 10h 이후 변화와 일치하는 사례가 있지만 모든 파일의 생성 시점 모델과 idv 로그는 없으므로 10h를 파일별 확정 사실로 취급하지 않았다. 그림의 점선은 이 명목 기준이다.

지속 편차 시각은 30개 trailing 평균의 |z|>3이 10개 연속 관찰된 확인 시각(t≥10h 검색)이다. 10번째 샘플에서 경보를 확인하도록 시간을 기록했다. 명목 10h 대비 차이를 a_deviation_onsets.csv에 분 단위로 제공하지만 주입 시각이 검증된 실제 탐지 지연으로 해석하면 안 된다. 명목 경계부터 조건을 만족하면 이전 반응은 좌측 검열된다. 이 고정 기술통계 규칙을 Phase 2 경보 임계값으로 채택한 것은 아니다.

## 주요 반응 (모델 중요도나 물리 원인이 아님)

{table(top[['fault','variable','standardized_mean_difference','unique_series']])}

![고장별 반응](../figures/fault_response_overview.png)

'''+ '\n'.join(f'- [Fault {i} 상세 6개 패널](../figures/fault_{i:02d}_analysis.png)' for i in range(1,22)))
    compatibility=pd.DataFrame([
        ['변수 개수','41 measured','41 measured + 11 manipulated','불일치'],
        ['변수 순서','Time, xmv-1~41 (실제 simout)','3 metadata, xmeas_1~41, xmv_1~11','부분 대응; 헤더 수정 근거 필요'],
        ['변수 정의','로컬 소스의 측정 정의','원래 변수명 유지; 상세 단위표 없음','공통 계열이나 동일 조건 미확정'],
        ['샘플링 주기','1분 (실측)','3분 (공식)','불일치'],
        ['Fault ID','1~21, 21 충돌','1~20','21 제외; 16~20 원인 UNKNOWN'],
        ['운전 모드','1,3,4; 고장은 1만','공식 API에 명시 없음','미확인'],
        ['주입 시각','모델 명목 10h; 파일별 미확인','Training 1h / Testing 8h (공식)','불일치'],
        ['시뮬레이터 조건','개정 C, Simulink; batch seed rand()','시드 분할 설명, 제어기/빌드 미기재','동일성 미확인'],
        ['독립성','중복 시계열 존재, seed 로그 없음','Train/Test seeds non-overlap (공식)','A-B 독립성 확증 불가']
    ],columns=['비교 항목','GitHub A','Harvard B','판정']);compatibility.to_csv(T/'compatibility.csv',index=False)
    write('05_compatibility_report.md',f'''# 두 데이터셋의 호환성

{table(compatibility)}

전체 측정값의 동일 길이·동일 순서·float64 정확 일치 시계열 비교에서 A-B 일치 {len(cross)}건이다. A를 3분 간격으로 phase 0/1/2분에서 추출하고 선두 500/960개를 B와 비교한 {len(cand)}개 후보 중 전체 일치 {int(cand.full_match.sum())}건, 선두 20개 일치 {int(cand.prefix20_match.sum())}건이다. 비교는 후보 xmeas 대응에만 사용했으며 두 데이터를 합친 학습 테이블을 만들지 않았다. 수치 정밀도/시작 시각/서로 다른 제어기 때문에 불일치는 독립 실행의 증명이 아니다. 모든 sliding window, 근사 일치, 동적 정합을 검사한 것은 아니다.

B는 공식 설명에 따라 Training 20개 샘플/Testing 160개 샘플까지 고장 주입 전이다(3분 간격). 이후를 명목 고장 구간으로 삼을 수 있으나 주입 경계의 정확한 기록 규약은 별도 확인해야 한다. faultNumber가 파일 전체에 부여되어도 주입 전 센서를 실제 고장 상태로 오인하면 안 된다. sample*3분 시간축은 공식 25/48h 길이와 일치하는 해석이며 첫 센서 기록이 t=0인지 3분인지에 대한 원 생성 로그는 없다.

A는 공정 이해와 가설 수립에 사용하고 B를 별도의 모델 개발 데이터로 사용하는 것은 가능하다. 두 자료의 행을 직접 합치거나 A 고장 반응을 B 원인 정답으로 사용하는 것은 현재 근거로 허용되지 않는다. B의 Test 기술통계를 탐색했으므로 향후 모델 선택에는 사용하지 않고 별도 train 실행 holdout 또는 새로운 검증 실행으로 분석 선택의 영향을 관리해야 한다.

A 정상 50h 전체와 B 정상 Training run 1의 공통 이름 후보에 대한 평균/표준편차 차이는 candidate_normal_distribution_comparison.csv에 기록했다. 기간·주기·제어 조건이 다른 기술통계이며 동일 분포 가설 검정이나 단위 동일성 검증을 대체하지 않는다. Mode 1/3/4 정상 통계는 a_normal_mode_statistics.csv에 분리했다.
''')
    write('06_phase1_summary.md',f'''# Phase 1 종합 보고서

## 연구 배경과 확보 데이터

TEP 시뮬레이션의 공정 이상 탐지 및 고장 진단을 준비하기 위해 데이터 품질과 버전 차이를 조사했다. Dataset A는 사용자 제공 폴더이며 LFS 원본 214개를 복원·검증했다. Dataset B는 사용자 제공 RData 4개다. 공식 V1.0 MD5가 모두 일치하는지 확인했다: {all(v['checksum_match'] for v in audits)}.

A는 {int(a.rows.sum()):,}개 샘플, 41개 측정값, 1분 간격이다. B는 {int(bs.rows.sum()):,}개 샘플, 52개 공정 변수, 3분 간격이다. B 실행 그룹은 {len(b):,}개다. A의 214개 파일은 고유 측정 시계열 {ah.full.nunique()}개이며 실제 난수 독립 실행 수는 UNKNOWN이다.

## 공정·고장·관측

Reactor, Condenser, Separator, Stripper, Compressor와 냉각수/원료/밸브 제어가 결합된 공정이다. 주요 변수는 유량, 압력, 수위, 온도, 조성 및 제어입력이다. 53개 원 시뮬레이터 측정/조작 변수의 정의를 02 보고서에 기록했다. A에는 조작변수가 저장되지 않았다.

Fault 1~15의 공개 시나리오에 근거하여 원료·냉각·동역학·밸브 고착 후보를 해석할 수 있다. Fault 16~20은 공개 원인 UNKNOWN으로 유지하고, Fault 21은 문서와 소스 정의가 충돌하므로 원인 평가에서 격리한다. 실제 반응 상위 변수는 04 보고서와 21개 고장 그림을 참고한다. 큰 표준화 변화는 예측 중요도나 실제 원인과 동일하지 않다.

## 품질과 불균형

A 결측 {a.missing.sum()}개, B 결측 {bs.missing.sum()}개다. A 전체 동일 시계열의 추가 중복 {ah.full.duplicated().sum()}개, B 추가 중복 {b.full.duplicated().sum()}개다. Fault 6은 약 17h에서 끝나 고장별 행 수가 다르다. A 고장 Mode 1 실행은 Fault별 파일 10개이나 중복 제거 후 수는 a_class_counts.csv를 따라야 한다. 정상 데이터에는 500h 한 실행이 있어 행 수가 독립 실험 수를 과장한다.

실측 예로 Fault 1에서 xmeas_1, Fault 6에서 xmeas_18, Fault 13에서 xmeas_7이 큰 평균 변화를 보였다. Fault 3/4/5/7/9/11/12/14/15/16/19/21 등은 평균 차이만으로 반응을 판단하면 놓칠 수 있어 시계열과 분산·상관 변화를 함께 확인해야 한다. Mode 1/3/4 정상 운전의 센서별 분포는 a_normal_mode_statistics.csv에 별도 기록했으며 고장 참조를 서로 섞지 않았다.

B는 전체 시계열 중복이 없지만 선두 20샘플 해시가 {b.prefix20.nunique()}개로 묶인다. 정상과 고장 실행이 같은 simulationRun에서 초기 구간을 공유하는 사실을 확인했다. 이 구조 때문에 21,000개 궤적 수와 독립 난수 실험 수를 동일시할 수 없다. Train/Test 간 이 초기 해시 중복은 {shared_prefix_count}개다.

B는 각 분할·고장별 500개 실행 그룹을 제공한다. faultNumber 기준 정상:고장 실행 비율은 1:20이지만 주입 전 구간을 정상으로 재라벨링하면 샘플 불균형이 달라진다. 명목 Training 기준 정상 250,000 + 고장실행 내 200,000 pre-fault = 450,000 정상 대 4,800,000 post-fault, Testing 기준 정상 480,000 + 1,600,000 pre-fault = 2,080,000 정상 대 8,000,000 post-fault다. 이는 공개 경계(20/160) 해석에 따른 산술이며 실제 센서 반응 시점을 뜻하지 않는다.

## Phase 2 설계 권고

현재값 모델은 Logistic Regression, Random Forest, XGBoost 순으로 비교하고, 인과적 trailing 통계 및 Temporal 윈도우는 동일 시점·실행 평가에 맞춘다. 시간 모델은 1D-CNN/GRU로 시작하고 데이터/성능 근거가 생기면 LSTM·Transformer를 비교한다. 이는 구조적 권고이며 이번 분석에서 성능을 검증한 결과가 아니다. 정상 학습 기반 PCA/잔차 기준도 미지 고장 실험의 비교 기준으로 고려할 수 있다.

동일 실행과 중복 시계열은 Train/Validation/Test를 가로질러 나누지 않는다. B 파일별 simulationRun 숫자는 독립적인 전역 ID가 아니다. 같은 run 번호의 서로 다른 Fault를 같은 분할에 묶는 보수적 정책을 우선 검토한다. scaler/특징선택은 Train만, 경보 임계값은 Validation만 사용한다. Test는 재최적화에 사용하지 않는다. 윈도우의 미래 참조와 실행 경계 침범은 자동 검증해야 한다. 현재 테스트는 실제 데이터의 구조와 중복 감사를 검증하며 아직 존재하지 않는 Phase 2 분할의 누수 검증을 대체하지 않는다.

## 진행 판단과 남은 문제

Phase 1의 확보·기술통계·정의 대조 및 보고서를 작성했다. A는 배포 모델의 명목 10h 주입을 기준으로 조건부 전후/편차 시각을 제공했다. 파일별 실제 주입 로그와 진정한 독립성 등 원본 증거가 부족한 항목은 미확인으로 남겼다. A-B 직접 결합은 권고하지 않는다. B 단독 모델 개발은 사용자 승인 후 조건부로 진행 가능하다.

1. A의 같은 Fault 내부 중복 파일과 공유 초기 구간을 묶고 가설 해석 범위를 제한한다. B는 같은 simulationRun의 Fault 간 주입 전 구간이 겹치므로 함께 분할한다.
2. Fault 21의 생성 시점 소스/MEX 및 고장 정의를 확인한다. 미해결이면 제외한다.
3. A 고장 주입 기록과 seed 로그, B 운전 모드/제어기 버전은 미확인으로 관리한다.
4. B 주입 경계 라벨 규칙, 보수적 실행 분할, 지연·경보 episode 규칙을 학습 전에 명시한다.
5. 모델 설명 기여, 관측 반응, 공개 물리 원인을 별도로 보고한다. 대응이 없는 원인 적중률은 계산하지 않는다.

모델 학습은 수행하지 않았다. 사용자 Phase 2 승인 전에는 시작하지 않는다.
''')
    make_notebooks()
    env={'python':sys.version,'platform':platform.platform(),'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','python-calamine','pyreadr','matplotlib','scipy','narwhals']}}
    (ROOT/'reports/sources/environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
    distributions={d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=[str(ROOT/'.deps')])}
    for name in ['numpy','pandas','openpyxl']: distributions[name]=importlib.metadata.version(name)
    (ROOT/'requirements-lock.txt').write_text('\n'.join(f'{n}=={v}' for n,v in sorted(distributions.items(),key=lambda x:x[0].lower()))+'\n',encoding='utf-8')
    print('Reports, figures, notebooks complete',flush=True)

def make_notebooks():
    titles=['01_data_inventory','02_process_understanding','03_fault_scenario_analysis','04_sensor_response_analysis','05_dataset_compatibility']
    reports=['01_dataset_audit.md','02_process_variables.md','03_fault_catalog.md','04_fault_sensor_response.md','05_compatibility_report.md']
    csvs=['a_inventory.csv','variable_dictionary.csv','fault_catalog.csv','a_fault_response.csv','compatibility.csv']
    for title,report,csv in zip(titles,reports,csvs):
        code=f"from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Markdown, Image\nROOT = Path.cwd()\nif not (ROOT / 'reports').exists(): ROOT = ROOT.parent\ndisplay(Markdown((ROOT / 'reports/markdown/{report}').read_text(encoding='utf-8')))\ndisplay(pd.read_csv(ROOT / 'reports/tables/{csv}').head(60))\n"
        cells=[dict(cell_type='markdown',id='intro',metadata={},source=[f'# {title}\n','Phase 1 보고서와 원자료 기반 감사 결과. 전체 재계산: 프로젝트 루트에서 `./run_phase1.ps1`.\n']),dict(cell_type='code',id='report-and-table',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))]
        if title.startswith('04'):
            cells.append(dict(cell_type='code',id='sensor-figures',metadata={},execution_count=None,outputs=[],source=["display(Image(filename=str(ROOT / 'reports/figures/fault_response_overview.png')))\n","for fault in range(1,22):\n","    display(Image(filename=str(ROOT / f'reports/figures/fault_{fault:02d}_analysis.png')))\n"]))
        notebook=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},nbformat=4,nbformat_minor=5)
        (ROOT/'notebooks'/f'{title}.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':build()
