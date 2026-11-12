"""12백만 Anchor를 청크 단위로 검토해 학습 전 분할/후보 목록을 생성한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
from src.common import *
from src.pretraining import *
from src.query import manifest

def main():
    PRE_DIR.mkdir(parents=True,exist_ok=True)
    events=pd.read_parquet(TABLES/'events.parquet');files=[];counts=defaultdict(lambda:np.zeros(5,dtype=np.int64));represented=defaultdict(set);bounds=[];checks=[]
    categories=defaultdict(set);source_files=[]
    for tool in sorted(events.Tool.unique()):
        source_meta=manifest(tool)
        am=json.loads((CACHE/'analysis'/tool/'manifest.json').read_text(encoding='utf-8'))
        own=events[events.Tool.eq(tool)]
        for scheme in ['temporal','group']:
            for split,start,end in partitions(tool,source_meta['min_time'],source_meta['max_time'],scheme):bounds.append(dict(Tool=tool,scheme=scheme,split=SPLITS[split],start=start,end_exclusive=end))
        for item in am['index']:
            path=CACHE/'analysis'/tool/item['file'];d=pd.read_parquet(path)
            assert d.time.is_monotonic_increasing and d.time.is_unique
            for family in 'ABC':assert np.isfinite(d.loc[d[family+'_available'],schema()[family]]).all().all()
            source_files.append({'file':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            for scheme in ['temporal','group']:
                parts=partitions(tool,source_meta['min_time'],source_meta['max_time'],scheme)
                assignment=pd.DataFrame({'time':d.time})
                for h in CFG['horizons']:
                    split,y=assign_and_label(d,parts,h);assignment['split']=split;assignment[f'y_h{h}']=y
                    for code in range(3):
                        inside=split==code
                        for cohort in ['maximum','common']:
                            for f in 'ABC':
                                avail=d['common_available' if cohort=='common' else f+'_available'].to_numpy()
                                mask=inside&avail;key=(scheme,SPLITS[code],tool,h,cohort,f)
                                counts[key]+=np.array([inside.sum(),mask.sum(),(mask&(y==POS)).sum(),(mask&(y==NEG)).sum(),(mask&np.isin(y,[NA,EXCLUDE])).sum()])
                                for k in FAULTS:
                                    positive=mask&(y==POS)&d[f'label__conservative__{k}__h{h}'].eq(POS).to_numpy()
                                    represented[key].update(d.loc[positive,'next_event_'+k])
                        base=inside&d.A_available.to_numpy()&np.isin(y,[POS,NEG])
                        if h==PRE['default_horizon']:
                            categories[(scheme,SPLITS[code])].update(map(tuple,d.loc[base,['recipe','stage']].to_numpy()))
                out=PRE_DIR/scheme/tool/item['file'];write_parquet(assignment,out)
                files.append({'Tool':tool,'scheme':scheme,'source':str(path.relative_to(ROOT)).replace('\\','/'),'assignment':str(out.relative_to(ROOT)).replace('\\','/'),'source_signature':[path.stat().st_size,path.stat().st_mtime_ns]})
        print(tool+' 분할·유한값·후보 검토 완료',flush=True)
    rows=[]
    for key,v in counts.items():
        scheme,split,tool,h,cohort,f=key
        rows.append(dict(scheme=scheme,split=split,Tool=tool,H=h,cohort=cohort,feature_set=f,guard_pass_anchors=v[0],available_anchors=v[1],positive_anchors=v[2],negative_anchors=v[3],unknown_or_excluded=v[4],positive_events=len(represented[key])))
    summary=pd.DataFrame(rows);save_table(summary,'pretraining_candidates');save_table(pd.DataFrame(bounds),'pretraining_split_boundaries')
    for scheme in ['temporal','group']:
        for h in CFG['horizons']:
            sets=[set().union(*(ids for key,ids in represented.items() if key[0]==scheme and key[1]==split and key[3]==h)) for split in SPLITS]
            assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            checks.append({'check':f'{scheme} H{h} 동일 양성 이벤트의 분할 중복 없음','passed':True})
    shift=[]
    for scheme in ['temporal','group']:
        train=categories[(scheme,'train')]
        for split in SPLITS:
            known=categories[(scheme,split)];shift.append(dict(scheme=scheme,split=split,condition_pairs=len(known),unseen_pairs=len(known-train),basis='H600 / A / 정책상 확정 라벨 / 조합 수 기준'))
    save_table(pd.DataFrame(shift),'pretraining_unseen_conditions')
    aggregate=summary.groupby(['scheme','split','H','cohort','feature_set'])[['positive_anchors','negative_anchors','positive_events']].sum().reset_index()
    aggregate['both_classes']=aggregate.positive_anchors.gt(0)&aggregate.negative_anchors.gt(0)
    save_table(aggregate,'pretraining_readiness')
    dump_json(PRE_DIR/'manifest.json',{'config_hash':hashlib.sha256(json.dumps(PRE,sort_keys=True).encode()).hexdigest(),'policy':'conservative','target':'any','files':files})
    dump_json(ROOT/'outputs/validation/pretraining_checks.json',{'checks':checks,'finite_available_features_checked':True,'source_anchor_hashes':source_files,'model_trained':False})
    print(aggregate[aggregate.H.eq(600)&aggregate.cohort.eq('maximum')].to_string(index=False))

if __name__=='__main__':main()
