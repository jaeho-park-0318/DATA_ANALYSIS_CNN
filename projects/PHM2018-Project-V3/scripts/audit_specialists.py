"""후반 평가를 읽지 않고 학습/검증 데이터 양으로 최대3대 선정."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.specialist_experiment import *


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'selection.json').exists(),'선정 기록 이미 존재'
    verify_previous()
    paths=list((ROOT/'outputs/stage6').glob('*'))+list(TABLES.glob('stage6_*'))
    dump_json(OUT/'prior_artifacts.json',{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in paths if p.is_file()})
    dump_json(OUT/'audit_started.json',dict(started_at_utc=utc(),protocol_sha256=digest(PROTOCOL_PATH),later_partition_read=False))
    tools=pd.read_parquet(TABLES/'tool_summary.parquet').Tool.sort_values().tolist();rows=[]
    for tool in tools:
        row={'Tool':tool}
        for split in ['train','validation']:
            x,m=load(split,tool);counts=event_input_counts(m,events_for(tool,split));counts.update(positive_rows=0 if m is None else int(m.audit_label.eq(1).sum()),negative_rows=0 if m is None else int(m.audit_label.eq(0).sum()),all_input_rows=0 if m is None else len(m))
            row.update({split+'_'+k:v for k,v in counts.items()})
            del x,m
        failures=[f'{k}={row[k]} < {v}' for k,v in PROTOCOL['eligibility'].items() if row[k]<v]
        row.update(eligible=not failures,reason='조건 충족' if not failures else '; '.join(failures));rows.append(row)
        print(tool,'train/validation 고장군집',row['train_positive_clusters'],row['validation_positive_clusters'],'자격',row['eligible'],flush=True)
    d=pd.DataFrame(rows);rank=d[d.eligible].sort_values(['train_positive_clusters','train_positive_rows','Tool'],ascending=[False,False,True]);selected=rank.head(PROTOCOL['maximum_tools']).Tool.tolist();d['selected']=d.Tool.isin(selected)
    save_table(d,'stage7_eligibility')
    dump_json(OUT/'selection.json',dict(selected_at_utc=utc(),selected_tools=selected,eligible_tools=rank.Tool.tolist(),criteria=PROTOCOL['eligibility'],protocol_sha256=digest(PROTOCOL_PATH),eligibility_table_sha256=digest(TABLES/'stage7_eligibility.parquet'),later_partition_used_for_selection=False))
    verify_previous();print('SELECTED',selected,flush=True)


if __name__=='__main__':main()
