"""실제 후보 로더의 반환 분모와 경계 계약을 점검한다. 모델 학습 없음."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pretraining import *

def main():
    summary=pd.read_parquet(TABLES/'pretraining_readiness.parquet');checks=[]
    for scheme in ['temporal','group']:
        for split in SPLITS:
            n=pos=0
            for x,y,meta in iter_batches(scheme,split,'A',600,'maximum'):
                assert list(x.columns)==schema()['A'];assert len(x)==len(y)==len(meta)
                assert y.isin([NEG,POS]).all()
                n+=len(y);pos+=int(y.eq(POS).sum())
            row=summary[summary.scheme.eq(scheme)&summary.split.eq(split)&summary.H.eq(600)&summary.feature_set.eq('A')&summary.cohort.eq('maximum')].iloc[0]
            assert n==row.positive_anchors+row.negative_anchors and pos==row.positive_anchors
            checks.append({'scheme':scheme,'split':split,'rows':n,'positive':pos,'passed':True})
        for family in 'ABC':
            batch=next(iter_batches(scheme,'train',family,600,'common'))
            assert list(batch[0].columns)==schema()[family]
    assert snapshot()==json.loads((ROOT/'outputs/source_snapshot_before.json').read_text(encoding='utf-8'))
    dump_json(ROOT/'outputs/validation/pretraining_loader_checks.json',{'full_loader_checks':checks,'common_feature_schemas_checked':6,'raw_metadata_unchanged':True})
    print('6개 전체 후보 로더 분모 및 6개 공통 특징 스키마 검증 통과')

if __name__=='__main__':main()
