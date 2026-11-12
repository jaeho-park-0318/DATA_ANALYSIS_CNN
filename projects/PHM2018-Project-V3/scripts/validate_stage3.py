"""공통 코호트 학습 결과와 동일 경보 상한 비교를 검증한다."""
from validate_stage2 import main
from src.common import *

if __name__=='__main__':
    main(stage='stage3')
    checks=json.loads((ROOT/'outputs/validation/stage3_checks.json').read_text(encoding='utf-8'))
    common=json.loads((ROOT/'outputs/validation/stage3_common_contract.json').read_text(encoding='utf-8'));assert all(c['passed'] for c in common['checks']);checks['checks']+=common['checks']
    p=pd.read_parquet(TABLES/'stage3_budget_points.parquet');c=pd.read_parquet(TABLES/'stage3_alarm_curve.parquet')
    assert c.available_events.nunique()==c.observed_support_raw.nunique()==c.available_anchors.nunique()==1
    for r in p.itertuples():
        candidates=c[c.run_id.eq(r.run_id)&c.alarm_episodes_per_10000_raw.le(r.budget_per_10000_raw)]
        assert r.detected_events==candidates.detected_events.max()
        assert r.alarm_episodes_per_10000_raw==candidates[candidates.detected_events.eq(r.detected_events)].alarm_episodes_per_10000_raw.min()
        checks['checks'].append(dict(check=r.run_id+f' 경보 상한 {r.budget_per_10000_raw}에서 최대 검출·동률 최소 부담',passed=True))
    d=pd.read_parquet(TABLES/'stage3_paired_deltas.parquet')
    assert np.array_equal(d.detected_after-d.detected_before,d.gained_events-d.lost_events)
    assert np.allclose(d.recall_delta,(d.detected_after-d.detected_before)/c.available_events.iloc[0])
    checks['checks'].append(dict(check='대응 비교 검출 증감·같은 이벤트 분모',passed=True))
    dump_json(ROOT/'outputs/validation/stage3_checks.json',checks);print('TOTAL STAGE3 CHECKS:',len(checks['checks']))
