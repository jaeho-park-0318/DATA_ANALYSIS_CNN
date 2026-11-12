"""기존 대표2대 Parquet만 재사용해 현재값/이력별 연구용 후보를 집계한다.

원본 CSV 읽기, 라벨 재생성, 학습, 승인 플래그 변경을 수행하지 않는다.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import psutil

from src.common import CONFIG, DEST, FAULTS, OUT, ROOT, SENSORS, digest, pl, save

RESEARCH = ROOT / 'data/processed/research_cohort_v1'
INTEGRATED = 'FlowCool 세 유형 중 하나라도 발생 (통합)'
WINDOWS = [0, 60, 300, 900]


def current_input_mask():
    return (pl.col('current_observed') & pl.col('sensor_age').is_between(0, CONFIG['gap_threshold'])
            & ~pl.col('state_conflict')
            & pl.all_horizontal([pl.col(s).is_not_null() & pl.col(s).is_finite() for s in SENSORS])
            & pl.col('recipe').is_not_null() & pl.col('stage').is_not_null()).fill_null(False)


def history_mask(w):
    mask = current_input_mask()
    if w:
        mask = mask & (pl.col('history_span_raw') >= w)
    return mask.fill_null(False)


def event_groups(times, horizon, history):
    """[고장-H-history,고장]의 연결성분 수. 통계적 독립성을 보증하지 않는다."""
    ends = sorted(set(float(t) for t in times))
    groups, previous_end = 0, None
    for end in ends:
        if previous_end is None or end - horizon - history > previous_end:
            groups += 1
        previous_end = end
    return groups


def target_support(code, horizon):
    if code != 'any':
        return pl.col(f'ttf_verified_source_{code}').fill_null(False)
    minimum = pl.min_horizontal([pl.col(f'next_event_time_{c}') for c in FAULTS])
    next_verified = pl.any_horizontal([
        (pl.col(f'next_event_time_{c}') == minimum) & pl.col(f'ttf_verified_source_{c}') for c in FAULTS])
    all_verified = pl.all_horizontal([pl.col(f'ttf_verified_source_{c}') for c in FAULTS])
    return pl.when(pl.col(f'candidate_any_h{horizon}raw') == 1).then(next_verified).otherwise(all_verified).fill_null(False)


def research_mask(code, horizon):
    label = f'candidate_{code}_h{horizon}raw'
    # 다른 유형의 고장이 더 임박한 구간도 첫 연구용 코호트에서 공통 제외한다.
    near_any = pl.any_horizontal([
        (pl.col(f'next_event_time_{c}') - pl.col('time')).is_between(0, CONFIG['minimum_lead'], closed='left')
        for c in FAULTS]).fill_null(False)
    return (pl.col(label).is_not_null() & target_support(code, horizon) & ~near_any).fill_null(False)


def positive_events(frame, code, horizon):
    pos = frame.filter((pl.col(f'candidate_{code}_h{horizon}raw') == 1) & research_mask(code, horizon))
    if code != 'any':
        return [(FAULTS[code], float(t)) for t in pos[f'next_event_time_{code}'].drop_nulls().unique().to_list()]
    result = []
    minimum = pl.min_horizontal([pl.col(f'next_event_time_{c}') for c in FAULTS])
    for c, name in FAULTS.items():
        result.extend((name, float(t)) for t in pos.filter(pl.col(f'next_event_time_{c}') == minimum)[f'next_event_time_{c}'].drop_nulls().unique().to_list())
    return sorted(set(result))


def protected_paths():
    files = [ROOT / 'config/preprocessing.json', OUT / 'release_status.json', OUT / 'run_status.json',
             DEST / 'feature_schema.json', OUT / 'audit_summary.json']
    files += [DEST / f'{tool}.parquet' for tool in CONFIG['pilot_tools']]
    return [p for p in files if p.exists()]


def run():
    started = time.monotonic()
    before = {str(p): digest(p) for p in protected_paths()}
    RESEARCH.mkdir(parents=True, exist_ok=True)
    rows, coverage, event_rows = [], [], []
    memory = []
    for tool in CONFIG['pilot_tools']:
        if psutil.virtual_memory().available < CONFIG['min_available_ram_gib'] * 1024**3:
            raise RuntimeError('가용 RAM 정책 미충족; 다음 장비 집계 중단')
        # 필요한 열만 읽고 한 장비씩 집계한다. Rolling을 Anchor에서 재계산하지 않는다.
        columns = ['tool', 'time', 'sensor_time', 'sensor_age', 'current_observed', 'state_conflict',
                   'stage', 'recipe', 'history_span_raw', 'maintenance_status', 'approved_for_training'] + SENSORS
        columns += [f'next_event_time_{c}' for c in FAULTS]
        columns += [f'ttf_verified_source_{c}' for c in FAULTS]
        columns += [f'candidate_{c}_h{h}raw{suffix}' for c in [*FAULTS, 'any'] for h in CONFIG['horizons']
                    for suffix in ['', '_excluded_lt60']]
        frame = pl.read_parquet(DEST / f'{tool}.parquet', columns=columns)
        assert not frame['approved_for_training'].any()
        valid = frame.filter(current_input_mask())
        membership = valid.select('tool', 'time', 'maintenance_status', 'approved_for_training')
        membership = membership.with_columns(*[valid.select(history_mask(w)).to_series().alias(f'input_history_{w}raw') for w in WINDOWS])
        membership = membership.with_columns(*[
            valid.select(research_mask(c, 600)).to_series().alias(f'research_label_supported::{name}')
            for c, name in [*FAULTS.items(), ('any', INTEGRATED)]])
        # 라벨별 선택 인덱스만 저장한다. 입력 센서·기존 후보 파일을 복제하거나 덮어쓰지 않는다.
        membership.write_parquet(RESEARCH / f'{tool}_membership.parquet', compression='zstd')
        for w in WINDOWS:
            cohort = frame.filter(history_mask(w))
            coverage.append({'tool': tool, 'history_raw': w, 'grid_anchors': frame.height,
                             'input_anchors': cohort.height, 'current_input_anchors': valid.height,
                             'dropped_vs_current': valid.height - cohort.height,
                             'stage_zero_anchors': cohort.filter(pl.col('stage') == 0).height})
            for h in CONFIG['horizons']:
                for code, name in [*FAULTS.items(), ('any', INTEGRATED)]:
                    label = f'candidate_{code}_h{h}raw'
                    excluded = pl.col(label + '_excluded_lt60')
                    chosen = cohort.filter(research_mask(code, h))
                    events = positive_events(cohort, code, h)
                    for fault_name, event_time in events:
                        event_rows.append({'tool': tool, 'target': name, 'horizon_raw': h, 'history_raw': w,
                                           'fault_name': fault_name, 'event_time': event_time})
                    cache_events = pl.read_parquet(OUT / 'audit' / f'{tool}_events.parquet')
                    known_events = {(e['canonical_name'], float(e['time'])) for e in cache_events.iter_rows(named=True) if not e['tagged']}
                    assert set(events) <= known_events
                    r = {'tool': tool, 'target': name, 'horizon_raw': h, 'history_raw': w,
                         'input_anchors': cohort.height,
                         'positive': cohort.filter(pl.col(label) == 1).height,
                         'negative': cohort.filter(pl.col(label) == 0).height,
                         'unknown': cohort.filter(pl.col(label).is_null() & ~excluded).height,
                         'excluded_lt60': cohort.filter(excluded).height,
                         'research_positive': chosen.filter(pl.col(label) == 1).height,
                         'research_negative': chosen.filter(pl.col(label) == 0).height,
                         'research_known': chosen.height, 'research_not_selected': cohort.height - chosen.height,
                         'research_post_fault_end_unknown': chosen.filter(pl.col('maintenance_status') == 'post_fault_end_unknown').height,
                         'positive_unique_events': len(events), 'positive_time_clusters': len(set(t for _, t in events)),
                         'nonoverlapping_event_groups': event_groups([t for _, t in events], h, w),
                         'candidate_known_not_research': cohort.filter(pl.col(label).is_not_null()).height - chosen.height}
                    assert sum(r[k] for k in ['positive', 'negative', 'unknown', 'excluded_lt60']) == cohort.height
                    rows.append(r)
        memory.append({'tool': tool, 'rss_gib_after_aggregation': psutil.Process().memory_info().rss / 1024**3,
                       'available_ram_gib': psutil.virtual_memory().available / 1024**3})
        print(f"연구 후보 집계 {tool}: 현재입력 {valid.height:,}", flush=True)
        del frame, valid, cohort, chosen, membership
    stats = pl.DataFrame(rows)
    sums = ['input_anchors','positive','negative','unknown','excluded_lt60','research_positive','research_negative',
            'research_known','research_not_selected','research_post_fault_end_unknown','positive_unique_events',
            'positive_time_clusters','nonoverlapping_event_groups','candidate_known_not_research']
    combined = stats.group_by('target','horizon_raw','history_raw').agg(*[pl.col(c).sum() for c in sums],
        (pl.col('research_positive') > 0).sum().alias('tools_with_positive'),
        (pl.col('research_negative') > 0).sum().alias('tools_with_negative'),
        ((pl.col('research_positive') > 0) & (pl.col('research_negative') > 0)).sum().alias('tools_with_both_classes'),
        pl.col('positive_unique_events').min().alias('min_events_per_tool'),
        pl.col('nonoverlapping_event_groups').min().alias('min_event_groups_per_tool')).sort('horizon_raw','history_raw','target')
    stats.write_parquet(RESEARCH / 'counts_by_tool.parquet')
    combined.write_parquet(RESEARCH / 'counts_combined.parquet')
    pl.DataFrame(coverage).write_parquet(RESEARCH / 'input_coverage.parquet')
    pl.DataFrame(event_rows).write_parquet(RESEARCH / 'positive_event_membership.parquet')
    after = {str(p): digest(p) for p in protected_paths()}
    assert before == after, '기존 진단/설정/승인 파일 변경'
    summary = {'status': 'RESEARCH_CANDIDATE_SPEC_ONLY', 'models_trained': 0, 'raw_files_read': 0,
               'input_diagnostic_files': [str(DEST / f'{t}.parquet') for t in CONFIG['pilot_tools']],
               'protected_sha256_before': before, 'protected_sha256_after': after,
               'target_name_mapping': FAULTS, 'integrated_name': INTEGRATED,
               'windows_raw': WINDOWS, 'counts': combined.to_dicts(), 'coverage': coverage,
               'resources': memory, 'resource_scope': '장비별 집계 직후 샘플; 실행중 최대치 측정 아님',
               'elapsed_seconds': round(time.monotonic() - started, 3)}
    save(OUT / 'research_cohort_summary.json', summary)
    save(RESEARCH / 'selection_spec.json', {'status': 'RESEARCH_CANDIDATE_SPEC_ONLY', 'operational_approval_changed': False,
         'current_input': '현재관측=true, 지연0~12raw, 센서6 유한, recipe/stage존재, 상태상충없음',
         'history': '현재입력 조건 + 동일 연속 세그먼트 history_span_raw>=0/60/300/900; 밀도100% 보장 아님',
         'target': '기존 candidate_* 재사용, 라벨 0/1, 해당 유형 TTF시점근거 일치, 어떤 유형이든 60미만이면 제외',
         'integrated': '양성은 가장 가까운 다음고장 유형의 TTF근거, 음성은 세 유형 모두의 근거 필요',
         'event_key': 'tool + 원문 정규화 fault_name + event_time; 통합은 가장 이른 시각(동시유형 보존)',
         'time_group': '[event-H-history,event] 중첩 연결성분; 통계적 독립/정비사이클 확인 아님',
         'history60_features_computed': False, 'horizon_raw': 600})
    return summary


if __name__ == '__main__':
    run()
