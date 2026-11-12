"""저장된 단일 Holdout 예측의 불리언 집계만 복구한다. 추론/학습 없음."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datetime import datetime, timezone
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from src.common import *
from src.final_evaluation import at_threshold

OUT = ROOT / 'outputs/stage6'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(frozen):
    for name, digest in frozen['sha256'].items():
        assert sha(ROOT / name) == digest, name


def main():
    frozen = json.loads((OUT / 'frozen_manifest.json').read_text(encoding='utf-8'))
    state = json.loads((OUT / 'holdout_access.json').read_text(encoding='utf-8'))
    assert state['status'] == 'opened' and state['full_prediction_passes'] == 1
    verify(frozen)
    prediction_hash = sha(OUT / 'holdout_predictions.parquet')
    recovery = dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
                    reason='빈 장비 이벤트 표를 concat한 후 tagged가 object bool이 되어 ~ 연산 실패',
                    correction='집계 이벤트 tagged/available/detected 열을 bool로 명시 변환',
                    postprocessing_only=True, additional_prediction_passes=0,
                    model_refit=False, threshold_search=False,
                    input_prediction_sha256=prediction_hash,
                    frozen_manifest_sha256=sha(OUT / 'frozen_manifest.json'),
                    original_failure_log_sha256=sha(ROOT / 'outputs/stage6_holdout_evaluation.log'),
                    recovery_script_sha256=sha(Path(__file__)))
    marker = OUT / 'postprocessing_recovery.json'
    with marker.open('x', encoding='utf-8') as handle:
        json.dump(recovery, handle, ensure_ascii=False, indent=2)
    p = pd.read_parquet(OUT / 'holdout_predictions.parquet')
    supports = json.loads((OUT / 'holdout_exposure.json').read_text(encoding='utf-8'))
    bounds = pd.read_parquet(TABLES / 'pretraining_split_boundaries.parquet')
    b = bounds[bounds.scheme.eq('temporal') & bounds.split.eq('holdout')].set_index('Tool')
    events = pd.read_parquet(TABLES / 'events.parquet')
    events = events[events.fault_time.ge(events.Tool.map(b.start)) & events.fault_time.lt(events.Tool.map(b.end_exclusive))].copy()
    summaries, evs, alarms, links, tools, uncertainty = [], [], [], [], [], []
    pair_ids = sorted(events.Tool.str[:2].unique())
    ix = np.random.default_rng(42).integers(0, len(pair_ids), (2000, len(pair_ids)))
    for point in frozen['operating_points']:
        row, e, a, l, t = at_threshold(p, events, point['threshold'], supports)
        for col in ['tagged', 'available', 'detected']:
            assert e[col].notna().all() and e[col].isin([True, False]).all()
            e[col] = e[col].astype(bool)
        row.update(operating_point=point['operating_point'],
                   validation_budget_per_10000_raw=point['validation_budget_per_10000_raw'], family=frozen['family'])
        summaries.append(row)
        for df, dest in [(e, evs), (a, alarms), (l, links), (t, tools)]:
            df['operating_point'] = point['operating_point']
            dest.append(df)
        z = e[~e.tagged].copy()
        z['pair'] = z.Tool.str[:2]
        counts = z.groupby('pair').agg(detected=('detected', 'sum'), available=('available', 'sum'), recorded=('event_id', 'size')).reindex(pair_ids, fill_value=0)
        sample = counts.to_numpy()[ix].sum(axis=1)
        for den, j in [('available', 1), ('recorded', 2)]:
            good = sample[:, j] > 0
            values = sample[good, 0] / sample[good, j]
            uncertainty.append(dict(operating_point=point['operating_point'], denominator=den,
                                    ci_low=np.quantile(values, .025), ci_high=np.quantile(values, .975),
                                    valid_samples=int(good.sum()), lead_p10=z.loc[z.detected, 'lead_raw'].quantile(.1),
                                    lead_median=z.loc[z.detected, 'lead_raw'].median()))
        print(json.dumps(row, ensure_ascii=False), flush=True)
    for name, data in [('summary', pd.DataFrame(summaries)), ('events', pd.concat(evs, ignore_index=True)),
                       ('episodes', pd.concat(alarms, ignore_index=True)), ('matches', pd.concat(links, ignore_index=True)),
                       ('by_tool', pd.concat(tools, ignore_index=True)), ('uncertainty', pd.DataFrame(uncertainty))]:
        save_table(data, 'stage6_holdout_' + name)
    known = p.audit_label.isin([0, 1])
    y, score = p.loc[known, 'audit_label'], p.loc[known, 'score']
    metrics = dict(all_anchors=len(p), known_anchors=int(known.sum()), positive=int(y.eq(1).sum()),
                   AP_known=float(average_precision_score(y, score)) if y.nunique() == 2 else None,
                   ROC_AUC_known=float(roc_auc_score(y, score)) if y.nunique() == 2 else None,
                   log_loss_known=float(log_loss(y, score, labels=[0, 1])),
                   label_counts={str(k): int(v) for k, v in p.audit_label.value_counts().items()})
    dump_json(OUT / 'holdout_classification.json', metrics)
    verify(frozen)
    assert sha(OUT / 'holdout_predictions.parquet') == prediction_hash
    assert snapshot() == json.loads((ROOT / 'outputs/source_snapshot_before.json').read_text(encoding='utf-8'))
    recovery.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), prediction_unchanged=True, frozen_files_unchanged=True)
    dump_json(marker, recovery)
    state.update(status='complete', completed_at_utc=recovery['completed_at_utc'],
                 prediction_sha256=prediction_hash,
                 summary_sha256=sha(TABLES / 'stage6_holdout_summary.parquet'),
                 postprocessing_recovery='outputs/stage6/postprocessing_recovery.json',
                 additional_prediction_passes=0)
    dump_json(OUT / 'holdout_access.json', state)
    print('저장 예측 집계 복구 완료. 추가 추론 0회, 재학습/임계값 탐색 없음.', flush=True)


if __name__ == '__main__':
    main()
