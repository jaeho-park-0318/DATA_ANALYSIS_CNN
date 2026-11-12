"""실제 원본/파생 표의 정합성 검증. 소형 합성 시험과 구분한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from src.common import *
from src.query import read_window,manifest

def main():
    tool=pd.read_parquet(TABLES/'tool_summary.parquet');labels=pd.read_parquet(TABLES/'label_status_summary.parquet');features=pd.read_parquet(TABLES/'feature_availability_summary.parquet');events=pd.read_parquet(TABLES/'events.parquet');ex=pd.read_parquet(TABLES/'event_exclusion_reasons.parquet')
    checks=[]
    def check(name,value):
        checks.append({'check':name,'passed':bool(value)})
        if not value:raise AssertionError(name)
    check('이벤트 ID 중복 없음',events.event_id.is_unique)
    check('원본 행 = 정규화 시각 + 합쳐진 중복 행',(tool.raw_rows==tool.normalized_rows+tool.duplicate_rows).all())
    for r in tool.itertuples():
        q=labels[labels.Tool.eq(r.Tool)]
        check(r.Tool+' 원본 라벨 합계',q[q.sample_level.eq('raw_rows')].denominator.eq(r.raw_rows).all())
        check(r.Tool+' Anchor 라벨 합계',q[q.sample_level.eq('anchor')].denominator.eq(r.anchors).all())
        m=manifest(r.Tool);check(r.Tool+' TTF 원본 검증',sum(m['ttf_mismatch'].values())==0)
    check('라벨 네 상태의 합',(labels.positive+labels.negative+labels.NA+labels.excluded==labels.denominator).all())
    check('특징 네 상태의 합',(features.positive_anchors+features.negative_anchors+features.NA_anchors+features.excluded_anchors==features.available_anchors).all())
    common=features[features.cohort.eq('common')]
    check('공통 코호트는 A/B/C 분모 동일',common.groupby(['Tool','policy','H','fault_type']).available_anchors.nunique().eq(1).all())
    check('공통 코호트는 제외 사유 분모도 동일',common.groupby(['Tool','policy','H','fault_type']).history_gap_anchors.nunique().eq(1).all())
    check('이벤트 보존은 양성 Anchor와 일치',(ex.preserved==(ex.usable_positive_anchors>0)).all())
    check('이벤트/정책/H/특징 중복 없음',not ex.duplicated(['event_id','policy','H','feature_set']).any())
    stages=pd.read_parquet(TABLES/'preprocessing_preservation.parquet')
    check('전처리 모든 단계에 입력 수와 제외 사유 기록',stages.input_rows.notna().all() and stages.exclusion_reason.str.len().gt(0).all())
    check('같은 단위 단계의 입력-출력=제외',stages.loc[~stages.denominator_changed].eval('input_rows - output_rows == excluded_rows').all())
    check('단위 변경을 행 삭제로 계산하지 않음',stages.loc[stages.denominator_changed,'excluded_rows'].isna().all())
    for _,g in stages.groupby(['Tool','policy','H','feature_set']):
        if not g.sort_values('stage_order').events_with_eligible_history.diff().dropna().le(0).all():raise AssertionError('가용성 단계에서 이벤트 수 증가')
    check('전처리 독립 분기별 가용 이벤트 단조 감소',True)
    for t in ['01M01','03M02']:
        if t not in tool.Tool.values:continue
        m=manifest(t);start=m['min_time']+1000;end=start+600
        d,a=read_window(t,start,end,'raw');check(t+' 요청 범위만 반환',d.empty or d.time.between(start,end).all())
        check(t+' 관련 없는 파일 로딩 안 함',a['files_selected']<a['files_total'])
        start=(m['min_time']+m['max_time'])//2
        d,a=read_window(t,start+.25,start+600.75,'raw');check(t+' 큰 원시시간의 소수 경계 조회',d.empty or d.time.between(start+.25,start+600.75).all())
        d,a=read_window(t,m['max_time']+100,m['max_time']+200,'raw');check(t+' 데이터 없는 범위',d.empty and a['files_selected']==0)
    from src.detail import get_detail
    empty=get_detail('01M01',0,100)
    check('빈 상세 구간을 오류나 실제0 센서로 표시하지 않음',empty['empty'] and empty['raw_rows']==0 and all(not v['raw'] for v in empty['series'].values()))
    check('빈 상세 구간이 표준 JSON으로 반환됨',bool(json.dumps(empty,allow_nan=False)))
    before=json.loads((ROOT/'outputs/source_snapshot_before.json').read_text(encoding='utf-8'))
    check('원본 65개 파일 크기/수정시각 보존',before==snapshot())
    dump_json(ROOT/'outputs/validation/actual_checks.json',{'tools':len(tool),'checks':checks,'full_file_content_hash_checked':False})
    print(f'실제 데이터 검증 {len(checks)}개 통과, 장비 {len(tool)}대')

if __name__=='__main__':main()
