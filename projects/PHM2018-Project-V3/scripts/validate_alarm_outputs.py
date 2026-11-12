"""첫 단계의 실제 이벤트·경보·그림·Notebook 산출물을 독립 대조한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
from PIL import Image
import nbformat

def main():
    summary=pd.read_parquet(TABLES/'alarm_stage1_summary.parquet');events=pd.read_parquet(TABLES/'alarm_stage1_events.parquet');alarms=pd.read_parquet(TABLES/'alarm_stage1_episodes.parquet');links=pd.read_parquet(TABLES/'alarm_stage1_matches.parquet');checks=[]
    keys=['scheme','split','Tool','feature_set','baseline']
    def check(name,ok):
        assert ok,name
        checks.append({'check':name,'passed':bool(ok)})
    for row in summary.itertuples():
        e=events;a=alarms;m=links
        for key in keys:
            e=e[e[key].eq(getattr(row,key))];a=a[a[key].eq(getattr(row,key))];m=m[m[key].eq(getattr(row,key))]
        check('/'.join(str(getattr(row,k)) for k in keys),len(e)==row.recorded_events and len(a)==row.alarm_episodes and e.detected.sum()==row.detected_events and m.event_id.nunique()==row.detected_events and not e.event_id.duplicated().any())
        if len(m):
            z=m.merge(e[['event_id','fault_time']],on='event_id').merge(a[['episode_id','start','start_label']],on='episode_id')
            check('실제 매칭 경계와 선행시간',z.lead_raw.between(60,600).all() and (z.fault_time-z.start).eq(z.lead_raw).all() and z.start_label.ne(3).all())
        if row.observed_support_raw:check('경보 부담 분모 재계산',np.isclose(row.alarm_episodes_per_10000_raw,10000*len(a)/row.observed_support_raw))
    check('holdout 점수화 없음',summary.split.eq('validation').all())
    figures=json.loads((ROOT/'outputs/alarm_stage1_figure_manifest.json').read_text(encoding='utf-8'))
    for f in figures:
        with Image.open(ROOT/f['file']) as im:check(f['file'],im.width>=1000 and im.height>=500)
    book=nbformat.read(ROOT/'notebooks/11_alarm_evaluation.ipynb',as_version=4)
    cells=[c for c in book.cells if c.cell_type=='code']
    check('Notebook 모든 코드 실제 실행',all(c.execution_count is not None for c in cells))
    check('Notebook 오류 없음',not any(o.output_type=='error' for c in cells for o in c.outputs))
    check('Notebook 실제 그림5개 포함',sum('image/png' in o.get('data',{}) for c in cells for o in c.outputs)>=5)
    check('원본 메타데이터 유지',snapshot()==json.loads((ROOT/'outputs/source_snapshot_before.json').read_text(encoding='utf-8')))
    dump_json(ROOT/'outputs/validation/alarm_stage1_output_checks.json',{'checks':checks,'figures':len(figures),'notebook_code_cells':len(cells),'model_training_performed':False})
    print(f'첫 단계 산출물 검증 {len(checks)}개 통과')

if __name__=='__main__':main()
