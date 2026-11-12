"""학습 전 별도 순차 구현으로 실제 시각의 경보 에피소드를 대조한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import *
from src.pretraining import iter_scoring_batches
from src.alarm_evaluation import alarm_episodes,EPISODE_COLUMNS

def reference(d,threshold):
    rows=[]
    for tool,g in d.groupby('Tool'):
        current=None;prev=None;number=0
        for r in g.sort_values('time').itertuples():
            on=r.score>=threshold
            if on:
                new=current is None or prev is None or r.time-prev.time!=60 or r.segment_id!=prev.segment_id
                if new:
                    number+=1;current=dict(Tool=tool,episode_id=f'{tool}-alarm-{number:06d}',start=r.time,end=r.time,anchors=0,start_label=r.audit_label,contains_unknown=False,contains_exclude=False,segment_id=r.segment_id);rows.append(current)
                current['end']=r.time;current['anchors']+=1;current['contains_unknown'] |= r.audit_label==2;current['contains_exclude'] |= r.audit_label==3
            else:current=None
            prev=r
    return pd.DataFrame(rows,columns=EPISODE_COLUMNS)

def main():
    checks=[]
    for tool in ['01M01','03M02','10M02']:
        chunks=[]
        for x,m in iter_scoring_batches('temporal','validation','A',tool=tool):
            chunks.append(m)
            if sum(map(len,chunks))>=15000:break
        d=pd.concat(chunks,ignore_index=True).iloc[:15000].copy()
        # 실제 시각에 적용한 시험용 점수이며 모델 성능 보고에 사용하지 않는다.
        for rule in ['zero','one','cyclic']:
            d['score']=0. if rule=='zero' else 1. if rule=='one' else (np.arange(len(d))%17)/16
            for threshold in [.1,.5,.9]:
                actual=alarm_episodes(d,threshold);expected=reference(d,threshold)
                pd.testing.assert_frame_equal(actual.reset_index(drop=True),expected.reset_index(drop=True),check_dtype=False)
                checks.append(dict(Tool=tool,rule=rule,threshold=threshold,anchors=len(d),episodes=len(actual),passed=True))
    dump_json(ROOT/'outputs/validation/alarm_independent_audit.json',{'checks':checks,'test_scores_are_not_model_predictions':True})
    print(f'독립 경보 구현 대조 {len(checks)}개 통과')

if __name__=='__main__':main()
