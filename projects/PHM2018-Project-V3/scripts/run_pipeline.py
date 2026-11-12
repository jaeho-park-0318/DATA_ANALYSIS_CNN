"""장비별 체크포인트를 재사용하는 로컬 전체 실행. 모델 학습 없음."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import time
from src.common import *
from src.ingest import ingest_tool
from src.analyze import analyze_tool,consolidate

def main():
    events=events_from_raw();save_table(events,'events')
    before=snapshot();dump_json(ROOT/'outputs/source_snapshot_before.json',before)
    dump_json(ROOT/'outputs/environment.json',environment());started=time.monotonic();journal=[]
    for i,tool in enumerate(sorted(events.Tool.unique())):
        ingest_tool(tool,events);analyze_tool(tool,events)
        elapsed=time.monotonic()-started
        journal.append({'Tool':tool,'completed':i+1,'total':20,'elapsed_seconds':elapsed})
        dump_json(ROOT/'outputs/run_progress.json',journal)
        print(f'PROGRESS {i+1}/20 {tool} elapsed={elapsed:.1f}s',flush=True)
    consolidate(events)
    if before!=snapshot():raise AssertionError('원본 메타데이터 변경')
    print('COMPLETE: 20 tools; no model training',flush=True)

if __name__=='__main__':main()
