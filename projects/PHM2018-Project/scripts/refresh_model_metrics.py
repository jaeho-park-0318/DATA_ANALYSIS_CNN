"""고정된 점수·임계값을 유지하고 경보 집계 정책을 모든 실험에 동일 적용한다."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from src.model_common import OUT,read_json,save_json
from src.evaluate_models import operational_metrics


def main():
    for path in sorted((OUT/"experiments").glob("*/*/result.json")):
        result=read_json(path)
        for split in ('validation','evaluation'):
            frame=pd.read_parquet(path.parent/f"{split}_predictions.parquet")
            result['metrics'][split].update(operational_metrics(frame,result['threshold'],result['horizon'],result['scheme'],split))
        result['alarm_policy']='동일 Tool·관측 세그먼트·Stage=0/>0 상태의 연속 양성; 72 초과 간격/음성에서 분리; H마다 재발행'
        save_json(path,result)
    for path in sorted((OUT/"prospective").glob("*/result.json")):
        result=read_json(path);frame=pd.read_parquet(path.parent/'predictions.parquet')
        result['metrics'].update(operational_metrics(frame,result['threshold'],600,result['scheme'],'evaluation'))
        save_json(path,result)
    print('고정 예측 점수의 경보 지표 재집계 완료',flush=True)


if __name__=='__main__':main()
